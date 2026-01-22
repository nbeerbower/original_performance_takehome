"""
# Anthropic's Original Performance Engineering Take-home (Release version)
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle: dict):
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel within real constraints:
        load:2, valu:6, alu:12, store:2, flow:1
        """
        n_chunks = batch_size // VLEN  # 32 chunks

        # =========================================================
        # PHASE 1: ALLOCATE ALL SCRATCH FIRST
        # =========================================================

        # Header variables
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Scalar temps
        tmp = [self.alloc_scratch(f"tmp{i}") for i in range(8)]

        # Vector registers
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)

        # Vector constants
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Hash stage vector constants
        v_hash = []
        for hi in range(len(HASH_STAGES)):
            v_c1 = self.alloc_scratch(f"v_hc1_{hi}", VLEN)
            v_c3 = self.alloc_scratch(f"v_hc3_{hi}", VLEN)
            v_hash.append((v_c1, v_c3))

        # Scalar constants
        zero_c = self.alloc_scratch("zero_c")
        one_c = self.alloc_scratch("one_c")
        two_c = self.alloc_scratch("two_c")
        vlen_c = self.alloc_scratch("vlen_c")
        n_chunks_c = self.alloc_scratch("n_chunks_c")
        rounds_c = self.alloc_scratch("rounds_c")

        # Hash scalar constants
        hash_c = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.alloc_scratch(f"hc1_{hi}")
            c3 = self.alloc_scratch(f"hc3_{hi}")
            hash_c.append((c1, c3, val1, val3))

        # Loop control
        chunk_i = self.alloc_scratch("chunk_i")
        round_i = self.alloc_scratch("round_i")
        cond = self.alloc_scratch("cond")

        # =========================================================
        # PHASE 2: LOAD CONSTANTS
        # =========================================================

        # Load basic scalar constants (2 per cycle)
        self.add_bundle({"load": [("const", zero_c, 0), ("const", one_c, 1)]})
        self.add_bundle({"load": [("const", two_c, 2), ("const", vlen_c, VLEN)]})
        self.add_bundle({"load": [("const", n_chunks_c, n_chunks), ("const", rounds_c, rounds)]})

        # Load hash constants
        for hi, (c1, c3, val1, val3) in enumerate(hash_c):
            self.add_bundle({"load": [("const", c1, val1), ("const", c3, val3)]})

        # Load header values
        for i in range(0, len(init_vars), 2):
            self.add_bundle({"load": [("const", tmp[0], i), ("const", tmp[1], i + 1 if i + 1 < len(init_vars) else 0)]})
            loads = [("load", self.scratch[init_vars[i]], tmp[0])]
            if i + 1 < len(init_vars):
                loads.append(("load", self.scratch[init_vars[i+1]], tmp[1]))
            self.add_bundle({"load": loads})

        # Broadcast vector constants
        self.add_bundle({"valu": [
            ("vbroadcast", v_zero, zero_c),
            ("vbroadcast", v_one, one_c),
            ("vbroadcast", v_two, two_c),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ]})

        # Broadcast hash constants (6 valu slots, can do 6 broadcasts per cycle)
        for hi in range(0, len(HASH_STAGES), 3):
            vops = []
            for j in range(min(3, len(HASH_STAGES) - hi)):
                v_c1, v_c3 = v_hash[hi + j]
                c1, c3, _, _ = hash_c[hi + j]
                vops.append(("vbroadcast", v_c1, c1))
                vops.append(("vbroadcast", v_c3, c3))
            self.add_bundle({"valu": vops})

        self.add("flow", ("pause",))

        # =========================================================
        # PHASE 3: MAIN LOOP
        # =========================================================

        self.add_bundle({"load": [("const", round_i, 0)]})
        round_loop = len(self.instrs)

        self.add_bundle({"load": [("const", chunk_i, 0)]})
        chunk_loop = len(self.instrs)

        # Compute addresses: addr = base + chunk_i * VLEN
        self.add_bundle({"alu": [("*", tmp[0], chunk_i, vlen_c)]})
        self.add_bundle({"alu": [
            ("+", tmp[0], self.scratch["inp_indices_p"], tmp[0]),
            ("+", tmp[1], self.scratch["inp_values_p"], tmp[0]),
        ]})

        # Load v_idx and v_val
        self.add_bundle({"load": [("vload", v_idx, tmp[0]), ("vload", v_val, tmp[1])]})

        # Gather tree nodes (2 scalar loads per cycle)
        for vi in range(0, VLEN, 2):
            self.add_bundle({"alu": [
                ("+", tmp[2], self.scratch["forest_values_p"], v_idx + vi),
                ("+", tmp[3], self.scratch["forest_values_p"], v_idx + vi + 1),
            ]})
            self.add_bundle({"load": [
                ("load", v_node_val + vi, tmp[2]),
                ("load", v_node_val + vi + 1, tmp[3]),
            ]})

        # Hash computation
        self.add_bundle({"valu": [("^", v_val, v_val, v_node_val)]})

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_c1, v_c3 = v_hash[hi]
            self.add_bundle({"valu": [
                (op1, v_tmp1, v_val, v_c1),
                (op3, v_tmp2, v_val, v_c3),
            ]})
            self.add_bundle({"valu": [(op2, v_val, v_tmp1, v_tmp2)]})

        # Tree step
        self.add_bundle({"valu": [("*", v_idx, v_idx, v_two), ("&", v_tmp1, v_val, v_one)]})
        self.add_bundle({"valu": [("+", v_idx, v_idx, v_one)]})
        self.add_bundle({"valu": [("+", v_idx, v_idx, v_tmp1)]})
        self.add_bundle({"valu": [("<", v_tmp1, v_idx, v_n_nodes)]})
        self.add_bundle({"flow": [("vselect", v_idx, v_tmp1, v_idx, v_zero)]})

        # Store results
        self.add_bundle({"alu": [("*", tmp[0], chunk_i, vlen_c)]})
        self.add_bundle({"alu": [
            ("+", tmp[0], self.scratch["inp_indices_p"], tmp[0]),
            ("+", tmp[1], self.scratch["inp_values_p"], tmp[0]),
        ]})
        self.add_bundle({"store": [("vstore", tmp[0], v_idx), ("vstore", tmp[1], v_val)]})

        # Loop control - increment FIRST, then compare in next cycle
        self.add_bundle({"flow": [("add_imm", chunk_i, chunk_i, 1)]})
        self.add_bundle({"alu": [("<", cond, chunk_i, n_chunks_c)]})
        self.add_bundle({"flow": [("cond_jump", cond, chunk_loop)]})

        self.add_bundle({"flow": [("add_imm", round_i, round_i, 1)]})
        self.add_bundle({"alu": [("<", cond, round_i, rounds_c)]})
        self.add_bundle({"flow": [("cond_jump", cond, round_loop)]})

        self.add("flow", ("pause",))


BASELINE = 147734

def do_kernel_test(
    forest_height: int, rounds: int, batch_size: int,
    seed: int = 123, trace: bool = False, prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES, trace=trace, prints=prints)

    for expected in reference_kernel2(mem, forest.height, len(inp.values), rounds):
        machine.run()
        inp_values_p = mem[6]
        assert machine.mem[inp_values_p:inp_values_p + len(inp.values)] == expected[inp_values_p:inp_values_p + len(inp.values)], f"Incorrect values"
        inp_indices_p = mem[5]
        assert machine.mem[inp_indices_p:inp_indices_p + len(inp.indices)] == expected[inp_indices_p:inp_indices_p + len(inp.indices)], f"Incorrect indices"
    return machine.cycle


class KernelTest(unittest.TestCase):
    def test_kernel_correctness(self):
        for seed in range(8):
            do_kernel_test(10, 16, 256, seed)

    def test_kernel_cycles(self):
        cycles = do_kernel_test(10, 16, 256)
        print("CYCLES: ", cycles)


if __name__ == "__main__":
    unittest.main()
