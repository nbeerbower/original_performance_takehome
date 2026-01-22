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
        Optimized kernel - key insight from "cheating" solution:
        Keep idx/val in scratch across all 16 rounds, only load/store once per chunk.
        """
        n_chunks = batch_size // VLEN  # 32

        # =========================================================
        # ALLOCATE SCRATCH
        # =========================================================

        # Header variables
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Temps
        tmp = [self.alloc_scratch(f"tmp{i}") for i in range(8)]

        # Vector registers - keep idx/val in scratch across rounds!
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

        # Multiply-add constants for hash optimization
        # Stage 0: val*4097 + c1 (since 1 + 2^12 = 4097)
        # Stage 2: val*33 + c1 (since 1 + 2^5 = 33)
        # Stage 4: val*9 + c1 (since 1 + 2^3 = 9)
        v_mul_0 = self.alloc_scratch("v_mul_0", VLEN)  # 4097
        v_mul_2 = self.alloc_scratch("v_mul_2", VLEN)  # 33
        v_mul_4 = self.alloc_scratch("v_mul_4", VLEN)  # 9

        # Hash constants (pre-broadcast)
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
        mul_0_c = self.alloc_scratch("mul_0_c")  # 4097
        mul_2_c = self.alloc_scratch("mul_2_c")  # 33
        mul_4_c = self.alloc_scratch("mul_4_c")  # 9

        hash_c = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.alloc_scratch(f"hc1_{hi}")
            c3 = self.alloc_scratch(f"hc3_{hi}")
            hash_c.append((c1, c3, val1, val3))

        # Loop control
        chunk_i = self.alloc_scratch("chunk_i")
        cond = self.alloc_scratch("cond")

        # =========================================================
        # SETUP: Load constants (batch 2 per cycle)
        # =========================================================

        self.add_bundle({"load": [("const", zero_c, 0), ("const", one_c, 1)]})
        self.add_bundle({"load": [("const", two_c, 2), ("const", vlen_c, VLEN)]})
        self.add_bundle({"load": [("const", n_chunks_c, n_chunks), ("const", mul_0_c, 4097)]})
        self.add_bundle({"load": [("const", mul_2_c, 33), ("const", mul_4_c, 9)]})

        for c1, c3, val1, val3 in hash_c:
            self.add_bundle({"load": [("const", c1, val1), ("const", c3, val3)]})

        # Load header (batch 2 per cycle)
        for i in range(0, len(init_vars), 2):
            self.add_bundle({"load": [("const", tmp[0], i), ("const", tmp[1], i+1 if i+1 < len(init_vars) else 0)]})
            loads = [("load", self.scratch[init_vars[i]], tmp[0])]
            if i + 1 < len(init_vars):
                loads.append(("load", self.scratch[init_vars[i+1]], tmp[1]))
            self.add_bundle({"load": loads})

        # Broadcast vector constants (max 6 valu per cycle)
        self.add_bundle({"valu": [
            ("vbroadcast", v_zero, zero_c),
            ("vbroadcast", v_one, one_c),
            ("vbroadcast", v_two, two_c),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_mul_0, mul_0_c),
            ("vbroadcast", v_mul_2, mul_2_c),
        ]})
        self.add_bundle({"valu": [("vbroadcast", v_mul_4, mul_4_c)]})

        for hi in range(0, len(HASH_STAGES), 3):
            vops = []
            for j in range(min(3, len(HASH_STAGES) - hi)):
                v_c1, v_c3 = v_hash[hi + j]
                c1, c3, _, _ = hash_c[hi + j]
                vops.extend([("vbroadcast", v_c1, c1), ("vbroadcast", v_c3, c3)])
            self.add_bundle({"valu": vops})

        self.add("flow", ("pause",))

        # =========================================================
        # MAIN LOOP: For each chunk, do ALL 16 rounds
        # Key insight: keep idx/val in scratch, minimize memory traffic
        # =========================================================

        self.add_bundle({"load": [("const", chunk_i, 0)]})
        chunk_loop = len(self.instrs)

        # Load idx and val for this chunk (ONCE per chunk)
        self.add_bundle({"alu": [("*", tmp[0], chunk_i, vlen_c)]})
        self.add_bundle({"alu": [
            ("+", tmp[0], self.scratch["inp_indices_p"], tmp[0]),
            ("+", tmp[1], self.scratch["inp_values_p"], tmp[0]),
        ]})
        self.add_bundle({"load": [("vload", v_idx, tmp[0]), ("vload", v_val, tmp[1])]})

        # FULLY UNROLLED: 16 rounds with idx/val staying in scratch
        for round_num in range(rounds):
            # Gather tree nodes - PIPELINED with XOR overlapped (5 cycles for gather+XOR)
            # Use scalar alu XOR to overlap with gather (12 alu slots available)

            # Cycle 1: compute addresses for elements 0,1
            self.add_bundle({"alu": [
                ("+", tmp[2], self.scratch["forest_values_p"], v_idx + 0),
                ("+", tmp[3], self.scratch["forest_values_p"], v_idx + 1),
            ]})
            # Cycle 2: load[0,1] + addr[2,3]
            self.add_bundle({
                "load": [
                    ("load", v_node_val + 0, tmp[2]),
                    ("load", v_node_val + 1, tmp[3]),
                ],
                "alu": [
                    ("+", tmp[2], self.scratch["forest_values_p"], v_idx + 2),
                    ("+", tmp[3], self.scratch["forest_values_p"], v_idx + 3),
                ]
            })
            # Cycle 3: load[2,3] + addr[4,5] + XOR[0,1]
            self.add_bundle({
                "load": [
                    ("load", v_node_val + 2, tmp[2]),
                    ("load", v_node_val + 3, tmp[3]),
                ],
                "alu": [
                    ("+", tmp[2], self.scratch["forest_values_p"], v_idx + 4),
                    ("+", tmp[3], self.scratch["forest_values_p"], v_idx + 5),
                    ("^", v_val + 0, v_val + 0, v_node_val + 0),
                    ("^", v_val + 1, v_val + 1, v_node_val + 1),
                ]
            })
            # Cycle 4: load[4,5] + addr[6,7] + XOR[2,3]
            self.add_bundle({
                "load": [
                    ("load", v_node_val + 4, tmp[2]),
                    ("load", v_node_val + 5, tmp[3]),
                ],
                "alu": [
                    ("+", tmp[2], self.scratch["forest_values_p"], v_idx + 6),
                    ("+", tmp[3], self.scratch["forest_values_p"], v_idx + 7),
                    ("^", v_val + 2, v_val + 2, v_node_val + 2),
                    ("^", v_val + 3, v_val + 3, v_node_val + 3),
                ]
            })
            # Cycle 5: load[6,7] + XOR[4,5]
            self.add_bundle({
                "load": [
                    ("load", v_node_val + 6, tmp[2]),
                    ("load", v_node_val + 7, tmp[3]),
                ],
                "alu": [
                    ("^", v_val + 4, v_val + 4, v_node_val + 4),
                    ("^", v_val + 5, v_val + 5, v_node_val + 5),
                ]
            })
            # Cycle 6: XOR[6,7] - must wait for load to complete
            self.add_bundle({
                "alu": [
                    ("^", v_val + 6, v_val + 6, v_node_val + 6),
                    ("^", v_val + 7, v_val + 7, v_node_val + 7),
                ]
            })

            # Use multiply_add for stages 0, 2, 4 (saves 1 cycle each)
            # Stage 0: ("+", c1, "+", "<<", 12) → val*4097 + c1
            v_c1_0, _ = v_hash[0]
            self.add_bundle({"valu": [("multiply_add", v_val, v_val, v_mul_0, v_c1_0)]})

            # Stage 1: ("^", c1, "^", ">>", 19) → (val^c1) ^ (val>>19)
            v_c1_1, v_c3_1 = v_hash[1]
            self.add_bundle({"valu": [("^", v_tmp1, v_val, v_c1_1), (">>", v_tmp2, v_val, v_c3_1)]})
            self.add_bundle({"valu": [("^", v_val, v_tmp1, v_tmp2)]})

            # Stage 2: ("+", c1, "+", "<<", 5) → val*33 + c1
            v_c1_2, _ = v_hash[2]
            self.add_bundle({"valu": [("multiply_add", v_val, v_val, v_mul_2, v_c1_2)]})

            # Stage 3: ("+", c1, "^", "<<", 9) → (val+c1) ^ (val<<9)
            v_c1_3, v_c3_3 = v_hash[3]
            self.add_bundle({"valu": [("+", v_tmp1, v_val, v_c1_3), ("<<", v_tmp2, v_val, v_c3_3)]})
            self.add_bundle({"valu": [("^", v_val, v_tmp1, v_tmp2)]})

            # Stage 4: ("+", c1, "+", "<<", 3) → val*9 + c1
            v_c1_4, _ = v_hash[4]
            self.add_bundle({"valu": [("multiply_add", v_val, v_val, v_mul_4, v_c1_4)]})

            # Stage 5: ("^", c1, "^", ">>", 16) → (val^c1) ^ (val>>16)
            v_c1_5, v_c3_5 = v_hash[5]
            self.add_bundle({"valu": [("^", v_tmp1, v_val, v_c1_5), (">>", v_tmp2, v_val, v_c3_5)]})
            self.add_bundle({"valu": [("^", v_val, v_tmp1, v_tmp2)]})

            # Tree step: idx = idx*2 + 1 + (val & 1), with bounds check
            # Use multiply_add: idx*2 + 1 in one cycle
            self.add_bundle({"valu": [
                ("multiply_add", v_idx, v_idx, v_two, v_one),  # idx = idx*2 + 1
                ("&", v_tmp1, v_val, v_one),  # tmp1 = val & 1
            ]})
            self.add_bundle({"valu": [("+", v_idx, v_idx, v_tmp1)]})  # idx = idx + tmp1
            self.add_bundle({"valu": [("<", v_tmp1, v_idx, v_n_nodes)]})
            self.add_bundle({"flow": [("vselect", v_idx, v_tmp1, v_idx, v_zero)]})

        # Store results ONCE after all 16 rounds complete
        self.add_bundle({"alu": [("*", tmp[0], chunk_i, vlen_c)]})
        self.add_bundle({"alu": [
            ("+", tmp[0], self.scratch["inp_indices_p"], tmp[0]),
            ("+", tmp[1], self.scratch["inp_values_p"], tmp[0]),
        ]})
        self.add_bundle({"store": [("vstore", tmp[0], v_idx), ("vstore", tmp[1], v_val)]})

        # Chunk loop control
        self.add_bundle({"flow": [("add_imm", chunk_i, chunk_i, 1)]})
        self.add_bundle({"alu": [("<", cond, chunk_i, n_chunks_c)]})
        self.add_bundle({"flow": [("cond_jump", cond, chunk_loop)]})

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
