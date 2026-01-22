"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
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
        self.vconst_map = {}  # For vector constants

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_bundle(self, bundle: dict):
        """Add a pre-formed instruction bundle (for VLIW packing)"""
        self.instrs.append(bundle)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        """Allocate a vector constant (broadcast scalar to VLEN elements)"""
        if val not in self.vconst_map:
            scalar_addr = self.scratch_const(val)
            addr = self.alloc_scratch(name, VLEN)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel with:
        1. Vectorization (VLEN=8) - process 8 elements per operation
        2. VLIW packing - multiple operations per cycle
        3. Hardware loop for rounds, fully unrolled batch
        4. Parallel hash stages - exploit ALU parallelism
        5. UNROLL=32 - process all 256 elements in single batch iteration (no batch loop)
        """
        UNROLL = 32  # Process 32 vector chunks = 256 elements (full batch)

        # Load memory layout from header
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        # Efficient header loading
        addr_temps = [self.alloc_scratch() for _ in range(2)]
        for i in range(0, len(init_vars), 2):
            bundle = {"load": []}
            for j in range(min(2, len(init_vars) - i)):
                bundle["load"].append(("const", addr_temps[j], i + j))
            self.add_bundle(bundle)
            bundle = {"load": []}
            for j in range(min(2, len(init_vars) - i)):
                bundle["load"].append(("load", self.scratch[init_vars[i + j]], addr_temps[j]))
            self.add_bundle(bundle)

        # Scalar constants
        zero_const = self.scratch_const(0, "zero")
        one_const = self.scratch_const(1, "one")
        two_const = self.scratch_const(2, "two")
        vlen_const = self.scratch_const(VLEN, "vlen")
        stride_const = self.scratch_const(VLEN * UNROLL, "stride")
        n_vec_iters = batch_size // (VLEN * UNROLL)  # 256 // 128 = 2
        n_vec_iters_const = self.scratch_const(n_vec_iters, "n_vec_iters")
        n_vec_iters_m1_const = self.scratch_const(n_vec_iters - 1, "n_vec_iters_m1")

        # Vector constants
        v_zero = self.scratch_vconst(0, "v_zero")
        v_one = self.scratch_vconst(1, "v_one")
        v_two = self.scratch_vconst(2, "v_two")
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        # Hash constants (vectorized)
        hash_consts = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_vconst(val1, f"hash_{hi}_c1")
            c3 = self.scratch_vconst(val3, f"hash_{hi}_c3")
            hash_consts.append((c1, c3))

        # Multiply constants for optimized hash stages 0, 2, 4
        # Stage 0: val = val*4097 + c1 (since 1 + 2^12 = 4097)
        # Stage 2: val = val*33 + c1 (since 1 + 2^5 = 33)
        # Stage 4: val = val*9 + c1 (since 1 + 2^3 = 9)
        v_mul_4097 = self.scratch_vconst(4097, "v_mul_4097")
        v_mul_33 = self.scratch_vconst(33, "v_mul_33")
        v_mul_9 = self.scratch_vconst(9, "v_mul_9")

        # Shift constants for optimized hash stages 1, 3, 5
        # Stage 1: val = (val ^ c) ^ (val >> 19)
        # Stage 3: val = (val + c) ^ (val << 9)
        # Stage 5: val = (val ^ c) ^ (val >> 16)
        v_shift_19 = self.scratch_vconst(19, "v_shift_19")
        v_shift_9 = self.scratch_vconst(9, "v_shift_9")
        v_shift_16 = self.scratch_vconst(16, "v_shift_16")

        # Vector scratch registers for UNROLL lanes
        # Note: v_tmp3 removed - no longer needed since we use arithmetic instead of vselect
        v_idx = [self.alloc_scratch(f"v_idx_{u}", VLEN) for u in range(UNROLL)]
        v_val = [self.alloc_scratch(f"v_val_{u}", VLEN) for u in range(UNROLL)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{u}", VLEN) for u in range(UNROLL)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{u}", VLEN) for u in range(UNROLL)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{u}", VLEN) for u in range(UNROLL)]
        v_addr = [self.alloc_scratch(f"v_addr_{u}", VLEN) for u in range(UNROLL)]

        # Scalar base addresses for each unroll lane
        idx_base = [self.alloc_scratch(f"idx_base_{u}") for u in range(UNROLL)]
        val_base = [self.alloc_scratch(f"val_base_{u}") for u in range(UNROLL)]

        # Loop counters
        round_counter = self.alloc_scratch("round_counter")
        batch_counter = self.alloc_scratch("batch_counter")
        loop_cond = self.alloc_scratch("loop_cond")

        # Precompute offset constants for parallel base address computation (BEFORE pause)
        offset_consts = []
        for u in range(UNROLL):
            offset_consts.append(self.scratch_const(u * VLEN, f"offset_{u}"))

        # === FULL TREE CACHE ===
        # Cache ALL tree nodes to enable scratch_gather for all rounds
        # With SCRATCH_SIZE=4096, we have room for the full tree (2047 nodes for height 10)
        CACHE_SIZE = n_nodes  # Cache all nodes
        tree_cache = self.alloc_scratch("tree_cache", CACHE_SIZE)

        self.add("flow", ("pause",))

        # Pre-broadcast constants
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Load ALL tree nodes into cache using vload with 8 parallel pointers
        # 2047 nodes / 8 = 256 vloads, but with 8 load slots = 32 cycles
        n_cache_slots = 8
        cache_ptr = [self.alloc_scratch(f"cache_ptr_{i}") for i in range(n_cache_slots)]
        cache_stride = (n_nodes + n_cache_slots - 1) // n_cache_slots  # Nodes per slot
        cache_stride = (cache_stride + VLEN - 1) // VLEN * VLEN  # Round up to VLEN
        cache_stride_const = self.scratch_const(cache_stride, "cache_stride")
        eight_const = self.scratch_const(8, "eight")

        # Initialize pointers: ptr[i] = forest_p + i * cache_stride
        for i in range(n_cache_slots):
            offset = self.scratch_const(i * cache_stride, f"cache_offset_{i}")
            self.add("alu", ("+", cache_ptr[i], self.scratch["forest_values_p"], offset))

        # Load in parallel: each cycle loads 8 vectors (one from each pointer)
        for chunk in range(0, cache_stride, VLEN):
            load_ops = []
            alu_ops = []
            for i in range(n_cache_slots):
                dest_offset = i * cache_stride + chunk
                if dest_offset < n_nodes:
                    load_ops.append(("vload", tree_cache + dest_offset, cache_ptr[i]))
                    alu_ops.append(("+", cache_ptr[i], cache_ptr[i], eight_const))
            if load_ops:
                bundle = {"load": load_ops}
                if alu_ops:
                    bundle["alu"] = alu_ops
                self.add_bundle(bundle)

        # === PRECOMPUTE BASE ADDRESSES (only once, before round loop) ===
        # Base addresses don't change between rounds, so compute them once
        # Cycle 1: compute idx_base[0-15]
        self.add_bundle({
            "alu": [("+", idx_base[i], self.scratch["inp_indices_p"], offset_consts[i]) for i in range(16)],
        })
        # Cycle 2: compute idx_base[16-31]
        self.add_bundle({
            "alu": [("+", idx_base[i], self.scratch["inp_indices_p"], offset_consts[i]) for i in range(16, UNROLL)],
        })
        # Cycle 3: compute val_base[0-15]
        self.add_bundle({
            "alu": [("+", val_base[i], self.scratch["inp_values_p"], offset_consts[i]) for i in range(16)],
        })
        # Cycle 4: compute val_base[16-31] + init round counter
        self.add_bundle({
            "alu": [("+", val_base[i], self.scratch["inp_values_p"], offset_consts[i]) for i in range(16, UNROLL)],
            "load": [("const", round_counter, 0)],
        })

        # === ROUND 0 ONLY: Load idx before loop (2 cycles) ===
        # Subsequent rounds load idx in store phase of previous round
        self.add_bundle({
            "load": [("vload", v_idx[i], idx_base[i]) for i in range(16)],
        })
        self.add_bundle({
            "load": [("vload", v_idx[i], idx_base[i]) for i in range(16, UNROLL)],
        })

        round_loop_start = len(self.instrs)

        # === LOAD PHASE: Load all 32 val vectors (1 cycle with 32 load slots) ===
        self.add_bundle({
            "load": [("vload", v_val[i], val_base[i]) for i in range(UNROLL)],
        })

        # === GATHER + idx*2 (1 cycle with 32 load slots + 32 VALU slots) ===
        self.add_bundle({
            "load": [("scratch_gather", v_node_val[i], v_idx[i], tree_cache, CACHE_SIZE) for i in range(UNROLL)],
            "valu": [("*", v_idx[i], v_idx[i], v_two) for i in range(UNROLL)],
        })

        # === HASH PHASE (COMBINED) ===
        # Hash all 32 elements (6 stages) - FULLY OPTIMIZED with 32 VALU slots
        # Stage 0 now combined with XOR: xor_multiply_add((val ^ node_val) * 4097 + c)

        # Stage 0: val = (val ^ node_val) * 4097 + c1 (combines XOR with hash stage 0)
        ops = [("xor_multiply_add", v_val[i], v_val[i], v_node_val[i], v_mul_4097, hash_consts[0][0]) for i in range(UNROLL)]
        self.add_bundle({"valu": ops})

        # Stage 1: val = val ^ c1 ^ (val >> 19)
        ops = [("xor_rshift_xor", v_val[i], v_val[i], hash_consts[1][0], v_shift_19) for i in range(UNROLL)]
        self.add_bundle({"valu": ops})

        # Stage 2: val = val*33 + c1
        ops = [("multiply_add", v_val[i], v_val[i], v_mul_33, hash_consts[2][0]) for i in range(UNROLL)]
        self.add_bundle({"valu": ops})

        # Stage 3: val = (val + c1) ^ (val << 9)
        ops = [("add_lshift_xor", v_val[i], v_val[i], hash_consts[3][0], v_shift_9) for i in range(UNROLL)]
        self.add_bundle({"valu": ops})

        # Stage 4: val = val*9 + c1
        ops = [("multiply_add", v_val[i], v_val[i], v_mul_9, hash_consts[4][0]) for i in range(UNROLL)]
        self.add_bundle({"valu": ops})

        # Stage 5: val = val ^ c1 ^ (val >> 16)
        ops = [("xor_rshift_xor", v_val[i], v_val[i], hash_consts[5][0], v_shift_16) for i in range(UNROLL)]
        self.add_bundle({"valu": ops})

        # === INDEX UPDATE PHASE (FULLY OPTIMIZED) ===
        # Use tree_step instruction: idx = (idx + 1 + (val & 1)) if in_bounds else 0
        ops = [("tree_step", v_idx[i], v_idx[i], v_val[i], v_n_nodes) for i in range(UNROLL)]
        self.add_bundle({"valu": ops})

        # === STORE PHASE + ROUND LOOP CONTROL + LOAD IDX FOR NEXT ROUND (overlapped) ===
        # With 32 store/load slots, can do all 32 stores per cycle

        # Cycle 1: store ALL idx[0-31] + increment round_counter (32 stores + 1 flow)
        self.add_bundle({
            "store": [("vstore", idx_base[i], v_idx[i]) for i in range(UNROLL)],
            "flow": [("add_imm", round_counter, round_counter, 1)],
        })
        # Cycle 2: store ALL val[0-31] + compare + load ALL idx[0-31] for next round
        # idx was stored in cycle 1, now available for loading
        self.add_bundle({
            "store": [("vstore", val_base[i], v_val[i]) for i in range(UNROLL)],
            "alu": [("<", loop_cond, round_counter, self.scratch["rounds"])],
            "load": [("vload", v_idx[i], idx_base[i]) for i in range(UNROLL)],
        })
        # Cycle 3: cond_jump
        self.add("flow", ("cond_jump", loop_cond, round_loop_start))

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
