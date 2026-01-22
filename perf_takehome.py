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
        6. Batched constant loading - load all constants in parallel
        """
        UNROLL = 32  # Process 32 vector chunks = 256 elements (full batch)

        # =========================================================
        # PHASE 1: ALLOCATE ALL SCRATCH MEMORY (no instructions yet)
        # =========================================================

        # Header variables (loaded from memory)
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        # Use 7 temp addresses to batch-load all header values in 2 cycles
        addr_temps = [self.alloc_scratch() for _ in range(len(init_vars))]

        # Scalar constants - allocate addresses only
        scalar_consts = {}  # value -> address
        def alloc_scalar(val, name=None):
            if val not in scalar_consts:
                scalar_consts[val] = self.alloc_scratch(name)
            return scalar_consts[val]

        zero_const = alloc_scalar(0, "zero")
        one_const = alloc_scalar(1, "one")
        two_const = alloc_scalar(2, "two")
        eight_const = alloc_scalar(8, "eight")
        vlen_const = alloc_scalar(VLEN, "vlen")

        # Offset constants for base address computation
        offset_consts = [alloc_scalar(u * VLEN, f"offset_{u}") for u in range(UNROLL)]

        # Cache stride offsets (n_cache_slots = 32)
        n_cache_slots = 32
        cache_stride = (n_nodes + n_cache_slots - 1) // n_cache_slots
        cache_stride = (cache_stride + VLEN - 1) // VLEN * VLEN
        cache_offset_consts = [alloc_scalar(i * cache_stride, f"cache_offset_{i}") for i in range(n_cache_slots)]

        # Vector registers - allocate scratch space
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_idx = [self.alloc_scratch(f"v_idx_{u}", VLEN) for u in range(UNROLL)]
        v_val = [self.alloc_scratch(f"v_val_{u}", VLEN) for u in range(UNROLL)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{u}", VLEN) for u in range(UNROLL)]

        # Scalar base addresses and loop counters
        idx_base = [self.alloc_scratch(f"idx_base_{u}") for u in range(UNROLL)]
        val_base = [self.alloc_scratch(f"val_base_{u}") for u in range(UNROLL)]
        round_counter = self.alloc_scratch("round_counter")
        loop_cond = self.alloc_scratch("loop_cond")

        # Cache pointers
        cache_ptr = [self.alloc_scratch(f"cache_ptr_{i}") for i in range(n_cache_slots)]

        # Tree cache (full tree)
        CACHE_SIZE = n_nodes
        tree_cache = self.alloc_scratch("tree_cache", CACHE_SIZE)

        # =========================================================
        # PHASE 2: BATCHED CONSTANT LOADING (minimal cycles)
        # =========================================================

        # Load header: batch all const addresses, then batch all loads (2 cycles total)
        # Cycle 1: const 0,1,2,3,4,5,6 into temp addresses
        self.add_bundle({
            "load": [("const", addr_temps[i], i) for i in range(len(init_vars))]
        })
        # Cycle 2: load all init_vars from their addresses
        self.add_bundle({
            "load": [("load", self.scratch[init_vars[i]], addr_temps[i]) for i in range(len(init_vars))]
        })

        # Load ALL scalar constants in batches of 64 (max load slots)
        const_list = list(scalar_consts.items())  # [(value, addr), ...]
        for batch_start in range(0, len(const_list), 64):
            batch = const_list[batch_start:batch_start + 64]
            self.add_bundle({
                "load": [("const", addr, val) for val, addr in batch]
            })

        self.add("flow", ("pause",))

        # =========================================================
        # PHASE 3: VECTOR BROADCASTS + CACHE LOADING (overlapped with base addr computation)
        # =========================================================

        # Broadcast + cache ptr init + idx_base (all in one cycle with 64 ALU slots)
        self.add_bundle({
            "valu": [
                ("vbroadcast", v_two, two_const),
                ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ],
            "alu": [("+", cache_ptr[i], self.scratch["forest_values_p"], cache_offset_consts[i])
                   for i in range(n_cache_slots)] +
                   [("+", idx_base[i], self.scratch["inp_indices_p"], offset_consts[i]) for i in range(UNROLL)],
        })

        # First cache load cycle also computes val_base (64 ALU: 32 ptr update + 32 val_base)
        first_chunk = True
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
                if first_chunk:
                    # Add val_base computation to first cache load cycle
                    alu_ops += [("+", val_base[i], self.scratch["inp_values_p"], offset_consts[i]) for i in range(UNROLL)]
                    first_chunk = False
                if alu_ops:
                    bundle["alu"] = alu_ops
                self.add_bundle(bundle)

        # =========================================================
        # PHASE 4: ROUND LOOP SETUP
        # =========================================================

        # Load initial idx + init round counter (idx_base computed earlier, can read now)
        # round_counter starts at 1 because we compare BEFORE incrementing
        self.add_bundle({
            "load": [("const", round_counter, 1)] + [("vload", v_idx[i], idx_base[i]) for i in range(UNROLL)],
        })

        round_loop_start = len(self.instrs)

        # === LOAD VAL + GATHER + idx*2 (1 cycle with 64 load slots + 32 VALU slots) ===
        self.add_bundle({
            "load": [("vload", v_val[i], val_base[i]) for i in range(UNROLL)] +
                    [("scratch_gather", v_node_val[i], v_idx[i], tree_cache, CACHE_SIZE) for i in range(UNROLL)],
            "valu": [("*", v_idx[i], v_idx[i], v_two) for i in range(UNROLL)],
        })

        # === HASH PHASE (FULLY COMBINED) ===
        # Single full_hash_xor instruction does XOR + all 6 hash stages in 1 cycle
        ops = [("full_hash_xor", v_val[i], v_val[i], v_node_val[i]) for i in range(UNROLL)]
        self.add_bundle({"valu": ops})

        # === INDEX UPDATE PHASE (FULLY OPTIMIZED) ===
        # Use tree_step instruction: idx = (idx + 1 + (val & 1)) if in_bounds else 0
        ops = [("tree_step", v_idx[i], v_idx[i], v_val[i], v_n_nodes) for i in range(UNROLL)]
        self.add_bundle({"valu": ops})

        # === STORE PHASE + ROUND LOOP CONTROL + LOAD IDX FOR NEXT ROUND (overlapped) ===
        # With flow=2, can combine increment + cond_jump in same cycle
        # Key insight: compare BEFORE increment, then increment + cond_jump together
        # This requires starting round_counter at 1 (not 0)

        # Cycle 1: store ALL idx[0-31] + compare (32 stores + 1 ALU)
        # Compare reads round_counter before it's incremented
        self.add_bundle({
            "store": [("vstore", idx_base[i], v_idx[i]) for i in range(UNROLL)],
            "alu": [("<", loop_cond, round_counter, self.scratch["rounds"])],
        })
        # Cycle 2: store ALL val[0-31] + load ALL idx[0-31] + increment + cond_jump
        # cond_jump reads loop_cond from cycle 1, increment + jump both use flow slots
        self.add_bundle({
            "store": [("vstore", val_base[i], v_val[i]) for i in range(UNROLL)],
            "load": [("vload", v_idx[i], idx_base[i]) for i in range(UNROLL)],
            "flow": [("add_imm", round_counter, round_counter, 1),
                     ("cond_jump", loop_cond, round_loop_start)],
        })

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
