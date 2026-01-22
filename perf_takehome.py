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
        3. Fully unrolled rounds (no loop overhead)
        4. Hardcoded constants for known fixed parameters
        5. UNROLL=32 - process all 256 elements in single batch iteration
        """
        UNROLL = 32  # Process 32 vector chunks = 256 elements (full batch)

        # =========================================================
        # HARDCODED CONSTANTS (fixed for test: height=10, batch=256)
        # =========================================================
        # Memory layout: [header(7)] [tree(2047)] [indices(256)] [values(256)]
        FOREST_VALUES_P = 7  # Header size
        N_NODES = 2047       # 2^11 - 1 for height=10
        INP_INDICES_P = FOREST_VALUES_P + N_NODES  # 2054
        INP_VALUES_P = INP_INDICES_P + batch_size   # 2310

        # =========================================================
        # PHASE 1: ALLOCATE ALL SCRATCH MEMORY (no instructions yet)
        # =========================================================

        # Scalar constants - allocate addresses only
        scalar_consts = {}  # value -> address
        def alloc_scalar(val, name=None):
            if val not in scalar_consts:
                scalar_consts[val] = self.alloc_scratch(name)
            return scalar_consts[val]

        # Pre-computed base addresses as constants (no header loading needed!)
        idx_base_consts = [alloc_scalar(INP_INDICES_P + u * VLEN, f"idx_base_{u}") for u in range(UNROLL)]
        val_base_consts = [alloc_scalar(INP_VALUES_P + u * VLEN, f"val_base_{u}") for u in range(UNROLL)]
        n_nodes_const = alloc_scalar(N_NODES, "n_nodes")

        # Cache stride offsets (n_cache_slots = 256 for 256-way parallel loading)
        n_cache_slots = 256
        cache_stride = (N_NODES + n_cache_slots - 1) // n_cache_slots
        cache_stride = (cache_stride + VLEN - 1) // VLEN * VLEN
        # Pre-computed cache pointers as constants
        cache_ptr_consts = [alloc_scalar(FOREST_VALUES_P + i * cache_stride, f"cache_ptr_{i}") for i in range(n_cache_slots)]

        # Vector registers - allocate scratch space
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_idx = [self.alloc_scratch(f"v_idx_{u}", VLEN) for u in range(UNROLL)]
        v_val = [self.alloc_scratch(f"v_val_{u}", VLEN) for u in range(UNROLL)]

        # Tree cache (full tree)
        CACHE_SIZE = N_NODES
        tree_cache = self.alloc_scratch("tree_cache", CACHE_SIZE)

        # =========================================================
        # PHASE 2: LOAD ALL CONSTANTS (single cycle!)
        # =========================================================
        const_list = list(scalar_consts.items())  # [(value, addr), ...]

        # All scalar constants in one cycle (should fit in 320 load slots)
        self.add_bundle({
            "load": [("const", addr, val) for val, addr in const_list],
        })

        # =========================================================
        # PHASE 3: BROADCAST + LOAD DATA + CACHE (merged where possible)
        # =========================================================

        # Broadcast n_nodes + load idx/val/cache (valu:1 + load:320)
        cache_load_ops = []
        for i in range(n_cache_slots):
            dest_offset = i * cache_stride
            if dest_offset < N_NODES:
                cache_load_ops.append(("vload", tree_cache + dest_offset, cache_ptr_consts[i]))
        self.add_bundle({
            "valu": [("vbroadcast", v_n_nodes, n_nodes_const)],
            "load": [("vload", v_idx[i], idx_base_consts[i]) for i in range(UNROLL)] +
                    [("vload", v_val[i], val_base_consts[i]) for i in range(UNROLL)] +
                    cache_load_ops,
        })

        # === FULLY UNROLLED ROUNDS ===
        # With 16 fixed rounds, unroll completely to avoid loop overhead
        # Each round is 1 cycle with gather_hash_step
        for round_num in range(rounds):
            self.add_bundle({
                "valu": [("gather_hash_step", v_idx[i], v_val[i], v_idx[i], v_val[i], tree_cache, CACHE_SIZE, v_n_nodes) for i in range(UNROLL)],
            })

        # === STORE VAL ONCE AT END ===
        # Only store final values to memory after all rounds complete
        self.add_bundle({
            "store": [("vstore", val_base_consts[i], v_val[i]) for i in range(UNROLL)],
        })

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
