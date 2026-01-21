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
        5. UNROLL=8 - process 64 elements per batch iteration (4 iterations total)
        """
        UNROLL = 16  # Process 16 vector chunks per iteration

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
        n_vec_iters = batch_size // (VLEN * UNROLL)  # 256 // 32 = 8
        n_vec_iters_const = self.scratch_const(n_vec_iters, "n_vec_iters")

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

        self.add("flow", ("pause",))

        # Pre-broadcast constants
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Initialize round counter
        self.add("load", ("const", round_counter, 0))

        round_loop_start = len(self.instrs)

        # Initialize batch counter
        self.add("load", ("const", batch_counter, 0))

        # Compute base addresses for all unroll lanes
        # idx_base[0] = inp_indices_p, idx_base[1] = inp_indices_p + VLEN, etc.
        self.add_bundle({
            "alu": [
                ("+", idx_base[0], self.scratch["inp_indices_p"], zero_const),
                ("+", val_base[0], self.scratch["inp_values_p"], zero_const),
            ]
        })
        for u in range(1, UNROLL):
            self.add_bundle({
                "alu": [
                    ("+", idx_base[u], idx_base[u-1], vlen_const),
                    ("+", val_base[u], val_base[u-1], vlen_const),
                ]
            })

        batch_loop_start = len(self.instrs)

        # === LOAD PHASE: Load idx and val, compute addresses ===
        # Interleave vloads with address computation for better pipelining
        # Load idx vectors in pairs
        for u in range(0, UNROLL, 2):
            if u == 0:
                self.add_bundle({"load": [
                    ("vload", v_idx[u], idx_base[u]),
                    ("vload", v_idx[u+1], idx_base[u+1]),
                ]})
            elif u >= 2:
                # Overlap with address computation for previous pair
                self.add_bundle({
                    "load": [
                        ("vload", v_idx[u], idx_base[u]),
                        ("vload", v_idx[u+1], idx_base[u+1]),
                    ],
                    "valu": [
                        ("+", v_addr[u-2], v_forest_p, v_idx[u-2]),
                        ("+", v_addr[u-1], v_forest_p, v_idx[u-1]),
                    ]
                })

        # Load val vectors with address computation overlap
        for u in range(0, UNROLL, 2):
            bundle = {"load": [
                ("vload", v_val[u], val_base[u]),
                ("vload", v_val[u+1], val_base[u+1]),
            ]}
            # Overlap with address computation if there are remaining indices
            valu_ops = []
            if u + 2 < UNROLL:
                valu_ops.append(("+", v_addr[u+2], v_forest_p, v_idx[u+2]))
            if u + 3 < UNROLL:
                valu_ops.append(("+", v_addr[u+3], v_forest_p, v_idx[u+3]))
            if valu_ops:
                bundle["valu"] = valu_ops
            self.add_bundle(bundle)

        # Compute final addresses if not done in loop above
        final_addrs = []
        for i in range(UNROLL - 2, UNROLL):
            if i >= (UNROLL // 2) * 2 + 2:  # These weren't computed in the loop
                final_addrs.append(("+", v_addr[i], v_forest_p, v_idx[i]))
        if final_addrs:
            self.add_bundle({"valu": final_addrs})

        # === PIPELINED GATHER + COMPUTE (CHUNK=4) ===
        # Process in chunks: gather chunk N, then XOR+hash chunk N while gathering chunk N+1
        CHUNK = 4  # Optimal chunk size (tested: 4 > 2 > 8)

        # Gather first chunk (no overlap possible yet)
        for i in range(0, VLEN, 2):
            for u in range(CHUNK):
                self.add_bundle({"load": [
                    ("load_offset", v_node_val[u], v_addr[u], i),
                    ("load_offset", v_node_val[u], v_addr[u], i + 1),
                ]})

        # Process remaining chunks with overlap
        for chunk_start in range(CHUNK, UNROLL, CHUNK):
            chunk_end = min(chunk_start + CHUNK, UNROLL)
            prev_start = chunk_start - CHUNK
            prev_end = chunk_start
            chunk_size = prev_end - prev_start

            # XOR previous chunk (pack 6 lanes max per cycle)
            for u in range(prev_start, prev_end, 6):
                xor_ops = [("^", v_val[i], v_val[i], v_node_val[i]) for i in range(u, min(u+6, prev_end))]
                self.add_bundle({"valu": xor_ops})

            # Hash previous chunk while gathering current chunk
            gather_idx = 0
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                c1, c3 = hash_consts[hi]
                # Compute tmp1 and tmp2: 3 lanes per cycle (6 ops)
                for u in range(prev_start, prev_end, 3):
                    hash_ops = []
                    for i in range(u, min(u+3, prev_end)):
                        hash_ops.append((op1, v_tmp1[i], v_val[i], c1))
                        hash_ops.append((op3, v_tmp2[i], v_val[i], c3))
                    # Overlap with gather
                    if gather_idx < VLEN * (chunk_end - chunk_start):
                        lane = chunk_start + gather_idx // VLEN
                        offset = gather_idx % VLEN
                        if lane < chunk_end and offset < VLEN - 1:
                            self.add_bundle({
                                "valu": hash_ops,
                                "load": [
                                    ("load_offset", v_node_val[lane], v_addr[lane], offset),
                                    ("load_offset", v_node_val[lane], v_addr[lane], offset + 1),
                                ]
                            })
                            gather_idx += 2
                        else:
                            self.add_bundle({"valu": hash_ops})
                    else:
                        self.add_bundle({"valu": hash_ops})

                # Combine: 6 lanes per cycle
                for u in range(prev_start, prev_end, 6):
                    combine_ops = [(op2, v_val[i], v_tmp1[i], v_tmp2[i]) for i in range(u, min(u+6, prev_end))]
                    if gather_idx < VLEN * (chunk_end - chunk_start):
                        lane = chunk_start + gather_idx // VLEN
                        offset = gather_idx % VLEN
                        if lane < chunk_end and offset < VLEN - 1:
                            self.add_bundle({
                                "valu": combine_ops,
                                "load": [
                                    ("load_offset", v_node_val[lane], v_addr[lane], offset),
                                    ("load_offset", v_node_val[lane], v_addr[lane], offset + 1),
                                ]
                            })
                            gather_idx += 2
                        else:
                            self.add_bundle({"valu": combine_ops})
                    else:
                        self.add_bundle({"valu": combine_ops})

            # Finish any remaining gather
            while gather_idx < VLEN * (chunk_end - chunk_start):
                lane = chunk_start + gather_idx // VLEN
                offset = gather_idx % VLEN
                if lane < chunk_end and offset < VLEN - 1:
                    self.add_bundle({"load": [
                        ("load_offset", v_node_val[lane], v_addr[lane], offset),
                        ("load_offset", v_node_val[lane], v_addr[lane], offset + 1),
                    ]})
                gather_idx += 2

        # XOR and hash last chunk (no more gather to overlap)
        last_start = UNROLL - CHUNK
        for u in range(last_start, UNROLL, 6):
            xor_ops = [("^", v_val[i], v_val[i], v_node_val[i]) for i in range(u, min(u+6, UNROLL))]
            self.add_bundle({"valu": xor_ops})

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1, c3 = hash_consts[hi]
            for u in range(last_start, UNROLL, 3):
                ops = []
                for i in range(u, min(u+3, UNROLL)):
                    ops.append((op1, v_tmp1[i], v_val[i], c1))
                    ops.append((op3, v_tmp2[i], v_val[i], c3))
                self.add_bundle({"valu": ops})
            for u in range(last_start, UNROLL, 6):
                ops = [(op2, v_val[i], v_tmp1[i], v_tmp2[i]) for i in range(u, min(u+6, UNROLL))]
                self.add_bundle({"valu": ops})

        # === INDEX UPDATE PHASE ===
        # tmp1 = val & 1, idx = idx * 2 (2 ops per lane, 6 slots max)
        for u in range(0, UNROLL, 3):
            ops = []
            for i in range(u, min(u+3, UNROLL)):
                ops.append(("&", v_tmp1[i], v_val[i], v_one))
                ops.append(("*", v_idx[i], v_idx[i], v_two))
            self.add_bundle({"valu": ops})

        # tmp1 = tmp1 + 1
        for u in range(0, UNROLL, 6):
            ops = [("+", v_tmp1[i], v_tmp1[i], v_one) for i in range(u, min(u+6, UNROLL))]
            self.add_bundle({"valu": ops})

        # idx = idx + tmp1
        for u in range(0, UNROLL, 6):
            ops = [("+", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(u, min(u+6, UNROLL))]
            self.add_bundle({"valu": ops})

        # tmp1 = (idx < n_nodes)
        for u in range(0, UNROLL, 6):
            ops = [("<", v_tmp1[i], v_idx[i], v_n_nodes) for i in range(u, min(u+6, UNROLL))]
            self.add_bundle({"valu": ops})

        # idx = idx * tmp1, OVERLAPPED with first idx stores
        # After computing idx[0..5], we can start storing them while computing idx[6..15]
        # Cycle 1: compute idx[0..5]
        ops = [("*", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(0, 6)]
        self.add_bundle({"valu": ops})
        # Cycle 2: compute idx[6..11], store idx[0..1]
        ops = [("*", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(6, 12)]
        self.add_bundle({"valu": ops, "store": [
            ("vstore", idx_base[0], v_idx[0]),
            ("vstore", idx_base[1], v_idx[1]),
        ]})
        # Cycle 3: compute idx[12..15], store idx[2..3]
        ops = [("*", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(12, min(16, UNROLL))]
        self.add_bundle({"valu": ops, "store": [
            ("vstore", idx_base[2], v_idx[2]),
            ("vstore", idx_base[3], v_idx[3]),
        ]})

        # === STORE PHASE with POINTER UPDATES OVERLAPPED ===
        # Store remaining idx, then store val with pointer updates overlapped
        # Must update pointers AFTER the corresponding val store uses them
        for u in range(4, UNROLL, 2):  # Start from 4, first 4 already stored
            self.add_bundle({"store": [
                ("vstore", idx_base[u], v_idx[u]),
                ("vstore", idx_base[u+1], v_idx[u+1]),
            ]})
        # Store val vectors and overlap pointer updates (safe: writes happen at end of cycle)
        for u in range(0, UNROLL, 2):
            bundle = {"store": [
                ("vstore", val_base[u], v_val[u]),
                ("vstore", val_base[u+1], v_val[u+1]),
            ]}
            # Update the pointers we just used (u and u+1) plus more to fill ALU slots
            # Update 4 pointers per cycle (8 ALU ops: 4 idx + 4 val updates)
            if u < UNROLL:
                alu_ops = []
                # Update pointers for indices u, u+1 (already used) plus u+2, u+3 (will use next cycle)
                # But we need to only update what's been used. Simpler: just update current pair
                alu_ops.append(("+", idx_base[u], idx_base[u], stride_const))
                alu_ops.append(("+", idx_base[u+1], idx_base[u+1], stride_const))
                alu_ops.append(("+", val_base[u], val_base[u], stride_const))
                alu_ops.append(("+", val_base[u+1], val_base[u+1], stride_const))
                bundle["alu"] = alu_ops
            self.add_bundle(bundle)

        # Loop control
        self.add("flow", ("add_imm", batch_counter, batch_counter, 1))
        self.add("alu", ("<", loop_cond, batch_counter, n_vec_iters_const))
        self.add("flow", ("cond_jump", loop_cond, batch_loop_start))

        # Round loop control
        self.add("flow", ("add_imm", round_counter, round_counter, 1))
        self.add("alu", ("<", loop_cond, round_counter, self.scratch["rounds"]))
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
