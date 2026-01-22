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

        # Precompute offset constants for parallel base address computation (BEFORE pause)
        offset_consts = []
        for u in range(UNROLL):
            offset_consts.append(self.scratch_const(u * VLEN, f"offset_{u}"))

        # === TREE CACHE ===
        # Cache first 511 nodes (levels 0-8) to enable scratch_gather
        # After round k, max idx = 2^(k+1) - 2. For idx < 512: k <= 8
        CACHE_SIZE = 511  # 2^9 - 1 = 511 nodes
        CACHED_ROUNDS = 8  # Rounds 1-8 use cache (idx < 511 guaranteed)
        tree_cache = self.alloc_scratch("tree_cache", CACHE_SIZE)
        cached_rounds_const = self.scratch_const(CACHED_ROUNDS, "cached_rounds")
        v_cache_size = self.scratch_vconst(CACHE_SIZE, "v_cache_size")  # For conditional caching

        # Reuse existing vectors for conditional caching:
        # v_tmp1 = cache mask, v_tmp2 = memory values (both free before hash phase)

        self.add("flow", ("pause",))

        # Pre-broadcast constants
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Load tree nodes into cache using vload (511 nodes, ~64 cycles at 8 values/vload)
        cache_load_ptr = self.alloc_scratch("cache_load_ptr")
        self.add("alu", ("+", cache_load_ptr, self.scratch["forest_values_p"], zero_const))  # cache_load_ptr = forest_values_p
        eight_const = self.scratch_const(8, "eight")
        for i in range(0, CACHE_SIZE, VLEN):
            remaining = min(VLEN, CACHE_SIZE - i)
            if remaining == VLEN:
                # Full vector load
                self.add_bundle({
                    "load": [("vload", tree_cache + i, cache_load_ptr)],
                    "alu": [("+", cache_load_ptr, cache_load_ptr, eight_const)]
                })
            else:
                # Partial load for last chunk (load individually)
                for j in range(remaining):
                    self.add_bundle({
                        "load": [("load", tree_cache + i + j, cache_load_ptr)],
                        "alu": [("+", cache_load_ptr, cache_load_ptr, one_const)]
                    })

        # Initialize round counter
        self.add("load", ("const", round_counter, 0))

        round_loop_start = len(self.instrs)

        # Initialize batch counter (each round starts with batch 0)
        self.add("load", ("const", batch_counter, 0))

        # Compute base addresses for all unroll lanes IN PARALLEL
        # idx_base[u] = inp_indices_p + u*VLEN, val_base[u] = inp_values_p + u*VLEN
        # With 12 ALU slots, we can do 6 idx + 6 val = 12 ops per cycle
        for u in range(0, UNROLL, 6):
            alu_ops = []
            for i in range(u, min(u + 6, UNROLL)):
                alu_ops.append(("+", idx_base[i], self.scratch["inp_indices_p"], offset_consts[i]))
                alu_ops.append(("+", val_base[i], self.scratch["inp_values_p"], offset_consts[i]))
            self.add_bundle({"alu": alu_ops})

        # === PROLOGUE: Load first iteration's idx vectors ===
        # This is done once before the loop; subsequent iterations load idx during stores
        for u in range(0, UNROLL, 2):
            self.add_bundle({"load": [
                ("vload", v_idx[u], idx_base[u]),
                ("vload", v_idx[u+1], idx_base[u+1]),
            ]})

        batch_loop_start = len(self.instrs)

        # === LOAD PHASE: Load val vectors and compute addresses ===
        # (idx vectors already loaded - either from prologue or cross-iteration pipeline)
        # Overlap all address computation with val loads
        for u in range(0, UNROLL, 2):
            bundle = {"load": [
                ("vload", v_val[u], val_base[u]),
                ("vload", v_val[u+1], val_base[u+1]),
            ]}
            # Compute addresses for indices u, u+1 (now available since idx was preloaded)
            valu_ops = [
                ("+", v_addr[u], v_forest_p, v_idx[u]),
                ("+", v_addr[u+1], v_forest_p, v_idx[u+1]),
            ]
            bundle["valu"] = valu_ops
            self.add_bundle(bundle)

        # === GATHER PHASE ===
        # Use scratch_gather for cached rounds (fast!), load_offset for non-cached
        # scratch_gather: v_dest[i] = scratch[base + v_idx[i]] - loads full vector in 1 op!
        CHUNK = 4

        # CACHED GATHER: 16 scratch_gather ops = 8 cycles (vs 64 cycles for load_offset)
        # Must do ALL gathers before modifying v_idx, since scratch_gather reads v_idx
        for u in range(0, UNROLL, 2):
            self.add_bundle({"load": [
                ("scratch_gather", v_node_val[u], v_idx[u], tree_cache, CACHE_SIZE),
                ("scratch_gather", v_node_val[u+1], v_idx[u+1], tree_cache, CACHE_SIZE),
            ]})

        # Now safe to modify v_idx: idx = idx * 2
        for u in range(0, UNROLL, 6):
            valu_ops = [("*", v_idx[j], v_idx[j], v_two) for j in range(u, min(u + 6, UNROLL))]
            self.add_bundle({"valu": valu_ops})

        # === XOR + HASH PHASE (all elements now gathered) ===
        # XOR all elements: val = val ^ node_val
        for u in range(0, UNROLL, 6):
            xor_ops = [("^", v_val[i], v_val[i], v_node_val[i]) for i in range(u, min(u+6, UNROLL))]
            self.add_bundle({"valu": xor_ops})

        # Hash all elements (6 stages)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1, c3 = hash_consts[hi]
            # Compute tmp1 and tmp2 for all elements: 3 lanes per cycle (6 ops)
            for u in range(0, UNROLL, 3):
                hash_ops = []
                for i in range(u, min(u+3, UNROLL)):
                    hash_ops.append((op1, v_tmp1[i], v_val[i], c1))
                    hash_ops.append((op3, v_tmp2[i], v_val[i], c3))
                self.add_bundle({"valu": hash_ops})
            # Combine: 6 lanes per cycle
            for u in range(0, UNROLL, 6):
                combine_ops = [(op2, v_val[i], v_tmp1[i], v_tmp2[i]) for i in range(u, min(u+6, UNROLL))]
                self.add_bundle({"valu": combine_ops})

        # === INDEX UPDATE PHASE ===
        # tmp1 = val & 1
        for u in range(0, UNROLL, 6):
            ops = [("&", v_tmp1[i], v_val[i], v_one) for i in range(u, min(u+6, UNROLL))]
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

        # === STORE PHASE with CROSS-ITERATION PIPELINING ===
        # Store remaining idx
        for u in range(4, UNROLL, 2):
            self.add_bundle({"store": [
                ("vstore", idx_base[u], v_idx[u]),
                ("vstore", idx_base[u+1], v_idx[u+1]),
            ]})

        # Store val vectors, update pointers, AND load next iteration's idx (if not last iter)
        # The loads use updated pointers from the PREVIOUS cycle (write-at-end semantics)
        # Cycle 0: store val[0,1], update ptrs[0,1] - can't load yet (ptrs not ready)
        # Cycle 1+: store val[u,u+1], update ptrs[u,u+1], load idx[u-2,u-1] for next iter
        for u in range(0, UNROLL, 2):
            bundle = {"store": [
                ("vstore", val_base[u], v_val[u]),
                ("vstore", val_base[u+1], v_val[u+1]),
            ]}
            # Update pointers
            alu_ops = [
                ("+", idx_base[u], idx_base[u], stride_const),
                ("+", idx_base[u+1], idx_base[u+1], stride_const),
                ("+", val_base[u], val_base[u], stride_const),
                ("+", val_base[u+1], val_base[u+1], stride_const),
            ]
            bundle["alu"] = alu_ops

            # Load next iteration's idx vectors (using pointers updated in previous cycle)
            # Skip first cycle (u=0) since pointers aren't ready yet
            # Skip if this is the last batch iteration (checked at runtime via conditional)
            if u >= 2:
                # Load idx[u-2, u-1] for next iteration
                bundle["load"] = [
                    ("vload", v_idx[u-2], idx_base[u-2]),
                    ("vload", v_idx[u-1], idx_base[u-1]),
                ]
            self.add_bundle(bundle)

        # Load the last idx vectors for next iteration (from the final pointer updates)
        # This overlaps with loop control
        self.add_bundle({
            "load": [
                ("vload", v_idx[UNROLL-2], idx_base[UNROLL-2]),
                ("vload", v_idx[UNROLL-1], idx_base[UNROLL-1]),
            ],
            "flow": [("add_imm", batch_counter, batch_counter, 1)],
        })

        # Check if we should continue (and skip the preloaded idx if last iteration)
        self.add("alu", ("<", loop_cond, batch_counter, n_vec_iters_const))
        self.add("flow", ("cond_jump", loop_cond, batch_loop_start))

        # Round loop control for CACHED rounds
        self.add("flow", ("add_imm", round_counter, round_counter, 1))
        self.add("alu", ("<", loop_cond, round_counter, cached_rounds_const))
        self.add("flow", ("cond_jump", loop_cond, round_loop_start))

        # ============================================================
        # NON-CACHED ROUNDS LOOP (rounds 9-16 use normal memory gather)
        # ============================================================
        noncached_round_loop_start = len(self.instrs)

        # Initialize batch counter
        self.add("load", ("const", batch_counter, 0))

        # Compute base addresses
        for u in range(0, UNROLL, 6):
            alu_ops = []
            for i in range(u, min(u + 6, UNROLL)):
                alu_ops.append(("+", idx_base[i], self.scratch["inp_indices_p"], offset_consts[i]))
                alu_ops.append(("+", val_base[i], self.scratch["inp_values_p"], offset_consts[i]))
            self.add_bundle({"alu": alu_ops})

        # Prologue: Load first iteration's idx vectors
        for u in range(0, UNROLL, 2):
            self.add_bundle({"load": [
                ("vload", v_idx[u], idx_base[u]),
                ("vload", v_idx[u+1], idx_base[u+1]),
            ]})

        noncached_batch_loop_start = len(self.instrs)

        # LOAD PHASE: Load val vectors and compute addresses
        for u in range(0, UNROLL, 2):
            bundle = {"load": [
                ("vload", v_val[u], val_base[u]),
                ("vload", v_val[u+1], val_base[u+1]),
            ]}
            valu_ops = [
                ("+", v_addr[u], v_forest_p, v_idx[u]),
                ("+", v_addr[u+1], v_forest_p, v_idx[u+1]),
            ]
            bundle["valu"] = valu_ops
            self.add_bundle(bundle)

        # NON-CACHED GATHER using load_offset with chunked overlapping
        # Process in chunks of CHUNK=4 vectors, overlap hash with gather
        CHUNK = 4

        # Gather first chunk + precompute idx*2
        idx_mult_done = 0
        for i in range(0, VLEN, 2):
            for u in range(CHUNK):
                bundle = {"load": [
                    ("load_offset", v_node_val[u], v_addr[u], i),
                    ("load_offset", v_node_val[u], v_addr[u], i + 1),
                ]}
                if idx_mult_done < UNROLL:
                    valu_ops = [("*", v_idx[j], v_idx[j], v_two) for j in range(idx_mult_done, min(idx_mult_done + 6, UNROLL))]
                    bundle["valu"] = valu_ops
                    idx_mult_done += 6
                self.add_bundle(bundle)

        # Process remaining chunks with XOR+hash overlap
        for chunk_start in range(CHUNK, UNROLL, CHUNK):
            chunk_end = min(chunk_start + CHUNK, UNROLL)
            prev_start = chunk_start - CHUNK
            prev_end = chunk_start

            # XOR previous chunk
            for u in range(prev_start, prev_end, 6):
                xor_ops = [("^", v_val[i], v_val[i], v_node_val[i]) for i in range(u, min(u+6, prev_end))]
                self.add_bundle({"valu": xor_ops})

            # Hash previous chunk while gathering current chunk
            gather_idx = 0
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                c1, c3 = hash_consts[hi]
                for u in range(prev_start, prev_end, 3):
                    hash_ops = []
                    for i in range(u, min(u+3, prev_end)):
                        hash_ops.append((op1, v_tmp1[i], v_val[i], c1))
                        hash_ops.append((op3, v_tmp2[i], v_val[i], c3))
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

            while gather_idx < VLEN * (chunk_end - chunk_start):
                lane = chunk_start + gather_idx // VLEN
                offset = gather_idx % VLEN
                if lane < chunk_end and offset < VLEN - 1:
                    self.add_bundle({"load": [
                        ("load_offset", v_node_val[lane], v_addr[lane], offset),
                        ("load_offset", v_node_val[lane], v_addr[lane], offset + 1),
                    ]})
                gather_idx += 2

        # Last chunk XOR + hash
        last_start = UNROLL - CHUNK
        xor_ops = [("^", v_val[i], v_val[i], v_node_val[i]) for i in range(last_start, UNROLL)]
        self.add_bundle({"valu": xor_ops})

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1, c3 = hash_consts[hi]
            for u in range(last_start, UNROLL, 3):
                hash_ops = []
                for i in range(u, min(u+3, UNROLL)):
                    hash_ops.append((op1, v_tmp1[i], v_val[i], c1))
                    hash_ops.append((op3, v_tmp2[i], v_val[i], c3))
                self.add_bundle({"valu": hash_ops})
            combine_ops = [(op2, v_val[i], v_tmp1[i], v_tmp2[i]) for i in range(last_start, UNROLL)]
            self.add_bundle({"valu": combine_ops})

        # INDEX UPDATE (same as cached)
        for u in range(0, UNROLL, 6):
            ops = [("&", v_tmp1[i], v_val[i], v_one) for i in range(u, min(u+6, UNROLL))]
            self.add_bundle({"valu": ops})

        for u in range(0, UNROLL, 6):
            ops = [("+", v_tmp1[i], v_tmp1[i], v_one) for i in range(u, min(u+6, UNROLL))]
            self.add_bundle({"valu": ops})

        for u in range(0, UNROLL, 6):
            ops = [("+", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(u, min(u+6, UNROLL))]
            self.add_bundle({"valu": ops})

        for u in range(0, UNROLL, 6):
            ops = [("<", v_tmp1[i], v_idx[i], v_n_nodes) for i in range(u, min(u+6, UNROLL))]
            self.add_bundle({"valu": ops})

        ops = [("*", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(0, 6)]
        self.add_bundle({"valu": ops})
        ops = [("*", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(6, 12)]
        self.add_bundle({"valu": ops, "store": [
            ("vstore", idx_base[0], v_idx[0]),
            ("vstore", idx_base[1], v_idx[1]),
        ]})
        ops = [("*", v_idx[i], v_idx[i], v_tmp1[i]) for i in range(12, min(16, UNROLL))]
        self.add_bundle({"valu": ops, "store": [
            ("vstore", idx_base[2], v_idx[2]),
            ("vstore", idx_base[3], v_idx[3]),
        ]})

        # STORE PHASE (same as cached)
        for u in range(4, UNROLL, 2):
            self.add_bundle({"store": [
                ("vstore", idx_base[u], v_idx[u]),
                ("vstore", idx_base[u+1], v_idx[u+1]),
            ]})

        for u in range(0, UNROLL, 2):
            bundle = {"store": [
                ("vstore", val_base[u], v_val[u]),
                ("vstore", val_base[u+1], v_val[u+1]),
            ]}
            alu_ops = [
                ("+", idx_base[u], idx_base[u], stride_const),
                ("+", idx_base[u+1], idx_base[u+1], stride_const),
                ("+", val_base[u], val_base[u], stride_const),
                ("+", val_base[u+1], val_base[u+1], stride_const),
            ]
            bundle["alu"] = alu_ops
            if u >= 2:
                bundle["load"] = [
                    ("vload", v_idx[u-2], idx_base[u-2]),
                    ("vload", v_idx[u-1], idx_base[u-1]),
                ]
            self.add_bundle(bundle)

        self.add_bundle({
            "load": [
                ("vload", v_idx[UNROLL-2], idx_base[UNROLL-2]),
                ("vload", v_idx[UNROLL-1], idx_base[UNROLL-1]),
            ],
            "flow": [("add_imm", batch_counter, batch_counter, 1)],
        })

        self.add("alu", ("<", loop_cond, batch_counter, n_vec_iters_const))
        self.add("flow", ("cond_jump", loop_cond, noncached_batch_loop_start))

        # Non-cached round loop control
        self.add("flow", ("add_imm", round_counter, round_counter, 1))
        self.add("alu", ("<", loop_cond, round_counter, self.scratch["rounds"]))
        self.add("flow", ("cond_jump", loop_cond, noncached_round_loop_start))

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
