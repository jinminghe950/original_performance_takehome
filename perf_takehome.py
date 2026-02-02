"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

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

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

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

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Heavily optimized kernel - keeps ALL chunks in scratch across rounds.
        This avoids intermediate memory traffic between rounds.
        """
        n_chunks = batch_size // VLEN  # 32 chunks
        PIPE = 8  # Process 8 chunks in parallel (maximizes load/compute overlap)

        # Allocate vector buffers for ALL 32 chunks (persistent across rounds)
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(n_chunks)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(n_chunks)]

        # Working buffers - double buffer for pipelining (A and B sets)
        v_node_val_A = [self.alloc_scratch(f"v_node_val_A_{i}", VLEN) for i in range(PIPE)]
        v_node_val_B = [self.alloc_scratch(f"v_node_val_B_{i}", VLEN) for i in range(PIPE)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(PIPE)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(PIPE)]
        v_node_addr_A = [self.alloc_scratch(f"v_node_addr_A_{i}", VLEN) for i in range(PIPE)]
        v_node_addr_B = [self.alloc_scratch(f"v_node_addr_B_{i}", VLEN) for i in range(PIPE)]
        # Aliases for backward compatibility with non-pipelined code
        v_node_val = v_node_val_A
        v_node_addr = v_node_addr_A

        # Constant vectors
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

        # Tree node vectors for round 1 optimization
        v_tree1 = self.alloc_scratch("v_tree1", VLEN)
        v_tree2 = self.alloc_scratch("v_tree2", VLEN)

        # Tree node vectors for round 2/13 optimization (indices 3-6)
        v_tree3 = self.alloc_scratch("v_tree3", VLEN)
        v_tree4 = self.alloc_scratch("v_tree4", VLEN)
        v_tree5 = self.alloc_scratch("v_tree5", VLEN)
        v_tree6 = self.alloc_scratch("v_tree6", VLEN)

        # Tree node vectors for round 3/14 optimization (indices 7-14)
        v_tree_level3 = [self.alloc_scratch(f"v_tree_{i}", VLEN) for i in range(7, 15)]

        # Extra constant vectors
        v_three = self.alloc_scratch("v_three", VLEN)
        v_seven = self.alloc_scratch("v_seven", VLEN)

        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = [self.alloc_scratch(f"tmp_addr_{i}") for i in range(PIPE)]
        tmp_addr2 = [self.alloc_scratch(f"tmp_addr2_{i}") for i in range(PIPE)]

        # Input parameters
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height",
                     "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars:
            self.alloc_scratch(v, 1)

        for i, v in enumerate(init_vars):
            self.instrs.append({"load": [("const", tmp1, i)]})
            self.instrs.append({"load": [("load", self.scratch[v], tmp1)]})

        # Constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        three_const = self.scratch_const(3)
        seven_const = self.scratch_const(7)
        offset_consts = [self.scratch_const(chunk_i * VLEN) for chunk_i in range(n_chunks)]

        # Scalar temps for tree node loading
        tmp_tree1 = self.alloc_scratch("tmp_tree1")
        tmp_tree2 = self.alloc_scratch("tmp_tree2")
        tmp_tree = [self.alloc_scratch(f"tmp_tree_{i}") for i in range(8)]  # For loading tree levels

        # Hash constants
        hash_scalars = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            hash_scalars.append((self.scratch_const(val1), self.scratch_const(val3), op1, op2, op3))

        v_hash_consts = []
        for stage_i, (c1_scalar, c3_scalar, op1, op2, op3) in enumerate(hash_scalars):
            v_c1 = self.alloc_scratch(f"v_hc1_{stage_i}", VLEN)
            v_c3 = self.alloc_scratch(f"v_hc3_{stage_i}", VLEN)
            self.instrs.append({"valu": [("vbroadcast", v_c1, c1_scalar), ("vbroadcast", v_c3, c3_scalar)]})
            v_hash_consts.append((v_c1, v_c3, op1, op2, op3))

        # Broadcast basic constants
        self.instrs.append({"valu": [
            ("vbroadcast", v_zero, zero_const),
            ("vbroadcast", v_one, one_const),
            ("vbroadcast", v_two, two_const),
            ("vbroadcast", v_three, three_const),
            ("vbroadcast", v_seven, seven_const),
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
        ]})

        self.add("flow", ("pause",))

        inp_idx_p = self.scratch["inp_indices_p"]
        inp_val_p = self.scratch["inp_values_p"]
        forest_p = self.scratch["forest_values_p"]

        # Preload tree levels 2 and 3 (indices 3-6 and 7-14)
        # Load addresses for tree[3-6]
        self.instrs.append({"alu": [
            ("+", tmp_tree[0], forest_p, three_const),
            ("+", tmp_tree[1], forest_p, self.scratch_const(4)),
            ("+", tmp_tree[2], forest_p, self.scratch_const(5)),
            ("+", tmp_tree[3], forest_p, self.scratch_const(6)),
        ]})
        self.instrs.append({"load": [
            ("load", tmp_tree[0], tmp_tree[0]),
            ("load", tmp_tree[1], tmp_tree[1]),
        ]})
        self.instrs.append({"load": [
            ("load", tmp_tree[2], tmp_tree[2]),
            ("load", tmp_tree[3], tmp_tree[3]),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree3, tmp_tree[0]),
            ("vbroadcast", v_tree4, tmp_tree[1]),
            ("vbroadcast", v_tree5, tmp_tree[2]),
            ("vbroadcast", v_tree6, tmp_tree[3]),
        ]})

        # Load tree[7-14] for level 3
        self.instrs.append({"alu": [
            ("+", tmp_tree[0], forest_p, seven_const),
            ("+", tmp_tree[1], forest_p, self.scratch_const(8)),
            ("+", tmp_tree[2], forest_p, self.scratch_const(9)),
            ("+", tmp_tree[3], forest_p, self.scratch_const(10)),
            ("+", tmp_tree[4], forest_p, self.scratch_const(11)),
            ("+", tmp_tree[5], forest_p, self.scratch_const(12)),
        ]})
        self.instrs.append({"alu": [
            ("+", tmp_tree[6], forest_p, self.scratch_const(13)),
            ("+", tmp_tree[7], forest_p, self.scratch_const(14)),
        ]})
        self.instrs.append({"load": [
            ("load", tmp_tree[0], tmp_tree[0]),
            ("load", tmp_tree[1], tmp_tree[1]),
        ]})
        self.instrs.append({"load": [
            ("load", tmp_tree[2], tmp_tree[2]),
            ("load", tmp_tree[3], tmp_tree[3]),
        ]})
        self.instrs.append({"load": [
            ("load", tmp_tree[4], tmp_tree[4]),
            ("load", tmp_tree[5], tmp_tree[5]),
        ]})
        self.instrs.append({"load": [
            ("load", tmp_tree[6], tmp_tree[6]),
            ("load", tmp_tree[7], tmp_tree[7]),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree_level3[0], tmp_tree[0]),
            ("vbroadcast", v_tree_level3[1], tmp_tree[1]),
            ("vbroadcast", v_tree_level3[2], tmp_tree[2]),
            ("vbroadcast", v_tree_level3[3], tmp_tree[3]),
            ("vbroadcast", v_tree_level3[4], tmp_tree[4]),
            ("vbroadcast", v_tree_level3[5], tmp_tree[5]),
        ]})
        self.instrs.append({"valu": [
            ("vbroadcast", v_tree_level3[6], tmp_tree[6]),
            ("vbroadcast", v_tree_level3[7], tmp_tree[7]),
        ]})

        # Helper to emit 6-packed hash for 8 chunks
        def emit_hash_8(c, p):
            """Emit hash for 8 chunks using 6 VALU slots per cycle"""
            for v_c1, v_c3, op1, op2, op3 in v_hash_consts:
                # op1+op3 for chunks 0-2 (6 ops)
                self.instrs.append({"valu": [
                    (op1, v_tmp1[p[0]], v_val[c[0]], v_c1), (op3, v_tmp2[p[0]], v_val[c[0]], v_c3),
                    (op1, v_tmp1[p[1]], v_val[c[1]], v_c1), (op3, v_tmp2[p[1]], v_val[c[1]], v_c3),
                    (op1, v_tmp1[p[2]], v_val[c[2]], v_c1), (op3, v_tmp2[p[2]], v_val[c[2]], v_c3),
                ]})
                # op1+op3 for chunks 3-5 (6 ops)
                self.instrs.append({"valu": [
                    (op1, v_tmp1[p[3]], v_val[c[3]], v_c1), (op3, v_tmp2[p[3]], v_val[c[3]], v_c3),
                    (op1, v_tmp1[p[4]], v_val[c[4]], v_c1), (op3, v_tmp2[p[4]], v_val[c[4]], v_c3),
                    (op1, v_tmp1[p[5]], v_val[c[5]], v_c1), (op3, v_tmp2[p[5]], v_val[c[5]], v_c3),
                ]})
                # op1+op3 for chunks 6-7, op2 for 0-3 (6 ops)
                self.instrs.append({"valu": [
                    (op1, v_tmp1[p[6]], v_val[c[6]], v_c1), (op3, v_tmp2[p[6]], v_val[c[6]], v_c3),
                    (op1, v_tmp1[p[7]], v_val[c[7]], v_c1), (op3, v_tmp2[p[7]], v_val[c[7]], v_c3),
                    (op2, v_val[c[0]], v_tmp1[p[0]], v_tmp2[p[0]]),
                    (op2, v_val[c[1]], v_tmp1[p[1]], v_tmp2[p[1]]),
                ]})
                # op2 for chunks 2-7 (6 ops)
                self.instrs.append({"valu": [
                    (op2, v_val[c[2]], v_tmp1[p[2]], v_tmp2[p[2]]),
                    (op2, v_val[c[3]], v_tmp1[p[3]], v_tmp2[p[3]]),
                    (op2, v_val[c[4]], v_tmp1[p[4]], v_tmp2[p[4]]),
                    (op2, v_val[c[5]], v_tmp1[p[5]], v_tmp2[p[5]]),
                    (op2, v_val[c[6]], v_tmp1[p[6]], v_tmp2[p[6]]),
                    (op2, v_val[c[7]], v_tmp1[p[7]], v_tmp2[p[7]]),
                ]})

        # Helper to emit 6-packed index computation for 8 chunks
        def emit_index_8(c, p):
            """Emit index computation for 8 chunks using 6 VALU slots per cycle"""
            # bit = val & 1, shifted = idx << 1 for chunks 0-2 (6 ops)
            self.instrs.append({"valu": [
                ("&", v_tmp1[p[0]], v_val[c[0]], v_one), ("<<", v_tmp2[p[0]], v_idx[c[0]], v_one),
                ("&", v_tmp1[p[1]], v_val[c[1]], v_one), ("<<", v_tmp2[p[1]], v_idx[c[1]], v_one),
                ("&", v_tmp1[p[2]], v_val[c[2]], v_one), ("<<", v_tmp2[p[2]], v_idx[c[2]], v_one),
            ]})
            # bit, shifted for chunks 3-5 (6 ops)
            self.instrs.append({"valu": [
                ("&", v_tmp1[p[3]], v_val[c[3]], v_one), ("<<", v_tmp2[p[3]], v_idx[c[3]], v_one),
                ("&", v_tmp1[p[4]], v_val[c[4]], v_one), ("<<", v_tmp2[p[4]], v_idx[c[4]], v_one),
                ("&", v_tmp1[p[5]], v_val[c[5]], v_one), ("<<", v_tmp2[p[5]], v_idx[c[5]], v_one),
            ]})
            # bit, shifted for chunks 6-7, bit+1 for 0-1 (6 ops)
            self.instrs.append({"valu": [
                ("&", v_tmp1[p[6]], v_val[c[6]], v_one), ("<<", v_tmp2[p[6]], v_idx[c[6]], v_one),
                ("&", v_tmp1[p[7]], v_val[c[7]], v_one), ("<<", v_tmp2[p[7]], v_idx[c[7]], v_one),
                ("+", v_tmp1[p[0]], v_tmp1[p[0]], v_one), ("+", v_tmp1[p[1]], v_tmp1[p[1]], v_one),
            ]})
            # bit+1 for chunks 2-7 (6 ops)
            self.instrs.append({"valu": [
                ("+", v_tmp1[p[2]], v_tmp1[p[2]], v_one), ("+", v_tmp1[p[3]], v_tmp1[p[3]], v_one),
                ("+", v_tmp1[p[4]], v_tmp1[p[4]], v_one), ("+", v_tmp1[p[5]], v_tmp1[p[5]], v_one),
                ("+", v_tmp1[p[6]], v_tmp1[p[6]], v_one), ("+", v_tmp1[p[7]], v_tmp1[p[7]], v_one),
            ]})
            # new_idx = shifted + bit for chunks 0-5 (6 ops)
            self.instrs.append({"valu": [
                ("+", v_idx[c[0]], v_tmp2[p[0]], v_tmp1[p[0]]), ("+", v_idx[c[1]], v_tmp2[p[1]], v_tmp1[p[1]]),
                ("+", v_idx[c[2]], v_tmp2[p[2]], v_tmp1[p[2]]), ("+", v_idx[c[3]], v_tmp2[p[3]], v_tmp1[p[3]]),
                ("+", v_idx[c[4]], v_tmp2[p[4]], v_tmp1[p[4]]), ("+", v_idx[c[5]], v_tmp2[p[5]], v_tmp1[p[5]]),
            ]})
            # new_idx for 6-7, cmp for 0-3 (6 ops)
            self.instrs.append({"valu": [
                ("+", v_idx[c[6]], v_tmp2[p[6]], v_tmp1[p[6]]), ("+", v_idx[c[7]], v_tmp2[p[7]], v_tmp1[p[7]]),
                ("<", v_tmp1[p[0]], v_idx[c[0]], v_n_nodes), ("<", v_tmp1[p[1]], v_idx[c[1]], v_n_nodes),
                ("<", v_tmp1[p[2]], v_idx[c[2]], v_n_nodes), ("<", v_tmp1[p[3]], v_idx[c[3]], v_n_nodes),
            ]})
            # cmp for 4-7, mask for 0-1 (6 ops)
            self.instrs.append({"valu": [
                ("<", v_tmp1[p[4]], v_idx[c[4]], v_n_nodes), ("<", v_tmp1[p[5]], v_idx[c[5]], v_n_nodes),
                ("<", v_tmp1[p[6]], v_idx[c[6]], v_n_nodes), ("<", v_tmp1[p[7]], v_idx[c[7]], v_n_nodes),
                ("*", v_idx[c[0]], v_idx[c[0]], v_tmp1[p[0]]), ("*", v_idx[c[1]], v_idx[c[1]], v_tmp1[p[1]]),
            ]})
            # mask for 2-7 (6 ops)
            self.instrs.append({"valu": [
                ("*", v_idx[c[2]], v_idx[c[2]], v_tmp1[p[2]]), ("*", v_idx[c[3]], v_idx[c[3]], v_tmp1[p[3]]),
                ("*", v_idx[c[4]], v_idx[c[4]], v_tmp1[p[4]]), ("*", v_idx[c[5]], v_idx[c[5]], v_tmp1[p[5]]),
                ("*", v_idx[c[6]], v_idx[c[6]], v_tmp1[p[6]]), ("*", v_idx[c[7]], v_idx[c[7]], v_tmp1[p[7]]),
            ]})

        # ========== ROUND 0: Load vals, set idx=0, compute hash - NO stores (keep in scratch) ==========
        self.instrs.append({"load": [("load", tmp_node_val, forest_p)]})

        # Load ALL 32 chunks' initial values into scratch (16 vloads = 8 cycles)
        for chunk_i in range(0, n_chunks, 2):
            off0 = offset_consts[chunk_i]
            off1 = offset_consts[chunk_i + 1]
            self.instrs.append({"alu": [("+", tmp_addr[0], inp_val_p, off0),
                                        ("+", tmp_addr[1], inp_val_p, off1)]})
            self.instrs.append({"load": [("vload", v_val[chunk_i], tmp_addr[0]),
                                         ("vload", v_val[chunk_i + 1], tmp_addr[1])]})

        # Process 8 chunks at a time for round 0
        for base in range(0, n_chunks, PIPE):
            c = [base + j for j in range(PIPE)]  # Chunk indices
            p = [j % PIPE for j in range(PIPE)]  # Working buffer indices

            # Broadcast node value for 8 chunks (2 cycles at 6 ops each, but broadcasts use less)
            self.instrs.append({"valu": [
                ("vbroadcast", v_node_val[p[0]], tmp_node_val),
                ("vbroadcast", v_node_val[p[1]], tmp_node_val),
                ("vbroadcast", v_node_val[p[2]], tmp_node_val),
                ("vbroadcast", v_node_val[p[3]], tmp_node_val),
                ("vbroadcast", v_node_val[p[4]], tmp_node_val),
                ("vbroadcast", v_node_val[p[5]], tmp_node_val),
            ]})
            self.instrs.append({"valu": [
                ("vbroadcast", v_node_val[p[6]], tmp_node_val),
                ("vbroadcast", v_node_val[p[7]], tmp_node_val),
                ("+", v_idx[c[0]], v_zero, v_zero),
                ("+", v_idx[c[1]], v_zero, v_zero),
                ("+", v_idx[c[2]], v_zero, v_zero),
                ("+", v_idx[c[3]], v_zero, v_zero),
            ]})
            self.instrs.append({"valu": [
                ("+", v_idx[c[4]], v_zero, v_zero),
                ("+", v_idx[c[5]], v_zero, v_zero),
                ("+", v_idx[c[6]], v_zero, v_zero),
                ("+", v_idx[c[7]], v_zero, v_zero),
                ("^", v_val[c[0]], v_val[c[0]], v_node_val[p[0]]),
                ("^", v_val[c[1]], v_val[c[1]], v_node_val[p[1]]),
            ]})
            self.instrs.append({"valu": [
                ("^", v_val[c[2]], v_val[c[2]], v_node_val[p[2]]),
                ("^", v_val[c[3]], v_val[c[3]], v_node_val[p[3]]),
                ("^", v_val[c[4]], v_val[c[4]], v_node_val[p[4]]),
                ("^", v_val[c[5]], v_val[c[5]], v_node_val[p[5]]),
                ("^", v_val[c[6]], v_val[c[6]], v_node_val[p[6]]),
                ("^", v_val[c[7]], v_val[c[7]], v_node_val[p[7]]),
            ]})

            # Hash all 8 (6 stages × 4 cycles = 24 cycles)
            emit_hash_8(c, p)
            # Index computation for all 8 (9 cycles)
            emit_index_8(c, p)
            # NO STORES - keep results in scratch

        # ========== ROUND 1: Indices are 1 or 2 - use arithmetic, keep in scratch ==========
        self.instrs.append({"alu": [("+", tmp_tree1, forest_p, one_const),
                                    ("+", tmp_tree2, forest_p, two_const)]})
        self.instrs.append({"load": [("load", tmp_tree1, tmp_tree1),
                                     ("load", tmp_tree2, tmp_tree2)]})
        self.instrs.append({"valu": [("vbroadcast", v_tree1, tmp_tree1),
                                     ("vbroadcast", v_tree2, tmp_tree2)]})

        for base in range(0, n_chunks, PIPE):
            c = [base + j for j in range(PIPE)]
            p = [j % PIPE for j in range(PIPE)]

            # Compute node_val using arithmetic (idx is 1 or 2)
            # tmp1 = idx - 1, tmp2 = 2 - idx
            self.instrs.append({"valu": [
                ("-", v_tmp1[p[0]], v_idx[c[0]], v_one), ("-", v_tmp2[p[0]], v_two, v_idx[c[0]]),
                ("-", v_tmp1[p[1]], v_idx[c[1]], v_one), ("-", v_tmp2[p[1]], v_two, v_idx[c[1]]),
                ("-", v_tmp1[p[2]], v_idx[c[2]], v_one), ("-", v_tmp2[p[2]], v_two, v_idx[c[2]]),
            ]})
            self.instrs.append({"valu": [
                ("-", v_tmp1[p[3]], v_idx[c[3]], v_one), ("-", v_tmp2[p[3]], v_two, v_idx[c[3]]),
                ("-", v_tmp1[p[4]], v_idx[c[4]], v_one), ("-", v_tmp2[p[4]], v_two, v_idx[c[4]]),
                ("-", v_tmp1[p[5]], v_idx[c[5]], v_one), ("-", v_tmp2[p[5]], v_two, v_idx[c[5]]),
            ]})
            self.instrs.append({"valu": [
                ("-", v_tmp1[p[6]], v_idx[c[6]], v_one), ("-", v_tmp2[p[6]], v_two, v_idx[c[6]]),
                ("-", v_tmp1[p[7]], v_idx[c[7]], v_one), ("-", v_tmp2[p[7]], v_two, v_idx[c[7]]),
                ("*", v_node_addr[p[0]], v_tree2, v_tmp1[p[0]]),
                ("*", v_node_addr[p[1]], v_tree2, v_tmp1[p[1]]),
            ]})
            self.instrs.append({"valu": [
                ("*", v_node_addr[p[2]], v_tree2, v_tmp1[p[2]]),
                ("*", v_node_addr[p[3]], v_tree2, v_tmp1[p[3]]),
                ("*", v_node_addr[p[4]], v_tree2, v_tmp1[p[4]]),
                ("*", v_node_addr[p[5]], v_tree2, v_tmp1[p[5]]),
                ("*", v_node_addr[p[6]], v_tree2, v_tmp1[p[6]]),
                ("*", v_node_addr[p[7]], v_tree2, v_tmp1[p[7]]),
            ]})
            self.instrs.append({"valu": [
                ("multiply_add", v_node_val[p[0]], v_tree1, v_tmp2[p[0]], v_node_addr[p[0]]),
                ("multiply_add", v_node_val[p[1]], v_tree1, v_tmp2[p[1]], v_node_addr[p[1]]),
                ("multiply_add", v_node_val[p[2]], v_tree1, v_tmp2[p[2]], v_node_addr[p[2]]),
                ("multiply_add", v_node_val[p[3]], v_tree1, v_tmp2[p[3]], v_node_addr[p[3]]),
                ("multiply_add", v_node_val[p[4]], v_tree1, v_tmp2[p[4]], v_node_addr[p[4]]),
                ("multiply_add", v_node_val[p[5]], v_tree1, v_tmp2[p[5]], v_node_addr[p[5]]),
            ]})
            self.instrs.append({"valu": [
                ("multiply_add", v_node_val[p[6]], v_tree1, v_tmp2[p[6]], v_node_addr[p[6]]),
                ("multiply_add", v_node_val[p[7]], v_tree1, v_tmp2[p[7]], v_node_addr[p[7]]),
                ("^", v_val[c[0]], v_val[c[0]], v_node_val[p[0]]),
                ("^", v_val[c[1]], v_val[c[1]], v_node_val[p[1]]),
                ("^", v_val[c[2]], v_val[c[2]], v_node_val[p[2]]),
                ("^", v_val[c[3]], v_val[c[3]], v_node_val[p[3]]),
            ]})
            self.instrs.append({"valu": [
                ("^", v_val[c[4]], v_val[c[4]], v_node_val[p[4]]),
                ("^", v_val[c[5]], v_val[c[5]], v_node_val[p[5]]),
                ("^", v_val[c[6]], v_val[c[6]], v_node_val[p[6]]),
                ("^", v_val[c[7]], v_val[c[7]], v_node_val[p[7]]),
            ]})

            # Hash and index
            emit_hash_8(c, p)
            emit_index_8(c, p)
            # NO STORES - keep in scratch

        # ========== ROUNDS 2+: Different handling based on round number ==========
        # Due to tree wrapping at n_nodes, after round 10 all indices reset to 0
        # Round 11 = like round 0, Round 12 = like round 1, etc.

        def emit_hash_and_index_8(c, p):
            """Emit hash and index computation for 8 chunks with 6-packed VALU"""
            emit_hash_8(c, p)
            emit_index_8(c, p)

        for round_i in range(2, rounds):
            # Determine effective round after considering tree wrap
            # After round 10, all indices wrap to 0, so round 11 behaves like round 0
            effective_round = round_i
            if round_i >= 11:
                effective_round = round_i - 11

            if effective_round == 0:
                # All indices are 0 - broadcast tree[0] once and reuse
                self.instrs.append({"valu": [("vbroadcast", v_node_val[0], tmp_node_val)]})

                for base in range(0, n_chunks, PIPE):
                    c = [base + j for j in range(PIPE)]
                    p = [j % PIPE for j in range(PIPE)]

                    # Set idx=0 and XOR with tree[0] for all 8 chunks
                    self.instrs.append({"valu": [
                        ("+", v_idx[c[0]], v_zero, v_zero), ("+", v_idx[c[1]], v_zero, v_zero),
                        ("+", v_idx[c[2]], v_zero, v_zero), ("+", v_idx[c[3]], v_zero, v_zero),
                        ("+", v_idx[c[4]], v_zero, v_zero), ("+", v_idx[c[5]], v_zero, v_zero),
                    ]})
                    self.instrs.append({"valu": [
                        ("+", v_idx[c[6]], v_zero, v_zero), ("+", v_idx[c[7]], v_zero, v_zero),
                        ("^", v_val[c[0]], v_val[c[0]], v_node_val[0]),
                        ("^", v_val[c[1]], v_val[c[1]], v_node_val[0]),
                        ("^", v_val[c[2]], v_val[c[2]], v_node_val[0]),
                        ("^", v_val[c[3]], v_val[c[3]], v_node_val[0]),
                    ]})
                    self.instrs.append({"valu": [
                        ("^", v_val[c[4]], v_val[c[4]], v_node_val[0]),
                        ("^", v_val[c[5]], v_val[c[5]], v_node_val[0]),
                        ("^", v_val[c[6]], v_val[c[6]], v_node_val[0]),
                        ("^", v_val[c[7]], v_val[c[7]], v_node_val[0]),
                    ]})
                    emit_hash_and_index_8(c, p)

            elif effective_round == 1:
                # Indices are 1 or 2 - use arithmetic selection
                for base in range(0, n_chunks, PIPE):
                    c = [base + j for j in range(PIPE)]
                    p = [j % PIPE for j in range(PIPE)]

                    # tmp1 = idx - 1, tmp2 = 2 - idx
                    self.instrs.append({"valu": [
                        ("-", v_tmp1[p[0]], v_idx[c[0]], v_one), ("-", v_tmp2[p[0]], v_two, v_idx[c[0]]),
                        ("-", v_tmp1[p[1]], v_idx[c[1]], v_one), ("-", v_tmp2[p[1]], v_two, v_idx[c[1]]),
                        ("-", v_tmp1[p[2]], v_idx[c[2]], v_one), ("-", v_tmp2[p[2]], v_two, v_idx[c[2]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("-", v_tmp1[p[3]], v_idx[c[3]], v_one), ("-", v_tmp2[p[3]], v_two, v_idx[c[3]]),
                        ("-", v_tmp1[p[4]], v_idx[c[4]], v_one), ("-", v_tmp2[p[4]], v_two, v_idx[c[4]]),
                        ("-", v_tmp1[p[5]], v_idx[c[5]], v_one), ("-", v_tmp2[p[5]], v_two, v_idx[c[5]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("-", v_tmp1[p[6]], v_idx[c[6]], v_one), ("-", v_tmp2[p[6]], v_two, v_idx[c[6]]),
                        ("-", v_tmp1[p[7]], v_idx[c[7]], v_one), ("-", v_tmp2[p[7]], v_two, v_idx[c[7]]),
                        ("*", v_node_addr[p[0]], v_tree2, v_tmp1[p[0]]),
                        ("*", v_node_addr[p[1]], v_tree2, v_tmp1[p[1]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("*", v_node_addr[p[2]], v_tree2, v_tmp1[p[2]]),
                        ("*", v_node_addr[p[3]], v_tree2, v_tmp1[p[3]]),
                        ("*", v_node_addr[p[4]], v_tree2, v_tmp1[p[4]]),
                        ("*", v_node_addr[p[5]], v_tree2, v_tmp1[p[5]]),
                        ("*", v_node_addr[p[6]], v_tree2, v_tmp1[p[6]]),
                        ("*", v_node_addr[p[7]], v_tree2, v_tmp1[p[7]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node_val[p[0]], v_tree1, v_tmp2[p[0]], v_node_addr[p[0]]),
                        ("multiply_add", v_node_val[p[1]], v_tree1, v_tmp2[p[1]], v_node_addr[p[1]]),
                        ("multiply_add", v_node_val[p[2]], v_tree1, v_tmp2[p[2]], v_node_addr[p[2]]),
                        ("multiply_add", v_node_val[p[3]], v_tree1, v_tmp2[p[3]], v_node_addr[p[3]]),
                        ("multiply_add", v_node_val[p[4]], v_tree1, v_tmp2[p[4]], v_node_addr[p[4]]),
                        ("multiply_add", v_node_val[p[5]], v_tree1, v_tmp2[p[5]], v_node_addr[p[5]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node_val[p[6]], v_tree1, v_tmp2[p[6]], v_node_addr[p[6]]),
                        ("multiply_add", v_node_val[p[7]], v_tree1, v_tmp2[p[7]], v_node_addr[p[7]]),
                        ("^", v_val[c[0]], v_val[c[0]], v_node_val[p[0]]),
                        ("^", v_val[c[1]], v_val[c[1]], v_node_val[p[1]]),
                        ("^", v_val[c[2]], v_val[c[2]], v_node_val[p[2]]),
                        ("^", v_val[c[3]], v_val[c[3]], v_node_val[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("^", v_val[c[4]], v_val[c[4]], v_node_val[p[4]]),
                        ("^", v_val[c[5]], v_val[c[5]], v_node_val[p[5]]),
                        ("^", v_val[c[6]], v_val[c[6]], v_node_val[p[6]]),
                        ("^", v_val[c[7]], v_val[c[7]], v_node_val[p[7]]),
                    ]})
                    emit_hash_and_index_8(c, p)

            elif effective_round == 2:
                # Indices are 3-6 - use 4-value arithmetic selection
                for base in range(0, n_chunks, PIPE):
                    c = [base + j for j in range(PIPE)]
                    p = [j % PIPE for j in range(PIPE)]

                    # offset = idx - 3, bit0 = offset & 1, bit1 = offset >> 1
                    self.instrs.append({"valu": [
                        ("-", v_tmp1[p[0]], v_idx[c[0]], v_three),
                        ("-", v_tmp1[p[1]], v_idx[c[1]], v_three),
                        ("-", v_tmp1[p[2]], v_idx[c[2]], v_three),
                        ("-", v_tmp1[p[3]], v_idx[c[3]], v_three),
                        ("-", v_tmp1[p[4]], v_idx[c[4]], v_three),
                        ("-", v_tmp1[p[5]], v_idx[c[5]], v_three),
                    ]})
                    self.instrs.append({"valu": [
                        ("-", v_tmp1[p[6]], v_idx[c[6]], v_three),
                        ("-", v_tmp1[p[7]], v_idx[c[7]], v_three),
                        ("&", v_tmp2[p[0]], v_tmp1[p[0]], v_one),
                        ("&", v_tmp2[p[1]], v_tmp1[p[1]], v_one),
                        ("&", v_tmp2[p[2]], v_tmp1[p[2]], v_one),
                        ("&", v_tmp2[p[3]], v_tmp1[p[3]], v_one),
                    ]})
                    self.instrs.append({"valu": [
                        ("&", v_tmp2[p[4]], v_tmp1[p[4]], v_one),
                        ("&", v_tmp2[p[5]], v_tmp1[p[5]], v_one),
                        ("&", v_tmp2[p[6]], v_tmp1[p[6]], v_one),
                        ("&", v_tmp2[p[7]], v_tmp1[p[7]], v_one),
                        (">>", v_tmp1[p[0]], v_tmp1[p[0]], v_one),
                        (">>", v_tmp1[p[1]], v_tmp1[p[1]], v_one),
                    ]})
                    self.instrs.append({"valu": [
                        (">>", v_tmp1[p[2]], v_tmp1[p[2]], v_one),
                        (">>", v_tmp1[p[3]], v_tmp1[p[3]], v_one),
                        (">>", v_tmp1[p[4]], v_tmp1[p[4]], v_one),
                        (">>", v_tmp1[p[5]], v_tmp1[p[5]], v_one),
                        (">>", v_tmp1[p[6]], v_tmp1[p[6]], v_one),
                        (">>", v_tmp1[p[7]], v_tmp1[p[7]], v_one),
                    ]})
                    # inv_bit0 = 1 - bit0
                    self.instrs.append({"valu": [
                        ("-", v_node_addr[p[0]], v_one, v_tmp2[p[0]]),
                        ("-", v_node_addr[p[1]], v_one, v_tmp2[p[1]]),
                        ("-", v_node_addr[p[2]], v_one, v_tmp2[p[2]]),
                        ("-", v_node_addr[p[3]], v_one, v_tmp2[p[3]]),
                        ("-", v_node_addr[p[4]], v_one, v_tmp2[p[4]]),
                        ("-", v_node_addr[p[5]], v_one, v_tmp2[p[5]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("-", v_node_addr[p[6]], v_one, v_tmp2[p[6]]),
                        ("-", v_node_addr[p[7]], v_one, v_tmp2[p[7]]),
                        ("*", v_node_val[p[0]], v_tree3, v_node_addr[p[0]]),
                        ("*", v_node_val[p[1]], v_tree3, v_node_addr[p[1]]),
                        ("*", v_node_val[p[2]], v_tree3, v_node_addr[p[2]]),
                        ("*", v_node_val[p[3]], v_tree3, v_node_addr[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("*", v_node_val[p[4]], v_tree3, v_node_addr[p[4]]),
                        ("*", v_node_val[p[5]], v_tree3, v_node_addr[p[5]]),
                        ("*", v_node_val[p[6]], v_tree3, v_node_addr[p[6]]),
                        ("*", v_node_val[p[7]], v_tree3, v_node_addr[p[7]]),
                        ("multiply_add", v_node_val[p[0]], v_tree4, v_tmp2[p[0]], v_node_val[p[0]]),
                        ("multiply_add", v_node_val[p[1]], v_tree4, v_tmp2[p[1]], v_node_val[p[1]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node_val[p[2]], v_tree4, v_tmp2[p[2]], v_node_val[p[2]]),
                        ("multiply_add", v_node_val[p[3]], v_tree4, v_tmp2[p[3]], v_node_val[p[3]]),
                        ("multiply_add", v_node_val[p[4]], v_tree4, v_tmp2[p[4]], v_node_val[p[4]]),
                        ("multiply_add", v_node_val[p[5]], v_tree4, v_tmp2[p[5]], v_node_val[p[5]]),
                        ("multiply_add", v_node_val[p[6]], v_tree4, v_tmp2[p[6]], v_node_val[p[6]]),
                        ("multiply_add", v_node_val[p[7]], v_tree4, v_tmp2[p[7]], v_node_val[p[7]]),
                    ]})
                    # second_half = tree5 * inv_bit0 + tree6 * bit0
                    self.instrs.append({"valu": [
                        ("*", v_node_addr[p[0]], v_tree5, v_node_addr[p[0]]),
                        ("*", v_node_addr[p[1]], v_tree5, v_node_addr[p[1]]),
                        ("*", v_node_addr[p[2]], v_tree5, v_node_addr[p[2]]),
                        ("*", v_node_addr[p[3]], v_tree5, v_node_addr[p[3]]),
                        ("*", v_node_addr[p[4]], v_tree5, v_node_addr[p[4]]),
                        ("*", v_node_addr[p[5]], v_tree5, v_node_addr[p[5]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("*", v_node_addr[p[6]], v_tree5, v_node_addr[p[6]]),
                        ("*", v_node_addr[p[7]], v_tree5, v_node_addr[p[7]]),
                        ("multiply_add", v_node_addr[p[0]], v_tree6, v_tmp2[p[0]], v_node_addr[p[0]]),
                        ("multiply_add", v_node_addr[p[1]], v_tree6, v_tmp2[p[1]], v_node_addr[p[1]]),
                        ("multiply_add", v_node_addr[p[2]], v_tree6, v_tmp2[p[2]], v_node_addr[p[2]]),
                        ("multiply_add", v_node_addr[p[3]], v_tree6, v_tmp2[p[3]], v_node_addr[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node_addr[p[4]], v_tree6, v_tmp2[p[4]], v_node_addr[p[4]]),
                        ("multiply_add", v_node_addr[p[5]], v_tree6, v_tmp2[p[5]], v_node_addr[p[5]]),
                        ("multiply_add", v_node_addr[p[6]], v_tree6, v_tmp2[p[6]], v_node_addr[p[6]]),
                        ("multiply_add", v_node_addr[p[7]], v_tree6, v_tmp2[p[7]], v_node_addr[p[7]]),
                        ("-", v_tmp2[p[0]], v_one, v_tmp1[p[0]]),
                        ("-", v_tmp2[p[1]], v_one, v_tmp1[p[1]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("-", v_tmp2[p[2]], v_one, v_tmp1[p[2]]),
                        ("-", v_tmp2[p[3]], v_one, v_tmp1[p[3]]),
                        ("-", v_tmp2[p[4]], v_one, v_tmp1[p[4]]),
                        ("-", v_tmp2[p[5]], v_one, v_tmp1[p[5]]),
                        ("-", v_tmp2[p[6]], v_one, v_tmp1[p[6]]),
                        ("-", v_tmp2[p[7]], v_one, v_tmp1[p[7]]),
                    ]})
                    # result = first_half * inv_bit1 + second_half * bit1
                    self.instrs.append({"valu": [
                        ("*", v_node_val[p[0]], v_node_val[p[0]], v_tmp2[p[0]]),
                        ("*", v_node_val[p[1]], v_node_val[p[1]], v_tmp2[p[1]]),
                        ("*", v_node_val[p[2]], v_node_val[p[2]], v_tmp2[p[2]]),
                        ("*", v_node_val[p[3]], v_node_val[p[3]], v_tmp2[p[3]]),
                        ("*", v_node_val[p[4]], v_node_val[p[4]], v_tmp2[p[4]]),
                        ("*", v_node_val[p[5]], v_node_val[p[5]], v_tmp2[p[5]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("*", v_node_val[p[6]], v_node_val[p[6]], v_tmp2[p[6]]),
                        ("*", v_node_val[p[7]], v_node_val[p[7]], v_tmp2[p[7]]),
                        ("multiply_add", v_node_val[p[0]], v_node_addr[p[0]], v_tmp1[p[0]], v_node_val[p[0]]),
                        ("multiply_add", v_node_val[p[1]], v_node_addr[p[1]], v_tmp1[p[1]], v_node_val[p[1]]),
                        ("multiply_add", v_node_val[p[2]], v_node_addr[p[2]], v_tmp1[p[2]], v_node_val[p[2]]),
                        ("multiply_add", v_node_val[p[3]], v_node_addr[p[3]], v_tmp1[p[3]], v_node_val[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node_val[p[4]], v_node_addr[p[4]], v_tmp1[p[4]], v_node_val[p[4]]),
                        ("multiply_add", v_node_val[p[5]], v_node_addr[p[5]], v_tmp1[p[5]], v_node_val[p[5]]),
                        ("multiply_add", v_node_val[p[6]], v_node_addr[p[6]], v_tmp1[p[6]], v_node_val[p[6]]),
                        ("multiply_add", v_node_val[p[7]], v_node_addr[p[7]], v_tmp1[p[7]], v_node_val[p[7]]),
                        ("^", v_val[c[0]], v_val[c[0]], v_node_val[p[0]]),
                        ("^", v_val[c[1]], v_val[c[1]], v_node_val[p[1]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("^", v_val[c[2]], v_val[c[2]], v_node_val[p[2]]),
                        ("^", v_val[c[3]], v_val[c[3]], v_node_val[p[3]]),
                        ("^", v_val[c[4]], v_val[c[4]], v_node_val[p[4]]),
                        ("^", v_val[c[5]], v_val[c[5]], v_node_val[p[5]]),
                        ("^", v_val[c[6]], v_val[c[6]], v_node_val[p[6]]),
                        ("^", v_val[c[7]], v_val[c[7]], v_node_val[p[7]]),
                    ]})
                    emit_hash_and_index_8(c, p)

            else:
                # General case: use pipelined scattered loads for 8 chunks
                n_groups = n_chunks // PIPE  # 4 groups with PIPE=8

                # We use v_node_addr_A for groups 0, 2 and v_node_addr_B for groups 1, 3
                # Precompute all addresses upfront during first group's loads

                c0 = [j for j in range(PIPE)]           # Group 0: chunks 0-7
                c1 = [8 + j for j in range(PIPE)]       # Group 1: chunks 8-15
                c2 = [16 + j for j in range(PIPE)]      # Group 2: chunks 16-23
                c3 = [24 + j for j in range(PIPE)]      # Group 3: chunks 24-31
                p = [j % PIPE for j in range(PIPE)]

                # First 2 cycles: compute addresses for group 0 (into A)
                self.instrs.append({"valu": [
                    ("+", v_node_addr_A[p[0]], v_idx[c0[0]], v_forest_p),
                    ("+", v_node_addr_A[p[1]], v_idx[c0[1]], v_forest_p),
                    ("+", v_node_addr_A[p[2]], v_idx[c0[2]], v_forest_p),
                    ("+", v_node_addr_A[p[3]], v_idx[c0[3]], v_forest_p),
                    ("+", v_node_addr_A[p[4]], v_idx[c0[4]], v_forest_p),
                    ("+", v_node_addr_A[p[5]], v_idx[c0[5]], v_forest_p),
                ]})
                self.instrs.append({"valu": [
                    ("+", v_node_addr_A[p[6]], v_idx[c0[6]], v_forest_p),
                    ("+", v_node_addr_A[p[7]], v_idx[c0[7]], v_forest_p),
                ]})

                # Load first group while computing addresses for groups 1, 2, 3
                load_idx = 0
                for chunk in range(8):
                    for pair in range(4):
                        instr = {"load": [
                            ("load_offset", v_node_val_A[p[chunk]], v_node_addr_A[p[chunk]], pair*2),
                            ("load_offset", v_node_val_A[p[chunk]], v_node_addr_A[p[chunk]], pair*2+1)
                        ]}
                        # Overlap with address computation for groups 1, 2, 3
                        if load_idx == 0:
                            instr["valu"] = [
                                ("+", v_node_addr_B[p[0]], v_idx[c1[0]], v_forest_p),
                                ("+", v_node_addr_B[p[1]], v_idx[c1[1]], v_forest_p),
                                ("+", v_node_addr_B[p[2]], v_idx[c1[2]], v_forest_p),
                                ("+", v_node_addr_B[p[3]], v_idx[c1[3]], v_forest_p),
                                ("+", v_node_addr_B[p[4]], v_idx[c1[4]], v_forest_p),
                                ("+", v_node_addr_B[p[5]], v_idx[c1[5]], v_forest_p),
                            ]
                        elif load_idx == 1:
                            instr["valu"] = [
                                ("+", v_node_addr_B[p[6]], v_idx[c1[6]], v_forest_p),
                                ("+", v_node_addr_B[p[7]], v_idx[c1[7]], v_forest_p),
                            ]
                        # Note: Groups 2, 3 addresses will be computed later during group 0-1 processing
                        load_idx += 1
                        self.instrs.append(instr)

                # Middle groups: overlap hash of current with loads of next
                # With PIPE=8: 32 load cycles needed, ~33 VALU cycles available (hash=24 + index=9)
                for group_i in range(n_groups - 1):
                    curr_base = group_i * PIPE
                    next_base = (group_i + 1) * PIPE
                    curr_c = [curr_base + j for j in range(PIPE)]
                    next_c = [next_base + j for j in range(PIPE)]
                    p = [j % PIPE for j in range(PIPE)]

                    # Use alternating buffers
                    if group_i % 2 == 0:
                        curr_val, curr_addr = v_node_val_A, v_node_addr_A
                        next_val, next_addr = v_node_val_B, v_node_addr_B
                    else:
                        curr_val, curr_addr = v_node_val_B, v_node_addr_B
                        next_val, next_addr = v_node_val_A, v_node_addr_A

                    # Prepare next group's addresses (2 cycles)
                    self.instrs.append({"valu": [
                        ("+", next_addr[p[0]], v_idx[next_c[0]], v_forest_p),
                        ("+", next_addr[p[1]], v_idx[next_c[1]], v_forest_p),
                        ("+", next_addr[p[2]], v_idx[next_c[2]], v_forest_p),
                        ("+", next_addr[p[3]], v_idx[next_c[3]], v_forest_p),
                        ("+", next_addr[p[4]], v_idx[next_c[4]], v_forest_p),
                        ("+", next_addr[p[5]], v_idx[next_c[5]], v_forest_p),
                    ]})
                    self.instrs.append({"valu": [
                        ("+", next_addr[p[6]], v_idx[next_c[6]], v_forest_p),
                        ("+", next_addr[p[7]], v_idx[next_c[7]], v_forest_p),
                        ("^", v_val[curr_c[0]], v_val[curr_c[0]], curr_val[p[0]]),
                        ("^", v_val[curr_c[1]], v_val[curr_c[1]], curr_val[p[1]]),
                        ("^", v_val[curr_c[2]], v_val[curr_c[2]], curr_val[p[2]]),
                        ("^", v_val[curr_c[3]], v_val[curr_c[3]], curr_val[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("^", v_val[curr_c[4]], v_val[curr_c[4]], curr_val[p[4]]),
                        ("^", v_val[curr_c[5]], v_val[curr_c[5]], curr_val[p[5]]),
                        ("^", v_val[curr_c[6]], v_val[curr_c[6]], curr_val[p[6]]),
                        ("^", v_val[curr_c[7]], v_val[curr_c[7]], curr_val[p[7]]),
                    ]})

                    # Emit hash for 8 chunks (24 cycles) overlapped with 32 scattered loads
                    # Hash has 6 stages, each with 4 cycles for 8 chunks
                    scatter_idx = 0
                    for v_c1, v_c3, op1, op2, op3 in v_hash_consts:
                        # Cycle 1: op1+op3 for chunks 0-2 + 2 loads
                        instr = {"valu": [
                            (op1, v_tmp1[p[0]], v_val[curr_c[0]], v_c1), (op3, v_tmp2[p[0]], v_val[curr_c[0]], v_c3),
                            (op1, v_tmp1[p[1]], v_val[curr_c[1]], v_c1), (op3, v_tmp2[p[1]], v_val[curr_c[1]], v_c3),
                            (op1, v_tmp1[p[2]], v_val[curr_c[2]], v_c1), (op3, v_tmp2[p[2]], v_val[curr_c[2]], v_c3),
                        ]}
                        if scatter_idx < 32:
                            chunk = scatter_idx // 4
                            pair = scatter_idx % 4
                            instr["load"] = [
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)
                            ]
                            scatter_idx += 1
                        self.instrs.append(instr)

                        # Cycle 2: op1+op3 for chunks 3-5 + 2 loads
                        instr = {"valu": [
                            (op1, v_tmp1[p[3]], v_val[curr_c[3]], v_c1), (op3, v_tmp2[p[3]], v_val[curr_c[3]], v_c3),
                            (op1, v_tmp1[p[4]], v_val[curr_c[4]], v_c1), (op3, v_tmp2[p[4]], v_val[curr_c[4]], v_c3),
                            (op1, v_tmp1[p[5]], v_val[curr_c[5]], v_c1), (op3, v_tmp2[p[5]], v_val[curr_c[5]], v_c3),
                        ]}
                        if scatter_idx < 32:
                            chunk = scatter_idx // 4
                            pair = scatter_idx % 4
                            instr["load"] = [
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)
                            ]
                            scatter_idx += 1
                        self.instrs.append(instr)

                        # Cycle 3: op1+op3 for chunks 6-7, op2 for 0-1 + 2 loads
                        instr = {"valu": [
                            (op1, v_tmp1[p[6]], v_val[curr_c[6]], v_c1), (op3, v_tmp2[p[6]], v_val[curr_c[6]], v_c3),
                            (op1, v_tmp1[p[7]], v_val[curr_c[7]], v_c1), (op3, v_tmp2[p[7]], v_val[curr_c[7]], v_c3),
                            (op2, v_val[curr_c[0]], v_tmp1[p[0]], v_tmp2[p[0]]),
                            (op2, v_val[curr_c[1]], v_tmp1[p[1]], v_tmp2[p[1]]),
                        ]}
                        if scatter_idx < 32:
                            chunk = scatter_idx // 4
                            pair = scatter_idx % 4
                            instr["load"] = [
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)
                            ]
                            scatter_idx += 1
                        self.instrs.append(instr)

                        # Cycle 4: op2 for chunks 2-7 + 2 loads
                        instr = {"valu": [
                            (op2, v_val[curr_c[2]], v_tmp1[p[2]], v_tmp2[p[2]]),
                            (op2, v_val[curr_c[3]], v_tmp1[p[3]], v_tmp2[p[3]]),
                            (op2, v_val[curr_c[4]], v_tmp1[p[4]], v_tmp2[p[4]]),
                            (op2, v_val[curr_c[5]], v_tmp1[p[5]], v_tmp2[p[5]]),
                            (op2, v_val[curr_c[6]], v_tmp1[p[6]], v_tmp2[p[6]]),
                            (op2, v_val[curr_c[7]], v_tmp1[p[7]], v_tmp2[p[7]]),
                        ]}
                        if scatter_idx < 32:
                            chunk = scatter_idx // 4
                            pair = scatter_idx % 4
                            instr["load"] = [
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)
                            ]
                            scatter_idx += 1
                        self.instrs.append(instr)

                    # Index computation for 8 chunks (9 cycles) with remaining loads
                    # bit = val & 1, shifted = idx << 1 for chunks 0-2
                    instr = {"valu": [
                        ("&", v_tmp1[p[0]], v_val[curr_c[0]], v_one), ("<<", v_tmp2[p[0]], v_idx[curr_c[0]], v_one),
                        ("&", v_tmp1[p[1]], v_val[curr_c[1]], v_one), ("<<", v_tmp2[p[1]], v_idx[curr_c[1]], v_one),
                        ("&", v_tmp1[p[2]], v_val[curr_c[2]], v_one), ("<<", v_tmp2[p[2]], v_idx[curr_c[2]], v_one),
                    ]}
                    if scatter_idx < 32:
                        chunk = scatter_idx // 4
                        pair = scatter_idx % 4
                        instr["load"] = [("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                         ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)]
                        scatter_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [
                        ("&", v_tmp1[p[3]], v_val[curr_c[3]], v_one), ("<<", v_tmp2[p[3]], v_idx[curr_c[3]], v_one),
                        ("&", v_tmp1[p[4]], v_val[curr_c[4]], v_one), ("<<", v_tmp2[p[4]], v_idx[curr_c[4]], v_one),
                        ("&", v_tmp1[p[5]], v_val[curr_c[5]], v_one), ("<<", v_tmp2[p[5]], v_idx[curr_c[5]], v_one),
                    ]}
                    if scatter_idx < 32:
                        chunk = scatter_idx // 4
                        pair = scatter_idx % 4
                        instr["load"] = [("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                         ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)]
                        scatter_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [
                        ("&", v_tmp1[p[6]], v_val[curr_c[6]], v_one), ("<<", v_tmp2[p[6]], v_idx[curr_c[6]], v_one),
                        ("&", v_tmp1[p[7]], v_val[curr_c[7]], v_one), ("<<", v_tmp2[p[7]], v_idx[curr_c[7]], v_one),
                        ("+", v_tmp1[p[0]], v_tmp1[p[0]], v_one), ("+", v_tmp1[p[1]], v_tmp1[p[1]], v_one),
                    ]}
                    if scatter_idx < 32:
                        chunk = scatter_idx // 4
                        pair = scatter_idx % 4
                        instr["load"] = [("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                         ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)]
                        scatter_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [
                        ("+", v_tmp1[p[2]], v_tmp1[p[2]], v_one), ("+", v_tmp1[p[3]], v_tmp1[p[3]], v_one),
                        ("+", v_tmp1[p[4]], v_tmp1[p[4]], v_one), ("+", v_tmp1[p[5]], v_tmp1[p[5]], v_one),
                        ("+", v_tmp1[p[6]], v_tmp1[p[6]], v_one), ("+", v_tmp1[p[7]], v_tmp1[p[7]], v_one),
                    ]}
                    if scatter_idx < 32:
                        chunk = scatter_idx // 4
                        pair = scatter_idx % 4
                        instr["load"] = [("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                         ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)]
                        scatter_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [
                        ("+", v_idx[curr_c[0]], v_tmp2[p[0]], v_tmp1[p[0]]), ("+", v_idx[curr_c[1]], v_tmp2[p[1]], v_tmp1[p[1]]),
                        ("+", v_idx[curr_c[2]], v_tmp2[p[2]], v_tmp1[p[2]]), ("+", v_idx[curr_c[3]], v_tmp2[p[3]], v_tmp1[p[3]]),
                        ("+", v_idx[curr_c[4]], v_tmp2[p[4]], v_tmp1[p[4]]), ("+", v_idx[curr_c[5]], v_tmp2[p[5]], v_tmp1[p[5]]),
                    ]}
                    if scatter_idx < 32:
                        chunk = scatter_idx // 4
                        pair = scatter_idx % 4
                        instr["load"] = [("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                         ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)]
                        scatter_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [
                        ("+", v_idx[curr_c[6]], v_tmp2[p[6]], v_tmp1[p[6]]), ("+", v_idx[curr_c[7]], v_tmp2[p[7]], v_tmp1[p[7]]),
                        ("<", v_tmp1[p[0]], v_idx[curr_c[0]], v_n_nodes), ("<", v_tmp1[p[1]], v_idx[curr_c[1]], v_n_nodes),
                        ("<", v_tmp1[p[2]], v_idx[curr_c[2]], v_n_nodes), ("<", v_tmp1[p[3]], v_idx[curr_c[3]], v_n_nodes),
                    ]}
                    if scatter_idx < 32:
                        chunk = scatter_idx // 4
                        pair = scatter_idx % 4
                        instr["load"] = [("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                         ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)]
                        scatter_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [
                        ("<", v_tmp1[p[4]], v_idx[curr_c[4]], v_n_nodes), ("<", v_tmp1[p[5]], v_idx[curr_c[5]], v_n_nodes),
                        ("<", v_tmp1[p[6]], v_idx[curr_c[6]], v_n_nodes), ("<", v_tmp1[p[7]], v_idx[curr_c[7]], v_n_nodes),
                        ("*", v_idx[curr_c[0]], v_idx[curr_c[0]], v_tmp1[p[0]]), ("*", v_idx[curr_c[1]], v_idx[curr_c[1]], v_tmp1[p[1]]),
                    ]}
                    if scatter_idx < 32:
                        chunk = scatter_idx // 4
                        pair = scatter_idx % 4
                        instr["load"] = [("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                         ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)]
                        scatter_idx += 1
                    self.instrs.append(instr)

                    instr = {"valu": [
                        ("*", v_idx[curr_c[2]], v_idx[curr_c[2]], v_tmp1[p[2]]), ("*", v_idx[curr_c[3]], v_idx[curr_c[3]], v_tmp1[p[3]]),
                        ("*", v_idx[curr_c[4]], v_idx[curr_c[4]], v_tmp1[p[4]]), ("*", v_idx[curr_c[5]], v_idx[curr_c[5]], v_tmp1[p[5]]),
                        ("*", v_idx[curr_c[6]], v_idx[curr_c[6]], v_tmp1[p[6]]), ("*", v_idx[curr_c[7]], v_idx[curr_c[7]], v_tmp1[p[7]]),
                    ]}
                    if scatter_idx < 32:
                        chunk = scatter_idx // 4
                        pair = scatter_idx % 4
                        instr["load"] = [("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                         ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)]
                        scatter_idx += 1
                    self.instrs.append(instr)

                    # Emit remaining loads if any
                    while scatter_idx < 32:
                        chunk = scatter_idx // 4
                        pair = scatter_idx % 4
                        self.instrs.append({"load": [
                            ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                            ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)
                        ]})
                        scatter_idx += 1

                # Last group: just hash (no next group to load)
                last_base = (n_groups - 1) * PIPE
                last_c = [last_base + j for j in range(PIPE)]
                p = [j % PIPE for j in range(PIPE)]
                if (n_groups - 1) % 2 == 0:
                    last_val = v_node_val_A
                else:
                    last_val = v_node_val_B

                # XOR for last group
                self.instrs.append({"valu": [
                    ("^", v_val[last_c[0]], v_val[last_c[0]], last_val[p[0]]),
                    ("^", v_val[last_c[1]], v_val[last_c[1]], last_val[p[1]]),
                    ("^", v_val[last_c[2]], v_val[last_c[2]], last_val[p[2]]),
                    ("^", v_val[last_c[3]], v_val[last_c[3]], last_val[p[3]]),
                    ("^", v_val[last_c[4]], v_val[last_c[4]], last_val[p[4]]),
                    ("^", v_val[last_c[5]], v_val[last_c[5]], last_val[p[5]]),
                ]})
                self.instrs.append({"valu": [
                    ("^", v_val[last_c[6]], v_val[last_c[6]], last_val[p[6]]),
                    ("^", v_val[last_c[7]], v_val[last_c[7]], last_val[p[7]]),
                ]})
                emit_hash_and_index_8(last_c, p)

        # ========== FINAL: Store all 32 chunks to memory ==========
        for chunk_i in range(0, n_chunks, 2):
            off0 = offset_consts[chunk_i]
            off1 = offset_consts[chunk_i + 1]
            self.instrs.append({"alu": [("+", tmp_addr[0], inp_idx_p, off0),
                                        ("+", tmp_addr2[0], inp_val_p, off0),
                                        ("+", tmp_addr[1], inp_idx_p, off1),
                                        ("+", tmp_addr2[1], inp_val_p, off1)]})
            self.instrs.append({"store": [("vstore", tmp_addr[0], v_idx[chunk_i]),
                                          ("vstore", tmp_addr2[0], v_val[chunk_i])]})
            self.instrs.append({"store": [("vstore", tmp_addr[1], v_idx[chunk_i + 1]),
                                          ("vstore", tmp_addr2[1], v_val[chunk_i + 1])]})

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
