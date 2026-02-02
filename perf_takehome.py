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
        PIPE = 4  # Process 4 chunks in parallel

        # Allocate vector buffers for ALL 32 chunks (persistent across rounds)
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(n_chunks)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(n_chunks)]

        # Working buffers - double buffer for pipelining (A and B sets)
        # For PIPE=8, we need larger buffers
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

        # Process 4 chunks at a time for round 0
        for base in range(0, n_chunks, PIPE):
            c = [base + j for j in range(PIPE)]  # Chunk indices
            p = [j % PIPE for j in range(PIPE)]  # Working buffer indices

            # Broadcast node value and set idx=0 for all 4 chunks
            self.instrs.append({"valu": [
                ("vbroadcast", v_node_val[p[0]], tmp_node_val),
                ("vbroadcast", v_node_val[p[1]], tmp_node_val),
                ("vbroadcast", v_node_val[p[2]], tmp_node_val),
                ("vbroadcast", v_node_val[p[3]], tmp_node_val),
            ]})
            self.instrs.append({"valu": [
                ("+", v_idx[c[0]], v_zero, v_zero),
                ("+", v_idx[c[1]], v_zero, v_zero),
                ("+", v_idx[c[2]], v_zero, v_zero),
                ("+", v_idx[c[3]], v_zero, v_zero),
            ]})

            # XOR all 4
            self.instrs.append({"valu": [
                ("^", v_val[c[0]], v_val[c[0]], v_node_val[p[0]]),
                ("^", v_val[c[1]], v_val[c[1]], v_node_val[p[1]]),
                ("^", v_val[c[2]], v_val[c[2]], v_node_val[p[2]]),
                ("^", v_val[c[3]], v_val[c[3]], v_node_val[p[3]]),
            ]})

            # Hash all 4 (6 stages × 3 cycles = 18 cycles)
            for v_c1, v_c3, op1, op2, op3 in v_hash_consts:
                self.instrs.append({"valu": [
                    (op1, v_tmp1[p[0]], v_val[c[0]], v_c1), (op3, v_tmp2[p[0]], v_val[c[0]], v_c3),
                    (op1, v_tmp1[p[1]], v_val[c[1]], v_c1), (op3, v_tmp2[p[1]], v_val[c[1]], v_c3),
                ]})
                self.instrs.append({"valu": [
                    (op1, v_tmp1[p[2]], v_val[c[2]], v_c1), (op3, v_tmp2[p[2]], v_val[c[2]], v_c3),
                    (op1, v_tmp1[p[3]], v_val[c[3]], v_c1), (op3, v_tmp2[p[3]], v_val[c[3]], v_c3),
                ]})
                self.instrs.append({"valu": [
                    (op2, v_val[c[0]], v_tmp1[p[0]], v_tmp2[p[0]]),
                    (op2, v_val[c[1]], v_tmp1[p[1]], v_tmp2[p[1]]),
                    (op2, v_val[c[2]], v_tmp1[p[2]], v_tmp2[p[2]]),
                    (op2, v_val[c[3]], v_tmp1[p[3]], v_tmp2[p[3]]),
                ]})

            # Index computation for all 4 (6 cycles)
            self.instrs.append({"valu": [
                ("&", v_tmp1[p[0]], v_val[c[0]], v_one), ("<<", v_tmp2[p[0]], v_idx[c[0]], v_one),
                ("&", v_tmp1[p[1]], v_val[c[1]], v_one), ("<<", v_tmp2[p[1]], v_idx[c[1]], v_one),
            ]})
            self.instrs.append({"valu": [
                ("&", v_tmp1[p[2]], v_val[c[2]], v_one), ("<<", v_tmp2[p[2]], v_idx[c[2]], v_one),
                ("&", v_tmp1[p[3]], v_val[c[3]], v_one), ("<<", v_tmp2[p[3]], v_idx[c[3]], v_one),
            ]})
            self.instrs.append({"valu": [
                ("+", v_tmp1[p[0]], v_tmp1[p[0]], v_one), ("+", v_tmp1[p[1]], v_tmp1[p[1]], v_one),
                ("+", v_tmp1[p[2]], v_tmp1[p[2]], v_one), ("+", v_tmp1[p[3]], v_tmp1[p[3]], v_one),
            ]})
            self.instrs.append({"valu": [
                ("+", v_idx[c[0]], v_tmp2[p[0]], v_tmp1[p[0]]), ("+", v_idx[c[1]], v_tmp2[p[1]], v_tmp1[p[1]]),
                ("+", v_idx[c[2]], v_tmp2[p[2]], v_tmp1[p[2]]), ("+", v_idx[c[3]], v_tmp2[p[3]], v_tmp1[p[3]]),
            ]})
            self.instrs.append({"valu": [
                ("<", v_tmp1[p[0]], v_idx[c[0]], v_n_nodes), ("<", v_tmp1[p[1]], v_idx[c[1]], v_n_nodes),
                ("<", v_tmp1[p[2]], v_idx[c[2]], v_n_nodes), ("<", v_tmp1[p[3]], v_idx[c[3]], v_n_nodes),
            ]})
            self.instrs.append({"valu": [
                ("*", v_idx[c[0]], v_idx[c[0]], v_tmp1[p[0]]), ("*", v_idx[c[1]], v_idx[c[1]], v_tmp1[p[1]]),
                ("*", v_idx[c[2]], v_idx[c[2]], v_tmp1[p[2]]), ("*", v_idx[c[3]], v_idx[c[3]], v_tmp1[p[3]]),
            ]})
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
            self.instrs.append({"valu": [
                ("-", v_tmp1[p[0]], v_idx[c[0]], v_one), ("-", v_tmp2[p[0]], v_two, v_idx[c[0]]),
                ("-", v_tmp1[p[1]], v_idx[c[1]], v_one), ("-", v_tmp2[p[1]], v_two, v_idx[c[1]]),
            ]})
            self.instrs.append({"valu": [
                ("-", v_tmp1[p[2]], v_idx[c[2]], v_one), ("-", v_tmp2[p[2]], v_two, v_idx[c[2]]),
                ("-", v_tmp1[p[3]], v_idx[c[3]], v_one), ("-", v_tmp2[p[3]], v_two, v_idx[c[3]]),
            ]})
            self.instrs.append({"valu": [
                ("*", v_node_addr[p[0]], v_tree2, v_tmp1[p[0]]),
                ("*", v_node_addr[p[1]], v_tree2, v_tmp1[p[1]]),
                ("*", v_node_addr[p[2]], v_tree2, v_tmp1[p[2]]),
                ("*", v_node_addr[p[3]], v_tree2, v_tmp1[p[3]]),
            ]})
            self.instrs.append({"valu": [
                ("multiply_add", v_node_val[p[0]], v_tree1, v_tmp2[p[0]], v_node_addr[p[0]]),
                ("multiply_add", v_node_val[p[1]], v_tree1, v_tmp2[p[1]], v_node_addr[p[1]]),
                ("multiply_add", v_node_val[p[2]], v_tree1, v_tmp2[p[2]], v_node_addr[p[2]]),
                ("multiply_add", v_node_val[p[3]], v_tree1, v_tmp2[p[3]], v_node_addr[p[3]]),
            ]})

            # XOR + Hash (using persistent buffers)
            self.instrs.append({"valu": [
                ("^", v_val[c[0]], v_val[c[0]], v_node_val[p[0]]),
                ("^", v_val[c[1]], v_val[c[1]], v_node_val[p[1]]),
                ("^", v_val[c[2]], v_val[c[2]], v_node_val[p[2]]),
                ("^", v_val[c[3]], v_val[c[3]], v_node_val[p[3]]),
            ]})
            for v_c1, v_c3, op1, op2, op3 in v_hash_consts:
                self.instrs.append({"valu": [
                    (op1, v_tmp1[p[0]], v_val[c[0]], v_c1), (op3, v_tmp2[p[0]], v_val[c[0]], v_c3),
                    (op1, v_tmp1[p[1]], v_val[c[1]], v_c1), (op3, v_tmp2[p[1]], v_val[c[1]], v_c3),
                ]})
                self.instrs.append({"valu": [
                    (op1, v_tmp1[p[2]], v_val[c[2]], v_c1), (op3, v_tmp2[p[2]], v_val[c[2]], v_c3),
                    (op1, v_tmp1[p[3]], v_val[c[3]], v_c1), (op3, v_tmp2[p[3]], v_val[c[3]], v_c3),
                ]})
                self.instrs.append({"valu": [
                    (op2, v_val[c[0]], v_tmp1[p[0]], v_tmp2[p[0]]),
                    (op2, v_val[c[1]], v_tmp1[p[1]], v_tmp2[p[1]]),
                    (op2, v_val[c[2]], v_tmp1[p[2]], v_tmp2[p[2]]),
                    (op2, v_val[c[3]], v_tmp1[p[3]], v_tmp2[p[3]]),
                ]})

            # Index computation
            self.instrs.append({"valu": [
                ("&", v_tmp1[p[0]], v_val[c[0]], v_one), ("<<", v_tmp2[p[0]], v_idx[c[0]], v_one),
                ("&", v_tmp1[p[1]], v_val[c[1]], v_one), ("<<", v_tmp2[p[1]], v_idx[c[1]], v_one),
            ]})
            self.instrs.append({"valu": [
                ("&", v_tmp1[p[2]], v_val[c[2]], v_one), ("<<", v_tmp2[p[2]], v_idx[c[2]], v_one),
                ("&", v_tmp1[p[3]], v_val[c[3]], v_one), ("<<", v_tmp2[p[3]], v_idx[c[3]], v_one),
            ]})
            self.instrs.append({"valu": [
                ("+", v_tmp1[p[0]], v_tmp1[p[0]], v_one), ("+", v_tmp1[p[1]], v_tmp1[p[1]], v_one),
                ("+", v_tmp1[p[2]], v_tmp1[p[2]], v_one), ("+", v_tmp1[p[3]], v_tmp1[p[3]], v_one),
            ]})
            self.instrs.append({"valu": [
                ("+", v_idx[c[0]], v_tmp2[p[0]], v_tmp1[p[0]]), ("+", v_idx[c[1]], v_tmp2[p[1]], v_tmp1[p[1]]),
                ("+", v_idx[c[2]], v_tmp2[p[2]], v_tmp1[p[2]]), ("+", v_idx[c[3]], v_tmp2[p[3]], v_tmp1[p[3]]),
            ]})
            self.instrs.append({"valu": [
                ("<", v_tmp1[p[0]], v_idx[c[0]], v_n_nodes), ("<", v_tmp1[p[1]], v_idx[c[1]], v_n_nodes),
                ("<", v_tmp1[p[2]], v_idx[c[2]], v_n_nodes), ("<", v_tmp1[p[3]], v_idx[c[3]], v_n_nodes),
            ]})
            self.instrs.append({"valu": [
                ("*", v_idx[c[0]], v_idx[c[0]], v_tmp1[p[0]]), ("*", v_idx[c[1]], v_idx[c[1]], v_tmp1[p[1]]),
                ("*", v_idx[c[2]], v_idx[c[2]], v_tmp1[p[2]]), ("*", v_idx[c[3]], v_idx[c[3]], v_tmp1[p[3]]),
            ]})
            # NO STORES - keep in scratch

        # ========== ROUNDS 2+: Different handling based on round number ==========
        # Due to tree wrapping at n_nodes, after round 10 all indices reset to 0
        # Round 11 = like round 0, Round 12 = like round 1, etc.

        def emit_hash_and_index(c, p):
            """Emit hash and index computation for 4 chunks"""
            for v_c1, v_c3, op1, op2, op3 in v_hash_consts:
                self.instrs.append({"valu": [
                    (op1, v_tmp1[p[0]], v_val[c[0]], v_c1), (op3, v_tmp2[p[0]], v_val[c[0]], v_c3),
                    (op1, v_tmp1[p[1]], v_val[c[1]], v_c1), (op3, v_tmp2[p[1]], v_val[c[1]], v_c3),
                ]})
                self.instrs.append({"valu": [
                    (op1, v_tmp1[p[2]], v_val[c[2]], v_c1), (op3, v_tmp2[p[2]], v_val[c[2]], v_c3),
                    (op1, v_tmp1[p[3]], v_val[c[3]], v_c1), (op3, v_tmp2[p[3]], v_val[c[3]], v_c3),
                ]})
                self.instrs.append({"valu": [
                    (op2, v_val[c[0]], v_tmp1[p[0]], v_tmp2[p[0]]),
                    (op2, v_val[c[1]], v_tmp1[p[1]], v_tmp2[p[1]]),
                    (op2, v_val[c[2]], v_tmp1[p[2]], v_tmp2[p[2]]),
                    (op2, v_val[c[3]], v_tmp1[p[3]], v_tmp2[p[3]]),
                ]})

            # Index computation: new_idx = 2*idx + 1 + (val & 1)
            self.instrs.append({"valu": [
                ("&", v_tmp1[p[0]], v_val[c[0]], v_one), ("<<", v_tmp2[p[0]], v_idx[c[0]], v_one),
                ("&", v_tmp1[p[1]], v_val[c[1]], v_one), ("<<", v_tmp2[p[1]], v_idx[c[1]], v_one),
            ]})
            self.instrs.append({"valu": [
                ("&", v_tmp1[p[2]], v_val[c[2]], v_one), ("<<", v_tmp2[p[2]], v_idx[c[2]], v_one),
                ("&", v_tmp1[p[3]], v_val[c[3]], v_one), ("<<", v_tmp2[p[3]], v_idx[c[3]], v_one),
            ]})
            self.instrs.append({"valu": [
                ("+", v_tmp1[p[0]], v_tmp1[p[0]], v_one), ("+", v_tmp1[p[1]], v_tmp1[p[1]], v_one),
                ("+", v_tmp1[p[2]], v_tmp1[p[2]], v_one), ("+", v_tmp1[p[3]], v_tmp1[p[3]], v_one),
            ]})
            self.instrs.append({"valu": [
                ("+", v_idx[c[0]], v_tmp2[p[0]], v_tmp1[p[0]]), ("+", v_idx[c[1]], v_tmp2[p[1]], v_tmp1[p[1]]),
                ("+", v_idx[c[2]], v_tmp2[p[2]], v_tmp1[p[2]]), ("+", v_idx[c[3]], v_tmp2[p[3]], v_tmp1[p[3]]),
            ]})
            self.instrs.append({"valu": [
                ("<", v_tmp1[p[0]], v_idx[c[0]], v_n_nodes), ("<", v_tmp1[p[1]], v_idx[c[1]], v_n_nodes),
                ("<", v_tmp1[p[2]], v_idx[c[2]], v_n_nodes), ("<", v_tmp1[p[3]], v_idx[c[3]], v_n_nodes),
            ]})
            self.instrs.append({"valu": [
                ("*", v_idx[c[0]], v_idx[c[0]], v_tmp1[p[0]]), ("*", v_idx[c[1]], v_idx[c[1]], v_tmp1[p[1]]),
                ("*", v_idx[c[2]], v_idx[c[2]], v_tmp1[p[2]]), ("*", v_idx[c[3]], v_idx[c[3]], v_tmp1[p[3]]),
            ]})

        for round_i in range(2, rounds):
            # Determine effective round after considering tree wrap
            # After round 10, all indices wrap to 0, so round 11 behaves like round 0
            effective_round = round_i
            if round_i >= 11:
                effective_round = round_i - 11

            if effective_round == 0:
                # All indices are 0 - use broadcast of tree[0]
                for base in range(0, n_chunks, PIPE):
                    c = [base + j for j in range(PIPE)]
                    p = [j % PIPE for j in range(PIPE)]

                    # Broadcast tree[0] and set idx=0
                    self.instrs.append({"valu": [
                        ("vbroadcast", v_node_val[p[0]], tmp_node_val),
                        ("vbroadcast", v_node_val[p[1]], tmp_node_val),
                        ("vbroadcast", v_node_val[p[2]], tmp_node_val),
                        ("vbroadcast", v_node_val[p[3]], tmp_node_val),
                    ]})
                    self.instrs.append({"valu": [
                        ("+", v_idx[c[0]], v_zero, v_zero),
                        ("+", v_idx[c[1]], v_zero, v_zero),
                        ("+", v_idx[c[2]], v_zero, v_zero),
                        ("+", v_idx[c[3]], v_zero, v_zero),
                    ]})

                    # XOR + Hash
                    self.instrs.append({"valu": [
                        ("^", v_val[c[0]], v_val[c[0]], v_node_val[p[0]]),
                        ("^", v_val[c[1]], v_val[c[1]], v_node_val[p[1]]),
                        ("^", v_val[c[2]], v_val[c[2]], v_node_val[p[2]]),
                        ("^", v_val[c[3]], v_val[c[3]], v_node_val[p[3]]),
                    ]})
                    emit_hash_and_index(c, p)

            elif effective_round == 1:
                # Indices are 1 or 2 - use arithmetic selection (like round 1)
                for base in range(0, n_chunks, PIPE):
                    c = [base + j for j in range(PIPE)]
                    p = [j % PIPE for j in range(PIPE)]

                    # Compute node_val using arithmetic (idx is 1 or 2)
                    self.instrs.append({"valu": [
                        ("-", v_tmp1[p[0]], v_idx[c[0]], v_one), ("-", v_tmp2[p[0]], v_two, v_idx[c[0]]),
                        ("-", v_tmp1[p[1]], v_idx[c[1]], v_one), ("-", v_tmp2[p[1]], v_two, v_idx[c[1]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("-", v_tmp1[p[2]], v_idx[c[2]], v_one), ("-", v_tmp2[p[2]], v_two, v_idx[c[2]]),
                        ("-", v_tmp1[p[3]], v_idx[c[3]], v_one), ("-", v_tmp2[p[3]], v_two, v_idx[c[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("*", v_node_addr[p[0]], v_tree2, v_tmp1[p[0]]),
                        ("*", v_node_addr[p[1]], v_tree2, v_tmp1[p[1]]),
                        ("*", v_node_addr[p[2]], v_tree2, v_tmp1[p[2]]),
                        ("*", v_node_addr[p[3]], v_tree2, v_tmp1[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node_val[p[0]], v_tree1, v_tmp2[p[0]], v_node_addr[p[0]]),
                        ("multiply_add", v_node_val[p[1]], v_tree1, v_tmp2[p[1]], v_node_addr[p[1]]),
                        ("multiply_add", v_node_val[p[2]], v_tree1, v_tmp2[p[2]], v_node_addr[p[2]]),
                        ("multiply_add", v_node_val[p[3]], v_tree1, v_tmp2[p[3]], v_node_addr[p[3]]),
                    ]})

                    # XOR + Hash
                    self.instrs.append({"valu": [
                        ("^", v_val[c[0]], v_val[c[0]], v_node_val[p[0]]),
                        ("^", v_val[c[1]], v_val[c[1]], v_node_val[p[1]]),
                        ("^", v_val[c[2]], v_val[c[2]], v_node_val[p[2]]),
                        ("^", v_val[c[3]], v_val[c[3]], v_node_val[p[3]]),
                    ]})
                    emit_hash_and_index(c, p)

            elif effective_round == 2:
                # Indices are 3, 4, 5, or 6 - use arithmetic selection from 4 values
                for base in range(0, n_chunks, PIPE):
                    c = [base + j for j in range(PIPE)]
                    p = [j % PIPE for j in range(PIPE)]

                    # Compute offset = idx - 3, then select using bit operations
                    # bit0 = offset & 1, bit1 = offset >> 1
                    # result = (tree3*(1-bit0) + tree4*bit0)*(1-bit1) + (tree5*(1-bit0) + tree6*bit0)*bit1
                    self.instrs.append({"valu": [
                        ("-", v_tmp1[p[0]], v_idx[c[0]], v_three),  # offset
                        ("-", v_tmp1[p[1]], v_idx[c[1]], v_three),
                        ("-", v_tmp1[p[2]], v_idx[c[2]], v_three),
                        ("-", v_tmp1[p[3]], v_idx[c[3]], v_three),
                    ]})
                    # bit0 = offset & 1
                    self.instrs.append({"valu": [
                        ("&", v_tmp2[p[0]], v_tmp1[p[0]], v_one),
                        ("&", v_tmp2[p[1]], v_tmp1[p[1]], v_one),
                        ("&", v_tmp2[p[2]], v_tmp1[p[2]], v_one),
                        ("&", v_tmp2[p[3]], v_tmp1[p[3]], v_one),
                    ]})
                    # bit1 = offset >> 1
                    self.instrs.append({"valu": [
                        (">>", v_tmp1[p[0]], v_tmp1[p[0]], v_one),
                        (">>", v_tmp1[p[1]], v_tmp1[p[1]], v_one),
                        (">>", v_tmp1[p[2]], v_tmp1[p[2]], v_one),
                        (">>", v_tmp1[p[3]], v_tmp1[p[3]], v_one),
                    ]})
                    # inv_bit0 = 1 - bit0 (store in v_node_addr)
                    self.instrs.append({"valu": [
                        ("-", v_node_addr[p[0]], v_one, v_tmp2[p[0]]),
                        ("-", v_node_addr[p[1]], v_one, v_tmp2[p[1]]),
                        ("-", v_node_addr[p[2]], v_one, v_tmp2[p[2]]),
                        ("-", v_node_addr[p[3]], v_one, v_tmp2[p[3]]),
                    ]})
                    # first_half = tree3 * inv_bit0 + tree4 * bit0
                    # Use v_node_val temporarily for tree3*inv_bit0
                    self.instrs.append({"valu": [
                        ("*", v_node_val[p[0]], v_tree3, v_node_addr[p[0]]),
                        ("*", v_node_val[p[1]], v_tree3, v_node_addr[p[1]]),
                        ("*", v_node_val[p[2]], v_tree3, v_node_addr[p[2]]),
                        ("*", v_node_val[p[3]], v_tree3, v_node_addr[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node_val[p[0]], v_tree4, v_tmp2[p[0]], v_node_val[p[0]]),
                        ("multiply_add", v_node_val[p[1]], v_tree4, v_tmp2[p[1]], v_node_val[p[1]]),
                        ("multiply_add", v_node_val[p[2]], v_tree4, v_tmp2[p[2]], v_node_val[p[2]]),
                        ("multiply_add", v_node_val[p[3]], v_tree4, v_tmp2[p[3]], v_node_val[p[3]]),
                    ]})
                    # second_half = tree5 * inv_bit0 + tree6 * bit0 (reuse v_node_addr for result)
                    self.instrs.append({"valu": [
                        ("*", v_node_addr[p[0]], v_tree5, v_node_addr[p[0]]),
                        ("*", v_node_addr[p[1]], v_tree5, v_node_addr[p[1]]),
                        ("*", v_node_addr[p[2]], v_tree5, v_node_addr[p[2]]),
                        ("*", v_node_addr[p[3]], v_tree5, v_node_addr[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node_addr[p[0]], v_tree6, v_tmp2[p[0]], v_node_addr[p[0]]),
                        ("multiply_add", v_node_addr[p[1]], v_tree6, v_tmp2[p[1]], v_node_addr[p[1]]),
                        ("multiply_add", v_node_addr[p[2]], v_tree6, v_tmp2[p[2]], v_node_addr[p[2]]),
                        ("multiply_add", v_node_addr[p[3]], v_tree6, v_tmp2[p[3]], v_node_addr[p[3]]),
                    ]})
                    # inv_bit1 = 1 - bit1 (reuse v_tmp2)
                    self.instrs.append({"valu": [
                        ("-", v_tmp2[p[0]], v_one, v_tmp1[p[0]]),
                        ("-", v_tmp2[p[1]], v_one, v_tmp1[p[1]]),
                        ("-", v_tmp2[p[2]], v_one, v_tmp1[p[2]]),
                        ("-", v_tmp2[p[3]], v_one, v_tmp1[p[3]]),
                    ]})
                    # result = first_half * inv_bit1 + second_half * bit1
                    self.instrs.append({"valu": [
                        ("*", v_node_val[p[0]], v_node_val[p[0]], v_tmp2[p[0]]),
                        ("*", v_node_val[p[1]], v_node_val[p[1]], v_tmp2[p[1]]),
                        ("*", v_node_val[p[2]], v_node_val[p[2]], v_tmp2[p[2]]),
                        ("*", v_node_val[p[3]], v_node_val[p[3]], v_tmp2[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("multiply_add", v_node_val[p[0]], v_node_addr[p[0]], v_tmp1[p[0]], v_node_val[p[0]]),
                        ("multiply_add", v_node_val[p[1]], v_node_addr[p[1]], v_tmp1[p[1]], v_node_val[p[1]]),
                        ("multiply_add", v_node_val[p[2]], v_node_addr[p[2]], v_tmp1[p[2]], v_node_val[p[2]]),
                        ("multiply_add", v_node_val[p[3]], v_node_addr[p[3]], v_tmp1[p[3]], v_node_val[p[3]]),
                    ]})

                    # XOR + Hash
                    self.instrs.append({"valu": [
                        ("^", v_val[c[0]], v_val[c[0]], v_node_val[p[0]]),
                        ("^", v_val[c[1]], v_val[c[1]], v_node_val[p[1]]),
                        ("^", v_val[c[2]], v_val[c[2]], v_node_val[p[2]]),
                        ("^", v_val[c[3]], v_val[c[3]], v_node_val[p[3]]),
                    ]})
                    emit_hash_and_index(c, p)

            else:
                # General case: use pipelined scattered loads
                # Overlap hash of group N with loads of group N+1
                n_groups = n_chunks // PIPE  # 8 groups

                # First group: just load (no prior hash to overlap with)
                base = 0
                c = [base + j for j in range(PIPE)]
                p = [j % PIPE for j in range(PIPE)]

                # Compute addresses and start loading for first group into buffer A
                self.instrs.append({"valu": [
                    ("+", v_node_addr_A[p[0]], v_idx[c[0]], v_forest_p),
                    ("+", v_node_addr_A[p[1]], v_idx[c[1]], v_forest_p),
                    ("+", v_node_addr_A[p[2]], v_idx[c[2]], v_forest_p),
                    ("+", v_node_addr_A[p[3]], v_idx[c[3]], v_forest_p),
                ]})
                for chunk in range(4):
                    for pair in range(4):
                        self.instrs.append({"load": [
                            ("load_offset", v_node_val_A[p[chunk]], v_node_addr_A[p[chunk]], pair*2),
                            ("load_offset", v_node_val_A[p[chunk]], v_node_addr_A[p[chunk]], pair*2+1)
                        ]})

                # Middle groups: overlap hash of current with loads of next
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

                    # Prepare next group's addresses
                    self.instrs.append({"valu": [
                        ("+", next_addr[p[0]], v_idx[next_c[0]], v_forest_p),
                        ("+", next_addr[p[1]], v_idx[next_c[1]], v_forest_p),
                        ("+", next_addr[p[2]], v_idx[next_c[2]], v_forest_p),
                        ("+", next_addr[p[3]], v_idx[next_c[3]], v_forest_p),
                    ]})

                    # XOR for current group
                    self.instrs.append({"valu": [
                        ("^", v_val[curr_c[0]], v_val[curr_c[0]], curr_val[p[0]]),
                        ("^", v_val[curr_c[1]], v_val[curr_c[1]], curr_val[p[1]]),
                        ("^", v_val[curr_c[2]], v_val[curr_c[2]], curr_val[p[2]]),
                        ("^", v_val[curr_c[3]], v_val[curr_c[3]], curr_val[p[3]]),
                    ]})

                    # Hash stages 0-5 overlapped with scattered loads for next group
                    scatter_idx = 0
                    for v_c1, v_c3, op1, op2, op3 in v_hash_consts:
                        # Hash part 1 + loads
                        instr = {"valu": [
                            (op1, v_tmp1[p[0]], v_val[curr_c[0]], v_c1), (op3, v_tmp2[p[0]], v_val[curr_c[0]], v_c3),
                            (op1, v_tmp1[p[1]], v_val[curr_c[1]], v_c1), (op3, v_tmp2[p[1]], v_val[curr_c[1]], v_c3),
                        ]}
                        if scatter_idx < 16:
                            chunk = scatter_idx // 4
                            pair = scatter_idx % 4
                            instr["load"] = [
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)
                            ]
                            scatter_idx += 1
                        self.instrs.append(instr)

                        # Hash part 2 + loads
                        instr = {"valu": [
                            (op1, v_tmp1[p[2]], v_val[curr_c[2]], v_c1), (op3, v_tmp2[p[2]], v_val[curr_c[2]], v_c3),
                            (op1, v_tmp1[p[3]], v_val[curr_c[3]], v_c1), (op3, v_tmp2[p[3]], v_val[curr_c[3]], v_c3),
                        ]}
                        if scatter_idx < 16:
                            chunk = scatter_idx // 4
                            pair = scatter_idx % 4
                            instr["load"] = [
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)
                            ]
                            scatter_idx += 1
                        self.instrs.append(instr)

                        # Hash part 3 + loads
                        instr = {"valu": [
                            (op2, v_val[curr_c[0]], v_tmp1[p[0]], v_tmp2[p[0]]),
                            (op2, v_val[curr_c[1]], v_tmp1[p[1]], v_tmp2[p[1]]),
                            (op2, v_val[curr_c[2]], v_tmp1[p[2]], v_tmp2[p[2]]),
                            (op2, v_val[curr_c[3]], v_tmp1[p[3]], v_tmp2[p[3]]),
                        ]}
                        if scatter_idx < 16:
                            chunk = scatter_idx // 4
                            pair = scatter_idx % 4
                            instr["load"] = [
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)
                            ]
                            scatter_idx += 1
                        self.instrs.append(instr)

                    # Index computation + remaining loads
                    index_instrs = [
                        {"valu": [("&", v_tmp1[p[0]], v_val[curr_c[0]], v_one), ("<<", v_tmp2[p[0]], v_idx[curr_c[0]], v_one),
                                  ("&", v_tmp1[p[1]], v_val[curr_c[1]], v_one), ("<<", v_tmp2[p[1]], v_idx[curr_c[1]], v_one)]},
                        {"valu": [("&", v_tmp1[p[2]], v_val[curr_c[2]], v_one), ("<<", v_tmp2[p[2]], v_idx[curr_c[2]], v_one),
                                  ("&", v_tmp1[p[3]], v_val[curr_c[3]], v_one), ("<<", v_tmp2[p[3]], v_idx[curr_c[3]], v_one)]},
                        {"valu": [("+", v_tmp1[p[0]], v_tmp1[p[0]], v_one), ("+", v_tmp1[p[1]], v_tmp1[p[1]], v_one),
                                  ("+", v_tmp1[p[2]], v_tmp1[p[2]], v_one), ("+", v_tmp1[p[3]], v_tmp1[p[3]], v_one)]},
                        {"valu": [("+", v_idx[curr_c[0]], v_tmp2[p[0]], v_tmp1[p[0]]), ("+", v_idx[curr_c[1]], v_tmp2[p[1]], v_tmp1[p[1]]),
                                  ("+", v_idx[curr_c[2]], v_tmp2[p[2]], v_tmp1[p[2]]), ("+", v_idx[curr_c[3]], v_tmp2[p[3]], v_tmp1[p[3]])]},
                        {"valu": [("<", v_tmp1[p[0]], v_idx[curr_c[0]], v_n_nodes), ("<", v_tmp1[p[1]], v_idx[curr_c[1]], v_n_nodes),
                                  ("<", v_tmp1[p[2]], v_idx[curr_c[2]], v_n_nodes), ("<", v_tmp1[p[3]], v_idx[curr_c[3]], v_n_nodes)]},
                        {"valu": [("*", v_idx[curr_c[0]], v_idx[curr_c[0]], v_tmp1[p[0]]), ("*", v_idx[curr_c[1]], v_idx[curr_c[1]], v_tmp1[p[1]]),
                                  ("*", v_idx[curr_c[2]], v_idx[curr_c[2]], v_tmp1[p[2]]), ("*", v_idx[curr_c[3]], v_idx[curr_c[3]], v_tmp1[p[3]])]},
                    ]
                    for instr in index_instrs:
                        if scatter_idx < 16:
                            chunk = scatter_idx // 4
                            pair = scatter_idx % 4
                            instr["load"] = [
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2),
                                ("load_offset", next_val[p[chunk]], next_addr[p[chunk]], pair*2+1)
                            ]
                            scatter_idx += 1
                        self.instrs.append(instr)

                    # Any remaining loads
                    while scatter_idx < 16:
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

                self.instrs.append({"valu": [
                    ("^", v_val[last_c[0]], v_val[last_c[0]], last_val[p[0]]),
                    ("^", v_val[last_c[1]], v_val[last_c[1]], last_val[p[1]]),
                    ("^", v_val[last_c[2]], v_val[last_c[2]], last_val[p[2]]),
                    ("^", v_val[last_c[3]], v_val[last_c[3]], last_val[p[3]]),
                ]})
                emit_hash_and_index(last_c, p)

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
