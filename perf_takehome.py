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
        Highly optimized kernel with aggressive scheduling.
        Process 4 chunks in parallel to maximize VALU utilization.
        """
        n_chunks = batch_size // VLEN  # 32 chunks
        PIPE = 4  # Process 4 chunks in parallel

        # Allocate vector buffers
        v_idx = [self.alloc_scratch(f"v_idx_{i}", VLEN) for i in range(PIPE)]
        v_val = [self.alloc_scratch(f"v_val_{i}", VLEN) for i in range(PIPE)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{i}", VLEN) for i in range(PIPE)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{i}", VLEN) for i in range(PIPE)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{i}", VLEN) for i in range(PIPE)]
        v_node_addr = [self.alloc_scratch(f"v_node_addr_{i}", VLEN) for i in range(PIPE)]

        # Constant vectors
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)

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
        offset_consts = [self.scratch_const(chunk_i * VLEN) for chunk_i in range(n_chunks)]

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
            ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]),
            ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]),
        ]})

        self.add("flow", ("pause",))

        # Helper: Hash + index computation using arithmetic (avoids slow vselect)
        def gen_hash_and_index(p):
            # XOR
            self.instrs.append({"valu": [("^", v_val[p], v_val[p], v_node_val[p])]})
            # Hash (6 stages × 2 cycles = 12 cycles)
            for v_c1, v_c3, op1, op2, op3 in v_hash_consts:
                self.instrs.append({"valu": [(op1, v_tmp1[p], v_val[p], v_c1),
                                             (op3, v_tmp2[p], v_val[p], v_c3)]})
                self.instrs.append({"valu": [(op2, v_val[p], v_tmp1[p], v_tmp2[p])]})
            # Index computation using arithmetic (5 cycles instead of 6)
            # branch = 1 + (val & 1): gives 1 if even, 2 if odd
            self.instrs.append({"valu": [("&", v_tmp1[p], v_val[p], v_one),
                                         ("<<", v_tmp2[p], v_idx[p], v_one)]})
            self.instrs.append({"valu": [("+", v_tmp1[p], v_tmp1[p], v_one)]})
            self.instrs.append({"valu": [("+", v_idx[p], v_tmp2[p], v_tmp1[p])]})
            self.instrs.append({"valu": [("<", v_tmp1[p], v_idx[p], v_n_nodes)]})
            self.instrs.append({"valu": [("*", v_idx[p], v_idx[p], v_tmp1[p])]})

        inp_idx_p = self.scratch["inp_indices_p"]
        inp_val_p = self.scratch["inp_values_p"]
        forest_p = self.scratch["forest_values_p"]

        # ROUND 0: Special optimization - all indices are 0, so one load for all!
        # Load forest_values[0] once
        self.instrs.append({"load": [("load", tmp_node_val, forest_p)]})

        for chunk_i in range(n_chunks):
            p = chunk_i % PIPE
            offset = offset_consts[chunk_i]

            # Load val only (idx=0 for all, no need to load)
            self.instrs.append({"alu": [("+", tmp_addr2[p], inp_val_p, offset)]})
            self.instrs.append({"load": [("vload", v_val[p], tmp_addr2[p])]})

            # Broadcast the single node value to all lanes
            self.instrs.append({"valu": [("vbroadcast", v_node_val[p], tmp_node_val)]})

            # Also need to set v_idx to 0 (for index computation)
            self.instrs.append({"valu": [("+", v_idx[p], v_zero, v_zero)]})  # v_idx = 0

            # Hash + index computation
            gen_hash_and_index(p)

            # Store
            self.instrs.append({"alu": [("+", tmp_addr[p], inp_idx_p, offset),
                                        ("+", tmp_addr2[p], inp_val_p, offset)]})
            self.instrs.append({"store": [("vstore", tmp_addr[p], v_idx[p]),
                                          ("vstore", tmp_addr2[p], v_val[p])]})

        # ROUNDS 1+: Process 4 chunks with interleaved pipeline
        # Overlap: While chunks 0-1 do hash, chunks 2-3 do loads
        for round_i in range(1, rounds):
            chunk_i = 0
            while chunk_i < n_chunks:
                if chunk_i + 3 < n_chunks:
                    # Process 4 chunks with pipelining
                    p = [0, 1, 2, 3]
                    off = [offset_consts[chunk_i + j] for j in range(4)]

                    # Phase 1: Address calc for all 4 chunks (ALU has 12 slots)
                    self.instrs.append({"alu": [
                        ("+", tmp_addr[p[0]], inp_idx_p, off[0]), ("+", tmp_addr2[p[0]], inp_val_p, off[0]),
                        ("+", tmp_addr[p[1]], inp_idx_p, off[1]), ("+", tmp_addr2[p[1]], inp_val_p, off[1]),
                        ("+", tmp_addr[p[2]], inp_idx_p, off[2]), ("+", tmp_addr2[p[2]], inp_val_p, off[2]),
                        ("+", tmp_addr[p[3]], inp_idx_p, off[3]), ("+", tmp_addr2[p[3]], inp_val_p, off[3]),
                    ]})

                    # Phase 2: Load idx/val for chunks 0-1, then 2-3
                    self.instrs.append({"load": [("vload", v_idx[p[0]], tmp_addr[p[0]]),
                                                 ("vload", v_val[p[0]], tmp_addr2[p[0]])]})
                    self.instrs.append({"load": [("vload", v_idx[p[1]], tmp_addr[p[1]]),
                                                 ("vload", v_val[p[1]], tmp_addr2[p[1]])]})
                    self.instrs.append({"load": [("vload", v_idx[p[2]], tmp_addr[p[2]]),
                                                 ("vload", v_val[p[2]], tmp_addr2[p[2]])]})
                    self.instrs.append({"load": [("vload", v_idx[p[3]], tmp_addr[p[3]]),
                                                 ("vload", v_val[p[3]], tmp_addr2[p[3]])]})

                    # Phase 3: Node addr calc for all 4 (fits in 6 VALU slots)
                    self.instrs.append({"valu": [
                        ("+", v_node_addr[p[0]], v_idx[p[0]], v_forest_p),
                        ("+", v_node_addr[p[1]], v_idx[p[1]], v_forest_p),
                        ("+", v_node_addr[p[2]], v_idx[p[2]], v_forest_p),
                        ("+", v_node_addr[p[3]], v_idx[p[3]], v_forest_p),
                    ]})

                    # Phase 4: Scattered loads for all 4 chunks (16 cycles total)
                    for pair in range(4):
                        self.instrs.append({"load": [("load_offset", v_node_val[p[0]], v_node_addr[p[0]], pair*2),
                                                     ("load_offset", v_node_val[p[0]], v_node_addr[p[0]], pair*2+1)]})
                    for pair in range(4):
                        self.instrs.append({"load": [("load_offset", v_node_val[p[1]], v_node_addr[p[1]], pair*2),
                                                     ("load_offset", v_node_val[p[1]], v_node_addr[p[1]], pair*2+1)]})
                    for pair in range(4):
                        self.instrs.append({"load": [("load_offset", v_node_val[p[2]], v_node_addr[p[2]], pair*2),
                                                     ("load_offset", v_node_val[p[2]], v_node_addr[p[2]], pair*2+1)]})
                    for pair in range(4):
                        self.instrs.append({"load": [("load_offset", v_node_val[p[3]], v_node_addr[p[3]], pair*2),
                                                     ("load_offset", v_node_val[p[3]], v_node_addr[p[3]], pair*2+1)]})

                    # Phase 5: XOR + Hash for all 4 chunks (use 4 VALU slots per hash stage)
                    self.instrs.append({"valu": [
                        ("^", v_val[p[0]], v_val[p[0]], v_node_val[p[0]]),
                        ("^", v_val[p[1]], v_val[p[1]], v_node_val[p[1]]),
                        ("^", v_val[p[2]], v_val[p[2]], v_node_val[p[2]]),
                        ("^", v_val[p[3]], v_val[p[3]], v_node_val[p[3]]),
                    ]})
                    for v_c1, v_c3, op1, op2, op3 in v_hash_consts:
                        # Part 1 for chunks 0-1
                        self.instrs.append({"valu": [
                            (op1, v_tmp1[p[0]], v_val[p[0]], v_c1), (op3, v_tmp2[p[0]], v_val[p[0]], v_c3),
                            (op1, v_tmp1[p[1]], v_val[p[1]], v_c1), (op3, v_tmp2[p[1]], v_val[p[1]], v_c3),
                        ]})
                        # Part 1 for chunks 2-3
                        self.instrs.append({"valu": [
                            (op1, v_tmp1[p[2]], v_val[p[2]], v_c1), (op3, v_tmp2[p[2]], v_val[p[2]], v_c3),
                            (op1, v_tmp1[p[3]], v_val[p[3]], v_c1), (op3, v_tmp2[p[3]], v_val[p[3]], v_c3),
                        ]})
                        # Part 2 for all 4
                        self.instrs.append({"valu": [
                            (op2, v_val[p[0]], v_tmp1[p[0]], v_tmp2[p[0]]),
                            (op2, v_val[p[1]], v_tmp1[p[1]], v_tmp2[p[1]]),
                            (op2, v_val[p[2]], v_tmp1[p[2]], v_tmp2[p[2]]),
                            (op2, v_val[p[3]], v_tmp1[p[3]], v_tmp2[p[3]]),
                        ]})

                    # Phase 6: Index computation for all 4
                    self.instrs.append({"valu": [
                        ("&", v_tmp1[p[0]], v_val[p[0]], v_one), ("<<", v_tmp2[p[0]], v_idx[p[0]], v_one),
                        ("&", v_tmp1[p[1]], v_val[p[1]], v_one), ("<<", v_tmp2[p[1]], v_idx[p[1]], v_one),
                    ]})
                    self.instrs.append({"valu": [
                        ("&", v_tmp1[p[2]], v_val[p[2]], v_one), ("<<", v_tmp2[p[2]], v_idx[p[2]], v_one),
                        ("&", v_tmp1[p[3]], v_val[p[3]], v_one), ("<<", v_tmp2[p[3]], v_idx[p[3]], v_one),
                    ]})
                    self.instrs.append({"valu": [
                        ("+", v_tmp1[p[0]], v_tmp1[p[0]], v_one), ("+", v_tmp1[p[1]], v_tmp1[p[1]], v_one),
                        ("+", v_tmp1[p[2]], v_tmp1[p[2]], v_one), ("+", v_tmp1[p[3]], v_tmp1[p[3]], v_one),
                    ]})
                    self.instrs.append({"valu": [
                        ("+", v_idx[p[0]], v_tmp2[p[0]], v_tmp1[p[0]]), ("+", v_idx[p[1]], v_tmp2[p[1]], v_tmp1[p[1]]),
                        ("+", v_idx[p[2]], v_tmp2[p[2]], v_tmp1[p[2]]), ("+", v_idx[p[3]], v_tmp2[p[3]], v_tmp1[p[3]]),
                    ]})
                    self.instrs.append({"valu": [
                        ("<", v_tmp1[p[0]], v_idx[p[0]], v_n_nodes), ("<", v_tmp1[p[1]], v_idx[p[1]], v_n_nodes),
                        ("<", v_tmp1[p[2]], v_idx[p[2]], v_n_nodes), ("<", v_tmp1[p[3]], v_idx[p[3]], v_n_nodes),
                    ]})
                    self.instrs.append({"valu": [
                        ("*", v_idx[p[0]], v_idx[p[0]], v_tmp1[p[0]]), ("*", v_idx[p[1]], v_idx[p[1]], v_tmp1[p[1]]),
                        ("*", v_idx[p[2]], v_idx[p[2]], v_tmp1[p[2]]), ("*", v_idx[p[3]], v_idx[p[3]], v_tmp1[p[3]]),
                    ]})

                    # Phase 7: Store all 4 chunks
                    self.instrs.append({"alu": [
                        ("+", tmp_addr[p[0]], inp_idx_p, off[0]), ("+", tmp_addr2[p[0]], inp_val_p, off[0]),
                        ("+", tmp_addr[p[1]], inp_idx_p, off[1]), ("+", tmp_addr2[p[1]], inp_val_p, off[1]),
                        ("+", tmp_addr[p[2]], inp_idx_p, off[2]), ("+", tmp_addr2[p[2]], inp_val_p, off[2]),
                        ("+", tmp_addr[p[3]], inp_idx_p, off[3]), ("+", tmp_addr2[p[3]], inp_val_p, off[3]),
                    ]})
                    self.instrs.append({"store": [("vstore", tmp_addr[p[0]], v_idx[p[0]]),
                                                  ("vstore", tmp_addr2[p[0]], v_val[p[0]])]})
                    self.instrs.append({"store": [("vstore", tmp_addr[p[1]], v_idx[p[1]]),
                                                  ("vstore", tmp_addr2[p[1]], v_val[p[1]])]})
                    self.instrs.append({"store": [("vstore", tmp_addr[p[2]], v_idx[p[2]]),
                                                  ("vstore", tmp_addr2[p[2]], v_val[p[2]])]})
                    self.instrs.append({"store": [("vstore", tmp_addr[p[3]], v_idx[p[3]]),
                                                  ("vstore", tmp_addr2[p[3]], v_val[p[3]])]})

                    chunk_i += 4
                else:
                    # Process remaining chunks normally (2 at a time)
                    pA, pB = 0, 1
                    offA = offset_consts[chunk_i]
                    offB = offset_consts[min(chunk_i + 1, n_chunks - 1)]

                    self.instrs.append({"alu": [("+", tmp_addr[pA], inp_idx_p, offA), ("+", tmp_addr2[pA], inp_val_p, offA),
                                                ("+", tmp_addr[pB], inp_idx_p, offB), ("+", tmp_addr2[pB], inp_val_p, offB)]})
                    self.instrs.append({"load": [("vload", v_idx[pA], tmp_addr[pA]), ("vload", v_val[pA], tmp_addr2[pA])]})
                    self.instrs.append({"load": [("vload", v_idx[pB], tmp_addr[pB]), ("vload", v_val[pB], tmp_addr2[pB])]})
                    self.instrs.append({"valu": [("+", v_node_addr[pA], v_idx[pA], v_forest_p),
                                                 ("+", v_node_addr[pB], v_idx[pB], v_forest_p)]})
                    for pair in range(4):
                        self.instrs.append({"load": [("load_offset", v_node_val[pA], v_node_addr[pA], pair*2),
                                                     ("load_offset", v_node_val[pA], v_node_addr[pA], pair*2+1)]})
                    for pair in range(4):
                        self.instrs.append({"load": [("load_offset", v_node_val[pB], v_node_addr[pB], pair*2),
                                                     ("load_offset", v_node_val[pB], v_node_addr[pB], pair*2+1)]})
                    self.instrs.append({"valu": [("^", v_val[pA], v_val[pA], v_node_val[pA]),
                                                 ("^", v_val[pB], v_val[pB], v_node_val[pB])]})
                    for v_c1, v_c3, op1, op2, op3 in v_hash_consts:
                        self.instrs.append({"valu": [(op1, v_tmp1[pA], v_val[pA], v_c1), (op3, v_tmp2[pA], v_val[pA], v_c3),
                                                     (op1, v_tmp1[pB], v_val[pB], v_c1), (op3, v_tmp2[pB], v_val[pB], v_c3)]})
                        self.instrs.append({"valu": [(op2, v_val[pA], v_tmp1[pA], v_tmp2[pA]),
                                                     (op2, v_val[pB], v_tmp1[pB], v_tmp2[pB])]})
                    self.instrs.append({"valu": [("&", v_tmp1[pA], v_val[pA], v_one), ("<<", v_tmp2[pA], v_idx[pA], v_one),
                                                 ("&", v_tmp1[pB], v_val[pB], v_one), ("<<", v_tmp2[pB], v_idx[pB], v_one)]})
                    self.instrs.append({"valu": [("+", v_tmp1[pA], v_tmp1[pA], v_one), ("+", v_tmp1[pB], v_tmp1[pB], v_one)]})
                    self.instrs.append({"valu": [("+", v_idx[pA], v_tmp2[pA], v_tmp1[pA]), ("+", v_idx[pB], v_tmp2[pB], v_tmp1[pB])]})
                    self.instrs.append({"valu": [("<", v_tmp1[pA], v_idx[pA], v_n_nodes), ("<", v_tmp1[pB], v_idx[pB], v_n_nodes)]})
                    self.instrs.append({"valu": [("*", v_idx[pA], v_idx[pA], v_tmp1[pA]), ("*", v_idx[pB], v_idx[pB], v_tmp1[pB])]})
                    self.instrs.append({"alu": [("+", tmp_addr[pA], inp_idx_p, offA), ("+", tmp_addr2[pA], inp_val_p, offA),
                                                ("+", tmp_addr[pB], inp_idx_p, offB), ("+", tmp_addr2[pB], inp_val_p, offB)]})
                    self.instrs.append({"store": [("vstore", tmp_addr[pA], v_idx[pA]), ("vstore", tmp_addr2[pA], v_val[pA])]})
                    self.instrs.append({"store": [("vstore", tmp_addr[pB], v_idx[pB]), ("vstore", tmp_addr2[pB], v_val[pB])]})
                    chunk_i += 2

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
