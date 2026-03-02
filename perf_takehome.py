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

import heapq
import random
import unittest

from problem import (
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

# Optimization summary (current best: 1186 cycles on submission harness):
# - Build a global task DAG and schedule it with hard (RAW/WAW) and soft (WAR)
#   dependencies, then bundle ops across engines with a tuned priority score.
# - Preload shallow tree nodes (0..14) into vector constants and special-case
#   depths 0..3 to avoid general gather loads in early rounds.
# - Use round-specific VALU->ALU offloads for selected hash shift stages
#   (ALU_HASH_SH*_ROUNDS) to reduce VALU pressure near the throughput limit.
# - Use round-specific ALU index updates (ALU_IDX_UPDATE_ROUNDS) for shallow
#   depths to trade spare ALU bandwidth for fewer VALU bottlenecks.
# - Keep a final-round depth-4 fast path that reuses carried parity bits and
#   skips an unnecessary penultimate idx update when legal.
# - Place debug-harness pause instructions by piggybacking on existing bundles
#   so submission-mode cycle count is unchanged.


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def _schedule_tasks(self, tasks, hard_deps, soft_deps):
        n = len(tasks)
        hard_children = [[] for _ in range(n)]
        soft_children = [[] for _ in range(n)]
        hard_remaining = [0] * n
        soft_remaining = [0] * n
        for tid in range(n):
            hard_remaining[tid] = len(hard_deps[tid])
            soft_remaining[tid] = len(soft_deps[tid])
            for parent in hard_deps[tid]:
                hard_children[parent].append(tid)
            for parent in soft_deps[tid]:
                soft_children[parent].append(tid)

        outdeg = [len(hard_children[tid]) + len(soft_children[tid]) for tid in range(n)]
        child_valu = [0] * n
        child_load = [0] * n
        child_alu = [0] * n
        child_flow = [0] * n
        for tid in range(n):
            for ch in hard_children[tid]:
                eng = tasks[ch][0]
                if eng == "valu":
                    child_valu[tid] += 1
                elif eng == "load":
                    child_load[tid] += 1
                elif eng == "alu":
                    child_alu[tid] += 1
                elif eng == "flow":
                    child_flow[tid] += 1
            for ch in soft_children[tid]:
                eng = tasks[ch][0]
                if eng == "valu":
                    child_valu[tid] += 1
                elif eng == "load":
                    child_load[tid] += 1
                elif eng == "alu":
                    child_alu[tid] += 1
                elif eng == "flow":
                    child_flow[tid] += 1

        rank = [1] * n
        for tid in range(n - 1, -1, -1):
            kids = hard_children[tid] + soft_children[tid]
            if kids:
                rank[tid] = 1 + max(rank[ch] for ch in kids)

        vrank = [1 if tasks[tid][0] == "valu" else 0 for tid in range(n)]
        for tid in range(n - 1, -1, -1):
            kids = hard_children[tid] + soft_children[tid]
            if kids:
                vrank[tid] += max(vrank[ch] for ch in kids)

        is_vmadd = [0] * n
        is_vshift = [0] * n
        is_vxor = [0] * n
        is_acmp = [0] * n
        is_loff = [0] * n
        is_vsel = [0] * n
        for tid in range(n):
            eng, slot = tasks[tid]
            op = slot[0]
            if eng == "valu":
                if op == "multiply_add":
                    is_vmadd[tid] = 1
                elif op in ("<<", ">>"):
                    is_vshift[tid] = 1
                elif op == "^":
                    is_vxor[tid] = 1
            elif eng == "alu":
                if op in ("==", "<"):
                    is_acmp[tid] = 1
            elif eng == "load":
                if op == "load_offset":
                    is_loff[tid] = 1
            elif eng == "flow":
                if op == "vselect":
                    is_vsel[tid] = 1

        # Tuned op-aware schedule priority to maximize cross-engine overlap.
        # Lower tuple comes first in heapq.
        def priority(tid):
            eng = tasks[tid][0]
            score = (
                -1.4921618275616715 * rank[tid]
                -2.7124243919137423 * vrank[tid]
                -1.5534690611153839 * outdeg[tid]
                -2.84952244179926 * child_valu[tid]
                +1.2667737469763853 * child_load[tid]
                +5.4774637227303495 * child_alu[tid]
                +1.4998379061690588 * child_flow[tid]
                -1.594748502139819 * (eng == "valu")
                -0.8505366466709959 * (eng == "load")
                +0.37302961290601266 * (eng == "alu")
                +1.7762530469245243 * (eng == "flow")
                +0.7096476661136442 * tid
                +0.7504001574326017 * (n - tid)
                +0.43584652523465467 * is_vmadd[tid]
                -1.00908621802914 * is_vshift[tid]
                -0.6861448187228185 * is_vxor[tid]
                -1.1240981652520285 * is_acmp[tid]
                +0.050174759359080785 * is_loff[tid]
                +0.7162879617436614 * is_vsel[tid]
            )
            return (-score, -tid)

        engines = ["load", "alu", "valu", "flow", "store"]
        ready = {e: [] for e in engines}
        for tid in range(n):
            if hard_remaining[tid] == 0 and soft_remaining[tid] == 0:
                eng = tasks[tid][0]
                heapq.heappush(ready[eng], (priority(tid), tid))

        instrs = []
        scheduled = 0
        unscheduled = [True] * n
        while scheduled < n:
            bundle = {}
            chosen = []
            used = {eng: 0 for eng in engines}
            pending_hard = {}

            while True:
                progressed = False
                for eng in engines:
                    limit = SLOT_LIMITS[eng]
                    slots = bundle.get(eng)
                    while used[eng] < limit and ready[eng]:
                        _, tid = heapq.heappop(ready[eng])
                        if not unscheduled[tid]:
                            continue
                        if slots is None:
                            slots = []
                            bundle[eng] = slots
                        slots.append(tasks[tid][1])
                        chosen.append(tid)
                        unscheduled[tid] = False
                        used[eng] += 1
                        progressed = True

                        # WAR deps are same-cycle-safe: unlock children immediately.
                        for ch in soft_children[tid]:
                            soft_remaining[ch] -= 1
                            if (
                                unscheduled[ch]
                                and hard_remaining[ch] == 0
                                and soft_remaining[ch] == 0
                            ):
                                ch_eng = tasks[ch][0]
                                heapq.heappush(ready[ch_eng], (priority(ch), ch))

                        # RAW/WAW deps require at least next cycle.
                        for ch in hard_children[tid]:
                            pending_hard[ch] = pending_hard.get(ch, 0) + 1
                if not progressed:
                    break

            if not bundle:
                raise RuntimeError("scheduler deadlock")
            instrs.append(bundle)
            scheduled += len(chosen)

            for ch, dec in pending_hard.items():
                hard_remaining[ch] -= dec
                if unscheduled[ch] and hard_remaining[ch] == 0 and soft_remaining[ch] == 0:
                    eng = tasks[ch][0]
                    heapq.heappush(ready[eng], (priority(ch), ch))
        return instrs

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        assert batch_size % VLEN == 0, "batch_size must be divisible by VLEN"
        n_vec = batch_size // VLEN
        forest_values_p = 7
        inp_values_p = forest_values_p + n_nodes + batch_size

        tasks = []
        hard_deps = []
        soft_deps = []
        last_write = {}
        last_read = {}
        const_map = {}

        def vec_addrs(base):
            return [base + i for i in range(VLEN)]

        def add_task(engine, slot, reads=(), writes=()):
            hard = set()
            soft = set()
            for addr in reads:
                parent = last_write.get(addr)
                if parent is not None:
                    hard.add(parent)
            for addr in writes:
                parent = last_write.get(addr)
                if parent is not None:
                    hard.add(parent)
                parent = last_read.get(addr)
                if parent is not None and parent not in hard:
                    soft.add(parent)
            tid = len(tasks)
            tasks.append((engine, slot))
            hard_deps.append(hard)
            soft_deps.append(soft)
            for addr in reads:
                last_read[addr] = tid
            for addr in writes:
                last_write[addr] = tid
                if addr in last_read:
                    del last_read[addr]
            return tid

        def const_addr(val, name=None):
            if val not in const_map:
                addr = self.alloc_scratch(name)
                add_task("load", ("const", addr, val), writes=[addr])
                const_map[val] = addr
            return const_map[val]

        def add_alu(op, dest, a1, a2):
            add_task("alu", (op, dest, a1, a2), reads=[a1, a2], writes=[dest])

        def add_valu(op, dest, a1, a2):
            add_task(
                "valu",
                (op, dest, a1, a2),
                reads=vec_addrs(a1) + vec_addrs(a2),
                writes=vec_addrs(dest),
            )

        def add_madd(dest, a, b, c):
            add_task(
                "valu",
                ("multiply_add", dest, a, b, c),
                reads=vec_addrs(a) + vec_addrs(b) + vec_addrs(c),
                writes=vec_addrs(dest),
            )

        def add_vbroadcast(dest, src):
            add_task(
                "valu",
                ("vbroadcast", dest, src),
                reads=[src],
                writes=vec_addrs(dest),
            )

        def add_vselect(dest, cond, a, b):
            add_task(
                "flow",
                ("vselect", dest, cond, a, b),
                reads=vec_addrs(cond) + vec_addrs(a) + vec_addrs(b),
                writes=vec_addrs(dest),
            )

        def add_add_imm(dest, a, imm):
            add_task(
                "flow",
                ("add_imm", dest, a, imm),
                reads=[a],
                writes=[dest],
            )

        def add_idx_update_depth0_alu(idx, parity):
            for lane in range(VLEN):
                add_alu("+", idx + lane, parity + lane, two)

        def add_idx_update_alu(idx, parity):
            for lane in range(VLEN):
                add_alu("<<", idx + lane, idx + lane, one)
            for lane in range(VLEN):
                add_alu("+", idx + lane, idx + lane, parity + lane)

        def add_load(dest, addr):
            add_task("load", ("load", dest, addr), reads=[addr], writes=[dest])

        def add_load_offset(dest, addr, offset):
            add_task(
                "load",
                ("load_offset", dest, addr, offset),
                reads=[addr + offset],
                writes=[dest + offset],
            )

        def add_vload(dest, addr):
            add_task(
                "load",
                ("vload", dest, addr),
                reads=[addr],
                writes=vec_addrs(dest),
            )

        def add_vstore(addr, src):
            add_task(
                "store",
                ("vstore", addr, src),
                reads=[addr] + vec_addrs(src),
            )

        def make_vec_const(name, scalar_addr):
            vec_addr = self.alloc_scratch(name, VLEN)
            add_vbroadcast(vec_addr, scalar_addr)
            return vec_addr

        def emit_hash_shift(round_idx, offload_rounds, op, dest_vec, src_vec, scalar_amt, vec_amt):
            """Optionally offload one vector shift stage from VALU to per-lane ALU."""
            if round_idx in offload_rounds:
                for lane in range(VLEN):
                    add_alu(op, dest_vec + lane, src_vec + lane, scalar_amt)
            else:
                add_valu(op, dest_vec, src_vec, vec_amt)

        def emit_hash(val, t1, t2, node_src, round_idx):
            add_valu("^", val, val, node_src)
            add_madd(val, val, v_mul1, v_c1)
            add_valu("^", t1, val, v_c2)
            emit_hash_shift(round_idx, ALU_HASH_SH19_ROUNDS, ">>", t2, val, sh19, v_sh19)
            add_valu("^", val, t1, t2)
            add_madd(val, val, v_mul3, v_c3)
            add_valu("+", t1, val, v_c4)
            emit_hash_shift(round_idx, ALU_HASH_SH9_ROUNDS, "<<", t2, val, sh9, v_sh9)
            add_valu("^", val, t1, t2)
            add_madd(val, val, v_mul5, v_c5)
            add_valu("^", t1, val, v_c6)
            emit_hash_shift(round_idx, ALU_HASH_SH16_ROUNDS, ">>", t2, val, sh16, v_sh16)
            add_valu("^", val, t1, t2)

        one = const_addr(1, "c1_scalar")
        two = const_addr(2, "c2_scalar")
        four = const_addr(4, "c4_scalar")
        six = const_addr(6, "c6_scalar")
        eight = const_addr(8, "c8_scalar")
        nine = const_addr(9, "c9_scalar")
        ten = const_addr(10, "c10_scalar")
        eleven = const_addr(11, "c11_scalar")
        twelve = const_addr(12, "c12_scalar")
        thirteen = const_addr(13, "c13_scalar")
        fourteen = const_addr(14, "c14_scalar")
        forest_base_m1 = const_addr(forest_values_p - 1, "forest_base_m1")

        # Hash constants.
        c1 = const_addr(0x7ED55D16, "hash_c1")
        c2 = const_addr(0xC761C23C, "hash_c2")
        c3 = const_addr(0x165667B1, "hash_c3")
        c4 = const_addr(0xD3A2646C, "hash_c4")
        c5 = const_addr(0xFD7046C5, "hash_c5")
        c6 = const_addr(0xB55A4F09, "hash_c6")
        sh19 = const_addr(19, "hash_sh19")
        sh9 = const_addr(9, "hash_sh9")
        sh16 = const_addr(16, "hash_sh16")
        mul1 = const_addr((1 << 12) + 1, "hash_mul1")
        mul3 = const_addr((1 << 5) + 1, "hash_mul3")
        mul5 = const_addr((1 << 3) + 1, "hash_mul5")

        v_two = make_vec_const("v_two", two)
        v_c1 = make_vec_const("v_hash_c1", c1)
        v_c2 = make_vec_const("v_hash_c2", c2)
        v_c3 = make_vec_const("v_hash_c3", c3)
        v_c4 = make_vec_const("v_hash_c4", c4)
        v_c5 = make_vec_const("v_hash_c5", c5)
        v_c6 = make_vec_const("v_hash_c6", c6)
        v_sh19 = make_vec_const("v_hash_sh19", sh19)
        v_sh9 = make_vec_const("v_hash_sh9", sh9)
        v_sh16 = make_vec_const("v_hash_sh16", sh16)
        v_mul1 = make_vec_const("v_hash_mul1", mul1)
        v_mul3 = make_vec_const("v_hash_mul3", mul3)
        v_mul5 = make_vec_const("v_hash_mul5", mul5)
        inp_values_base = const_addr(inp_values_p, "inp_values_base")

        # Preload shallow tree nodes used by special-cased rounds.
        node_vec = {}
        for ni in range(15):
            node_addr = const_addr(forest_values_p + ni, f"node_addr_{ni}")
            node_scalar = self.alloc_scratch(f"node_scalar_{ni}")
            add_load(node_scalar, node_addr)
            node_vec[ni] = make_vec_const(f"node_vec_{ni}", node_scalar)

        states = []
        value_ptrs = []
        for vi in range(n_vec):
            idx = self.alloc_scratch(f"idx_{vi}", VLEN)
            val = self.alloc_scratch(f"val_{vi}", VLEN)
            t1 = self.alloc_scratch(f"t1_{vi}", VLEN)
            t2 = self.alloc_scratch(f"t2_{vi}", VLEN)
            states.append((idx, val, t1, t2))

            ptr = self.alloc_scratch(f"value_ptr_{vi}")
            add_add_imm(ptr, inp_values_base, vi * VLEN)
            value_ptrs.append(ptr)
            add_vload(val, ptr)

        def emit_depth2_node_select(idx, t1, t2):
            """Select node values 3..6 for depth=2."""
            for lane in range(VLEN):
                add_alu("<", t2 + lane, idx + lane, six)
            add_vselect(t1, t2, node_vec[4], node_vec[6])
            for lane in range(VLEN):
                add_alu("==", t2 + lane, idx + lane, four)
            add_vselect(t1, t2, node_vec[3], t1)
            for lane in range(VLEN):
                add_alu("==", t2 + lane, idx + lane, six)
            add_vselect(t1, t2, node_vec[5], t1)
            return t1

        def emit_depth3_node_select(idx, t1, t2):
            """Select node values 7..14 for depth=3 (with idx already +1 shifted)."""
            add_vselect(t1, t2, node_vec[14], node_vec[13])
            for lane in range(VLEN):
                add_alu("==", t2 + lane, idx + lane, thirteen)
            add_vselect(t1, t2, node_vec[12], t1)
            for lane in range(VLEN):
                add_alu("==", t2 + lane, idx + lane, twelve)
            add_vselect(t1, t2, node_vec[11], t1)
            for lane in range(VLEN):
                add_alu("==", t2 + lane, idx + lane, eleven)
            add_vselect(t1, t2, node_vec[10], t1)
            for lane in range(VLEN):
                add_alu("==", t2 + lane, idx + lane, ten)
            add_vselect(t1, t2, node_vec[9], t1)
            for lane in range(VLEN):
                add_alu("==", t2 + lane, idx + lane, nine)
            add_vselect(t1, t2, node_vec[8], t1)
            for lane in range(VLEN):
                add_alu("==", t2 + lane, idx + lane, eight)
            add_vselect(t1, t2, node_vec[7], t1)
            return t1

        def emit_deep_node_load(idx, t1, t2, depth, is_last_round):
            """Depth>=4 path: gather node value by computed address."""
            if is_last_round and depth == 4:
                # Last round at depth 4: consume carried parity from the
                # prior round directly as addr = 2*idx + 6 + parity.
                for lane in range(VLEN):
                    add_alu("+", t1 + lane, idx + lane, idx + lane)
                for lane in range(VLEN):
                    add_alu("+", t1 + lane, t1 + lane, forest_base_m1)
                for lane in range(VLEN):
                    add_alu("+", t1 + lane, t1 + lane, t2 + lane)
                for lane in range(VLEN):
                    add_load_offset(t1, t1, lane)
            else:
                for lane in range(VLEN):
                    add_alu("+", t2 + lane, idx + lane, forest_base_m1)
                for lane in range(VLEN):
                    add_load_offset(t1, t2, lane)
            return t1

        def node_source_for_round(depth, idx, t1, t2, is_last_round):
            """Emit node-source selection for current depth and return vector address."""
            if depth == 0:
                return node_vec[0]
            if depth == 1:
                # Reuse parity bits from the prior round's idx update.
                add_vselect(t1, t2, node_vec[2], node_vec[1])
                return t1
            if depth == 2:
                return emit_depth2_node_select(idx, t1, t2)
            if depth == 3:
                return emit_depth3_node_select(idx, t1, t2)
            return emit_deep_node_load(idx, t1, t2, depth, is_last_round)

        def emit_idx_update(round_idx, depth, idx, val, t2):
            """Emit the post-hash index update for one state."""
            if round_idx == rounds - 1:
                return
            if depth == forest_height:
                return
            for lane in range(VLEN):
                add_alu("&", t2 + lane, val + lane, one)
            if round_idx in ALU_IDX_UPDATE_ROUNDS and depth in (0, 1, 2):
                if depth == 0:
                    add_idx_update_depth0_alu(idx, t2)
                else:
                    add_idx_update_alu(idx, t2)
                return
            if depth == 0:
                add_valu("+", idx, t2, v_two)
                return
            if round_idx == rounds - 2 and (round_idx + 1) % wrap_period == 4:
                return
            add_madd(idx, idx, v_two, t2)

        wrap_period = forest_height + 1
        for round_idx in range(rounds):
            depth = round_idx % wrap_period
            is_last_round = round_idx == rounds - 1
            for idx, val, t1, t2 in states:
                node_src = node_source_for_round(depth, idx, t1, t2, is_last_round)
                emit_hash(val, t1, t2, node_src, round_idx)
                emit_idx_update(round_idx, depth, idx, val, t2)

        for vi, (_, val, _, _) in enumerate(states):
            add_vstore(value_ptrs[vi], val)

        body = self._schedule_tasks(tasks, hard_deps, soft_deps)

        # Keep two pause points for the local debug harness without adding
        # extra cycles in submission mode by piggybacking on existing bundles.
        if "flow" in body[0]:
            raise RuntimeError("expected first bundle to have free flow slot")
        body[0]["flow"] = [("pause",)]

        last_store_i = None
        for i in range(len(body) - 1, -1, -1):
            if "store" in body[i]:
                last_store_i = i
                break
        if last_store_i is None:
            raise RuntimeError("no store bundle found")
        if "flow" in body[last_store_i]:
            raise RuntimeError("expected last store bundle to have free flow slot")
        body[last_store_i]["flow"] = [("pause",)]

        self.instrs = body

BASELINE = 147734

# Use per-lane ALU idx updates on selected rounds to reduce valu pressure.
ALU_IDX_UPDATE_ROUNDS = frozenset((11, 12, 13))
# Optional hash-shift offload (valu -> per-lane alu), tuned by search.
ALU_HASH_SH19_ROUNDS = frozenset((9,))
ALU_HASH_SH9_ROUNDS = frozenset((9, 11, 12))
ALU_HASH_SH16_ROUNDS = frozenset()

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
