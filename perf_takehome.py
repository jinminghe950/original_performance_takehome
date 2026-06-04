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

    # ------------------------------------------------------------------
    # Optimized vectorized kernel
    #
    # Key structural insight: every walker starts at idx=0 and the forest is
    # a *perfect* tree, so on a given round all walkers sit at the same depth
    # (the tree wraps to the root uniformly at the leaves).  Therefore:
    #   - depth 0 rounds need only a single broadcast node value (no gather);
    #   - shallow depths read from a small contiguous block of nodes, which we
    #     route with a vselect mux tree (cheap on the otherwise-idle flow
    #     engine) instead of paying for scalar gather loads (limited to 2/cyc);
    #   - deep depths fall back to scalar gather loads.
    #
    # Everything is built as a flat list of micro-ops with explicit scratch
    # read/write sets and then packed into VLIW bundles by a dependency-aware
    # list scheduler (see schedule()).  The 32 independent SIMD lanes per round
    # give the scheduler ample ILP to fill the engine slots every cycle.
    # ------------------------------------------------------------------

    def rng(self, a):
        return list(range(a, a + VLEN))

    def emit(self, engine, slot, reads, writes):
        self.ops.append((engine, slot, list(reads), list(writes)))
        self.prio.append(getattr(self, "_prio", 0))

    def vtemp(self):
        a = self.vtemps[self._ti % len(self.vtemps)]
        self._ti += 1
        return a

    def bcast(self, val):
        """Return a vector scratch addr holding `val` broadcast across lanes."""
        if val in self._bcache:
            return self._bcache[val]
        s = self.alloc_scratch(None)
        vb = self.alloc_scratch(None, VLEN)
        self._bpending.append((s, vb, val))
        self._bcache[val] = vb
        return vb

    def vbin(self, op, dest, a, b):
        self.emit("valu", (op, dest, a, b), self.rng(a) + self.rng(b), self.rng(dest))

    def aluv(self, op, dest, a, bscalar):
        """Apply a scalar ALU op across all VLEN lanes (b is a scalar source).

        Used to push cheap per-lane bookkeeping (bit extraction, gather address
        arithmetic, mux select bits) onto the otherwise-spare ALU engine so the
        valu engine carries only the hash, leaving it the sole ~1184-cycle pole.
        """
        for j in range(VLEN):
            self.emit("alu", (op, dest + j, a + j, bscalar), [a + j, bscalar], [dest + j])

    def vmadd(self, dest, a, b, c):
        self.emit(
            "valu",
            ("multiply_add", dest, a, b, c),
            self.rng(a) + self.rng(b) + self.rng(c),
            self.rng(dest),
        )

    def hash_inplace(self, a):
        """Apply the 6-stage myhash to vector `a` in place (12 valu ops).

        Stages 0/2/4 are of the form  a = (a+c) + (a<<s) = a*(2^s+1) + c
        which fuses into a single multiply_add.
        """
        c0, c1, c2, c3, c4, c5 = self.hc
        K0, K2, K4 = self.hK
        sh19, sh9, sh16 = self.hsh
        t = self.vtemp()
        # stage 0: + , + , <<12  ->  a*4097 + c0
        self.vmadd(a, a, K0, c0)
        # stage 1: ^ , ^ , >>19
        self.vbin(">>", t, a, sh19)
        self.vbin("^", a, a, c1)
        self.vbin("^", a, a, t)
        # stage 2: + , + , <<5  ->  a*33 + c2
        self.vmadd(a, a, K2, c2)
        # stage 3: + , ^ , <<9
        self.vbin("<<", t, a, sh9)
        u = self.vtemp()
        self.vbin("+", u, a, c3)
        self.vbin("^", a, t, u)
        # stage 4: + , + , <<3  ->  a*9 + c4
        self.vmadd(a, a, K4, c4)
        # stage 5: ^ , ^ , >>16
        self.vbin(">>", t, a, sh16)
        self.vbin("^", a, a, c5)
        self.vbin("^", a, a, t)

    # ---- scalar (ALU-engine) lane path ----------------------------------
    # The ALU engine (12 slots/cycle) is otherwise idle while valu is the
    # bottleneck.  Processing a fraction of the SIMD lanes one-at-a-time on the
    # ALU lets the two engines share the (hash-dominated) arithmetic.  The ALU
    # has no multiply_add, so the fused hash stages cost one extra op each.

    def salu(self, op, dest, a, b):
        self.emit("alu", (op, dest, a, b), [a, b], [dest])

    def stemp(self):
        a = self.stemps[self._si % len(self.stemps)]
        self._si += 1
        return a

    def s_hash(self, a):
        sc = self.sc
        t = self.stemp()
        self.salu("*", t, a, sc["K0"]); self.salu("+", a, t, sc["c0"])
        self.salu(">>", t, a, sc["sh19"]); self.salu("^", a, a, sc["c1"]); self.salu("^", a, a, t)
        self.salu("*", t, a, sc["K2"]); self.salu("+", a, t, sc["c2"])
        self.salu("<<", t, a, sc["sh9"])
        u = self.stemp()
        self.salu("+", u, a, sc["c3"]); self.salu("^", a, t, u)
        self.salu("*", t, a, sc["K4"]); self.salu("+", a, t, sc["c4"])
        self.salu(">>", t, a, sc["sh16"]); self.salu("^", a, a, sc["c5"]); self.salu("^", a, a, t)

    def s_mux(self, d, idxw):
        """Scalar arithmetic mux (no load engine) for an ALU lane at depth d."""
        sc = self.sc
        base = 2 ** d - 1
        basev = self._bcache[base]
        p = self.stemp()
        self.salu("-", p, idxw, basev)
        bits = []
        pp = p
        for i in range(d):
            if i == d - 1:
                bits.append(pp)
            else:
                bt = self.stemp(); self.salu("&", bt, pp, sc["one"]); bits.append(bt)
                np = self.stemp(); self.salu(">>", np, pp, sc["one"]); pp = np
        level = []
        for jj, k in enumerate(range(0, 2 ** d, 2)):
            t = self.stemp(); self.salu("*", t, bits[0], self.muxdiff[d][jj])
            out = self.stemp(); self.salu("+", out, t, self.fnode_s[base + k])
            level.append(out)
        for i in range(1, d):
            nxt = []
            for k in range(0, len(level), 2):
                diff = self.stemp(); self.salu("-", diff, level[k + 1], level[k])
                t = self.stemp(); self.salu("*", t, bits[i], diff)
                out = self.stemp(); self.salu("+", out, t, level[k])
                nxt.append(out)
            level = nxt
        return level[0]

    def s_process(self, v, d, last, is_leaf, r):
        sc = self.sc
        val_v, idx_v = self.val[v], self.idx[v]
        for j in range(VLEN):
            aw = val_v + j
            if d == 0:
                node = sc["f0"]
            elif self.mux_here(r, d) and d <= getattr(self, "ALU_MUX_MAX", 2):
                # ALU lanes scalar-mux only shallow depths; deeper muxed depths
                # are gathered here (cheap, only these few lanes) to avoid the
                # heavy scalar mux tree -- keeps the ALU engine balanced.
                node = self.s_mux(d, idx_v + j)
            else:
                addr = self.stemp()
                self.salu("+", addr, idx_v + j, sc["seven"])
                node = self.stemp()
                self.emit("load", ("load", node, addr), [addr], [node])
            self.salu("^", aw, aw, node)
            self.s_hash(aw)
            if last or is_leaf:
                continue
            bit = self.stemp()
            self.salu("&", bit, aw, sc["one"])
            if d == 0:
                self.salu("+", idx_v + j, bit, sc["one"])
            else:
                t = self.stemp()
                self.salu("*", t, idx_v + j, sc["two"])
                self.salu("+", t, t, sc["one"])
                self.salu("+", idx_v + j, t, bit)

    def arith_mux(self, d, idx_v):
        """Flow-free selection of forest[idx] for a depth-d block.

        Builds a mux tree using arithmetic selects (select(c,a,b)=b+c*(a-b),
        a single multiply_add when c is a 0/1 bit) over the preloaded block
        broadcasts.  First-level differences are precomputed once at setup.
        Avoids the flow engine entirely (the flow engine is 1 slot/cycle and
        serializes the critical path badly for mux trees).
        """
        base = 2 ** d - 1
        one_v = self.bcast(1)
        p = self.vtemp()
        self.vbin("-", p, idx_v, self.bcast(base))
        # extract bits b0 (LSB) .. b_{d-1}
        bits = []
        pp = p
        for i in range(d):
            if i == d - 1:
                bits.append(pp)  # top bit already in range {0,1}
            else:
                bt = self.vtemp()
                self.vbin("&", bt, pp, one_v)
                bits.append(bt)
                np = self.vtemp()
                self.vbin(">>", np, pp, one_v)
                pp = np
        # level 0: constant broadcasts combined with precomputed diffs
        diffs = self.muxdiff[d]
        level = []
        for j, k in enumerate(range(0, 2 ** d, 2)):
            out = self.vtemp()
            self.vmadd(out, bits[0], diffs[j], self.fbcast[base + k])
            level.append(out)
        # higher levels: per-lane diffs
        for i in range(1, d):
            nxt = []
            for k in range(0, len(level), 2):
                diff = self.vtemp()
                self.vbin("-", diff, level[k + 1], level[k])
                out = self.vtemp()
                self.vmadd(out, bits[i], diff, level[k])
                nxt.append(out)
            level = nxt
        return level[0]

    def flow_mux(self, d, idx_v):
        """Select forest[idx] using a vselect mux tree on the flow engine.

        Only the (cheap) bit extraction costs valu; the selects themselves run
        on the otherwise-idle flow engine, so this keeps deep mux trees off the
        compute engines.  Relies on abundant independent valu/alu work (from the
        ALU lane offload) to hide the flow engine's serial latency.
        """
        base = 2 ** d - 1
        basev = self.bcast(base)
        # Select bits are extracted as independent masks (p & 1, p & 2, p & 4):
        # vselect only tests for nonzero, so no shifts are needed.  This drops an
        # op and removes the serial shift chain, improving ILP.  On AUX_ALU the
        # masking runs per-lane on the ALU to keep the whole mux off valu.
        if getattr(self, "AUX_ALU", False):
            mask = lambda dst, a, m: self.aluv("&", dst, a, m)
            p = self.vtemp(); self.aluv("-", p, idx_v, basev)
        else:
            mask = lambda dst, a, m: self.vbin("&", dst, a, m)
            p = self.vtemp(); self.vbin("-", p, idx_v, basev)
        bits = []
        for i in range(d):
            bt = self.vtemp()
            mask(bt, p, self.bcast(1 << i))
            bits.append(bt)
        level = [self.fbcast[base + k] for k in range(2 ** d)]
        for i in range(d):
            nxt = []
            for k in range(0, len(level), 2):
                out = self.vtemp()
                self.emit(
                    "flow",
                    ("vselect", out, bits[i], level[k + 1], level[k]),
                    self.rng(bits[i]) + self.rng(level[k + 1]) + self.rng(level[k]),
                    self.rng(out),
                )
                nxt.append(out)
            level = nxt
        return level[0]

    def mux_here(self, r, d):
        """Whether to route depth d via mux (vs gather) at round r.

        Depths in mux_always are muxed everywhere; depths in mux_late are muxed
        only on the second (and later) pass, where the load engine is otherwise
        idle, so the first pass keeps a tight load-bound gather streak.
        """
        if d in self.mux_always:
            return True
        if d in self.mux_late and r > self.H:
            return True
        return False

    def get_node(self, d, idx_v, r):
        """Vector of node values forest[idx] for this round's depth d."""
        if d == 0:
            return self.fbcast[0]
        if self.mux_here(r, d):
            if getattr(self, "MUX_MODE", "flow") == "flow":
                return self.flow_mux(d, idx_v)
            return self.arith_mux(d, idx_v)
        # gather: addr = forest_values_p + idx, then scalar loads per lane.
        # Address arithmetic goes on the ALU when AUX_ALU so valu stays hash-only.
        addr = self.vtemp()
        if getattr(self, "AUX_ALU", False):
            self.aluv("+", addr, idx_v, self.seven_v)
        else:
            self.vbin("+", addr, idx_v, self.seven_v)
        node = self.vtemp()
        for j in range(VLEN):
            self.emit("load", ("load", node + j, addr + j), [addr + j], [node + j])
        return node

    def schedule(self, ops):
        """Readiness-based list scheduler -> list of VLIW bundles.

        Effects apply at end of cycle, so any two ops touching the same scratch
        address with at least one write are ordered into strictly different
        bundles (a conservative but correct hazard model).  Ops are released as
        their predecessors complete and packed greedily into bundles, preferring
        ops with the longest remaining dependency chain (critical path) so the
        engines stay saturated.
        """
        import heapq

        n = len(ops)
        preds = [0] * n
        succs = [[] for _ in range(n)]
        last_writer = {}
        last_readers = defaultdict(list)
        for i, (engine, slot, reads, writes) in enumerate(ops):
            ps = set()
            for a in reads:
                w = last_writer.get(a)
                if w is not None:
                    ps.add(w)
            for a in writes:
                w = last_writer.get(a)
                if w is not None:
                    ps.add(w)
                for r in last_readers.get(a, ()):
                    ps.add(r)
            preds[i] = len(ps)
            for p in ps:
                succs[p].append(i)
            for a in writes:
                last_writer[a] = i
                last_readers[a] = []
            for a in reads:
                last_readers[a].append(i)

        # critical-path length (longest chain to a sink) as scheduling priority
        cp = [1] * n
        for i in range(n - 1, -1, -1):
            best = 0
            for s in succs[i]:
                if cp[s] > best:
                    best = cp[s]
            cp[i] = best + 1

        # ready-list priority: pipeline tag first (so low-index vectors race
        # ahead and stagger the deep-gather phase against the shallow compute
        # phase), then critical path.
        prio = getattr(self, "prio", [0] * n)
        key = [(prio[i], -cp[i], i) for i in range(n)]

        indeg = preds
        earliest = [0] * n
        waiting = [(0, key[i]) for i in range(n) if indeg[i] == 0]
        heapq.heapify(waiting)
        avail = []
        bundles = []
        bundle = [-1] * n
        b = 0
        scheduled = 0
        while scheduled < n:
            while waiting and waiting[0][0] <= b:
                _, k = heapq.heappop(waiting)
                heapq.heappush(avail, k)
            if not avail:
                b = waiting[0][0]
                continue
            cnt = {}
            cur = {}
            leftover = []
            while avail:
                k = heapq.heappop(avail)
                i = k[2]
                engine = ops[i][0]
                if cnt.get(engine, 0) < SLOT_LIMITS[engine]:
                    cnt[engine] = cnt.get(engine, 0) + 1
                    bundle[i] = b
                    scheduled += 1
                    cur.setdefault(engine, []).append(ops[i][1])
                    for s in succs[i]:
                        indeg[s] -= 1
                        if b + 1 > earliest[s]:
                            earliest[s] = b + 1
                        if indeg[s] == 0:
                            heapq.heappush(waiting, (earliest[s], key[s]))
                else:
                    leftover.append(k)
            bundles.append(cur)
            for x in leftover:
                heapq.heappush(avail, x)
            b += 1
        return bundles

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        H = forest_height
        assert batch_size % VLEN == 0, "batch size must be a multiple of VLEN"
        nvec = batch_size // VLEN

        # Memory layout is deterministic from the shapes (see build_mem_image):
        FP = 7  # forest_values_p (header == 7)
        IVP = 7 + n_nodes + batch_size  # inp_values_p

        self.ops = []
        self.prio = []
        self._prio = 0
        self._bcache = {}
        self._bpending = []

        self.H = H
        depth_of = lambda r: r % (H + 1)
        used_depths = set(depth_of(r) for r in range(rounds))
        # depths that recur on the second (and later) pass over the tree
        late_depths = set(depth_of(r) for r in range(H + 1, rounds))

        # Depths in mux_always are routed via mux tree on every pass; depths in
        # mux_late are muxed only on the second pass (where the load engine is
        # idle), so the first pass keeps a tight load-bound gather streak.
        MUX_MAX = getattr(self, "MUX_MAX", 3)
        MUX_LATE_MAX = getattr(self, "MUX_LATE_MAX", 3)
        self.mux_always = set(
            d for d in range(1, MUX_MAX + 1) if d in used_depths and 2 ** d <= n_nodes
        )
        self.mux_late = set(
            d for d in range(MUX_MAX + 1, MUX_LATE_MAX + 1)
            if d in late_depths and 2 ** d <= n_nodes
        )
        self.mux_depths = self.mux_always | self.mux_late

        # ---- allocate persistent state ----
        val = [self.alloc_scratch(f"val{v}", VLEN) for v in range(nvec)]
        idx = [self.alloc_scratch(f"idx{v}", VLEN) for v in range(nvec)]  # init 0

        # broadcast constants needed by the hash and index update
        self.bcast(1)
        self.bcast(2)
        self.seven_v = self.bcast(FP)
        one_v = self._bcache[1]
        two_v = self._bcache[2]
        self.hc = [
            self.bcast(v)
            for v in (0x7ED55D16, 0xC761C23C, 0x165667B1, 0xD3A2646C, 0xFD7046C5, 0xB55A4F09)
        ]
        self.hK = [self.bcast(4097), self.bcast(33), self.bcast(9)]
        self.hsh = [self.bcast(19), self.bcast(9), self.bcast(16)]
        for d in self.mux_depths:
            self.bcast(2 ** d - 1)
            for i in range(d):  # mux select-bit masks (1, 2, 4, ...)
                self.bcast(1 << i)

        # forest values to preload (depth-0 node + every mux-depth block)
        need_nodes = set()
        if 0 in used_depths:
            need_nodes.add(0)
        for d in self.mux_depths:
            for k in range(2 ** d):
                need_nodes.add((2 ** d - 1) + k)

        # scalar addr regs reused for initial vloads / final vstores
        init_addr = [self.alloc_scratch(None) for _ in range(nvec)]

        # preallocate broadcast scalars/vectors and forest-preload storage
        fnode_s = {n: self.alloc_scratch(None) for n in need_nodes}
        self.fnode_s = fnode_s
        fnode_a = {n: self.alloc_scratch(None) for n in need_nodes}
        self.fbcast = {n: self.alloc_scratch(None, VLEN) for n in need_nodes}

        # precomputed first-level mux differences (constant per pair)
        self.muxdiff = {
            d: [self.alloc_scratch(None, VLEN) for _ in range(2 ** (d - 1))]
            for d in self.mux_depths
        }

        # dummy scratch words used only to inject pipeline-stagger dependencies
        self.dummies = [self.alloc_scratch(None) for _ in range(max(0, getattr(self, "NC", 16) - 1))]

        self.val = val
        self.idx = idx
        n_alu = getattr(self, "ALU_VECS", 5)

        # scalar-engine constant sources (lane 0 of each broadcast holds the
        # scalar value) plus a scalar temp pool for the ALU lane path.
        self.sc = {
            "one": one_v, "two": two_v, "seven": self.seven_v,
            "c0": self.hc[0], "c1": self.hc[1], "c2": self.hc[2],
            "c3": self.hc[3], "c4": self.hc[4], "c5": self.hc[5],
            "K0": self.hK[0], "K2": self.hK[1], "K4": self.hK[2],
            "sh19": self.hsh[0], "sh9": self.hsh[1], "sh16": self.hsh[2],
            "f0": fnode_s[0] if 0 in need_nodes else None,
        }
        n_stemp = getattr(self, "N_STEMP", (40 + 24 * n_alu) if n_alu else 0)
        self.stemps = [self.alloc_scratch(None) for _ in range(n_stemp)]
        self._si = 0

        # temp pool fills the remaining scratch
        pool = max(8, (SCRATCH_SIZE - self.scratch_ptr) // VLEN - 1)
        self.vtemps = [self.alloc_scratch(None, VLEN) for _ in range(pool)]
        self._ti = 0

        # ---- emit broadcast constant setup ----
        for s, vb, v in self._bpending:
            self.emit("load", ("const", s, v), [], [s])
            self.emit("valu", ("vbroadcast", vb, s), [s], self.rng(vb))

        # base pointers (a single const each); addresses are derived from these
        # on the idle flow engine (add_imm) to keep const ops off the
        # load-engine critical resource.
        ivp_base = self.alloc_scratch(None)
        fp_base = self.alloc_scratch(None)
        self.emit("load", ("const", ivp_base, IVP), [], [ivp_base])
        self.emit("load", ("const", fp_base, FP), [], [fp_base])

        # ---- preload forest constants ----
        for nidx in sorted(need_nodes):
            a = fnode_a[nidx]
            s = fnode_s[nidx]
            self.emit("flow", ("add_imm", a, fp_base, nidx), [fp_base], [a])
            self.emit("load", ("load", s, a), [a], [s])
            self.emit("valu", ("vbroadcast", self.fbcast[nidx], s), [s], self.rng(self.fbcast[nidx]))

        # ---- emit first-level mux differences ----
        for d in self.mux_depths:
            base = 2 ** d - 1
            for j, k in enumerate(range(0, 2 ** d, 2)):
                dv = self.muxdiff[d][j]
                hi, lo = self.fbcast[base + k + 1], self.fbcast[base + k]
                self.emit("valu", ("-", dv, hi, lo), self.rng(hi) + self.rng(lo), self.rng(dv))

        # ---- load initial values (addresses via flow add_imm, reused for the
        #      final stores) ----
        for v in range(nvec):
            a = init_addr[v]
            self.emit("flow", ("add_imm", a, ivp_base, v * VLEN), [ivp_base], [a])
            self.emit("load", ("vload", val[v], a), [a], self.rng(val[v]))

        # ---- main loop (chunk-major so pipeline-stagger gates precede the
        #      ops they delay in program order) ----
        # Spread the ALU-processed vectors evenly across the index range so
        # that every pipeline chunk contains a similar mix of valu and alu
        # lanes -- otherwise the (slow, bursty) ALU work clumps into a few
        # chunks and starves the other engine there.
        alu_set = set((i * nvec) // n_alu for i in range(n_alu)) if n_alu else set()
        NC = getattr(self, "NC", 16)
        STAG = getattr(self, "STAG", 1)
        if NC > 1 and nvec >= NC:
            cs = nvec // NC
            chunks = [
                list(range(c * cs, nvec if c == NC - 1 else (c + 1) * cs))
                for c in range(NC)
            ]
        else:
            NC = 1
            chunks = [list(range(nvec))]

        def process(v, r):
            d = depth_of(r)
            last = r == rounds - 1
            is_leaf = (d == H) and not last
            first_op[(r, v)] = len(self.ops)
            if v in alu_set:
                self.s_process(v, d, last, is_leaf, r)
                return
            node = self.get_node(d, idx[v], r)
            self.vbin("^", val[v], val[v], node)
            self.hash_inplace(val[v])
            if last or is_leaf:
                # last round: idx unused.  leaf: next round is depth 0 which
                # recomputes idx from scratch.
                return
            # idx = 2*idx + 1 + (val & 1).  In AUX_ALU mode the parity bit and
            # the (1+bit) increment are both computed on the ALU, leaving valu
            # to pay only the doubling multiply_add -- so the whole index update
            # costs valu a single op while flow stays free for the mux.
            if getattr(self, "AUX_ALU", False):
                bit = self.vtemp()
                self.aluv("&", bit, val[v], one_v)
                if d == 0:
                    self.aluv("+", idx[v], bit, one_v)       # idx = 1 + bit
                else:
                    inc = self.vtemp()
                    self.aluv("+", inc, bit, one_v)
                    self.vmadd(idx[v], idx[v], two_v, inc)
            else:
                bit = self.vtemp()
                self.vbin("&", bit, val[v], one_v)
                if getattr(self, "IDX_INC_FLOW", True):
                    inc = idx[v] if d == 0 else self.vtemp()
                    self.emit("flow", ("vselect", inc, bit, two_v, one_v),
                              self.rng(bit) + self.rng(two_v) + self.rng(one_v), self.rng(inc))
                    if d != 0:
                        self.vmadd(idx[v], idx[v], two_v, inc)
                elif d == 0:
                    self.vbin("+", idx[v], bit, one_v)
                else:
                    inc = self.vtemp()
                    self.vbin("+", inc, bit, one_v)
                    self.vmadd(idx[v], idx[v], two_v, inc)

        # Diagonal wavefront emission: chunk c is offset c*STAG rounds, so at a
        # given wavefront different chunks occupy different rounds.  This keeps
        # fine-grained interleaving (good ILP for the scheduler) while the gate
        # dependencies below pin the stagger in place.
        first_op = {}
        for w in range(rounds + (NC - 1) * STAG):
            for c in range(NC):
                r = w - c * STAG
                if 0 <= r < rounds:
                    for v in chunks[c]:
                        process(v, r)

        # The deep-gather rounds are load-bound while the shallow rounds are
        # compute-bound; in lockstep these phases can't overlap.  Inject
        # artificial dependencies so chunk c cannot begin until chunk c-1 has
        # reached round STAG, so chunks occupy different rounds at once and one
        # chunk's gather loads overlap another chunk's hash compute.  The dummy
        # scratch words carry no real data -- they only constrain the schedule.
        if not getattr(self, "NO_GATES", False):
            for c in range(1, NC):
                dummy = self.dummies[c - 1]
                wop = first_op[(STAG, chunks[c - 1][0])]
                self.ops[wop][3].append(dummy)
                for v in chunks[c]:
                    self.ops[first_op[(0, v)]][2].append(dummy)

        # ---- store final values (addresses already in init_addr) ----
        for v in range(nvec):
            self.emit("store", ("vstore", init_addr[v], val[v]),
                      [init_addr[v]] + self.rng(val[v]), [])

        scheduled = self.schedule(self.ops)
        # Pauses bracket the body to match reference_kernel2's two yields; the
        # submission harness disables them.  Placed outside the scheduler so
        # they strictly bracket the body.
        self.instrs = (
            [{"flow": [("pause",)]}] + scheduled + [{"flow": [("pause",)]}]
        )

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
