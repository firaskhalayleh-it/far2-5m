"""The FAR² core (blueprint III·C + Part IV). Encode once, reason in a slot
state with a weight-shared core, decode from the final state.

Stage map (matches the blueprint's numbered diagram):
  ① encoder  ② goal pool (read-only g)  ③ slot seeds  ④ shared core loop
  ⑤ collision monitor  ⑥ repr controller (KEEP/EXPAND/SPLIT)
  ⑦ halt controller (CONTINUE/RECONSIDER/HALT)  ⑧ epistemic head
  ⑨ decoder  ⑩ answer head (shared with the per-slot probe)

With every mechanism flag off this IS the Slot-Recurrent baseline (baseline C):
same encoder, H, slots, cross-attention, decoder — no operators. The D−C delta
is the FAR² contribution under test. Modules are created only when their flag
is on, so parameter counts are honest per configuration.

H is immutable across recurrence but differentiable through training — it is
computed once per episode inside the graph and never detached.

v0.3 causal controls (see docs/DESIGN.md):
  · repr_policy      learned | random (rate-matched, task-blind decisions AND a
                     random split source) | oracle (SPLIT forced by the task
                     condition at t=T//2 — the perfect-decision upper bound)
  · reconsider_policy learned | random (reset-at-random control)
  · goal_scrambled   goal channel intact, content rolled to the wrong episode
  · forward(..., intervene=...) episode-level do()-operator for causal replay:
                     {"block": {"split",...}} turns chosen actions into no-ops,
                     {"force": [("split", t), ...]} imposes an action at
                     iteration t. Eval-only (gradients are not preserved).
Non-learned policies still create their controller modules so parameter counts
stay identical to far2_full; they simply never consult them."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from far2.config import INCONSISTENT, UNDERDET, ModelConfig
from far2.data.vocab import N_VARS, VOCAB
from far2.models.base import ModelOutput, ReasoningModel
from far2.models.layers import (MASK_FILL, MHA, MLP, Block, attn_entropy,
                                n_params, sig_embedding)
from far2.models.monitor import collision_signals

KEEP, EXPAND, SPLIT = 0, 1, 2
CONTINUE, RECONSIDER, HALT = 0, 1, 2


def _ctrl(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


def _st_choice(logits: torch.Tensor, tau: float, training: bool) -> tuple:
    """Straight-through action selection. Returns (y [B,K] one-hot with soft
    gradient, hard indices [B], probs [B,K])."""
    probs = F.softmax(logits / tau, dim=-1)
    if training:
        y = F.gumbel_softmax(logits, tau=tau, hard=True)
        return y, y.argmax(-1), probs
    hard = logits.argmax(-1)
    return F.one_hot(hard, logits.shape[-1]).float(), hard, probs


def _apply_intervention(intervene, t: int, hard: torch.Tensor, y: torch.Tensor,
                        names: dict) -> tuple:
    """do()-operator on a decided action. 'block' maps listed actions to the
    no-op (index 0: KEEP/CONTINUE); 'force' imposes an action at iteration t
    (overriding block). Rebuilds y as a plain one-hot — causal replay only."""
    if not intervene:
        return hard, y
    changed = False
    for name, idx in names.items():
        if name in intervene.get("block", ()):
            hard = torch.where(hard == idx, torch.zeros_like(hard), hard)
            changed = True
    for fname, ft in intervene.get("force", ()):
        if fname in names and ft == t:
            hard = torch.full_like(hard, names[fname])
            changed = True
    if changed:
        y = F.one_hot(hard, y.shape[-1]).float()
    return hard, y


def _forced_at(intervene, name: str, t: int) -> bool:
    return bool(intervene) and any(n == name and ft == t
                                   for n, ft in intervene.get("force", ()))


class FAR2Model(ReasoningModel):
    kind = "far2"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, S, dg, h = cfg.d_model, cfg.max_slots, cfg.goal_dim, cfg.controller_hidden
        self._needs_halt_ctrl = cfg.halting == "learned" or cfg.reconsider
        assert cfg.repr_policy in ("learned", "random", "oracle")
        assert cfg.reconsider_policy in ("learned", "random")
        assert cfg.oracle_action in ("split", "expand")
        assert cfg.cross_mode in ("standard", "competitive")
        assert cfg.n_registers + cfg.init_slots <= cfg.max_slots
        if cfg.reconsider_policy == "random":
            # the reset-at-random control runs at a fixed budget by design
            assert cfg.halting != "learned"

        # ① encoder
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Embedding(cfg.max_seq_len, d)
        self.sig = sig_embedding(d) if cfg.digit_sig else None
        self.encoder = nn.ModuleList(
            Block(d, cfg.n_heads, cfg.mlp_mult) for _ in range(cfg.n_encoder_layers))
        self.enc_ln = nn.LayerNorm(d)

        # ② slot seeds (+ fixed unit directions for deterministic eval-time SPLIT)
        self.seeds = nn.Parameter(torch.randn(S, d) * 0.02)
        self.register_buffer("split_dirs", F.normalize(torch.randn(S, d), dim=-1))

        # ③ shared core.  Every module shared by far2_5m, far2_full, and
        # far2_bind is constructed before optional mechanisms.  Consequently
        # the same declared seed means the same shared initialization across
        # ablations; adding a controller must not silently reseed the core.
        self.ln_read = nn.LayerNorm(d)
        self.self_attn = MHA(d, cfg.n_heads)
        self.ln_cross = nn.LayerNorm(d)
        self.cross_attn = MHA(d, cfg.n_heads)
        self.ln_mlp = nn.LayerNorm(d)
        self.mlp = MLP(d, cfg.mlp_mult, cfg.core_act)
        self.gate = nn.Linear(2 * d, d)
        nn.init.normal_(self.gate.weight, std=1e-3)   # bias dominates at init
        nn.init.constant_(self.gate.bias, cfg.gate_bias_init)
        self.rms_state = nn.RMSNorm(d)
        if cfg.iter_embedding:                        # ablation flag, default off
            self.core_iter_emb = nn.Embedding(cfg.n_iters_max, d)

        # ④ decoder + shared head (also before optional mechanisms for the
        # same initialization-parity contract).
        self.ln_q = nn.LayerNorm(d)
        self.dec_kv_ln = nn.LayerNorm(d)
        self.dec_cross = MHA(d, cfg.n_heads)
        self.ln_d2 = nn.LayerNorm(d)
        self.dec_mlp = MLP(d, cfg.mlp_mult)
        self.dec_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size)
        # Output queries are deliberately non-contextual.  Reusing H at ANS
        # positions gives the bidirectional encoder a direct residual path to
        # the answer head and makes workspace causality impossible to prove.
        self.out_pos = nn.Embedding(cfg.max_seq_len, d)

        # ⑤⑥ controllers  ⑦ epistemic — created only when their mechanism is on
        if cfg.adaptive_slots:
            self.repr_ctrl = _ctrl(d + 4, h, 3)
        if self._needs_halt_ctrl:
            self.halt_ctrl = _ctrl(d + 4, h, 3)
            self.halt_q = _ctrl(d + 4, h, 1)
        if cfg.epistemic:
            self.ans_epi_head = _ctrl(d, h, 4)        # answer-level: THE 5M mechanism
        if cfg.epistemic and cfg.slot_epistemic:      # per-slot tags: unsupervised
            self.epi_head = _ctrl(d, h, 5)            # extra, off by default
            self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)

        # ⑧ protected goal (read-only by construction: nothing writes g)
        if cfg.protected_goal:
            self.goal_query = nn.Parameter(torch.randn(d) * 0.02)
            self.goal_proj = nn.Linear(d, dg)
            self.goal_inject = nn.Linear(dg, d)
            self.goal_tok = nn.Linear(dg, d)

        # ⑨ binding substrate: variable->slot ownership and the learned
        # direction for binding-directed SPLIT.  It is last so enabling the
        # substrate cannot perturb any shared or controller initialization.
        if cfg.binding:
            self.bind_W = nn.Linear(d, d, bias=False)
            self.split_dir_proj = nn.Linear(d, d, bias=False)
            # Entity path: aggregate every occurrence of X_e, then let that
            # query retrieve its full relational context from H.  This is the
            # upstream representation used by ownership and collision
            # detection; it does not replace the general content reader.
            self.entity_ln_q = nn.LayerNorm(d)
            self.entity_attn = MHA(d, cfg.n_heads)
            self.entity_ln_mlp = nn.LayerNorm(d)
            self.entity_mlp = MLP(d, 1, cfg.core_act)
            self.entity_ln = nn.LayerNorm(d)
            self.entity_value_head = nn.Linear(d, 10)

        # runtime state set by the curriculum
        self.unlocked: set[str] = set()
        self.tau: float = 1.0

    # ---- runtime knobs -------------------------------------------------
    def set_unlocked(self, ops: set[str]) -> None:
        # disable_actions wins over any curriculum unlock (fine-grained ablations)
        self.unlocked = set(ops) - set(self.cfg.disable_actions)

    def set_tau(self, tau: float) -> None:
        self.tau = max(tau, 0.5)

    # ---- pieces --------------------------------------------------------
    def _encode(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        ids = batch["input_ids"]
        L = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        if self.sig is not None:
            x = x + self.sig(batch["digit_sig"])
        mask = torch.where(batch["pad_mask"], 0.0, MASK_FILL)[:, None, None, :]
        for blk in self.encoder:
            x = blk(x, mask)
        return self.enc_ln(x), mask

    def _entity_states(self, H, pad_mask, batch):
        """Contextual entity states E[e] from every rendered occurrence.

        Occurrence masks are token-derived by collate; meta.vars supplies only
        annealed training labels. Absent rows are exactly zero so no synthetic
        entity can enter binding or collision search.
        """
        occ = batch["var_occurrence"].float()                     # [B,V,L]
        present = occ.any(-1)
        q = torch.einsum("bvl,bld->bvd", occ, H)
        q = q / occ.sum(-1, keepdim=True).clamp_min(1.0)
        ctx, _ = self.entity_attn(self.entity_ln_q(q), H, pad_mask)
        E = q + ctx
        E = E + self.entity_mlp(self.entity_ln_mlp(E))
        return self.entity_ln(E) * present[..., None], present

    def _bind_matrix(self, E, Z, active, ownership_bias=None):
        """B[e,s] = softmax_s(E_e^T W_B z_s / sqrt(d)) over active CONCEPT
        slots. Registers are masked before the softmax and can never own a
        variable. Differentiable through both the entity and slot paths."""
        scores = self.bind_W(E) @ Z.transpose(1, 2) / (self.cfg.d_model ** 0.5)
        if ownership_bias is not None:
            scores = scores + ownership_bias
        scores = scores + torch.where(active.bool(), 0.0, MASK_FILL)[:, None, :]
        return F.softmax(scores, dim=-1)                          # [B,V,S]

    def _semantic_split_children(self, parent, entity_x, entity_y, scale):
        """Create a conservative, swap-equivariant two-child bifurcation.

        The existing binding projection supplies the shared midpoint direction;
        the dedicated split projection supplies the antisymmetric direction.
        A 2x2 Sinkhorn matrix then assigns the two entities to different child
        columns.  No ground-truth identity or value enters this operation.
        """
        midpoint = 0.5 * (entity_x + entity_y)
        difference = 0.5 * (entity_x - entity_y)
        center_dir = F.normalize(self.bind_W(midpoint), dim=-1)
        split_dir = F.normalize(self.split_dir_proj(difference), dim=-1)
        center = parent + scale[:, None] * center_dir
        child_x = self.rms_state(center + scale[:, None] * split_dir)
        child_y = self.rms_state(center - scale[:, None] * split_dir)
        children = torch.stack([child_x, child_y], dim=1)          # [B,2,d]
        entities = torch.stack([entity_x, entity_y], dim=1)       # [B,2,d]
        scores = (self.bind_W(entities) @ children.transpose(1, 2)
                  / (self.cfg.d_model ** 0.5))
        assignment = torch.exp(scores - scores.amax(dim=(1, 2), keepdim=True))
        for _ in range(6):
            assignment = assignment / assignment.sum(-1, keepdim=True).clamp_min(1e-8)
            assignment = assignment / assignment.sum(-2, keepdim=True).clamp_min(1e-8)
        return children, assignment

    def _goal(self, H, batch):
        # "goal" in disable_actions: the channel's parameters exist (param
        # identity with the reference model) but it is NEVER consulted —
        # required for slot_recurrent_bind to be a true zero-mechanism C
        if (not self.cfg.protected_goal or "goal" not in self.unlocked
                or "goal" in self.cfg.disable_actions):
            return None
        gm = batch["goal_mask"]
        scores = (H @ self.goal_query) / (self.cfg.d_model ** 0.5)
        scores = scores.masked_fill(~gm, MASK_FILL)
        pooled = (F.softmax(scores, dim=-1)[..., None] * H).sum(1)
        has_goal = gm.any(dim=-1, keepdim=True).float()
        g = self.goal_proj(pooled) * has_goal         # zero vector when no goal span
        if self.cfg.goal_scrambled:
            # capacity control: the channel stays (same params, same compute,
            # SAME gradient path — no detach, else the arms differ in
            # trainability, not just content) but carries ANOTHER episode's
            # goal. Meaningful with batch > 1 (roll is identity at B=1).
            g = torch.roll(g, shifts=1, dims=0)
        return g

    def _answer_queries(self, batch):
        """Syntax-only output queries: query-token identity + answer rank.

        No contextual encoder state enters this path.  Mathematical content
        must first pass through Z; a variable token may address its bound slot
        but cannot carry the variable's value itself.
        """
        ids = batch["answer_query_ids"]
        rank = torch.arange(ids.shape[1], device=ids.device)
        return self.tok(ids) + self.out_pos(rank)[None]

    def _answer_binding(self, B_mat, batch):
        """Gather query-variable ownership rows for answer routing.

        Returns ([B,A,S] weights, [B,A] valid).  Queryless arithmetic families
        retain the general decoder cross-attention; variable-query families
        must read the slot their token-derived entity binding selects.
        """
        qv = batch["answer_query_vars"]
        valid = qv >= 0
        idx = qv.clamp_min(0)[..., None].expand(-1, -1, B_mat.shape[-1])
        rows = B_mat.gather(1, idx)
        return rows * valid[..., None], valid

    def _init_workspace(self, H, pad_mask, active):
        """Initialize the workspace without opening a decoder bypass from H.

        ``input`` performs one read through the same cross-attention used by
        the recurrent core. ``input_shuffled`` preserves the computation and
        distribution while deliberately supplying another episode's evidence.
        Inactive slots remain exact learned seeds for the drift contract.
        """
        cfg = self.cfg
        B = H.shape[0]
        Z = self.seeds[None].expand(B, -1, -1).clone()
        if cfg.workspace_init not in ("learned", "input", "input_shuffled"):
            raise ValueError(f"unknown workspace_init={cfg.workspace_init!r}")
        if cfg.workspace_init == "learned":
            return Z
        init_H = torch.roll(H, 1, dims=0) if cfg.workspace_init == "input_shuffled" else H
        init_mask = (torch.roll(pad_mask, 1, dims=0)
                     if cfg.workspace_init == "input_shuffled" else pad_mask)
        competitive = cfg.cross_mode == "competitive"
        slot_q_mask = (torch.where(active.bool(), 0.0, MASK_FILL)
                       [:, None, :, None] if competitive else None)
        warm, _ = self.cross_attn(
            self.ln_cross(Z), init_H, init_mask, competitive=competitive,
            slot_mask=slot_q_mask)
        candidate = Z + warm * active[..., None]
        if cfg.state_norm == "post_rms":
            candidate = torch.where(active[..., None].bool(),
                                    self.rms_state(candidate), candidate)
        return candidate

    def _decode(self, Q, Z, active, g_tok=None, answer_binding=None):
        kv = self.dec_kv_ln(Z)
        mask = torch.where(active.bool(), 0.0, MASK_FILL)[:, None, None, :]
        if g_tok is not None:
            kv = torch.cat([kv, g_tok[:, None]], dim=1)
            mask = F.pad(mask, (0, 1))                # goal token always visible
        a, _ = self.dec_cross(self.ln_q(Q), kv, mask)
        if answer_binding is not None:
            weights, valid = answer_binding
            routed = weights @ self.dec_kv_ln(Z)
            a = torch.where(valid[..., None], routed, a)
        x = Q + a
        x = x + self.dec_mlp(self.ln_d2(x))
        return self.head(self.dec_ln(x))

    def _slot_probs(self, Z) -> torch.Tensor:
        return F.softmax(self.head(self.dec_ln(Z)), dim=-1)

    def _ctrl_rand(self, shape, dev, salt: int) -> torch.Tensor:
        """Uniform noise for the non-learned control policies. Training uses
        the live RNG; eval derives it from a fixed per-iteration seed so
        control arms replay deterministically. Either way the draw carries no
        task information — that is the point of the control."""
        if self.training:
            return torch.rand(shape, device=dev)
        g = torch.Generator().manual_seed(0x5EED + salt)
        return torch.rand(shape, generator=g).to(dev)

    def _policy_repr(self, batch, t: int, T: int, B: int, dev) -> torch.Tensor:
        """Representation action for the non-learned policies. [B] in
        {KEEP, EXPAND, SPLIT}; slot-availability and curriculum unlocks are
        enforced by the executor, exactly as for the learned controller."""
        keep = torch.zeros(B, dtype=torch.long, device=dev)
        if self.cfg.repr_policy == "random":
            p_s = self.cfg.random_split_p if "split" in self.unlocked else 0.0
            p_e = self.cfg.random_expand_p if "expand" in self.unlocked else 0.0
            u = self._ctrl_rand((B,), dev, 31 * t)
            return torch.where(u < p_s, torch.full_like(keep, SPLIT),
                               torch.where(u < p_s + p_e,
                                           torch.full_like(keep, EXPAND), keep))
        # oracle: the perfect-decision upper bound.  One action is scheduled
        # for EVERY broken pair, starting after the first workspace update.
        # Ground truth is legal only in this upper-bound policy, never in the
        # learned/random controllers or their signals.
        act = SPLIT if self.cfg.oracle_action == "split" else EXPAND
        name = self.cfg.oracle_action
        demand = batch.get("oracle_split_count")
        action_index = t - 1
        if (demand is None or name not in self.unlocked or action_index < 0
                or action_index >= T - 1):
            return keep
        return torch.where(demand > action_index,
                           torch.full_like(keep, act), keep)

    def _bind_collision(self, B_mat, entity_probs, present, relation, allowed):
        """Binding-space collision searched over every INPUT-DERIVED relation:
        C(x,y) = Overlap(B_x, B_y) * JSD(P(E_x), P(E_y)).

        Incompatibility is decoded from encoder-side entity states, before
        binding can collapse two concepts into the same slot. Candidate edges
        are variables co-occurring in rendered clauses. The episode's declared
        pair and requires_distinction label are never read here.

        Returns (C [B], SPLIT source slot [B], argmax pair [B,2] variable
        indices, valid mask [B] bool, max overlap [B], max incompatibility
        [B]). Heuristic signal: no gradient."""
        with torch.no_grad():
            Bm = B_mat * allowed[:, None, :]                      # [B,V,S]
            ov = torch.minimum(Bm[:, :, None], Bm[:, None]).sum(-1)   # [B,V,V]
            P = entity_probs.clamp_min(1e-9)
            pi_, pj_ = P[:, :, None], P[:, None]
            m = 0.5 * (pi_ + pj_)
            jsd = 0.5 * ((pi_ * (pi_ / m).log()).sum(-1)
                         + (pj_ * (pj_ / m).log()).sum(-1))
            V = ov.shape[1]
            ok = present[:, :, None] * present[:, None] * relation
            ok = ok * (1 - torch.eye(V, device=ov.device))[None]
            C = ov * jsd * ok
            flat = C.flatten(1)
            bC = flat.max(-1).values
            am = flat.argmax(-1)
            xy = torch.stack([am // V, am % V], dim=1)            # [B,2]
            ar = torch.arange(len(Bm), device=Bm.device)
            Bx, By = Bm[ar, xy[:, 0]], Bm[ar, xy[:, 1]]           # [B,S]
            src = (torch.minimum(Bx, By) - (1 - allowed) * 1e9).argmax(-1)
            valid = (ok.flatten(1) > 0).any(-1)
            neg = torch.full_like(ov, -1.0)
            max_ov = torch.where(ok.bool(), ov, neg).flatten(1).max(-1).values
            max_jsd = torch.where(ok.bool(), jsd, neg).flatten(1).max(-1).values
            return bC, src, xy, valid, max_ov.clamp_min(0), max_jsd.clamp_min(0)

    # ---- forward -------------------------------------------------------
    def forward(self, batch, n_iters=None, ipt=False, collect=False,
                intervene=None, all_cycle_logits=False) -> ModelOutput:
        cfg = self.cfg
        B = batch["input_ids"].shape[0]
        S, d = cfg.max_slots, cfg.d_model
        dev = batch["input_ids"].device
        # n_iters=0 is a deliberate evaluation-only readout of the initialized
        # workspace.  It is the v0.4.52 Z0 shortcut control; treating zero as
        # false and silently replacing it with the training depth would make
        # initialization-dominant models look recurrent.
        T = (cfg.n_iters_train if n_iters is None
             else min(max(int(n_iters), 0), cfg.n_iters_max))
        if self.training and T == 0:
            raise ValueError("n_iters=0 is an evaluation-only Z0 diagnostic")

        H, pad_mask = self._encode(batch)
        Q = self._answer_queries(batch)
        g = self._goal(H, batch)
        gd = self.goal_inject(g) if g is not None else None
        g_tok = self.goal_tok(g) if g is not None else None
        entity_states = None
        entity_present = None
        entity_logits = None
        entity_probs = None
        if cfg.binding:
            # Computed once from immutable H.  Recurrent binding may change as
            # slots change, but the entity-side evidence cannot be erased by a
            # slot collision and is therefore a legitimate detector input.
            entity_states, entity_present = self._entity_states(H, pad_mask, batch)
            entity_logits = self.entity_value_head(entity_states)
            entity_probs = F.softmax(entity_logits, dim=-1)
        if cfg.split_impl == "semantic" and not cfg.binding:
            raise ValueError("semantic SPLIT requires the binding substrate")

        active = torch.zeros(B, S, device=dev)
        active[:, : cfg.n_registers + cfg.init_slots] = 1.0
        Z = self._init_workspace(H, pad_mask, active)
        if self.training:                      # jitter keeps seed roles flexible;
            Z = Z + cfg.slot_jitter * torch.randn_like(Z)   # eval is deterministic
        Z0_init = Z.detach().clone()
        # registers: always-active background slots — never SPLIT sources,
        # excluded from the collision pair search
        reg_mask = torch.zeros(S, device=dev)
        reg_mask[: cfg.n_registers] = 1.0
        alive = torch.ones(B, device=dev)

        # binding-directed SPLIT direction: computed PER ITERATION from the
        # SEARCHED collision pair (never batch["pair_idx"] — see
        # _bind_collision). Initialized None; set in the loop when binding.
        pair_dir = None
        split_pair = None
        ownership_bias = (torch.zeros(B, N_VARS, S, device=dev)
                          if cfg.binding else None)
        semantic_assignments = []
        iters_used = torch.full((B,), T, dtype=torch.long, device=dev)
        counts = {k: torch.zeros(B, device=dev) for k in ("expand", "split", "reconsider")}

        adaptive = (cfg.adaptive_slots
                    and ("expand" in self.unlocked or "split" in self.unlocked))
        learned_halt = cfg.halting == "learned" and "halt" in self.unlocked
        use_reconsider = cfg.reconsider and "reconsider" in self.unlocked
        use_halt_ctrl = learned_halt or use_reconsider
        use_slot_epi = (cfg.epistemic and cfg.slot_epistemic
                        and "epistemic" in self.unlocked)
        need_cross_probs = collect or (adaptive
                                       and cfg.collision_signal.startswith("attn"))

        detach_after, ipt_at = -1, -1
        if ipt and self.training and T > 1:
            detach_after = random.randint(0, T - 1)
            ipt_at = random.randint(detach_after + 1, T)

        state_sq = H.new_zeros(())
        grow_pressure = H.new_zeros(())
        cont_pressure = H.new_zeros(())
        entropies = H.new_zeros(())
        split_logit = torch.full((B,), -10.0, device=dev)
        reconsider_logit = torch.full((B,), -10.0, device=dev)
        norms, deltas, attn_ents = [], [], []
        probe_preds = []
        C_max = torch.zeros(B, device=dev)
        ipt_logits = None
        cycle_logits = []
        last_delta_sq = H.new_zeros(())
        halt_feats_last = None
        signals = None
        cross_final = None
        B_mat = None
        bind_src = None
        # PRE-ACTION signal window: the signal a controller could have seen
        # BEFORE any representation action executed. The P4 tournament scores
        # these — post-action state must not vote for the detector that
        # (supposedly) triggered it.
        acted = torch.zeros(B, device=dev)
        C_max_pre = torch.zeros(B, device=dev)
        sig_pre: dict = {}

        for t in range(T):
            Z_prev = Z
            upd_mask = (active * alive[:, None])[..., None]        # [B,S,1]

            # read view (per-slot epistemic tags only if that flag is on)
            if use_slot_epi:
                epi = self.epi_head(Z)
                Z_read = Z + F.softmax(epi[..., :4], dim=-1) @ self.type_emb
            else:
                Z_read = Z

            hs = self.ln_read(Z_read)
            slot_mask = torch.where(active.bool(), 0.0, MASK_FILL)[:, None, None, :]
            kv = hs
            m = slot_mask
            if g_tok is not None:
                kv = torch.cat([kv, self.ln_read(g_tok)[:, None]], dim=1)
                m = F.pad(slot_mask, (0, 1))
            sa, _ = self.self_attn(hs, kv, m)
            u = Z_read + sa

            competitive = cfg.cross_mode == "competitive"
            slot_q_mask = (torch.where(active.bool(), 0.0, MASK_FILL)
                           [:, None, :, None] if competitive else None)
            ca, cross_probs = self.cross_attn(
                self.ln_cross(u), H, pad_mask, need_probs=need_cross_probs,
                competitive=competitive, slot_mask=slot_q_mask)
            u = u + ca
            if collect and cross_probs is not None:
                attn_ents.append(attn_entropy(cross_probs))
                cross_final = cross_probs.detach().mean(1)          # [B,S,L]

            mlp_in = self.ln_mlp(u)
            if gd is not None:
                mlp_in = mlp_in + gd[:, None]
            if cfg.iter_embedding:
                mlp_in = mlp_in + self.core_iter_emb.weight[
                    t % cfg.n_iters_train][None, None]
            dz = self.mlp(mlp_in)

            G = torch.sigmoid(self.gate(torch.cat([Z, dz], dim=-1)))
            if cfg.gate_style == "convex":
                Z_new = upd_mask * ((1 - G) * Z + G * dz) + (1 - upd_mask) * Z
            else:
                Z_new = Z + upd_mask * G * dz
            if cfg.state_norm == "post_rms":
                Z = torch.where(upd_mask > 0, self.rms_state(Z_new), Z_new)
            else:
                Z = Z_new
            if t + 1 == detach_after:
                Z = Z.detach()

            # stats (active slots only)
            n_active = active.sum().clamp_min(1)
            sq = (Z.pow(2).mean(-1) * active).sum() / n_active
            state_sq = state_sq + sq
            last_delta_sq = ((Z - Z_prev).pow(2).mean(-1) * active).sum() / n_active
            norms.append(float(sq.detach().sqrt()))
            deltas.append(float(last_delta_sq.detach().sqrt()))

            # binding matrix (differentiable — the binding loss and directed
            # SPLIT train through it; the SIGNAL path below uses a detached copy)
            act_pair = active * (1 - reg_mask)[None]
            if cfg.binding:
                # Variable ownership is over CONCEPT slots only. Registers
                # absorb background content and must never own a variable;
                # otherwise collision search silently loses probability mass
                # when it later excludes registers.
                B_mat = self._bind_matrix(entity_states, Z, act_pair,
                                          ownership_bias)

            # ⑤ collision monitor — heuristic signal, no gradient path by design
            with torch.no_grad():
                slot_probs = self._slot_probs(Z)
                cp = cross_probs.mean(1) if cross_probs is not None else None
                sig = collision_signals(Z, act_pair, slot_probs, cp)
            bind_src = None
            pair_dir = None
            if B_mat is not None:
                bC, bsrc, bxy, bvalid, bov, bej = self._bind_collision(
                    B_mat.detach(), entity_probs.detach(),
                    entity_present.float(), batch["var_rel"].float(), act_pair)
                sig["bind"] = bC        # searched signal: a P4 candidate
                # Ownership overlap is also runtime-legitimate. P4 may select
                # it independently; only the DECLARED-pair metric added by the
                # evaluator carries the non-selectable `_probe` suffix.
                sig["binding_overlap"] = bov
                # Runtime-legitimate demand signal: unlike overlap, this asks
                # whether a rendered relation contains incompatible entities.
                # P4 may select it.  `bind` remains the localized
                # demand-times-collision score used to choose the split source.
                sig["entity_relation"] = bej
                if cfg.collision_signal in ("bind", "entity_relation",
                                            "binding_overlap"):
                    bind_src = (bsrc, bvalid)
                    split_pair = (bxy, bvalid)
                # SPLIT direction from the SEARCHED pair's contextual entity
                # states — differentiable through E (only search is no-grad).
                arB = torch.arange(B, device=dev)
                eb_ea = (entity_states[arB, bxy[:, 1]]
                         - entity_states[arB, bxy[:, 0]])
                pair_dir = (F.normalize(self.split_dir_proj(eb_ea), dim=-1),
                            bvalid.float())
                # The oracle upper bound receives the complete broken-pair
                # sequence, so its target is genuinely perfect.  Free and
                # random policies stay on the searched, token-derived path.
                oi = t - 1
                if (cfg.repr_policy == "oracle"
                        and cfg.oracle_action == "split"
                        and 0 <= oi < batch["oracle_split_pairs"].shape[1]):
                    opair = batch["oracle_split_pairs"][:, oi]
                    oval = (opair >= 0).all(-1)
                    safe = opair.clamp_min(0)
                    ox = B_mat[arB, safe[:, 0]] * act_pair
                    oy = B_mat[arB, safe[:, 1]] * act_pair
                    osrc = torch.minimum(ox, oy).argmax(-1)
                    bind_src = (osrc, oval)
                    split_pair = (safe, oval)
                    odiff = (entity_states[arB, safe[:, 1]]
                             - entity_states[arB, safe[:, 0]])
                    pair_dir = (F.normalize(self.split_dir_proj(odiff), dim=-1),
                                oval.float())
            chosen = sig.get(cfg.collision_signal, sig["C"])
            chosen_max = (chosen if chosen.dim() == 1 else
                          chosen.flatten(1).max(-1).values)         # [B]
            if (B_mat is not None
                    and cfg.collision_signal in ("bind", "entity_relation",
                                                 "binding_overlap")):
                chosen_max = torch.where(bvalid, chosen_max,
                                         torch.zeros_like(chosen_max))
            C_max = torch.maximum(C_max, chosen_max)
            if collect:
                # the signal at t informs the action at t, so t belongs to the
                # pre-action window of an episode acting at t (acted is
                # updated AFTER execution below)
                pre = acted < 0.5
                C_max_pre = torch.where(pre, torch.maximum(C_max_pre, chosen_max),
                                        C_max_pre)
                for k, v in sig.items():
                    if k in ("C_max", "C_argmax_pair") or v.dim() not in (1, 3):
                        continue          # dim 1: per-sample (searched bind C)
                    flat = v.flatten(1).max(-1).values if v.dim() == 3 else v
                    prev = sig_pre.get(k)
                    sig_pre[k] = (flat.clone() if prev is None else
                                  torch.where(pre, torch.maximum(prev, flat),
                                              prev))
            if collect and t == T - 1:
                signals = sig

            pooled = (Z * active[..., None]).sum(1) / n_active_per(active)
            frac = active.sum(-1, keepdim=True) / S
            tt = torch.full((B, 1), (t + 1) / T, device=dev)
            drms = ((Z - Z_prev).pow(2).mean(-1) * active).sum(-1, keepdim=True) \
                / active.sum(-1, keepdim=True).clamp_min(1)

            # ⑥ representation action — decided by the configured policy
            # (learned | random | oracle), then optionally overridden by a
            # do()-intervention; execution is identical for every policy
            run_repr = adaptive or _forced_at(intervene, "expand", t) \
                or _forced_at(intervene, "split", t)
            if run_repr:
                free = (active < 0.5).any(dim=-1, keepdim=True).float()
                free_s = free.squeeze(-1)
                if adaptive and cfg.repr_policy == "learned":
                    feats = torch.cat([pooled, chosen_max[:, None], frac, tt,
                                       drms.detach()], dim=-1)
                    logits = self.repr_ctrl(feats)
                    amask = torch.zeros_like(logits)
                    if "expand" not in self.unlocked:
                        amask[:, EXPAND] = MASK_FILL
                    if "split" not in self.unlocked:
                        amask[:, SPLIT] = MASK_FILL
                    amask[:, EXPAND] = amask[:, EXPAND] + (1 - free_s) * MASK_FILL
                    amask[:, SPLIT] = amask[:, SPLIT] + (1 - free_s) * MASK_FILL
                    y, hard, probs = _st_choice(logits + amask, self.tau, self.training)
                    split_logit = torch.maximum(split_logit, logits[:, SPLIT])
                    grow_pressure = grow_pressure + (probs[:, EXPAND] + probs[:, SPLIT]).mean()
                    entropies = entropies + (-(probs.clamp_min(1e-9).log() * probs).sum(-1)).mean()
                else:
                    # control policies (or a pure forced replay): the learned
                    # controller is never consulted
                    hard = (self._policy_repr(batch, t, T, B, dev) if adaptive
                            else torch.zeros(B, dtype=torch.long, device=dev))
                    y = F.one_hot(hard, 3).float()
                hard, y = _apply_intervention(intervene, t, hard, y,
                                              {"expand": EXPAND, "split": SPLIT})

                # halted samples act no more; a full workspace cannot grow
                gate_ok = alive * free_s
                k_idx = ((active < 0.5).float()
                         * torch.arange(S, 0, -1, device=dev)[None]).argmax(-1)
                k_onehot = F.one_hot(k_idx, S).float()

                # EXPAND: activate the next seed (+ jitter in training)
                y_e = (y[:, EXPAND] * gate_ok)[:, None, None]
                new_seed = self.seeds[k_idx]
                if self.training:
                    new_seed = new_seed + cfg.slot_jitter * torch.randn(B, d, device=dev)
                Z = Z * (1 - k_onehot[..., None] * y_e) + new_seed[:, None] * k_onehot[..., None] * y_e

                # SPLIT source: the slot the CHOSEN signal marks as most
                # collided (binding mode: the slot both pair members own most)
                # — except under the random policy, where the source is a
                # random ACTIVE non-register slot (a control must not aim)
                if adaptive and cfg.repr_policy == "random":
                    r = self._ctrl_rand((B, S), dev, 97 * t + 1)
                    src_idx = (r - (1 - act_pair) * 1e9).argmax(-1)
                else:
                    row_max = chosen.max(dim=-1).values + (act_pair - 1) * 1e9
                    src_idx = row_max.argmax(-1)
                    if bind_src is not None:
                        bsrc, bvalid = bind_src
                        src_idx = torch.where(bvalid, bsrc, src_idx)
                src = Z[torch.arange(B, device=dev), src_idx]
                d_near = _nearest_dist(Z.detach(), active, src_idx)
                # symmetry breaking is mandatory (identical copies are a saddle
                # point); at eval the direction is a fixed buffer so runs replay.
                # With binding, the clone direction is the LEARNED projection of
                # the pair's contextual-entity difference — S_src stays biased
                # toward x, the clone starts biased toward y.
                if self.training:
                    noise = F.normalize(torch.randn(B, d, device=dev), dim=-1)
                else:
                    noise = self.split_dirs[k_idx]
                if pair_dir is not None:
                    dirv, pdv = pair_dir
                    noise = pdv[:, None] * dirv + (1 - pdv[:, None]) * noise
                semantic = cfg.split_impl == "semantic" and split_pair is not None
                pair_valid = (split_pair[1].float() if semantic
                              else torch.ones_like(gate_ok))
                split_ok = gate_ok * pair_valid
                y_s = (y[:, SPLIT] * split_ok)[:, None, None]
                if semantic:
                    pair, _ = split_pair
                    ar = torch.arange(B, device=dev)
                    ex = entity_states[ar, pair[:, 0]]
                    ey = entity_states[ar, pair[:, 1]]
                    children, assignment = self._semantic_split_children(
                        src, ex, ey, cfg.split_beta * d_near)
                    src_onehot = F.one_hot(src_idx, S).float()
                    split_slots = src_onehot + k_onehot
                    Z = Z * (1 - split_slots[..., None] * y_s)
                    Z = Z + children[:, 0, None] * src_onehot[..., None] * y_s
                    Z = Z + children[:, 1, None] * k_onehot[..., None] * y_s

                    # Persist the discrete one-to-one ownership selected by
                    # Sinkhorn. This prevents the next ordinary binding softmax
                    # from silently collapsing both entities back onto one slot.
                    fired = ((hard == SPLIT).float() * split_ok).bool()
                    for bi in fired.nonzero(as_tuple=False).flatten().tolist():
                        x, yy = pair[bi].tolist()
                        x_child = int(assignment[bi, 0].argmax())
                        slots = (int(src_idx[bi]), int(k_idx[bi]))
                        sx, sy = slots[x_child], slots[1 - x_child]
                        ownership_bias[bi, x].fill_(-12.0)
                        ownership_bias[bi, yy].fill_(-12.0)
                        ownership_bias[bi, x, sx] = 12.0
                        ownership_bias[bi, yy, sy] = 12.0
                        semantic_assignments.append({
                            "batch": bi, "pair": [x, yy],
                            "slots": [sx, sy],
                            "sinkhorn": assignment[bi].detach().cpu().tolist(),
                        })
                else:
                    clone = src + cfg.split_beta * d_near[:, None] * noise
                    Z = (Z * (1 - k_onehot[..., None] * y_s)
                         + clone[:, None] * k_onehot[..., None] * y_s)

                hard_grow = ((hard == EXPAND).float() * gate_ok
                             + (hard == SPLIT).float() * split_ok)
                active = (active + k_onehot * hard_grow[:, None]).clamp(max=1.0)
                counts["expand"] = counts["expand"] + (hard == EXPAND).float() * gate_ok
                counts["split"] = counts["split"] + (hard == SPLIT).float() * split_ok
                acted = torch.clamp(acted + hard_grow, max=1.0)

            # ⑦ halt controller — RECONSIDER can run at a FIXED budget (P5a);
            # the HALT action only exists when halting is learned (P5b).
            # Policies: learned ctrl | random reset control | forced replay.
            recon_fire = None                                   # [B] in [0,1]
            if use_halt_ctrl and cfg.reconsider_policy == "learned":
                hfeats = torch.cat([pooled, drms.detach(), frac, tt,
                                    C_max[:, None]], dim=-1)
                hlogits = self.halt_ctrl(hfeats)
                hmask = torch.zeros_like(hlogits)
                if not use_reconsider:
                    hmask[:, RECONSIDER] = MASK_FILL
                if not learned_halt:
                    hmask[:, HALT] = MASK_FILL
                yh, hardh, probsh = _st_choice(hlogits + hmask, self.tau, self.training)
                reconsider_logit = torch.maximum(reconsider_logit, hlogits[:, RECONSIDER])
                entropies = entropies + (-(probsh.clamp_min(1e-9).log() * probsh).sum(-1)).mean()
                halt_feats_last = hfeats
                if learned_halt:
                    cont_pressure = cont_pressure + (probsh[:, CONTINUE] * alive).mean()
                hardh, yh = _apply_intervention(intervene, t, hardh, yh,
                                                {"reconsider": RECONSIDER})
                recon_fire = yh[:, RECONSIDER]
                if learned_halt:
                    just_halted = (hardh == HALT).float() * alive
                    iters_used = torch.where(just_halted.bool(),
                                             torch.tensor(t + 1, device=dev),
                                             iters_used)
                    alive = alive * (1 - just_halted)
            elif use_reconsider:                                # reset-at-random
                fired = (self._ctrl_rand((B,), dev, 7919 * (t + 1))
                         < cfg.random_reconsider_p).long() * RECONSIDER
                fired, y_rr = _apply_intervention(
                    intervene, t, fired, F.one_hot(fired, 3).float(),
                    {"reconsider": RECONSIDER})
                recon_fire = y_rr[:, RECONSIDER]
            elif _forced_at(intervene, "reconsider", t):
                # do(RECONSIDER) works on any FAR² model — the partial reset
                # needs no controller module
                recon_fire = torch.ones(B, device=dev)

            if recon_fire is not None:
                # RECONSIDER: partial reset toward this episode's initialized
                # workspace.  With input warm-start, resetting to generic seeds
                # would erase the very evidence the foundation repair preserves.
                y_r = (recon_fire * alive)[:, None, None]
                blend = (cfg.reconsider_alpha * Z
                         + (1 - cfg.reconsider_alpha) * Z0_init)
                blend = active[..., None] * blend + (1 - active[..., None]) * Z
                Z = y_r * blend + (1 - y_r) * Z
                counts["reconsider"] = counts["reconsider"] + recon_fire * alive

            if (cfg.halting == "converge" and not use_halt_ctrl
                    and not self.training and t >= 1):
                r = (Z - Z_prev).abs().amax(dim=(1, 2)) / Z.abs().amax(dim=(1, 2)).clamp_min(1e-6)
                just = (r < cfg.converge_tol).float() * alive
                iters_used = torch.where(just.bool(), torch.tensor(t + 1, device=dev), iters_used)
                alive = alive * (1 - just)

            decode_binding = None
            if cfg.binding:
                decode_B = self._bind_matrix(
                    entity_states, Z, active * (1 - reg_mask)[None],
                    ownership_bias)
                decode_binding = self._answer_binding(decode_B, batch)
            if collect:
                with torch.no_grad():
                    probe_preds.append(self._decode(
                        Q, Z, active, g_tok, decode_binding).argmax(-1))
            if t + 1 == ipt_at and ipt_at != T:
                ipt_logits = self._decode(Q, Z, active, g_tok, decode_binding)
            if all_cycle_logits and t + 1 < T:
                cycle_logits.append(
                    self._decode(Q, Z, active, g_tok, decode_binding))
            if not self.training and float(alive.sum()) == 0:
                break

        answer_binding = None
        if cfg.binding:
            # Recompute after the last action/reset.  The previous B_mat was
            # measured before that state mutation and is stale for decoding,
            # binding loss, and telemetry.
            act_pair = active * (1 - reg_mask)[None]
            B_mat = self._bind_matrix(entity_states, Z, act_pair,
                                      ownership_bias)
            answer_binding = self._answer_binding(B_mat, batch)
        logits = self._decode(Q, Z, active, g_tok, answer_binding)

        # ⑧ OPERATIONAL epistemic decision (review #3): the answer-level head
        # is a decision component, not a regularizer — at inference a
        # non-DERIVED verdict ROUTES the output to the corresponding
        # abstention token instead of the decoded digits. Training keeps the
        # auxiliary CE below (targets still teach the decoder the abstention
        # tokens; the head decides when to emit them).
        epi_logits = None
        if cfg.epistemic and "epistemic" in self.unlocked:
            pooled_f = (Z * active[..., None]).sum(1) / n_active_per(active)
            epi_logits = self.ans_epi_head(pooled_f)
            if not self.training:
                route = epi_logits.argmax(-1)                     # [B]
                for cls, tok in ((UNDERDET, "UNDERDETERMINED"),
                                 (INCONSISTENT, "NO_SOLUTION")):
                    msk = route == cls
                    if msk.any():
                        one = torch.full_like(logits[msk], -1e4)
                        one[:, 0, VOCAB.t(tok)] = 1e4
                        if one.shape[1] > 1:
                            one[:, 1:, VOCAB.t("PAD")] = 1e4
                        logits[msk] = one

        loss_terms = {
            "state_sq": state_sq / max(T, 1),
            "fp_delta": last_delta_sq,
            "slot_cost": grow_pressure / max(T, 1),
            "iter_cost": cont_pressure / max(T, 1),
            "ctrl_entropy": entropies / max(T, 1),
            "split_logit": split_logit,
            "reconsider_logit": reconsider_logit,
        }
        if ipt_logits is not None:
            loss_terms["ipt_logits"] = ipt_logits
        if cycle_logits:
            loss_terms["cycle_logits"] = torch.stack(cycle_logits, dim=1)
        # binding-curriculum loss: SOME slot bound to variable e must decode
        # e's value — a permutation-free soft assignment CE (mixture over the
        # ownership distribution). Annealed to zero before transfer claims.
        if cfg.binding and B_mat is not None:
            vv = batch["var_values"]
            vmask = ((vv >= 0) & entity_present).float()
            if vmask.sum() > 0:
                p_slots = F.softmax(self.head(self.dec_ln(Z)), dim=-1)
                tgt = vv.clamp_min(0)
                pt = p_slots.gather(
                    2, tgt[:, None, :].expand(-1, S, -1)).transpose(1, 2)
                mix = (B_mat * pt).sum(-1).clamp_min(1e-9)
                loss_terms["bind_nll"] = -(mix.log() * vmask).sum() / vmask.sum()
            # Direct supervision makes E a testable entity representation,
            # instead of hoping a first-token anchor incidentally contains the
            # variable's relational value.  Labels never enter forward control.
            emask = (vv >= 0) & entity_present
            if emask.any():
                digit0 = VOCAB.t("D0")
                loss_terms["entity_nll"] = F.cross_entropy(
                    entity_logits[emask], vv[emask] - digit0)
        if epi_logits is not None:
            loss_terms["ans_epi_logits"] = epi_logits
        if learned_halt and halt_feats_last is not None:
            loss_terms["halt_q_logit"] = self.halt_q(halt_feats_last).squeeze(-1)

        telemetry = {
            "iters_used": iters_used.tolist(),
            "iters_used_mean": float(iters_used.float().mean()),
            "active_slots": active.sum(-1).tolist(),
            "active_frac": float((active.sum(-1) / S).mean()),
            "expand_count": counts["expand"].tolist(),
            "split_count": counts["split"].tolist(),
            "reconsider_count": counts["reconsider"].tolist(),
            "state_norm": norms, "state_delta": deltas,
            "gate_mean": (float(G.detach().mean()) if T > 0
                          else float("nan")),
            "C_max": C_max.tolist(),
            "attn_entropy": attn_ents,
        }
        if collect and probe_preds:
            telemetry["probe_preds"] = torch.stack(probe_preds, dim=1)   # [B,T,A]
        if collect and cross_final is not None:
            telemetry["cross_attn"] = cross_final                        # [B,S,L]
        if collect:
            # Required even at the v0.4.52 Z0 readout, where no recurrent
            # cross-attention step has produced `cross_final` yet.
            telemetry["active_mask"] = active.detach()                   # [B,S]
            telemetry["Z_final"] = Z.detach()      # [B,S,d] for linear probes
            telemetry["H"] = H.detach()            # [B,L,d] for location probes
            telemetry["n_registers"] = cfg.n_registers
            if B_mat is not None:
                telemetry["bind_matrix"] = B_mat.detach()   # [B,V,S] ownership
                telemetry["entity_states"] = entity_states.detach()  # [B,V,d]
                telemetry["entity_present"] = entity_present.detach()
                telemetry["entity_value_logits"] = entity_logits.detach()
                telemetry["semantic_split_assignments"] = semantic_assignments
        if collect and signals is not None:
            telemetry["signal_max"] = {
                k: (v.flatten(1).max(-1).values if v.dim() == 3 else v).tolist()
                for k, v in signals.items()
                if k not in ("C_max", "C_argmax_pair") and v.dim() in (1, 3)}
        if collect and sig_pre:
            telemetry["signal_max_pre"] = {k: v.tolist()
                                           for k, v in sig_pre.items()}
            telemetry["C_max_pre"] = C_max_pre.tolist()
        if collect:
            # inactive slots must sit exactly at their init — any drift means
            # the update mask leaks (classic implementation bug)
            drift = ((Z - Z0_init).abs() * (1 - active)[..., None]).max()
            telemetry["inactive_drift"] = float(drift)
        return ModelOutput(logits, loss_terms, telemetry)

    def flops_per_sample(self, seq_len, n_iters, n_ans=1, active_slots=None,
                         n_decodes=1):
        cfg = self.cfg
        d, vocab = cfg.d_model, cfg.vocab_size
        S = active_slots if active_slots is not None else cfg.max_slots
        enc = sum(2 * n_params(b) * seq_len for b in self.encoder) + 4 * seq_len ** 2 * d
        entity = 0
        if cfg.binding:
            # One immutable-H entity read: projections + VxL attention, the
            # compact entity MLP, and the ten-way value head.
            entity = (4 * d * d * (N_VARS + seq_len)
                      + 4 * N_VARS * seq_len * d
                      + 2 * n_params(self.entity_mlp) * N_VARS
                      + 2 * d * 10 * N_VARS)
        core_p = (n_params(self.self_attn) + n_params(self.cross_attn)
                  + n_params(self.mlp) + n_params(self.gate))
        per_iter = 2 * core_p * S + 4 * S * S * d + 4 * S * seq_len * d
        dec = 2 * (n_params(self.dec_cross) + n_params(self.dec_mlp)) * n_ans \
            + 4 * n_ans * (S + 1) * d + 2 * d * vocab * n_ans
        return enc + entity + n_iters * per_iter + n_decodes * dec


def n_active_per(active: torch.Tensor) -> torch.Tensor:
    return active.sum(-1, keepdim=True).clamp_min(1)


def _nearest_dist(Z: torch.Tensor, active: torch.Tensor, src_idx: torch.Tensor) -> torch.Tensor:
    """Distance from each sample's source slot to its nearest other active slot
    (SPLIT noise scale). [B]"""
    B, S, _ = Z.shape
    src = Z[torch.arange(B, device=Z.device), src_idx]           # [B,d]
    dist = (Z - src[:, None]).pow(2).sum(-1).sqrt()              # [B,S]
    dist = dist + (1 - active) * 1e9
    dist.scatter_(1, src_idx[:, None], 1e9)                      # exclude self
    near = dist.min(-1).values
    return torch.where(near > 1e8, torch.ones_like(near), near)  # lone slot: unit scale
