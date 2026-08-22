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

from far2.config import ModelConfig
from far2.models.base import ModelOutput, ReasoningModel, gather_ans
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

        # binding substrate (v0.4): variable->slot ownership matrix and the
        # learned direction for binding-directed SPLIT
        if cfg.binding:
            self.bind_W = nn.Linear(d, d, bias=False)
            self.split_dir_proj = nn.Linear(d, d, bias=False)

        # ② goal pool + inject (read-only by construction: nothing writes g)
        if cfg.protected_goal:
            self.goal_query = nn.Parameter(torch.randn(d) * 0.02)
            self.goal_proj = nn.Linear(d, dg)
            self.goal_inject = nn.Linear(dg, d)
            self.goal_tok = nn.Linear(dg, d)

        # ③ slot seeds (+ fixed unit directions for deterministic eval-time SPLIT)
        self.seeds = nn.Parameter(torch.randn(S, d) * 0.02)
        self.register_buffer("split_dirs", F.normalize(torch.randn(S, d), dim=-1))

        # ④ shared core
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

        # ⑥⑦ controllers  ⑧ epistemic — created only when their mechanism is on
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

        # ⑨⑩ decoder + shared head
        self.ln_q = nn.LayerNorm(d)
        self.dec_kv_ln = nn.LayerNorm(d)
        self.dec_cross = MHA(d, cfg.n_heads)
        self.ln_d2 = nn.LayerNorm(d)
        self.dec_mlp = MLP(d, cfg.mlp_mult)
        self.dec_ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size)

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

    def _bind_matrix(self, H, Z, active, batch):
        """B[e,s] = softmax_s(h_e^T W_B z_s / sqrt(d)) over ACTIVE slots.
        h_e = encoder state at the variable's anchor position. Rows of
        variables absent from the episode are meaningless — consumers mask
        with var_valid. Differentiable: the binding-curriculum loss and the
        directed-SPLIT projection train through it."""
        pos = batch["var_pos"].clamp_min(0)                       # [B,V]
        hv = torch.gather(H, 1, pos[..., None].expand(-1, -1, H.shape[-1]))
        scores = self.bind_W(hv) @ Z.transpose(1, 2) / (self.cfg.d_model ** 0.5)
        scores = scores + torch.where(active.bool(), 0.0, MASK_FILL)[:, None, :]
        return F.softmax(scores, dim=-1)                          # [B,V,S]

    def _goal(self, H, batch):
        if not self.cfg.protected_goal:
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

    def _decode(self, Qa, Z, active, g_tok=None):
        kv = self.dec_kv_ln(Z)
        mask = torch.where(active.bool(), 0.0, MASK_FILL)[:, None, None, :]
        if g_tok is not None:
            kv = torch.cat([kv, g_tok[:, None]], dim=1)
            mask = F.pad(mask, (0, 1))                # goal token always visible
        a, _ = self.dec_cross(self.ln_q(Qa), kv, mask)
        x = Qa + a
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
        # oracle: the perfect-decision upper bound — force cfg.oracle_action
        # exactly when the task condition says the pair must be kept distinct,
        # once, mid-loop. Uses ground truth at run time: a bound, never a win.
        act = SPLIT if self.cfg.oracle_action == "split" else EXPAND
        name = self.cfg.oracle_action
        req = batch.get("requires_distinction")
        if req is None or name not in self.unlocked or t != max(1, T // 2):
            return keep
        return torch.where(req == 1, torch.full_like(keep, act), keep)

    def _bind_collision(self, B_mat, Z, batch, allowed):
        """Binding-space collision for the episode's declared pair:
        C = Overlap(B_x, B_y) * JSD(P_x, P_y), where P_e decodes the
        binding-weighted state. This compares CONCEPTS (who owns the
        variables), not slots with slots. Returns (C [B], SPLIT source =
        argmax_s min(Bx, By) over allowed slots, valid mask [B] bool).
        Heuristic signal: computed without gradient."""
        with torch.no_grad():
            pi = batch["pair_idx"]
            valid = pi[:, 0] >= 0
            idx = pi.clamp_min(0)
            ar = torch.arange(len(B_mat), device=B_mat.device)
            Ba, Bb = B_mat[ar, idx[:, 0]], B_mat[ar, idx[:, 1]]     # [B,S]
            overlap = torch.minimum(Ba, Bb)
            za = torch.einsum("bs,bsd->bd", Ba, Z)
            zb = torch.einsum("bs,bsd->bd", Bb, Z)
            pa = F.softmax(self.head(self.dec_ln(za)), dim=-1)
            pb = F.softmax(self.head(self.dec_ln(zb)), dim=-1)
            m = (0.5 * (pa + pb)).clamp_min(1e-9)
            jsd = 0.5 * ((pa * (pa.clamp_min(1e-9) / m).log()).sum(-1)
                         + (pb * (pb.clamp_min(1e-9) / m).log()).sum(-1))
            src = (overlap - (1 - allowed) * 1e9).argmax(-1)
            return overlap.sum(-1) * jsd, src, valid

    # ---- forward -------------------------------------------------------
    def forward(self, batch, n_iters=None, ipt=False, collect=False,
                intervene=None) -> ModelOutput:
        cfg = self.cfg
        B = batch["input_ids"].shape[0]
        S, d = cfg.max_slots, cfg.d_model
        dev = batch["input_ids"].device
        T = min(n_iters or cfg.n_iters_train, cfg.n_iters_max)

        H, pad_mask = self._encode(batch)
        Qa = gather_ans(H, batch["ans_pos"])
        g = self._goal(H, batch)
        gd = self.goal_inject(g) if g is not None else None
        g_tok = self.goal_tok(g) if g is not None else None

        Z = self.seeds[None].expand(B, S, d).clone()
        if self.training:                      # jitter keeps seed roles flexible;
            Z = Z + cfg.slot_jitter * torch.randn_like(Z)   # eval is deterministic
        Z0_init = Z.detach().clone()
        active = torch.zeros(B, S, device=dev)
        active[:, : cfg.n_registers + cfg.init_slots] = 1.0
        # registers: always-active background slots — never SPLIT sources,
        # excluded from the collision pair search
        reg_mask = torch.zeros(S, device=dev)
        reg_mask[: cfg.n_registers] = 1.0
        alive = torch.ones(B, device=dev)

        # binding-directed SPLIT direction: learned projection of the pair's
        # anchor difference (h_b - h_a); trains through the clone content
        pair_dir = None
        if cfg.binding:
            vp = batch["var_pos"]
            ppos = torch.gather(vp, 1, batch["pair_idx"].clamp_min(0))
            pd_valid = ((batch["pair_idx"][:, 0] >= 0)
                        & (ppos.min(dim=1).values >= 0)).float()
            ppos = ppos.clamp_min(0)
            ar = torch.arange(B, device=dev)
            hb_ha = H[ar, ppos[:, 1]] - H[ar, ppos[:, 0]]
            pair_dir = (F.normalize(self.split_dir_proj(hb_ha), dim=-1), pd_valid)
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
                B_mat = self._bind_matrix(H, Z, active, batch)

            # ⑤ collision monitor — heuristic signal, no gradient path by design
            with torch.no_grad():
                slot_probs = self._slot_probs(Z)
                cp = cross_probs.mean(1) if cross_probs is not None else None
                sig = collision_signals(Z, act_pair, slot_probs, cp)
            chosen = sig.get(cfg.collision_signal, sig["C"])
            chosen_max = chosen.flatten(1).max(-1).values           # [B]
            bind_src = None
            if cfg.collision_signal == "bind" and B_mat is not None:
                bC, bsrc, bvalid = self._bind_collision(
                    B_mat.detach(), Z.detach(), batch, act_pair)
                chosen_max = torch.where(bvalid, bC, chosen_max)
                bind_src = (bsrc, bvalid)
            C_max = torch.maximum(C_max, chosen_max)
            if collect:
                # the signal at t informs the action at t, so t belongs to the
                # pre-action window of an episode acting at t (acted is
                # updated AFTER execution below)
                pre = acted < 0.5
                C_max_pre = torch.where(pre, torch.maximum(C_max_pre, chosen_max),
                                        C_max_pre)
                for k, v in sig.items():
                    if k in ("C_max", "C_argmax_pair") or v.dim() != 3:
                        continue
                    flat = v.flatten(1).max(-1).values
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
                # the pair's anchor difference — S_src stays biased toward x,
                # the clone starts biased toward y.
                if self.training:
                    noise = F.normalize(torch.randn(B, d, device=dev), dim=-1)
                else:
                    noise = self.split_dirs[k_idx]
                if pair_dir is not None:
                    dirv, pdv = pair_dir
                    noise = pdv[:, None] * dirv + (1 - pdv[:, None]) * noise
                clone = src + cfg.split_beta * d_near[:, None] * noise
                y_s = (y[:, SPLIT] * gate_ok)[:, None, None]
                Z = Z * (1 - k_onehot[..., None] * y_s) + clone[:, None] * k_onehot[..., None] * y_s

                hard_grow = ((hard == EXPAND) | (hard == SPLIT)).float() * gate_ok
                active = (active + k_onehot * hard_grow[:, None]).clamp(max=1.0)
                counts["expand"] = counts["expand"] + (hard == EXPAND).float() * gate_ok
                counts["split"] = counts["split"] + (hard == SPLIT).float() * gate_ok
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
                # RECONSIDER: partial reset of active slots toward seeds; H untouched
                y_r = (recon_fire * alive)[:, None, None]
                blend = cfg.reconsider_alpha * Z + (1 - cfg.reconsider_alpha) * self.seeds[None]
                blend = active[..., None] * blend + (1 - active[..., None]) * Z
                Z = y_r * blend + (1 - y_r) * Z
                counts["reconsider"] = counts["reconsider"] + recon_fire * alive

            if (cfg.halting == "converge" and not use_halt_ctrl
                    and not self.training and t >= 1):
                r = (Z - Z_prev).abs().amax(dim=(1, 2)) / Z.abs().amax(dim=(1, 2)).clamp_min(1e-6)
                just = (r < cfg.converge_tol).float() * alive
                iters_used = torch.where(just.bool(), torch.tensor(t + 1, device=dev), iters_used)
                alive = alive * (1 - just)

            if collect:
                with torch.no_grad():
                    probe_preds.append(self._decode(Qa, Z, active, g_tok).argmax(-1))
            if t + 1 == ipt_at and ipt_at != T:
                ipt_logits = self._decode(Qa, Z, active, g_tok)
            if not self.training and float(alive.sum()) == 0:
                break

        logits = self._decode(Qa, Z, active, g_tok)

        loss_terms = {
            "state_sq": state_sq / T,
            "fp_delta": last_delta_sq,
            "slot_cost": grow_pressure / T,
            "iter_cost": cont_pressure / T,
            "ctrl_entropy": entropies / T,
            "split_logit": split_logit,
            "reconsider_logit": reconsider_logit,
        }
        if ipt_logits is not None:
            loss_terms["ipt_logits"] = ipt_logits
        # binding-curriculum loss: SOME slot bound to variable e must decode
        # e's value — a permutation-free soft assignment CE (mixture over the
        # ownership distribution). Annealed to zero before transfer claims.
        if cfg.binding and B_mat is not None:
            vv = batch["var_values"]
            vmask = ((vv >= 0) & (batch["var_pos"] >= 0)).float()
            if vmask.sum() > 0:
                p_slots = F.softmax(self.head(self.dec_ln(Z)), dim=-1)
                tgt = vv.clamp_min(0)
                pt = p_slots.gather(
                    2, tgt[:, None, :].expand(-1, S, -1)).transpose(1, 2)
                mix = (B_mat * pt).sum(-1).clamp_min(1e-9)
                loss_terms["bind_nll"] = -(mix.log() * vmask).sum() / vmask.sum()
        if cfg.epistemic and "epistemic" in self.unlocked:
            pooled_f = (Z * active[..., None]).sum(1) / n_active_per(active)
            loss_terms["ans_epi_logits"] = self.ans_epi_head(pooled_f)
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
            "gate_mean": float(G.detach().mean()),
            "C_max": C_max.tolist(),
            "attn_entropy": attn_ents,
        }
        if collect and probe_preds:
            telemetry["probe_preds"] = torch.stack(probe_preds, dim=1)   # [B,T,A]
        if collect and cross_final is not None:
            telemetry["cross_attn"] = cross_final                        # [B,S,L]
            telemetry["active_mask"] = active.detach()                   # [B,S]
        if collect:
            telemetry["Z_final"] = Z.detach()      # [B,S,d] for linear probes
            telemetry["H"] = H.detach()            # [B,L,d] for location probes
            telemetry["n_registers"] = cfg.n_registers
            if B_mat is not None:
                telemetry["bind_matrix"] = B_mat.detach()   # [B,V,S] ownership
        if collect and signals is not None:
            telemetry["signal_max"] = {
                k: v.flatten(1).max(-1).values.tolist()
                for k, v in signals.items()
                if k not in ("C_max", "C_argmax_pair") and v.dim() == 3}
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

    def flops_per_sample(self, seq_len, n_iters, n_ans=1, active_slots=None):
        cfg = self.cfg
        d, V = cfg.d_model, cfg.vocab_size
        S = active_slots if active_slots is not None else cfg.max_slots
        enc = sum(2 * n_params(b) * seq_len for b in self.encoder) + 4 * seq_len ** 2 * d
        core_p = (n_params(self.self_attn) + n_params(self.cross_attn)
                  + n_params(self.mlp) + n_params(self.gate))
        per_iter = 2 * core_p * S + 4 * S * S * d + 4 * S * seq_len * d
        dec = 2 * (n_params(self.dec_cross) + n_params(self.dec_mlp)) * n_ans \
            + 4 * n_ans * (S + 1) * d + 2 * d * V * n_ans
        return enc + n_iters * per_iter + dec


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
