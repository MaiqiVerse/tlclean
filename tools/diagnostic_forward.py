"""OOM-safe drop-in for kernel_retrieval.extract_from_model.

Problem solved
--------------
extract_from_model passes output_attentions=True at the model level, which
makes HF Llama accumulate every layer's N x N attention matrix into
all_self_attns. For Llama-2-7B with 32 layers and 32 heads at bf16:

    accumulated = 32 (layers) * 32 (heads) * 2 bytes * N^2  =  2048 N^2 bytes

    N      accumulated
    1500   ~9.2 GB          OK on a 40 GB GPU
    2500   ~26 GB           tight even on A100
    4000   ~65 GB           OOM on most GPUs
    6600   ~178 GB          impossible (TREC 10/cls case from the doc)

Strategy
--------
Don't pass output_attentions=True at the model level. Instead, monkey-patch
self_attn.forward at the TL layers only, internally pass output_attentions=
True so the layer produces attn_weights, capture alpha[answer_pos, :] from
that tensor while it is transiently alive, and discard the rest by returning
None for the attn_weights position. The transient per-layer N x N matrix
still gets allocated (it's inherent to eager attention) but it is freed
when the layer's forward returns -- no cross-layer accumulation.

Memory budget on Llama-2-7B (bf16):
    Per-layer TRANSIENT (alive during one layer's fwd, freed on return):
        ~128 N^2 bytes  =  0.5 GB at N=2000,  5.6 GB at N=6600
    PERSISTENT state captured here (per TL layer, on CPU in fp32):
        alpha row:     N * num_heads * 4 bytes  =  128 N bytes (~0.25 MB at N=2000)
        v cache:       N * num_kv_heads * d_head * 2 bytes  =  8192 N bytes  (~16 MB at N=2000)
        u (computed):  N * d_model * 4 bytes  =  16384 N bytes (~32 MB at N=2000)

    Total persistent CPU memory for |T| TL layers: ~50 * N * |T| MB
        e.g., |T|=5, N=2000 ->  ~250 MB CPU
              |T|=5, N=6600 ->  ~825 MB CPU

    GPU peak (model + per-layer transient):
        14 GB (Llama-2-7B bf16) + per-layer transient (above)
        -> N=2500: ~15 GB    fits on a 24 GB GPU
        -> N=4000: ~16 GB    still fits
        -> N=6600: ~20 GB    fits on a 24 GB GPU; well within 40+
        -> N>7000: per-layer transient alone exceeds typical card; would
                    need sdpa/flash attention (out of scope here)

Returns the same TLExtraction object as extract_from_model -> drop-in
replacement at calling sites.

Future optimization (not implemented): load model with
attn_implementation='sdpa' (no N^2 transient) and hook q_proj, k_proj,
v_proj instead. Then apply RoPE manually to captured Q[N] and K[all] and
compute alpha[N, :] = softmax(Q[N] @ K.T / sqrt(d) + causal_mask). This
brings transient down to O(N * d_head) per layer but requires reimplementing
the model's RoPE call.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import Tensor

from tools.icl_common import head_dim
from tools.kernel_retrieval import TLExtraction
from tools.tl_heads import TLHeadSet


@contextmanager
def _patch_tl_attention_capture(model, layers_needed, answer_position, key_pad=None,
                                row_mode=None):
    """Monkey-patch self_attn.forward at the given layers so the answer
    position's attention row alpha[N, :] is captured, per layer, as a CPU
    fp32 Tensor[num_heads, n_tokens]; yields the dict layer_idx -> row.

    Two ways, chosen by the model's attention kernel:

    eager  -- the layer is told output_attentions=True, it computes the full
              attn_weights internally, the answer row is sliced out and the
              matrix is dropped at layer exit (the historical path; every
              registered number was read this way, bit for bit).
    sdpa   -- the fused kernels return no weights, and transformers 4.52
              drops the model-level causal mask under sdpa whenever nothing
              is padded, so forcing output_attentions=True here made those
              layers fall back to eager attention WITH NO MASK: non-causal
              rows, and a wrong layer output downstream (RESULTS 63.17c).
              Instead the row is recomputed from the layer's own q_proj /
              k_proj (q_norm / k_norm when the module has them, Qwen3), the
              module's rotary embedding and scaling, in float32 -- the
              arithmetic the fused kernels do internally -- against every key
              at or before the answer position; the layer itself runs
              untouched under sdpa. Mirrors HF's Llama-family attention
              forward; a module without apply_rotary_pos_emb / repeat_kv in
              its file, or with a sliding window, is refused rather than
              approximated.

    `key_pad` (bool Tensor[n_tokens], True = padded key) closes keys the 2-D
    attention_mask closed; None when nothing is padded. `row_mode` overrides
    the choice -- "weights" (slice attn_weights) or "recompute" -- so a test
    can read an eager forward both ways; None = by the kernel.
    """
    captured: dict[int, Tensor] = {}
    originals: dict[int, callable] = {}
    impl = getattr(model.config, "_attn_implementation", "eager") or "eager"
    mode = row_mode or ("weights" if impl == "eager" else "recompute")
    if mode not in ("weights", "recompute"):
        raise ValueError(f"row_mode {mode!r}: expected 'weights' or 'recompute'")

    def make_patched(_orig, _lidx):
        def patched(*args, **kwargs):
            # Force this layer to produce attn_weights internally even
            # though the model-level call did not request them.
            kwargs = dict(kwargs)
            kwargs["output_attentions"] = True
            out = _orig(*args, **kwargs)
            # Output is (attn_output, attn_weights, [past_key_value]).
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn_weights = out[1]
                # [B=1, num_heads, q_len, kv_len]; slice answer-row, ship to CPU
                captured[_lidx] = (
                    attn_weights[0, :, answer_position, :]
                    .detach().to(torch.float32).cpu()
                )
                # Replace with None to drop the GPU reference at layer exit.
                return (out[0], None) + tuple(out[2:])
            return out
        return patched

    def make_row_patched(_orig, _lidx, _attn):
        import sys
        mod = sys.modules[type(_attn).__module__]
        rot = getattr(mod, "apply_rotary_pos_emb", None)
        rep = getattr(mod, "repeat_kv", None)
        if rot is None or rep is None:
            raise RuntimeError(
                f"layer {_lidx}: {type(_attn).__name__} ({mod.__name__}) has no "
                f"apply_rotary_pos_emb / repeat_kv; the answer-row capture under "
                f"{impl!r} mirrors HF's Llama-family attention and will not guess "
                "at this module. Run the model under eager attention.")
        if getattr(_attn, "sliding_window", None):
            raise RuntimeError(
                f"layer {_lidx}: sliding_window={_attn.sliding_window}; the "
                f"answer-row capture under {impl!r} attends to every earlier key")
        for name in ("q_proj", "k_proj", "head_dim", "num_key_value_groups", "scaling"):
            if not hasattr(_attn, name):
                raise RuntimeError(f"layer {_lidx}: {type(_attn).__name__} has no {name}; "
                                   f"the answer-row capture under {impl!r} needs it")

        def patched(*args, **kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
            if hs is None or pe is None:
                raise RuntimeError(
                    f"layer {_lidx}: the answer-row capture needs hidden_states and "
                    "position_embeddings, which transformers 4.52's decoder layer "
                    "passes by keyword; got neither")
            with torch.no_grad():
                shape = (*hs.shape[:-1], -1, _attn.head_dim)
                q = _attn.q_proj(hs).view(shape)
                k = _attn.k_proj(hs).view(shape)
                if hasattr(_attn, "q_norm"):
                    q = _attn.q_norm(q)
                if hasattr(_attn, "k_norm"):
                    k = _attn.k_norm(k)
                q, k = q.transpose(1, 2), k.transpose(1, 2)
                cos, sin = pe
                q, k = rot(q, k, cos, sin)
                k = rep(k, _attn.num_key_value_groups)
                qr = q[0, :, answer_position, :].to(torch.float32)        # [H, d]
                kk = k[0].to(torch.float32)                                # [H, T, d]
                s = torch.einsum("hd,htd->ht", qr, kk) * float(_attn.scaling)
                s[:, answer_position + 1:] = float("-inf")
                if key_pad is not None:
                    s[:, key_pad.to(s.device)] = float("-inf")
                captured[_lidx] = torch.softmax(s, dim=-1).detach().cpu()
            return _orig(*args, **kwargs)
        return patched

    for layer_idx in layers_needed:
        attn = model.model.layers[layer_idx].self_attn
        orig_fn = attn.forward
        originals[layer_idx] = orig_fn
        if mode == "weights":
            attn.forward = make_patched(orig_fn, layer_idx)
        else:
            attn.forward = make_row_patched(orig_fn, layer_idx, attn)

    try:
        yield captured
    finally:
        for layer_idx, orig_fn in originals.items():
            model.model.layers[layer_idx].self_attn.forward = orig_fn


def v_patch_columns(heads, n_attn_heads, n_kv_heads, d_head,
                    allow_kv_group=False):
    """Columns of v_proj's output belonging to the given ATTENTION heads.

    v_proj emits [n_tokens, n_kv_heads * d_head] -- one slot per KEY-VALUE head.

    Under MHA (n_kv == n_attn) every attention head owns a slot, so the slice is
    exact and this is a genuine per-head write.

    Under GQA several attention heads share a slot. There is no per-query-head
    value anywhere in the model's computation: repeat_kv runs inside
    LlamaAttention.forward with no module boundary between it and the attention
    itself, so no v_proj hook can reach the post-repeat tensor, and patching
    o_proj's input would replace the head's OUTPUT rather than its value. The
    write is therefore a KV-GROUP intervention, and rather than widen the group
    silently this refuses unless the caller says it accepts that -- and names
    the heads that would be dragged in.

    Returns a sorted LongTensor of column indices; heads=None means all of them.
    """
    if heads is None:
        return None                       # signals "whole row", the old path
    heads = sorted({int(h) for h in heads})
    if not heads:
        return torch.empty(0, dtype=torch.long)
    bad = [h for h in heads if not 0 <= h < n_attn_heads]
    if bad:
        raise ValueError(f"attention head index out of range: {bad} "
                         f"(model has {n_attn_heads})")
    group = n_attn_heads // n_kv_heads
    if group > 1:
        kv = sorted({h // group for h in heads})
        dragged = sorted({q for k in kv for q in range(k * group, (k + 1) * group)}
                         - set(heads))
        if dragged and not allow_kv_group:
            raise ValueError(
                f"GQA: this model has {n_attn_heads} attention heads over "
                f"{n_kv_heads} KV heads ({group} per group), so v_proj has no "
                f"per-query-head slot. Writing heads {heads} necessarily also "
                f"writes {dragged}, because they read the same KV slots "
                f"{kv}. Pass allow_kv_group=True to accept KV-GROUP semantics "
                f"and say so wherever the result is reported, or choose whole "
                f"KV groups so nothing is dragged in.")
        slots = kv
    else:
        slots = heads                     # MHA: attention head == KV slot
    cols = torch.cat([torch.arange(k * d_head, (k + 1) * d_head,
                                   dtype=torch.long) for k in slots])
    return cols


def _run_capture_forward(
    model,
    input_ids: Tensor,
    tl_heads: TLHeadSet,
    answer_position: Optional[int],
    attention_mask: Optional[Tensor],
    inject: Optional[tuple] = None,
    patch_v: Optional[dict] = None,
    capture_resid: bool = False,
    allow_kv_group: bool = False,
    row_mode: Optional[str] = None,
) -> tuple[dict[int, Tensor], dict[int, Tensor], dict]:
    """Shared core for diagnostic_forward and extract_per_head_logit_contributions.

    Runs one model forward with:
      - output_attentions=False at the model level (no cross-layer accumulation)
      - monkey-patched self_attn.forward at TL layers to capture alpha[N, :]
      - v_proj forward hooks at TL layers to capture V at all positions

    Returns:
        alpha_cache: dict layer -> Tensor[num_heads, n_tokens]   (fp32 CPU)
        v_cache:     dict layer -> Tensor[num_heads, n_tokens, d_head]  (CPU, orig dtype)
        dims:        {'n_tokens', 'answer_position', 'd_head', 'd_model',
                      'n_attn_heads', 'n_kv_heads'}
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"input_ids must be [1, n_tokens]; got {tuple(input_ids.shape)}"
        )
    n_tokens = int(input_ids.shape[1])
    if answer_position is None:
        answer_position = n_tokens - 1
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    cfg = model.config
    n_attn_heads = cfg.num_attention_heads
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_attn_heads)
    d_model = cfg.hidden_size
    # NOT hidden_size // n_attn_heads. That is the Llama convention and it is
    # wrong wherever config.head_dim is set on its own (Qwen3-4B: 2560 / 32 =
    # 80 against a real head_dim of 128, and the v_proj view below fails).
    # icl_common.head_dim reads the width off o_proj; same number on Llama.
    d_head = head_dim(model)
    layers_needed = tl_heads.layers()

    v_cache: dict[int, Tensor] = {}
    v_raw_cache: dict[int, Tensor] = {}

    def make_v_hook(layer_idx: int):
        def hook(module, _input, output):
            is_tensor = isinstance(output, torch.Tensor)
            out = output if is_tensor else output[0]
            raw = out[0]                                     # [n_tokens, n_kv*d_head]
            # ---- optional VALUE-STREAM PATCH -------------------------------
            # Overwrite v_proj's output at chosen positions. This is the whole
            # point of the intervention: alpha comes from Q.K and phi from V, so
            # replacing V leaves every KEY untouched and therefore leaves this
            # layer's alpha bit-identical. (A prompt edit cannot do that: it
            # changes the hidden state, hence k_i = W_K x_i, hence alpha -- which
            # is why the prompt-edit design could never reach slope 1.)
            if patch_v and layer_idx in patch_v:
                # (pos, vals) writes every head, exactly as before; a third
                # element selects ATTENTION heads and leaves the rest at their
                # natural value. The old 2-tuple form and heads=None take the
                # identical branch, so sections 29-32 remain bit-reproducible --
                # that is gate 1 of #28a, and it is why this is not written as
                # "always compute columns, sometimes all of them".
                spec = patch_v[layer_idx]
                pos, vals = spec[0], spec[1]
                heads = spec[2] if len(spec) > 2 else None
                cols = v_patch_columns(heads, n_attn_heads, n_kv_heads, d_head,
                                       allow_kv_group=allow_kv_group)
                raw = raw.clone()
                vals = vals.to(raw.device, raw.dtype)
                if cols is None:
                    raw[pos] = vals
                elif cols.numel():
                    p = torch.as_tensor(pos, dtype=torch.long,
                                        device=raw.device).reshape(-1)
                    c = cols.to(raw.device)
                    raw[p[:, None], c[None, :]] = vals[:, c]
                # cols.numel() == 0 is the SHAM arm: the clone is returned
                # untouched, so the whole patch path executes and changes
                # nothing. Gate 2.
                out = raw.unsqueeze(0)
            v_raw_cache[layer_idx] = raw.detach().cpu()      # donors come from here
            v = raw.view(n_tokens, n_kv_heads, d_head)
            v = v.transpose(0, 1).contiguous()               # [n_kv_heads, n_tok, d_head]
            if n_kv_heads != n_attn_heads:
                v = v.repeat_interleave(n_attn_heads // n_kv_heads, dim=0)
            v_cache[layer_idx] = v.detach().cpu()
            if patch_v and layer_idx in patch_v:
                # returning a value replaces the module's output downstream
                return out if is_tensor else (out,) + tuple(output[1:])
            return None
        return hook

    v_handles = []
    for l in layers_needed:
        h = model.model.layers[l].self_attn.v_proj.register_forward_hook(make_v_hook(l))
        v_handles.append(h)

    # optional residual injection at one layer's OUTPUT (answer position), so the
    # captured alpha/V of DOWNSTREAM layers reflect the injected direction.
    inject_handle = None
    if inject is not None:
        _inj_layer, _inj_vec = inject

        def _inj_hook(module, _i, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[0, answer_position, :] = h[0, answer_position, :] + _inj_vec.to(h.dtype)
            return (h,) + output[1:] if isinstance(output, tuple) else h

        inject_handle = model.model.layers[_inj_layer].register_forward_hook(_inj_hook)

    # optional RESIDUAL-STREAM capture at the answer position, one row per layer.
    # This is what lets the same forward answer "does anything downstream of the
    # TL heads absorb their write" -- project each row on the readout direction
    # and the slope becomes a function of depth instead of a single number at
    # the end. Costs a 16 KB slice copy per layer; output_hidden_states=True
    # would instead materialize the whole [n_tokens, d_model] stack per layer.
    resid_rows: dict[int, Tensor] = {}
    resid_handles = []
    if capture_resid:
        def make_resid_hook(slot: int):
            def hook(_m, _i, output):
                h = output[0] if isinstance(output, tuple) else output
                resid_rows[slot] = h[0, answer_position, :].detach().to(
                    torch.float32).cpu()
                return None
            return hook

        # slot 0 = embedding output (the stream before any layer writes to it),
        # slot l+1 = output of decoder layer l. So the last slot is the vector
        # the final RMSNorm sees, and <last, v> / rms(last) must reproduce the
        # measured logit margin exactly -- the gate on this whole capture.
        resid_handles.append(
            model.model.embed_tokens.register_forward_hook(make_resid_hook(0)))
        for _li in range(len(model.model.layers)):
            resid_handles.append(model.model.layers[_li]
                                 .register_forward_hook(make_resid_hook(_li + 1)))

    answer_logits = None
    try:
        key_pad = (attention_mask[0] == 0) if bool((attention_mask == 0).any()) else None
        with _patch_tl_attention_capture(model, layers_needed, answer_position,
                                         key_pad=key_pad, row_mode=row_mode) as alpha_cache:
            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,     # IMPORTANT: keep this False
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
                # The model already computed these; keeping the answer-position
                # row costs nothing and is the only way to read the OUTPUT layer
                # in the same forward as the direct-write layer. Comparing the
                # two needs them from ONE pass -- a second forward would differ
                # by bf16 nondeterminism, which is the size of the effects here.
                if getattr(out, "logits", None) is not None:
                    answer_logits = out.logits[0, answer_position, :].detach(
                    ).to(torch.float32).cpu()
    finally:
        for h in v_handles:
            h.remove()
        if inject_handle is not None:
            inject_handle.remove()
        for h in resid_handles:
            h.remove()

    resid_at_answer = None
    if capture_resid:
        n_slots = len(model.model.layers) + 1
        missing = [s for s in range(n_slots) if s not in resid_rows]
        if missing:
            raise RuntimeError(
                f"_run_capture_forward: residual stream not captured at slots "
                f"{missing[:8]} of {n_slots}. The hook fires on every decoder "
                f"layer plus embed_tokens, so a gap means the module tree is "
                f"not the expected model.model.layers / model.model.embed_tokens.")
        resid_at_answer = torch.stack([resid_rows[s] for s in range(n_slots)])

    missing = [l for l in layers_needed if l not in alpha_cache]
    if missing:
        raise RuntimeError(
            f"_run_capture_forward: alpha not captured at layers {missing}. "
            f"under eager the layer returned attn_weights=None although output_attentions "
            f"was forced True; under sdpa the answer-row recomputation did not run. "
            f"Kernel: {getattr(model.config, '_attn_implementation', None)!r}."
        )

    dims = {
        "n_tokens": n_tokens,
        "answer_position": answer_position,
        "d_head": d_head,
        "d_model": d_model,
        "n_attn_heads": n_attn_heads,
        "n_kv_heads": n_kv_heads,
        # Tensor[vocab] fp32 CPU, or None if the model returned no logits.
        # Not an int like the rest -- callers that only want shapes ignore it.
        "answer_logits": answer_logits,
        # raw v_proj output per layer, [n_tokens, n_kv*d_head] fp CPU. Donor
        # values for a value-stream patch are sliced straight out of this.
        "v_raw": v_raw_cache,
        # Tensor[n_layers+1, d_model] fp32 CPU, or None. Residual stream at the
        # answer position, slot 0 = embeddings, slot l+1 = after decoder layer l.
        "resid_at_answer": resid_at_answer,
    }
    return alpha_cache, v_cache, dims


def diagnostic_forward(
    model,
    input_ids: Tensor,
    tl_heads: TLHeadSet,
    answer_position: Optional[int] = None,
    attention_mask: Optional[Tensor] = None,
    patch_v: Optional[dict] = None,
    capture_resid: bool = False,
    allow_kv_group: bool = False,
) -> TLExtraction:
    """OOM-safe drop-in for tools.kernel_retrieval.extract_from_model.

    Same arguments and return type. Avoids the all-layers attention
    accumulation by NOT passing output_attentions=True at the model
    level, and instead capturing the answer-position alpha row per TL
    layer via monkey-patched self_attn.forward.

    Memory note: this builds u_per_head = [n_tokens, d_model] per head.
    For a SMALL head set (a deployed TLHeadSet of ~8 heads) this is fine
    (~32 MB/head at n_tokens=2000). For an ALL-HEADS sweep (1024 heads)
    this is ~100 GB of CPU churn -- use extract_per_head_logit_contributions
    instead, which never materializes u.
    """
    alpha_cache, v_cache, dims = _run_capture_forward(
        model, input_ids, tl_heads, answer_position, attention_mask,
        patch_v=patch_v, capture_resid=capture_resid,
        allow_kv_group=allow_kv_group
    )
    d_head = dims["d_head"]
    d_model = dims["d_model"]

    alpha_per_head: dict[tuple[int, int], Tensor] = {}
    u_per_head: dict[tuple[int, int], Tensor] = {}
    delta_tl = torch.zeros(d_model, dtype=torch.float32)

    for (l, k) in sorted(tl_heads.heads):
        alpha_lk = alpha_cache[l][k]                      # [n_tokens] fp32 CPU
        alpha_per_head[(l, k)] = alpha_lk

        v_lk = v_cache[l][k].to(torch.float32)            # [n_tokens, d_head]
        o_weight = model.model.layers[l].self_attn.o_proj.weight.detach()
        o_k = o_weight[:, k * d_head:(k + 1) * d_head].to(torch.float32).cpu()  # [d_model, d_head]

        # u_i^{(l,k)} = O_k @ V_{l,k,i}  ->  u = V_{l,k} @ O_k.T
        u = v_lk @ o_k.T                                  # [n_tokens, d_model]
        u_per_head[(l, k)] = u

        # delta_lk = O_k @ (sum_i alpha_lk[i] V_i) = alpha_lk @ u
        delta_lk = alpha_lk @ u                           # [d_model]
        delta_tl = delta_tl + delta_lk

    return TLExtraction(
        alpha_per_head=alpha_per_head,
        u_per_head=u_per_head,
        delta_tl=delta_tl,
        answer_position=dims["answer_position"],
        n_tokens=dims["n_tokens"],
        answer_logits=dims.get("answer_logits"),
        v_raw=dims.get("v_raw"),
        resid_at_answer=dims.get("resid_at_answer"),
    )


def extract_answer_attention(
    model,
    input_ids: Tensor,
    tl_heads: TLHeadSet,
    answer_position: Optional[int] = None,
    attention_mask: Optional[Tensor] = None,
) -> dict[tuple[int, int], Tensor]:
    """{(layer, head) -> alpha[n_tokens]} at the answer position. Nothing else.

    This is the kernel kappa on its own. diagnostic_forward returns it too, but
    also materializes u_per_head = [n_tokens, d_model] per head, which is ~32 MB
    per head at 3300 tokens and pure waste when only the attention row is
    wanted (the position-vs-content test, attention-mass diagnostics, and any
    kappa-only analysis). Same capture path, so the alpha values are identical
    to diagnostic_forward's by construction -- not a reimplementation.
    """
    alpha_cache, _v_cache, _dims = _run_capture_forward(
        model, input_ids, tl_heads, answer_position, attention_mask
    )
    return {(l, k): alpha_cache[l][k] for (l, k) in sorted(tl_heads.heads)}


def precompute_head_projections(
    model,
    tl_heads: TLHeadSet,
    target_token_ids=None,
    target_directions: Optional[Tensor] = None,
    device="cpu",
) -> dict[tuple[int, int], Tensor]:
    """proj[(l,k)] = O_k^T @ D^T, shape [d_head, n_targets].

    This quantity is PROMPT-INDEPENDENT -- it is a pure function of the model
    weights and the readout directions -- yet the per-prompt loop below used to
    recompute it for every head of every prompt. At 1024 heads it is
    [128, 4096] @ [4096, n_targets] per head, i.e. ~99 % of the arithmetic in
    an all-heads sweep, repeated once per prompt for nothing (a 250-prompt run
    did ~13 TFLOP of it, all on CPU -- which is why SLURM reported 0 % GPU
    utilization for job 822524).

    Hoist it out with this, pass the result as head_proj=, and the per-prompt
    work drops to two small contractions per head. Numerically identical: the
    same deterministic product, computed once instead of Q times.
    """
    if (target_token_ids is None) == (target_directions is None):
        raise ValueError("pass exactly one of target_token_ids / target_directions")
    d_model = model.config.hidden_size
    d_head = head_dim(model)   # not hidden_size // n_heads; see _run_capture_forward
    if target_directions is not None:
        dirs = target_directions.detach().to(torch.float32).to(device)
        if dirs.dim() != 2 or dirs.shape[1] != d_model:
            raise ValueError(
                f"target_directions must be [n_targets, {d_model}]; "
                f"got {tuple(dirs.shape)}")
    else:
        w_u = model.lm_head.weight.detach()
        dirs = w_u[torch.as_tensor(list(target_token_ids), dtype=torch.long)]
        dirs = dirs.to(torch.float32).to(device)
    dirs_t = dirs.T.contiguous()                              # [d_model, n_targets]
    proj: dict[tuple[int, int], Tensor] = {}
    for (l, k) in sorted(tl_heads.heads):
        o_weight = model.model.layers[l].self_attn.o_proj.weight.detach()
        o_k = o_weight[:, k * d_head:(k + 1) * d_head].to(torch.float32).to(device)
        proj[(l, k)] = o_k.T @ dirs_t                         # [d_head, n_targets]
    return proj


def extract_per_head_logit_contributions(
    model,
    input_ids: Tensor,
    tl_heads: TLHeadSet,
    target_token_ids=None,
    answer_position: Optional[int] = None,
    attention_mask: Optional[Tensor] = None,
    inject: Optional[tuple] = None,
    target_directions: Optional[Tensor] = None,
    head_proj: Optional[dict] = None,
    compute_device=None,
    row_mode: Optional[str] = None,
) -> dict[tuple[int, int], Tensor]:
    """Memory-light: per-head contribution to each target-token logit,
    WITHOUT materializing u_per_head.

    For head (l, k):
        a_{N,k}^{(l)} = O_k @ (sum_i alpha[i] V_i) = O_k @ weighted_v
        <a_{N,k}^{(l)}, W_U[c]> = weighted_v @ (O_k^T @ W_U[c])

    Memory: O(|heads| * n_targets + n_layers * n_tokens * d_head) instead
    of O(|heads| * n_tokens * d_model). For an all-heads sweep (1024 heads),
    n_tokens=6000, n_targets=1: ~1.5 GB CPU (v_cache, bf16) instead of ~100 GB.

    Args:
        target_token_ids: sequence of token IDs to project onto (e.g. the
                          single true-class token for TL-score, or the full
                          label vocabulary).
        target_directions: [n_targets, d_model] tensor of ARBITRARY readout
                          directions, used INSTEAD of W_U[target_token_ids].
                          This is what the normalized-DLA convention needs:
                          pass gamma * (W_U[t1] - W_U[t2]) (final-LN gamma
                          folded in) and divide the result by the per-prompt
                          rms to get contributions in real logit units that
                          sum exactly to the model's margin. Exactly one of
                          target_token_ids / target_directions is required.
        head_proj:        output of precompute_head_projections. When given,
                          O_k^T @ D^T is reused instead of being rebuilt for
                          every head of every prompt, and target_token_ids /
                          target_directions are not needed (the directions are
                          already baked in). Numerically identical, and the
                          only way an all-heads sweep is not CPU-bound.
        compute_device:   device for the per-head contractions in the fast
                          path (e.g. "cuda"). Default None = CPU, i.e. the
                          historical behaviour.

    Returns:
        dict (l, k) -> Tensor[n_targets], where
            result[(l,k)][j] = <a_{N,k}^{(l)}, direction_j>
    """
    if head_proj is None and (target_token_ids is None) == (target_directions is None):
        raise ValueError(
            "pass exactly one of target_token_ids / target_directions"
        )
    alpha_cache, v_cache, dims = _run_capture_forward(
        model, input_ids, tl_heads, answer_position, attention_mask, inject=inject,
        row_mode=row_mode,
    )
    d_head = dims["d_head"]

    # ---- fast path: projections hoisted out of the prompt loop --------------
    # Same arithmetic, but O_k^T @ D^T is taken from head_proj instead of being
    # rebuilt per prompt, and the two remaining contractions are batched per
    # layer on compute_device. See precompute_head_projections for why.
    if head_proj is not None:
        dev = compute_device or "cpu"
        by_layer: dict[int, list[int]] = {}
        for (l, k) in sorted(tl_heads.heads):
            by_layer.setdefault(l, []).append(k)
        out_fast: dict[tuple[int, int], Tensor] = {}
        for l, ks in by_layer.items():
            idx = torch.as_tensor(ks, dtype=torch.long)
            a = alpha_cache[l][idx].to(dev, torch.float32)      # [h, T]
            v = v_cache[l][idx].to(dev, torch.float32)          # [h, T, d_head]
            wv = torch.einsum("ht,htd->hd", a, v)               # [h, d_head]
            pj = torch.stack([head_proj[(l, k)].to(dev, torch.float32)
                              for k in ks])                     # [h, d_head, n_t]
            res = torch.einsum("hd,hdn->hn", wv, pj)            # [h, n_targets]
            res = res.cpu()
            for j, k in enumerate(ks):
                out_fast[(l, k)] = res[j]
        return out_fast

    # Readout directions: [n_targets, d_model]
    if target_directions is not None:
        w_u_targets = target_directions.detach().to(torch.float32).cpu()
        if w_u_targets.dim() != 2 or w_u_targets.shape[1] != dims["d_model"]:
            raise ValueError(
                f"target_directions must be [n_targets, {dims['d_model']}]; "
                f"got {tuple(w_u_targets.shape)}"
            )
    else:
        w_u = model.lm_head.weight.detach()
        w_u_targets = w_u[torch.as_tensor(list(target_token_ids), dtype=torch.long)]
        w_u_targets = w_u_targets.to(torch.float32).cpu()  # [n_targets, d_model]

    out: dict[tuple[int, int], Tensor] = {}
    for (l, k) in sorted(tl_heads.heads):
        alpha_lk = alpha_cache[l][k]                       # [n_tokens] fp32 CPU
        v_lk = v_cache[l][k].to(torch.float32)             # [n_tokens, d_head]
        o_weight = model.model.layers[l].self_attn.o_proj.weight.detach()
        o_k = o_weight[:, k * d_head:(k + 1) * d_head].to(torch.float32).cpu()  # [d_model, d_head]

        weighted_v = alpha_lk @ v_lk                       # [d_head]
        # proj[:, j] = O_k^T @ W_U[target_j]  -> [d_head, n_targets]
        proj = o_k.T @ w_u_targets.T                       # [d_head, n_targets]
        out[(l, k)] = weighted_v @ proj                    # [n_targets]
    return out


# ---------------------------------------------------------------------------
# Sanity (no real model needed; can run with a tiny mock model if extended)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # We can't run diagnostic_forward without a real model since it monkey-
    # patches HF Llama layers. The lightweight sanity is to confirm the
    # imports work and the context manager round-trips cleanly.
    class _MockAttn:
        def __init__(self):
            self.forward = lambda *a, **kw: ("output", "weights", "kv")

    class _MockLayer:
        def __init__(self):
            self.self_attn = _MockAttn()

    class _MockModelModel:
        def __init__(self, n):
            self.layers = [_MockLayer() for _ in range(n)]

    class _MockModel:
        def __init__(self, n):
            self.model = _MockModelModel(n)

    m = _MockModel(8)
    layers = {2, 5}

    with _patch_tl_attention_capture(m, layers, answer_position=0) as cache:
        # Call the patched forward to verify it routes correctly
        original_id = id(m.model.layers[2].self_attn.forward)
        # Note: in real use the model's forward would call these.
        # Here we just verify the patch installed and the dict is empty.
        assert isinstance(cache, dict)
        assert cache == {}
    # After exit, originals restored:
    print("context-manager install / restore: OK")
