"""Kernel-retrieval measurement and identity verification.

Implements §3-§4 of `method_kernel_retrieval.md`. The core claim is
Theorem 1:

    <Delta_TL, W_U^c>  =  sum_i  kappa(x_q, x_i) * phi_i^c(x_q)

where for each TL head (l, k) in T:
    kappa^{(l,k)}(x_q, x_i) := alpha_{N, i}^{(l,k)}                    [attention row]
    u_i^{(l,k)}             := W_O^{(l,k)} W_V^{(l,k)} h_i^{(l-1)}     [OV-projected hidden]
    phi_i^{c,(l,k)}         := <u_i^{(l,k)}, W_U^c>                    [per-head vote]
and the aggregate kappa / phi sum / weight-average over heads in T.

The identity is a structural consequence of the residual-stream
decomposition (Elhage 2021); the empirical content is Corollary 1 (UUID
identification) and the kernel-structure characterization (§6).

Module shape:
    extract_from_model(model, input_ids, tl_heads)  -- forward + hooks
        -> TLExtraction
    compute_kappa_phi(extraction, label_token_ids, w_u, position_mask=None)
        -> AggregateKernel{kappa, phi, phi_per_head}
    predicted_logit_contribution(extraction, label_token_ids, w_u, ...)
        -> Tensor[n_classes]     (Theorem 1 RHS)
    actual_logit_contribution(extraction, label_token_ids, w_u, ...)
        -> Tensor[n_classes]     (Theorem 1 LHS, from extracted Delta_TL)
    verify_identity(model, ...)
        -> dict with predicted, actual, agreement metrics

The "model" version requires `transformers` with Llama attention in eager
mode (`attn_implementation='eager'`). For pure-math testing without a model,
build a TLExtraction by hand and pass it directly to compute_kappa_phi /
predicted_logit_contribution; see __main__ smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TLExtraction:
    """One prompt's forward-pass extraction at TL heads.

    Shapes are documented per attribute. n_tokens is the prompt length;
    positions are indices into [0, n_tokens). The answer position is the
    one whose next-token prediction is the label (typically n_tokens - 1
    for an lm-eval-style prompt that ends right before the label).
    """
    alpha_per_head: dict[tuple[int, int], Tensor]
    """(l, k) -> Tensor[n_tokens]. alpha_{N, i}^{(l, k)} for i in [0, n_tokens).
    The attention WEIGHT at the answer position to each key position i.
    Causally masked positions (i > answer_position) should be zero."""

    u_per_head: dict[tuple[int, int], Tensor]
    """(l, k) -> Tensor[n_tokens, d_model]. u_i^{(l, k)} = W_O^{(l,k)} W_V^{(l,k)} h_i^{(l-1)}
    for each token position i. OV-projected hidden states at TL heads.
    Stored on CPU to avoid GPU memory pressure across many prompts."""

    delta_tl: Tensor
    """Tensor[d_model]. Sum over TL heads of head output at the answer
    position: sum_{(l,k) in T} a_{N,k}^{(l)}. By construction equals
    sum_{(l,k)} sum_i alpha_per_head[(l,k)][i] * u_per_head[(l,k)][i] -- this
    is Theorem 1's LHS in residual-stream form."""

    answer_position: int
    n_tokens: int

    answer_logits: "Tensor | None" = None
    """Tensor[vocab] fp32 CPU, or None. The model's OWN logits at the answer
    position, from the same forward that produced delta_tl. Optional with a
    default so existing constructors are unaffected. It exists because any
    comparison of the direct-write layer against the output layer must read
    both from ONE pass: a second forward differs by bf16 nondeterminism on the
    scale of the effects such comparisons measure."""

    v_raw: "dict[int, Tensor] | None" = None
    """layer -> Tensor[n_tokens, n_kv_heads * d_head] fp CPU, or None. The RAW
    v_proj output, before the head split. Donor values for a value-stream patch
    are sliced straight out of this and handed back as patch_v, so the patch is
    written in exactly the layout the hook intercepts -- no reshape in between
    to get wrong. Optional with a default so existing constructors are
    unaffected."""

    resid_at_answer: "Tensor | None" = None
    """Tensor[n_layers+1, d_model] fp32 CPU, or None. The residual stream at the
    answer position: slot 0 is the embedding output, slot l+1 is the output of
    decoder layer l. Projecting each slot on a readout direction turns a single
    end-of-network number into a function of depth, which is what separates
    "downstream absorbed the write" from "downstream merely rescaled it".
    The last slot is what the final RMSNorm sees, so
    <last, gamma * dW_U> / rms(last) must reproduce the measured logit margin --
    an exact identity, and the gate on this capture."""

    def head_set(self) -> set[tuple[int, int]]:
        return set(self.alpha_per_head.keys())

    def d_model(self) -> int:
        return self.delta_tl.shape[0]


@dataclass
class AggregateKernel:
    """Output of compute_kappa_phi (Definition 3.1)."""
    kappa: Tensor                                       # [n_positions]
    phi: Tensor                                         # [n_positions, n_classes]
    phi_per_head: dict[tuple[int, int], Tensor]          # (l,k) -> [n_positions, n_classes]
    kappa_per_head: dict[tuple[int, int], Tensor]        # (l,k) -> [n_positions]
    positions: Tensor                                   # [n_positions], indices into the original sequence


# ---------------------------------------------------------------------------
# Aggregation (Definition 3.1)
# ---------------------------------------------------------------------------

def compute_kappa_phi(
    extraction: TLExtraction,
    label_token_ids: Iterable[int],
    w_u: Tensor,
    position_mask: Optional[Tensor] = None,
    eps: float = 1e-12,
) -> AggregateKernel:
    """Compute the aggregate kernel kappa and per-class votes phi.

    Args:
        extraction: output of extract_from_model (or manual construction).
        label_token_ids: token IDs of the label vocabulary Y. Order defines
                         the class index in `phi` and downstream logits.
        w_u: unembedding matrix of shape [vocab_size, d_model] OR
             [|Y|, d_model] if already restricted to label tokens.
        position_mask: optional boolean Tensor[n_tokens] selecting which
                       positions to include in the kernel-retrieval sum.
                       If None, all positions are included.
        eps: floor for kappa to avoid 0/0 in the aggregate vote.

    Returns:
        AggregateKernel.
    """
    label_token_ids = list(label_token_ids)
    n_classes = len(label_token_ids)
    n_tokens = extraction.n_tokens

    # Slice W_U down to label rows: [n_classes, d_model]
    if w_u.shape[0] == n_classes:
        w_u_labels = w_u
    else:
        w_u_labels = w_u[torch.as_tensor(label_token_ids, dtype=torch.long)]
    w_u_labels = w_u_labels.to(extraction.delta_tl.dtype).cpu()

    # Position mask: which keys i to include
    if position_mask is None:
        positions = torch.arange(n_tokens, dtype=torch.long)
    else:
        if position_mask.shape != (n_tokens,):
            raise ValueError(
                f"position_mask shape {position_mask.shape} != ({n_tokens},)"
            )
        positions = torch.nonzero(position_mask, as_tuple=False).squeeze(-1)

    n_positions = positions.shape[0]
    if n_positions == 0:
        raise ValueError("position_mask selects zero positions")

    # Per-head kernel and vote, restricted to chosen positions
    kappa_per_head: dict[tuple[int, int], Tensor] = {}
    phi_per_head: dict[tuple[int, int], Tensor] = {}
    for lk, alpha in extraction.alpha_per_head.items():
        kappa_per_head[lk] = alpha[positions]                       # [n_positions]
        u = extraction.u_per_head[lk][positions]                    # [n_positions, d_model]
        phi_per_head[lk] = u.to(w_u_labels.dtype) @ w_u_labels.T    # [n_positions, n_classes]

    # Aggregate kernel: sum across heads
    kappa = torch.zeros(n_positions, dtype=w_u_labels.dtype)
    for v in kappa_per_head.values():
        kappa = kappa + v.to(kappa.dtype)

    # Aggregate vote: kernel-weighted average across heads.
    #   phi_i^c = (1 / kappa_i) * sum_{(l,k)} kappa^{(l,k)}_i * phi^{(l,k),c}_i
    # When kappa_i is below eps we set phi to zero (no contribution anyway,
    # since kappa appears as a multiplier in predicted_logit_contribution).
    weighted = torch.zeros(n_positions, n_classes, dtype=w_u_labels.dtype)
    for lk in extraction.alpha_per_head.keys():
        weighted = weighted + kappa_per_head[lk].to(weighted.dtype).unsqueeze(-1) * phi_per_head[lk]
    safe_kappa = kappa.clamp(min=eps)
    phi = weighted / safe_kappa.unsqueeze(-1)
    # Where kappa is effectively zero, zero out phi (avoid spurious large values).
    phi = torch.where(kappa.unsqueeze(-1) > eps, phi, torch.zeros_like(phi))

    return AggregateKernel(
        kappa=kappa,
        phi=phi,
        phi_per_head=phi_per_head,
        kappa_per_head=kappa_per_head,
        positions=positions,
    )


# ---------------------------------------------------------------------------
# Theorem 1 LHS / RHS
# ---------------------------------------------------------------------------

def predicted_logit_contribution(agg: AggregateKernel) -> Tensor:
    """Theorem 1 RHS: sum_i kappa(x_q, x_i) * phi_i^c -> Tensor[n_classes].

    Equivalent forms (used internally):
        = sum_i kappa_i * phi_i^c
        = sum_{(l,k)} sum_i kappa^{(l,k)}_i * phi^{(l,k),c}_i
    """
    # Use the per-head form because it bypasses the kappa-divide-and-reweight
    # round-trip in compute_kappa_phi, giving cleaner fp behaviour.
    out: Tensor | None = None
    for lk, kappa_lk in agg.kappa_per_head.items():
        contrib = (kappa_lk.unsqueeze(-1) * agg.phi_per_head[lk]).sum(dim=0)
        out = contrib if out is None else out + contrib
    assert out is not None, "no TL heads in aggregate"
    return out


def actual_logit_contribution(
    extraction: TLExtraction,
    label_token_ids: Iterable[int],
    w_u: Tensor,
) -> Tensor:
    """Theorem 1 LHS: <Delta_TL, W_U^c> for c in label_token_ids.

    Uses the extracted Delta_TL directly. Returns Tensor[n_classes].
    """
    label_token_ids = list(label_token_ids)
    n_classes = len(label_token_ids)
    if w_u.shape[0] == n_classes:
        w_u_labels = w_u
    else:
        w_u_labels = w_u[torch.as_tensor(label_token_ids, dtype=torch.long)]
    w_u_labels = w_u_labels.to(extraction.delta_tl.dtype).cpu()
    return extraction.delta_tl.cpu() @ w_u_labels.T


# ---------------------------------------------------------------------------
# Identity verification (§4.1)
# ---------------------------------------------------------------------------

@dataclass
class IdentityReport:
    predicted: Tensor          # [n_classes]
    actual: Tensor             # [n_classes]
    max_abs_diff: float
    rel_diff: float            # max_abs_diff / max(|actual|)
    inter_class_range: float   # max(actual) - min(actual)
    strict_pass: bool          # max_abs_diff < strict_tol * inter_class_range
    approx_pass: bool          # max_abs_diff < approx_tol * inter_class_range
    n_positions_used: int


def verify_identity(
    extraction: TLExtraction,
    label_token_ids: Iterable[int],
    w_u: Tensor,
    position_mask: Optional[Tensor] = None,
    strict_tol: float = 0.01,
    approx_tol: float = 0.05,
) -> IdentityReport:
    """End-to-end verification of Theorem 1 for one prompt.

    When `position_mask` is None (all positions used), the identity must
    hold within floating-point noise (it is a structural identity from
    bilinearity). Non-trivial discrepancies indicate hook / extraction
    bugs.

    When `position_mask` restricts to a subset (e.g. demo positions only),
    the comparison is between the demo-restricted RHS and the FULL LHS
    (extraction.delta_tl summed over all keys) -- so the gap is meaningful:
    it measures how much TL contribution comes from outside the masked
    positions.

    Args:
        extraction: per-prompt extraction.
        label_token_ids: token IDs of Y.
        w_u: unembedding matrix.
        position_mask: optional bool Tensor[n_tokens].
        strict_tol: |diff| / inter-class range threshold for strict success
                    per §A.4 result-roadmap pre-registration (0.01).
        approx_tol: looser threshold for approximate success (0.05).
    """
    agg = compute_kappa_phi(extraction, label_token_ids, w_u, position_mask=position_mask)
    predicted = predicted_logit_contribution(agg)
    actual = actual_logit_contribution(extraction, label_token_ids, w_u)

    diff = (predicted - actual).abs()
    max_abs = float(diff.max().item())
    inter_class_range = float((actual.max() - actual.min()).abs().item())
    rel = max_abs / max(1e-12, float(actual.abs().max().item()))

    return IdentityReport(
        predicted=predicted,
        actual=actual,
        max_abs_diff=max_abs,
        rel_diff=rel,
        inter_class_range=inter_class_range,
        strict_pass=max_abs < strict_tol * max(1e-12, inter_class_range),
        approx_pass=max_abs < approx_tol * max(1e-12, inter_class_range),
        n_positions_used=int(agg.positions.shape[0]),
    )


# ---------------------------------------------------------------------------
# Model integration (requires transformers + a Llama-family model in eager mode)
# ---------------------------------------------------------------------------

def extract_from_model(
    model,
    input_ids: Tensor,
    tl_heads,
    answer_position: Optional[int] = None,
    attention_mask: Optional[Tensor] = None,
    keep_on_cpu: bool = True,
) -> TLExtraction:
    """Eager forward + hooks. Capture per-TL-head alpha (at answer position
    to all keys) and u (= O_k V_k h_i, at every key position).

    Requires:
        - `transformers` installed.
        - model loaded with `attn_implementation='eager'` so that
          `output_attentions=True` returns a true per-head attention tensor.
        - Llama-family architecture: each layer has self_attn.v_proj and
          self_attn.o_proj with standard shapes.
        - GQA (Llama-3-8B, Qwen, ...) is handled by repeat_interleave on
          num_key_value_heads -> num_attention_heads.

    Args:
        model: a HF causal LM.
        input_ids: LongTensor [1, n_tokens] (single prompt only).
        tl_heads: TLHeadSet from tools.tl_heads.
        answer_position: which position is the query/answer. Default:
                         input_ids.shape[1] - 1 (last token of the prompt).
        attention_mask: optional [1, n_tokens] mask. Default: all ones.
        keep_on_cpu: move captured tensors to CPU after forward (default
                     True, recommended for long prompts).

    Returns:
        TLExtraction.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"input_ids must have shape [1, n_tokens]; got {tuple(input_ids.shape)}"
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
    if d_model % n_attn_heads != 0:
        raise ValueError(
            f"hidden_size {d_model} not divisible by num_attention_heads {n_attn_heads}"
        )
    d_head = d_model // n_attn_heads

    layers_needed = tl_heads.layers()
    v_cache: dict[int, Tensor] = {}     # layer -> [n_kv_heads, n_tokens, d_head]

    def make_v_hook(layer_idx: int):
        def hook(module, _input, output):
            # v_proj output: [B=1, n_tokens, n_kv_heads * d_head]
            out = output if isinstance(output, torch.Tensor) else output[0]
            v = out[0]                                          # [n_tokens, n_kv_heads * d_head]
            v = v.view(n_tokens, n_kv_heads, d_head)            # [n_tokens, n_kv_heads, d_head]
            v = v.transpose(0, 1).contiguous()                   # [n_kv_heads, n_tokens, d_head]
            if n_kv_heads != n_attn_heads:
                # GQA: expand kv heads to attention heads
                v = v.repeat_interleave(n_attn_heads // n_kv_heads, dim=0)
            v_cache[layer_idx] = v.detach().cpu() if keep_on_cpu else v.detach()
        return hook

    handles = []
    for l in layers_needed:
        h = model.model.layers[l].self_attn.v_proj.register_forward_hook(make_v_hook(l))
        handles.append(h)

    try:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        attentions = out.attentions  # tuple of len L, each [1, n_heads, n_tokens, n_tokens]
    finally:
        for h in handles:
            h.remove()

    alpha_per_head: dict[tuple[int, int], Tensor] = {}
    u_per_head: dict[tuple[int, int], Tensor] = {}
    delta_tl = torch.zeros(d_model, dtype=torch.float32)

    for (l, k) in sorted(tl_heads.heads):
        attn_l = attentions[l]
        if attn_l is None:
            raise RuntimeError(
                f"Layer {l}: model.config.attn_implementation must be 'eager' for "
                f"output_attentions=True to return real per-head weights."
            )
        # alpha_{N, :, k} -- row N of head k, full set of keys.
        alpha_full = attn_l[0, k, answer_position, :].detach().to(torch.float32).cpu()  # [n_tokens]
        alpha_per_head[(l, k)] = alpha_full

        # V_{l, k} at all key positions: [n_tokens, d_head]
        v_lk = v_cache[l][k].to(torch.float32)  # [n_tokens, d_head]
        # O_k = o_proj.weight[:, k*d_head:(k+1)*d_head] -- [d_model, d_head]
        o_weight = model.model.layers[l].self_attn.o_proj.weight.detach()
        o_k = o_weight[:, k * d_head:(k + 1) * d_head].to(torch.float32).cpu()  # [d_model, d_head]

        # u_i^{(l,k)} = O_k @ V_{l,k,i}  =>  u = V_{l,k} @ O_k^T  (batched over i)
        u = v_lk @ o_k.T                                       # [n_tokens, d_model]
        u_per_head[(l, k)] = u

        # Per-head TL contribution at answer position:
        # a_{N, k}^{(l)} = O_k @ (sum_i alpha_full[i] * V_{l,k,i})
        weighted_v = alpha_full @ v_lk                          # [d_head]
        a_lk_N = o_k @ weighted_v                               # [d_model]
        delta_tl = delta_tl + a_lk_N

    return TLExtraction(
        alpha_per_head=alpha_per_head,
        u_per_head=u_per_head,
        delta_tl=delta_tl,
        answer_position=answer_position,
        n_tokens=n_tokens,
    )


def verify_identity_on_model(
    model,
    input_ids: Tensor,
    tl_heads,
    label_token_ids: Iterable[int],
    answer_position: Optional[int] = None,
    position_mask: Optional[Tensor] = None,
    strict_tol: float = 0.01,
    approx_tol: float = 0.05,
) -> IdentityReport:
    """One-shot: extract + verify. Convenience wrapper around extract_from_model
    and verify_identity."""
    extraction = extract_from_model(
        model, input_ids, tl_heads, answer_position=answer_position
    )
    w_u = model.lm_head.weight.detach()  # [vocab, d_model]
    return verify_identity(
        extraction, label_token_ids, w_u,
        position_mask=position_mask,
        strict_tol=strict_tol, approx_tol=approx_tol,
    )


# ---------------------------------------------------------------------------
# Smoke test (no transformers required)
# ---------------------------------------------------------------------------

def _build_mock_extraction(
    n_heads: int = 3,
    n_tokens: int = 12,
    d_model: int = 16,
    answer_position: Optional[int] = None,
    seed: int = 0,
) -> tuple[TLExtraction, Tensor, list[int]]:
    """Build a self-consistent TLExtraction + W_U + label_token_ids for
    smoke testing the math. By construction Theorem 1 holds exactly.
    """
    torch.manual_seed(seed)
    if answer_position is None:
        answer_position = n_tokens - 1

    head_ids = [(7 + i, 11 + 2 * i) for i in range(n_heads)]  # arbitrary (l, k)

    alpha_per_head: dict[tuple[int, int], Tensor] = {}
    u_per_head: dict[tuple[int, int], Tensor] = {}
    delta_tl = torch.zeros(d_model, dtype=torch.float32)

    for lk in head_ids:
        # Random attention row over keys; causally zero entries past answer pos.
        raw = torch.rand(n_tokens, dtype=torch.float32)
        raw[answer_position + 1:] = 0.0
        alpha = raw / raw.sum().clamp(min=1e-12)
        u = torch.randn(n_tokens, d_model, dtype=torch.float32)

        alpha_per_head[lk] = alpha
        u_per_head[lk] = u
        # By construction Delta_TL = sum_{(l,k)} sum_i alpha[i] * u[i]
        delta_tl = delta_tl + alpha @ u    # [n_tokens] @ [n_tokens, d_model] -> [d_model]

    extraction = TLExtraction(
        alpha_per_head=alpha_per_head,
        u_per_head=u_per_head,
        delta_tl=delta_tl,
        answer_position=answer_position,
        n_tokens=n_tokens,
    )

    # Build a fake unembedding with 5 label tokens at known positions.
    vocab_size = 50
    w_u = torch.randn(vocab_size, d_model, dtype=torch.float32)
    label_token_ids = [3, 11, 17, 22, 41]

    return extraction, w_u, label_token_ids


if __name__ == "__main__":
    # 1) Math identity: by construction, predicted == actual exactly (fp noise).
    extraction, w_u, label_ids = _build_mock_extraction()
    report = verify_identity(extraction, label_ids, w_u)
    print(f"[mock-full] predicted - actual max-abs-diff:  {report.max_abs_diff:.3e}")
    print(f"[mock-full] inter-class range:                {report.inter_class_range:.3e}")
    print(f"[mock-full] strict_pass: {report.strict_pass}, approx_pass: {report.approx_pass}")
    print(f"[mock-full] n_positions_used: {report.n_positions_used}")
    assert report.strict_pass, "by construction the full-position identity must be strict"

    # 2) Position-masked: restrict to even positions only.
    mask = torch.zeros(extraction.n_tokens, dtype=torch.bool)
    mask[::2] = True
    report_demo = verify_identity(extraction, label_ids, w_u, position_mask=mask)
    print(f"\n[mock-restr] positions used: {report_demo.n_positions_used} of {extraction.n_tokens}")
    print(f"[mock-restr] gap (full LHS - restricted RHS) max-abs-diff: {report_demo.max_abs_diff:.3e}")
    # The gap measures the TL contribution from MASKED-OUT positions.
    # Since we're using mock data with attention on all positions, the gap is non-trivial.
    assert not report_demo.strict_pass, "with mask, restricted RHS misses non-mask TL contribution"

    # 3) Inspect aggregate kernel.
    agg = compute_kappa_phi(extraction, label_ids, w_u)
    print(f"\n[mock] kappa shape: {tuple(agg.kappa.shape)},  "
          f"sum = {agg.kappa.sum().item():.4f}  (== n_heads since each row sums to 1)")
    print(f"[mock] phi shape:   {tuple(agg.phi.shape)}")
    n_heads_expected = len(extraction.head_set())
    assert abs(agg.kappa.sum().item() - n_heads_expected) < 1e-4, \
        "kappa should sum to n_heads (each per-head row is a softmax)"

    # 4) Sanity-check predicted_logit_contribution per-head equivalence.
    predicted = predicted_logit_contribution(agg)
    actual = actual_logit_contribution(extraction, label_ids, w_u)
    print(f"\n[mock] predicted (first 5 classes): {predicted.tolist()}")
    print(f"[mock] actual    (first 5 classes): {actual.tolist()}")
    print(f"[mock] |diff|_max:                  {(predicted - actual).abs().max().item():.3e}")

    print("\nAll math sanity checks passed.")
