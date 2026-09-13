"""Vector-baseline arms for the receivers: Function Vectors and Task Vectors.

The receivers (run_k0_receiver, run_k10_increment) host TSLA as families of
answer-row injections on ARM_BASE's mask. FV and TV are the same kind of
arm -- offline information carried by one vector, the increment columns
closed to every head -- with their own operator and layer:

    FV-K<full> a=<alpha>      alpha * v_FV added at the answer row of the
                              sidecar's layer (run_fv_increment: L // 3);
                              alpha = 0 is the bitwise gate, 1 the main arm
    TV-K<full> L=<layer> a=1  the answer-row hidden state at decoder layer
                              L REPLACED by theta_L (run_tv_increment);
                              `a=1` only so the summaries' "alpha = 1"
                              filter lists it -- there is no strength.
                              (TV-K<full> L=<layer> a=0 = no replacement,
                              the gate.)

One grammar, one loader per sidecar, one hook each. The receivers ask
`parse_vector_arm` and treat every hit like a TSLA arm for masks and gates.
Specs: tools/baselines/specs/ (the adapted FV / TV / ICV / I2CL protocols).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ARM_RE = re.compile(r"^(?P<method>FV|TV|ICV|I2CL)-(?P<family>\S+?)(?: L=(?P<layer>\d+))? a=(?P<alpha>[-+0-9.eE]+)$")

#     I2CL-K<full> a=1          Li et al.'s implicit in-context learning: on
#                               every decoder layer's self_attn and mlp
#                               outputs, every position, out <- beta * out +
#                               lambda * cv with 4 L calibrated scalars
#                               (run_i2cl_increment); a=0 is the gate (no
#                               hook). Its prefix cache is rebuilt with the
#                               hooks, once per seed.
I2CL_MODULES = ("attn", "mlp")

#     ICV-K<full> a=<lambda>    Liu et al.'s in-context vector: the upstream
#                               ICVLayer on every decoder layer's MLP output,
#                               every position, strength lambda (their --lam);
#                               a=0 is the bitwise gate (no hook). The prefix
#                               cache is rebuilt per lambda WITH the hooks
#                               (run_icv_increment, the receivers' icv caches).
ICV_LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)   # the spec's grid, plus the gate


def parse_vector_arm(arm):
    """(method, family, layer or None, alpha) for an FV / TV / ICV arm, else None."""
    m = ARM_RE.match(str(arm))
    if not m:
        return None
    layer = int(m.group("layer")) if m.group("layer") is not None else None
    return m.group("method"), m.group("family"), layer, float(m.group("alpha"))


def vector_family_label(arm):
    """The curve an arm belongs to: `FV-K10`, `TV-K10 L=14`; None if not one.
    Everything in the name but the alpha, so one label = one vector."""
    p = parse_vector_arm(arm)
    if p is None:
        return None
    method, family, layer, _alpha = p
    return f"{method}-{family}" + (f" L={layer}" if layer is not None else "")


def fv_arm(alpha, k_full):
    return f"FV-K{int(k_full)} a={float(alpha):g}"


def fv_arms(k_full, alphas):
    return [fv_arm(a, k_full) for a in alphas]


def tv_family(k_full, suffix=""):
    """`K10` for the main (single dummy query) family, `K10m5` for the
    descriptive mean-of-five family (run_tv_increment)."""
    return f"K{int(k_full)}{suffix}"


def tv_arm(layer, k_full, on=True, suffix=""):
    return f"TV-{tv_family(k_full, suffix)} L={int(layer)} a={1 if on else 0}"


def tv_arms(layers, k_full, families=("",)):
    """Every (family, layer) at a=1, plus ONE gate arm `a=0` on the main
    family's first layer: a replacement that is not installed is the same
    forward whichever layer it names, so one gate covers them all."""
    out = [tv_arm(l, k_full, suffix=sfx) for sfx in families for l in layers]
    if layers:
        out.append(tv_arm(layers[0], k_full, on=False))
    return out


CANDIDATE_FRACTIONS = (8 / 32, 11 / 32, 14 / 32, 17 / 32, 20 / 32)


def tv_candidate_layers(n_layers):
    """The spec's {8, 11, 14, 17, 20} on 32 layers -- at most five
    configurations, the middle of the stack -- and the same fractions of the
    depth on other models (28: 7 10 12 15 18; 36: 9 12 16 19 22), duplicates
    dropped. 0-based decoder layers whose OUTPUT is replaced."""
    out = []
    for f in CANDIDATE_FRACTIONS:
        l = int(round(f * int(n_layers)))
        if l not in out:
            out.append(l)
    return out


def parse_layer_list(spec):
    """'8,11,14' -> [8, 11, 14]; '' -> []."""
    return [int(x) for x in str(spec or "").split(",") if x.strip()]


def parse_float_list(spec):
    """'0.05,0.1' -> [0.05, 0.1]; '' -> []."""
    return [float(x) for x in str(spec or "").split(",") if x.strip()]


def icv_arm(lam, k_full):
    return f"ICV-K{int(k_full)} a={float(lam):g}"


def icv_arms(k_full, lams):
    return [icv_arm(a, k_full) for a in lams]


def load_icv_sidecar(path, seeds, k_base=None, k_full=None):
    """(lambda grid, {seed: V float32 [n_layers, d]}) from run_icv_increment's
    sidecar; V[l] is the direction's segment for decoder layer l (the
    embedding row already dropped, as the upstream's `icv[1:]`)."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--icv-vectors {path}: not a file (run_icv_increment writes it)")
    doc = json.loads(p.read_text(encoding="utf-8"))
    runner = doc.get("runner") or {}
    vecs = runner.get("icv_vectors")
    if not isinstance(vecs, dict) or not vecs:
        raise SystemExit(f"{path}: no 'icv_vectors' under 'runner' (keys there: {sorted(runner)})")
    for want, field in ((k_full, "K"), (k_base, "K_base")):
        if want is None:
            continue
        got = runner.get(field)
        if got is None or int(got) != int(want):
            raise SystemExit(f"{path}: runner.{field} = {got!r}, this run is at {field} = {int(want)}; "
                             "the in-context vector is built per (K_base, K_full) level")
    grid = [float(x) for x in (runner.get("lambda_grid") or [])]
    if not grid:
        raise SystemExit(f"{path}: no 'lambda_grid' under 'runner' (keys there: {sorted(runner)})")
    out = {}
    for s in seeds:
        cell = vecs.get(str(s), vecs.get(s))
        if cell is None:
            raise SystemExit(f"{path}: icv_vectors: no entry for seed {s}; seeds present {sorted(vecs)}")
        v = np.asarray(cell, dtype=np.float32)
        if v.ndim != 2 or v.shape[0] == 0 or not np.all(np.isfinite(v)):
            raise SystemExit(f"{path}: icv_vectors: seed {s} is shape {v.shape} with "
                             f"{int((~np.isfinite(v)).sum())} non-finite entries; expected "
                             "[n_layers, hidden]")
        out[int(s)] = v
    return grid, out


def i2cl_arm(k_full, on=True):
    return f"I2CL-K{int(k_full)} a={1 if on else 0}"


def i2cl_module(model, layer, module):
    """The upstream's module paths for a Llama-family model (LlamaWrapper
    ._get_arribute_path): 'attn' -> layers[l].self_attn, 'mlp' -> layers[l].mlp."""
    lay = model.model.layers[int(layer)]
    if module == "attn":
        return lay.self_attn
    if module == "mlp":
        return lay.mlp
    raise ValueError(f"module {module!r}: expected one of {I2CL_MODULES}")


def load_i2cl_sidecar(path, seeds, k_base=None, k_full=None):
    """(n_layers, {seed: {"cv": {module: [n_layers, d] float32}, "coef":
    [n_layers, 2, 2] float32}}) from run_i2cl_increment's npz: arrays
    cv_attn_seed<s>, cv_mlp_seed<s>, coef_seed<s> and a 0-d `meta` json (the
    receivers' own convention). Refuses by name."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--i2cl-vectors {path}: not a file (run_i2cl_increment writes it)")
    with np.load(p, allow_pickle=False) as z:
        keys = list(z.files)
        if "meta" not in keys:
            raise SystemExit(f"{path}: no 'meta' array (arrays: {keys[:12]}...)")
        meta = json.loads(str(z["meta"].item() if z["meta"].ndim == 0 else z["meta"].reshape(-1)[0]))
        for want, field in ((k_full, "K"), (k_base, "K_base")):
            if want is None:
                continue
            got = meta.get(field)
            if got is None or int(got) != int(want):
                raise SystemExit(f"{path}: meta.{field} = {got!r}, this run is at {field} = {int(want)}; "
                                 "the I2CL vectors and coefficients are calibrated per (K_base, K_full) level")
        n_l = int(meta.get("n_layers", 0))
        out = {}
        for s in seeds:
            need = [f"cv_attn_seed{s}", f"cv_mlp_seed{s}", f"coef_seed{s}"]
            lack = [k for k in need if k not in keys]
            if lack:
                raise SystemExit(f"{path}: lacks {lack} (seeds present: "
                                 f"{sorted({k.split('seed')[-1] for k in keys if k.startswith('coef_seed')})})")
            cv = {"attn": np.asarray(z[need[0]], dtype=np.float32),
                  "mlp": np.asarray(z[need[1]], dtype=np.float32)}
            coef = np.asarray(z[need[2]], dtype=np.float32)
            for name, arr in (("cv_attn", cv["attn"]), ("cv_mlp", cv["mlp"])):
                if arr.ndim != 2 or (n_l and arr.shape[0] != n_l) or not np.all(np.isfinite(arr)):
                    raise SystemExit(f"{path}: {name}_seed{s} is shape {arr.shape} (n_layers {n_l}) with "
                                     f"{int((~np.isfinite(arr)).sum())} non-finite entries")
            if coef.shape != (cv["attn"].shape[0], 2, 2) or not np.all(np.isfinite(coef)):
                raise SystemExit(f"{path}: coef_seed{s} is shape {coef.shape}, expected "
                                 f"({cv['attn'].shape[0]}, 2, 2) finite")
            out[int(s)] = {"cv": cv, "coef": coef}
    return n_l or cv["attn"].shape[0], out


def i2cl_module_hooks(model, cv, coef, torch, train=False, noise_scale=0.0):
    """The upstream inject_hook_func ('linear', inject_pos 'all') on every
    decoder layer's self_attn and mlp outputs:

        out <- coef[l, m, 1] * out + coef[l, m, 0] * cv[m][l]      (float32, cast back)
        train: out <- out + randn * ||out||_2 * noise_scale        (per position; the noise
                                                                     and the norm carry no grad)

    `coef` is [n_layers, 2, 2] (modules in I2CL_MODULES order, then
    [lambda, beta]); a torch Parameter during calibration, an array at
    inference. Returns the handles."""
    n_l = len(model.model.layers)
    cvt = {m: torch.as_tensor(np.asarray(cv[m], dtype=np.float32)) for m in I2CL_MODULES}
    for m in I2CL_MODULES:
        if cvt[m].shape[0] != n_l:
            raise SystemExit(f"the {m} context vector has {cvt[m].shape[0]} layer rows, the model {n_l} layers")
    ct = coef if torch.is_tensor(coef) else torch.as_tensor(np.asarray(coef, dtype=np.float32))
    if tuple(ct.shape) != (n_l, 2, 2):
        raise SystemExit(f"coef is shape {tuple(ct.shape)}, expected ({n_l}, 2, 2)")
    handles = []

    def make(l, mi, m):
        def fn(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            dev = hs.device
            v = cvt[m][l].to(dev)
            c = ct.to(dev) if not torch.is_tensor(coef) else ct
            new = c[l, mi, 1] * hs.to(torch.float32) + c[l, mi, 0] * v
            if train and noise_scale > 0:
                norm = torch.norm(new.detach(), p=2, dim=-1, keepdim=True)
                new = new + torch.randn_like(new).detach() * norm * float(noise_scale)
            new = new.to(hs.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return fn

    for l in range(n_l):
        for mi, m in enumerate(I2CL_MODULES):
            handles.append(i2cl_module(model, l, m).register_forward_hook(make(l, mi, m)))
    return handles


def icv_transform(x, v_l, lam, torch):
    """The upstream ICVLayer.forward for ONE layer (shengliu66/ICV
    utils/llm_layers.py, b187c63), on the MLP output x [..., d] and the
    layer's direction segment v_l [d], in float32:

        n     = ||x||
        y     = lam * (1 + max(0, cos(x, -v_l))) * v_l / ||v_l||
        x_new = (x / ||x|| + y) / ||x / ||x|| + y|| * n

    Per-layer unit normalisation of the segment and the renormalisation to
    the MLP output's norm are the upstream's; the spec's open scale question
    is settled by this code. Returns float32; the caller casts back."""
    xf = x.to(torch.float32)
    vf = torch.as_tensor(np.asarray(v_l, dtype=np.float32)).to(xf.device)
    n = torch.norm(xf, p=2, dim=-1, keepdim=True)
    cos = torch.nn.functional.cosine_similarity(xf, -vf.expand_as(xf), dim=-1)
    lam_sim = (1.0 + torch.clamp(cos, min=0.0)).unsqueeze(-1)
    y = float(lam) * lam_sim * torch.nn.functional.normalize(vf, dim=-1)
    xn = torch.nn.functional.normalize(xf, p=2, dim=-1)
    return torch.nn.functional.normalize(xn + y, p=2, dim=-1) * n


def icv_hooks(model, v_layers, lam, torch):
    """Forward hooks on EVERY decoder layer's mlp module (all positions),
    the upstream's placement (`layer.mlp = Sequential(mlp, ICVLayer)`).
    lam = 0 installs nothing: the gate is the untouched forward."""
    if float(lam) == 0.0:
        return []
    v = np.asarray(v_layers, dtype=np.float32)
    n_l = len(model.model.layers)
    if v.shape[0] != n_l:
        raise SystemExit(f"the in-context vector has {v.shape[0]} layer rows, the model {n_l} layers")
    handles = []

    def make(l):
        def fn(_mod, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            new = icv_transform(hs, v[l], lam, torch).to(hs.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return fn

    for l in range(n_l):
        handles.append(model.model.layers[l].mlp.register_forward_hook(make(l)))
    return handles


def load_fv_sidecar(path, seeds, k_base=None, k_full=None):
    """{seed: (layer, v float64)} from run_fv_increment's sidecar. Refuses by
    name at every step, with the keys that are there (working rules 3.6); with
    k_base / k_full given, also refuses a sidecar built for another level."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--fv-vectors {path}: not a file (run_fv_increment writes it)")
    doc = json.loads(p.read_text(encoding="utf-8"))
    runner = doc.get("runner") or {}
    vecs = runner.get("fv_vectors")
    if not isinstance(vecs, dict) or not vecs:
        raise SystemExit(f"{path}: no 'fv_vectors' under 'runner' (keys there: {sorted(runner)})")
    for want, key in ((k_full, "K"), (k_base, "K_base")):
        if want is None:
            continue
        got = runner.get(key)
        if got is None or int(got) != int(want):
            raise SystemExit(f"{path}: runner.{key} = {got!r}, this run is at {key} = {int(want)}; "
                             "the FV vector is built per (K_base, K_full) level and does not "
                             "transfer between levels")
    layer_default = runner.get("edit_layer")
    out = {}
    for s in seeds:
        cell = vecs.get(str(s), vecs.get(s))
        if not isinstance(cell, dict) or "v" not in cell:
            keys = sorted(cell) if isinstance(cell, dict) else type(cell).__name__
            raise SystemExit(f"{path}: seed {s} has no 'v' (has {keys}); seeds present {sorted(vecs)}")
        layer = cell.get("layer", layer_default)
        if layer is None:
            raise SystemExit(f"{path}: seed {s}: no 'layer' and no runner.edit_layer")
        v = np.asarray(cell["v"], dtype=np.float64)
        if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
            raise SystemExit(f"{path}: seed {s}: 'v' is shape {v.shape} with "
                             f"{int((~np.isfinite(v)).sum())} non-finite entries; expected a finite "
                             "hidden-size vector")
        out[int(s)] = (int(layer), v)
    return out


def load_tv_sidecar(path, seeds, k_base=None, k_full=None, key="tv_vectors"):
    """(candidate layers, {seed: {layer: theta float64}}) from
    run_tv_increment's sidecar; `key` is tv_vectors (the main, single-dummy
    family) or tv_vectors_m5 (the descriptive mean). Refuses by name."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--tv-vectors {path}: not a file (run_tv_increment writes it)")
    doc = json.loads(p.read_text(encoding="utf-8"))
    runner = doc.get("runner") or {}
    vecs = runner.get(key)
    if not isinstance(vecs, dict) or not vecs:
        raise SystemExit(f"{path}: no '{key}' under 'runner' (keys there: {sorted(runner)})")
    for want, field in ((k_full, "K"), (k_base, "K_base")):
        if want is None:
            continue
        got = runner.get(field)
        if got is None or int(got) != int(want):
            raise SystemExit(f"{path}: runner.{field} = {got!r}, this run is at {field} = {int(want)}; "
                             "the task vector is built per (K_base, K_full) level and does not "
                             "transfer between levels")
    layers = [int(l) for l in (runner.get("candidate_layers") or [])]
    if not layers:
        raise SystemExit(f"{path}: no 'candidate_layers' under 'runner' (keys there: {sorted(runner)})")
    out = {}
    for s in seeds:
        cell = vecs.get(str(s), vecs.get(s))
        if not isinstance(cell, dict) or not cell:
            raise SystemExit(f"{path}: {key}: no entry for seed {s}; seeds present {sorted(vecs)}")
        got = {int(l): np.asarray(v, dtype=np.float64) for l, v in cell.items()}
        missing = [l for l in layers if l not in got]
        if missing:
            raise SystemExit(f"{path}: {key}: seed {s} lacks layers {missing} (has {sorted(got)})")
        for l, v in got.items():
            if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
                raise SystemExit(f"{path}: {key}: seed {s} layer {l}: theta is shape {v.shape} with "
                                 f"{int((~np.isfinite(v)).sum())} non-finite entries")
        out[int(s)] = got
    return layers, out


def replace_hook(model, layer, theta, torch):
    """The answer row of decoder layer `layer`'s output REPLACED by theta
    (Hendel et al.: the task vector overwrites the hidden state at the
    separator). Returns the handle. The row is -1: the cached forward runs
    only the receiver's tokens, so the last row is the answer row."""
    th = torch.as_tensor(np.asarray(theta))

    def fn(_mod, _args, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs.clone()
        hs[:, -1, :] = th.to(hs.dtype).to(hs.device)
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    return model.model.layers[int(layer)].register_forward_hook(fn)


def install_vector_arm(model, arm, seed, fv, tv, inject_hook, torch, icv=None, i2cl=None):
    """The hook(s) an FV / TV / ICV / I2CL arm needs on this forward, or []
    for alpha = 0 (the gate: nothing installed, so the arm is ARM_BASE bit
    for bit).

    `fv` is {seed: (layer, v)} -- one FV family per run; `tv` is
    {family: {seed: {layer: theta}}} -- the main and the m5 family side by
    side (load_tv_sidecar twice); `icv` is {seed: V [n_layers, d]}; `i2cl`
    is {seed: {"cv": {module: [n_layers, d]}, "coef": [n_layers, 2, 2]}}.
    An ICV / I2CL arm's hooks cover every layer and position, so its PREFIX
    cache must have been built with the same hooks (the receivers keep one
    per lambda / per seed)."""
    p = parse_vector_arm(arm)
    if p is None:
        return []
    method, family, layer, alpha = p
    if alpha == 0.0:
        return []
    if method == "FV":
        if fv is None or int(seed) not in fv:
            raise SystemExit(f"{arm!r}: no FV vector for seed {seed} (--fv-vectors)")
        lay, v = fv[int(seed)]
        return [inject_hook(model, lay, alpha, v)]
    if method == "ICV":
        if icv is None or int(seed) not in icv:
            raise SystemExit(f"{arm!r}: no in-context vector for seed {seed} (--icv-vectors)")
        return icv_hooks(model, icv[int(seed)], alpha, torch)
    if method == "I2CL":
        if i2cl is None or int(seed) not in i2cl:
            raise SystemExit(f"{arm!r}: no I2CL vectors for seed {seed} (--i2cl-vectors)")
        if alpha != 1.0:
            raise SystemExit(f"{arm!r}: I2CL's strengths are its calibrated coefficients; only a=0 "
                             "(gate) and a=1 exist")
        return i2cl_module_hooks(model, i2cl[int(seed)]["cv"], i2cl[int(seed)]["coef"], torch)
    fam = (tv or {}).get(family)
    if fam is None or int(seed) not in fam or layer not in fam[int(seed)]:
        have = sorted(tv) if tv else []
        raise SystemExit(f"{arm!r}: no TV vector for family {family!r} seed {seed} layer {layer} "
                         f"(--tv-vectors; families loaded {have})")
    if alpha != 1.0:
        raise SystemExit(f"{arm!r}: a task vector has no strength; only a=0 (gate) and a=1 exist")
    return [replace_hook(model, layer, fam[int(seed)][layer], torch)]


def alpha_zero_faults(zero, base, zero_arm, base_arm):
    """13.5.4(4) for a vector arm: the alpha = 0 arm must reproduce the base
    arm ELEMENTWISE. Nothing is installed at alpha = 0 (install_vector_arm
    returns []), so the two forwards are the same forward and this is an
    identity, not a tolerance. Returns the fault text, or None."""
    a, b = np.asarray(zero), np.asarray(base)
    if a.shape != b.shape:
        return f"{zero_arm!r} is shape {a.shape}, {base_arm!r} is {b.shape}"
    if np.array_equal(a, b):
        return None
    d = np.abs(a - b)
    return (f"{zero_arm!r} does not reproduce {base_arm!r} elementwise "
            f"({int((d > 0).sum())} of {d.size} candidates differ, max {d.max():.3e}). "
            "alpha=0 installs no hook, so this is an identity: a difference means the "
            "arm ran a different forward from the base arm")
