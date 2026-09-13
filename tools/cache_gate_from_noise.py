"""The receivers' cache-equivalence bound, from the model's own two-path noise.

13.5.4(2) holds the cached construction to 1 bf16 ulp of the monolithic
prompt (run_k0_receiver.cache_equivalence_faults, derived in 14.0b-22). A
model whose OWN prefill-and-continue path differs from its one-pass forward
by more than that -- Qwen3-14B in bf16: 1 ulp at K=1, 2-3 ulp at K=5
(RESULTS 63.3) -- cannot meet it however correct the construction is, and
the gate would then measure the model, not the cache. So the bound becomes

    max(1, ceil(worst ulp of tools/check_two_path_noise on the cell's largest K,
                in the cell's dtype))

and the measurement, not the failure, is what raises it (working rules 11:
the tolerance comes from an independent reason). A wrong cache still moves
values by 8+ ulp (test_k0_cache_gate's world B), so the gate keeps its
discriminating power at 2 or 3.

    python tools/cache_gate_from_noise.py --noise-json results/method/Q314c36/precheck_two_path_bfloat16_K5.json \\
        --K 5 --dtype bfloat16 --out results/method/Q314c36/cache_gate.json
    python tools/cache_gate_from_noise.py --read results/method/Q314c36/cache_gate.json --field max_ulp

Without a readable noise json (the measurement did not run: an OOM, a
missing file) the gate stays at 1 and says so in `source`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

FLOOR = 1.0


def gate_from_noise(doc, K, dtype):
    """(max_ulp, source, attn) from check_two_path_noise's json (a list, one
    entry per dtype). `attn` is the kernel the measurement ran under; a json
    from before the kernel was recorded (2026-09-13) was an eager one."""
    entries = [r for r in doc if r.get("dtype") == dtype] or list(doc)
    if not entries:
        return FLOOR, f"no {dtype} entry in the noise json; the derived 1 ulp stands", "eager"
    r = entries[-1]
    attn = str(r.get("attn", "eager"))
    worst = max(float(v["max_ulp"]) for v in r["by_q"].values())
    gate = max(FLOOR, float(math.ceil(worst - 1e-9)))
    tails = sorted({int(str(k).split(":")[-1]) for k in r["by_q"]})   # keys "<q>" or "<prompt>:<q>"
    return gate, (f"check_two_path_noise K={K} {dtype} {attn} on {r.get('n_prompts', 1)} prompt(s) of "
                  f"{r.get('n_tokens', '?')} tokens: worst {worst:.2f} ulp over live tails {tails}"
                  + ("" if gate > FLOOR else "; the derived 1 ulp stands")), attn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-json", nargs="+",
                    help="check_two_path_noise jsons; several (one per registered seed's K_max "
                         "prompt) give the gate as the worst over them (14.0b-24 addendum, "
                         "2026-09-13: one prompt under-measured clinc150 by 1 ulp, job 845173)")
    ap.add_argument("--K", type=int)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out")
    ap.add_argument("--attn", choices=("eager", "sdpa"), default=None,
                    help="the kernel the cell runs; recorded in the gate file when the "
                         "measurement could not run (otherwise the noise json's own kernel "
                         "is recorded, and a mismatch with this is refused)")
    ap.add_argument("--read", help="print one field of an existing cache_gate.json")
    ap.add_argument("--field", choices=("max_ulp", "source", "attn", "dtype"), default="max_ulp")
    args = ap.parse_args(argv)
    if args.read:
        d = json.loads(Path(args.read).read_text(encoding="utf-8"))
        # a gate file from before the kernel was recorded was measured under eager
        v = d.get(args.field, "eager" if args.field == "attn" else "")
        print(f"{v:g}" if isinstance(v, float) else v)
        return 0
    if not (args.noise_json and args.out):
        print("give --noise-json and --out, or --read", file=sys.stderr)
        return 2
    attn = args.attn or "eager"
    gate, sources = FLOOR, []
    for p in (Path(x) for x in args.noise_json):
        if p.is_file():
            try:
                g, src, a = gate_from_noise(json.loads(p.read_text(encoding="utf-8")),
                                            args.K, args.dtype)
                src += f" ({p})"
            except (ValueError, KeyError, TypeError) as e:
                g, src, a = FLOOR, f"{p}: unreadable ({type(e).__name__}); the derived 1 ulp stands", attn
            if args.attn and a != args.attn:
                print(f"{p} was measured under {a}; this cell runs {args.attn}", file=sys.stderr)
                return 1
            attn = a
        else:
            g, src = FLOOR, f"{p}: the K={args.K} measurement did not run; the derived 1 ulp stands"
        gate = max(gate, g)
        sources.append(src)
    source = " | ".join(sources) if len(sources) > 1 else sources[0]
    if len(sources) > 1:
        source = f"worst of {len(sources)} prompts: " + source
    Path(args.out).write_text(json.dumps({"max_ulp": gate, "source": source, "floor": FLOOR,
                                          "dtype": args.dtype, "attn": attn}, indent=2),
                              encoding="utf-8")
    print(f"  cache gate {gate:g} ulp ({attn})  <- {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
