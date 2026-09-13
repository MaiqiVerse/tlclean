"""The six H6 baselines as DATA: candidate configs, hook boundary, status.

ONE PLACE, ON PURPOSE. Section 2.4(5) caps each baseline at five candidate
configurations, and section 13.6.1 requires those configs to be hashed before
the viability gate runs. If each runner carried its own grid, "the frozen
config list" would be a claim about six files that nobody re-derives, and the
freeze would hash the wrong thing. This project has already paid for a
convention living in two places (working rules 2.6.2). So the grids live here, the
runners import them, and `freeze_baseline_spec.py` hashes this module.

HOOK BOUNDARY IS PART OF THE SPEC, NOT AN IMPLEMENTATION DETAIL. "Which tensor
does this baseline modify" is the single most adjustable thing about an
adapted baseline and the easiest to move after seeing a result -- shift the
injection one layer, read the head output post-W_O instead of pre-, and the
number changes with a perfectly good story attached. It is therefore written
down as frozen text and hashed with everything else.

Nothing here is executable configuration in the sense of being tuned. These are
the values section 13.2 and baseline_under_review.md section 3 already fix; this
module is where they stop being prose.
"""

from __future__ import annotations

# The registered candidate-config cap (section 2.4(5)): at most five TOTAL
# configurations per baseline, not five points per hyperparameter.
MAX_CONFIGS = 5

BASELINES = {
    "zerotuning_level1": {
        # THIS repository's commit at which the runner's state was recorded.
        # Whether run_zerotuning_level1.py exists there is what
        # NOT-IMPLEMENTED means, so nobody maintains a status by hand.
        # Filled in the commit that adds the runner; see the note below on
        # why it is the PARENT commit that gets recorded afterwards.
        "runner_commit": "f9e7e88df21a73f7e009d0aa91756a55124971c7",

        # NOTHING per seed: ZeroTuning selects no heads and discovers no
        # vector. r is a global hyperparameter, chosen like gamma*.
        "per_seed_artifacts": (),

        "arm_name": "ZeroTuning-Level-1 (paper-defined; protocol-adapted)",
        "citation": "Han et al., ZeroTuning, ICLR 2026",
        "upstream": "icl/ZeroTuning (FeijiangHan/ZeroTuning)",
        "repo_dir": "ZeroTuning",
        "registered_commit": "51df35b435bff8784241ae651394c0e71f46fa85",
        "needs_adapt_diff": True,   # the attention hook is grafted into the upstream module
        "status": "NOT-IMPLEMENTED",
        "configs": [{"r": 0.5}, {"r": 1.0}, {"r": 2.0}, {"r": 4.0},
                    {"r": 8.0}],
        "config_axis": "r, the shared multiplier on the post-softmax BOS weight",
        "placebo": {"r": 1.0},
        "placebo_kind": "identity-bitwise-vs-full-prompt-natural",
        "hook_boundary":
            "post-softmax, pre-dropout attention weights in every layer and "
            "head. A~_q0 = r A_q0 / [1 + (r-1) A_q0]; A~_qj = A_qj / "
            "[1 + (r-1) A_q0] for j != 0; the head output is then recomputed "
            "as A~ V. Touches attention WEIGHTS only -- never V, never the "
            "residual stream.",
        "selection": "three-seed-mean validation candidate NLL",
    },
    "inductive_bc": {
        # THIS repository's commit at which the runner's state was
        # recorded. Whether run_inductive_bc.py exists there is what
        # NOT-IMPLEMENTED means, so nobody maintains a status by hand.
        # The other five are None: not frozen at code level yet, and each
        # freezes on its own without waiting for the rest.
        "runner_commit": "1e5b5a95a1a4525681a81befebf32b7b7f63eb9c",

        # The bias is estimated on validation, per seed: run_inductive_bc
        # already writes bias_by_seed, and this is the field that says it
        # must.
        "per_seed_artifacts": ("bias",),

        "arm_name": "inductive probability-mean calibration (adapted from BC)",
        "citation": "Zhou et al., Batch Calibration, ICLR 2024, 49-70",
        "upstream": "icl/StaICC (hc495/StaICC) "
                    "prefabricate_inference/standard_calibration.py:108-134",
        "repo_dir": "StaICC",
        "registered_commit": "9d135329eb4c9a8fae7e561997c964bd1da724fa",
        "needs_adapt_diff": False,   # a post-hoc formula reimplemented from standard_calibration.py; no upstream file is modified
        "status": "IMPLEMENTED",
        "configs": [{}],
        "config_axis": "none -- BC has no hyperparameter",
        "placebo": None,
        "placebo_kind": "none-possible",
        "hook_boundary":
            "no model hook at all. A post-hoc transform of the candidate "
            "logits: p = softmax within the candidate set, out = softmax(p - "
            "b_s), b_s the per-class mean of p frozen on validation for prefix "
            "seed s. The forward is the stock natural one.",
        "selection": "nothing is selected; b_s is estimated, not tuned",
        "known_ceiling":
            "p and b are both probability vectors, so d = p - b sums to zero. "
            "Maximising softmax(d)_g under that constraint gives d_g = 1 and, "
            "by Jensen, the other C-1 coordinates equal at -1/(C-1): p_max = "
            "e^(1+1/(C-1))/(e^(1+1/(C-1))+C-1). At C=36 that is 0.074002, an "
            "NLL floor of 2.6037 nats against ln(36) = 3.5835 for chance; a "
            "uniform bias gives 2.6301, within 0.03 nats of the optimum. This "
            "is the official formula's genuine limit in this setting and is "
            "reported as context. It does NOT demote the comparison: H6 "
            "registers candidate-conditional NLL as primary (6.1, 9) and "
            "accuracy is reported alongside, not instead.",
    },
    "fv_on_icl": {
        # THIS repository's commit at which the runner's state was
        # recorded. None = not frozen at code level yet. Whether
        # run_fv_on_icl.py exists there is what NOT-IMPLEMENTED
        # means, so nobody maintains a status by hand.
        "runner_commit": None,

        # 2.4(4): the FV vector is extracted on THIS seed's 144 validation
        # queries, so there are three of them. alpha is the global
        # hyperparameter and is not listed here.
        "per_seed_artifacts": ("fv_vector",),

        "arm_name": "Function Vectors (adapted)",
        "citation": "Todd et al., Function Vectors in LLMs, ICLR 2024",
        "upstream": "icl/function_vectors (ericwtodd/function_vectors)",
        "repo_dir": "function_vectors",
        "registered_commit": "fb9eac7b6dc707ea1475a717379916007fe448d5",
        "needs_adapt_diff": True,   # the extraction/CIE entry points are adapted to take external prompts
        "status": "NOT-IMPLEMENTED",
        "configs": [{"alpha": 0.0}, {"alpha": 0.25}, {"alpha": 0.5},
                    {"alpha": 1.0}, {"alpha": 2.0}],
        "config_axis": "alpha, the injection scale. The injection LAYER is "
                       "fixed at 9 and is NOT searched -- searching it would "
                       "make the grid a Cartesian product while still being "
                       "called five points (section 2.4(5) forbids exactly "
                       "this).",
        "placebo": {"alpha": 0.0},
        "placebo_kind": "identity-bitwise-vs-full-prompt-natural",
        "hook_boundary":
            "v_FV = sum over the CIE top-10 heads of their mean answer-position "
            "output passed through that head's own W_O slice. Added as "
            "alpha * v_FV to the ANSWER ROW ONLY of decoder layer 9's residual "
            "stream. Discovery uses the first 100 discovery-split queries with "
            "prefix seeds cycling 42/43/44, and 25 label-shuffled corrupted "
            "prompts for the CIE.",
        "selection": "three-seed-mean validation candidate NLL",
    },
    "tsla_tl_on_icl": {
        # THIS repository's commit at which the runner's state was
        # recorded. None = not frozen at code level yet. Whether
        # run_tsla_tl_on_icl.py exists there is what NOT-IMPLEMENTED
        # means, so nobody maintains a status by hand.
        "runner_commit": None,

        # 2.4(4): same shape as FV -- a per-seed vector, a global alpha.
        "per_seed_artifacts": ("tsla_vector",),

        "arm_name": "TSLA TL-head steering vector (adapted)",
        "citation": "Yang et al., Localizing Task Recognition and Task "
                    "Learning in ICL, ICLR 2026",
        "upstream": "icl/Localizing_TR_TL (HLYang2001/Localizing_TR_TL)",
        "repo_dir": "Localizing_TR_TL",
        "registered_commit": "39aea6d48bacc4c241de0aeabb0f4f9858050054",
        "needs_adapt_diff": True,   # scoring is adapted to this label subspace and the pinv rule
        "status": "NOT-IMPLEMENTED",
        "configs": [{"alpha": 0.0}, {"alpha": 0.25}, {"alpha": 0.5},
                    {"alpha": 1.0}, {"alpha": 2.0}],
        "config_axis": "alpha only; injection layer fixed at 16, head count "
                       "fixed at the official top-3% (30 heads)",
        "placebo": {"alpha": 0.0},
        "placebo_kind": "identity-bitwise-vs-full-prompt-natural",
        "hook_boundary":
            "same injection code as fv_on_icl -- answer row of one decoder "
            "layer's residual stream, layer 16. The vector is the mean OV "
            "output of the 30 TSLA-TL heads. Label subspace = this project's "
            "label token unembeddings, projected with a frozen pinv rule whose "
            "rank is recorded. Discovery uses the first 50 discovery-split "
            "queries.",
        "selection": "three-seed-mean validation candidate NLL",
    },
    "unibias_code": {
        # THIS repository's commit at which the runner's state was
        # recorded. None = not frozen at code level yet. Whether
        # run_unibias_code.py exists there is what NOT-IMPLEMENTED
        # means, so nobody maintains a status by hand.
        "runner_commit": None,

        # 2.4(4): components are DISCOVERED on this seed's validation, and
        # 13.2.5's threshold/mask/alpha pass runs entirely there, so the mask
        # is per seed. It has no global hyperparameter at all.
        "per_seed_artifacts": ("attention_components", "ffn_components"),

        "arm_name": "UniBias-code (official-code reconstruction)",
        "citation": "Zhou et al., UniBias, NeurIPS 2024",
        "upstream": "icl/UniBias (hzzhou01/UniBias) "
                    "commit bd7736238d8a030ae27437d1b56d5da816ec9e02",
        "repo_dir": "UniBias",
        # section 13.2.5 names this commit; the freeze refuses to write
        # unless it is the one checked out.
        "registered_commit": "bd7736238d8a030ae27437d1b56d5da816ec9e02",
        "needs_adapt_diff": True,   # the missing custom_head_output/self_attn.mask interfaces are rebuilt
        "status": "NOT-IMPLEMENTED",
        "configs": [{"discovery_search": "author flow, discovery support only"}],
        "config_axis": "the repository's internal threshold/mask/alpha search "
                       "counts as ONE frozen configuration (section 2.4(5) "
                       "exception). It does NOT read validation NLL.",
        "placebo": {"attention_components": [], "ffn_components": []},
        "placebo_kind": "identity-bitwise-vs-full-prompt-natural",
        "hook_boundary":
            "attention: o_proj INPUT pre-hook, split into 32 query heads; "
            "identification multiplies each head by its own W_O slice to "
            "rebuild the post-W_O contribution to the answer row; intervention "
            "multiplies the selected head's channels by alpha BEFORE o_proj. "
            "Per QUERY head under GQA -- the other members of the same KV "
            "group must be bitwise unchanged. FFN: identification reads the "
            "true coefficient from a down_proj input pre-hook; elimination "
            "REPLACES the selected dimensions with alpha on the up_proj OUTPUT "
            "hook (code-faithful; NOT the paper's down_proj coefficient "
            "zeroing). Order fixed: discover and install FFN first, then "
            "discover attention on the FFN-intervened model. Layers 16-31 "
            "only, zero-based.",
        "selection": "none from validation; discovery-only",
    },
    "deepthinking_dev": {
        # THIS repository's commit at which the runner's state was
        # recorded. None = not frozen at code level yet. Whether
        # run_deepthinking_dev.py exists there is what NOT-IMPLEMENTED
        # means, so nobody maintains a status by hand.
        "runner_commit": None,

        # The 15 KV snapshots ARE per prefix -- one trajectory per demo
        # prefix -- but they are query-independent caches, not a fitted
        # artifact, and T* is a single global choice across the three.
        "per_seed_artifacts": ("kv_snapshots",),

        "arm_name": "Deep-Thinking-dev (paper-defined; cache-adapted)",
        "citation": "Yang et al., Iterative Forward Tuning Boosts ICL, ACL 2024",
        "upstream": "icl/DeepThinking (Yangjiaxi/DeepThinking)",
        "repo_dir": "DeepThinking",
        "registered_commit": "36fada14cfcaf873486ff422783d86c4644a9eae",
        "needs_adapt_diff": True,   # the legacy cache container is adapted to DynamicCache
        "status": "NOT-IMPLEMENTED",
        "configs": [{"T": t} for t in range(1, 16)],
        "config_axis": "T, the deep-thinking round. eta=0.01 and beta=0.9 are "
                       "FIXED (no reproducible with-dev eta grid exists). The "
                       "15 rounds are ONE nested trajectory, the section "
                       "2.4(5) exception -- not fifteen independent configs.",
        "placebo": {"T": 1},
        "placebo_kind": "identity-bitwise-vs-same-prefix-cache-natural",
        "path_tolerance": 0.02,   # section 11.9's ONLY numeric allowance
        "hook_boundary":
            "prefix KV cache only. Round 1 builds the natural prefix KV; each "
            "later round runs the same prefix behind the previous KV, takes "
            "the new L positions, forms the pseudo-gradient as new-old and "
            "updates M_t = G_t + 0.9 M_{t-1}, C_t = C_{t-1} + 0.01 M_t. "
            "Canonical state is a detached bf16 legacy tuple with per-layer "
            "K/V shaped [1, num_key_value_heads, L, head_dim] -- 8 KV heads on "
            "L3.1, never expanded to 32. The query is never part of the "
            "trajectory: the deep-thinking pass is query-INDEPENDENT.",
        "selection": "three-seed-mean validation candidate NLL over the 15 "
                     "snapshots; ties within 1e-8 take the smaller T",
    },
}

# Audited but deliberately NOT an H6 arm. Recorded here so the freeze shows it
# was excluded before any result was seen, not dropped after one.
EXCLUDED = {
    "yu_ananiadou": {
        "citation": "Yu & Ananiadou, EMNLP 2024",
        "upstream": "icl/in-context-mechanism (zepingyu0512/in-context-mechanism)",
        "repo_dir": "in-context-mechanism",
        "registered_commit": "f5b00cec1723d11e83b669146485c105e6f49220",
        "status": "N/A-SETTING + BLOCKED-CODE",
        "why": "the majority-bias intervention needs a two-class imbalanced "
               "prompt with a well-defined minority label position. This "
               "setting is 36-class and class-balanced at K=5, so that target "
               "set does not exist, and the foo/bar head taxonomy does not "
               "extend. Scaling every label uniformly, or inventing a "
               "multi-class head taxonomy, would be a NEW method wearing the "
               "authors' name. The missing code is the secondary reason; the "
               "setting is the primary one.",
    },
}


def config_count(name):
    """How many candidate configurations a baseline declares."""
    return len(BASELINES[name]["configs"])


def check_caps():
    """Section 2.4(5): at most five configs, with two named exceptions.

    Returns a list of violations; empty means the registry obeys the cap. The
    exceptions are enumerated by NAME rather than by a property of the entry,
    so a third baseline cannot quietly acquire one by describing itself the
    right way.
    """
    exempt = {"unibias_code", "deepthinking_dev"}
    bad = []
    for name, spec in sorted(BASELINES.items()):
        n = len(spec["configs"])
        if n > MAX_CONFIGS and name not in exempt:
            bad.append(f"{name}: {n} configs exceeds the cap of {MAX_CONFIGS} "
                       "and is not one of the two registered exceptions")
        if n == 0:
            bad.append(f"{name}: declares no configuration at all")
        # 2.4(4): every entry must SAY whether it fits anything per seed.
        # An absent field is not "nothing per seed" -- it is a baseline whose
        # author did not answer the question, and a run producing one shared
        # vector for three prefixes would then pass every check.
        if "runner_commit" not in spec:
            bad.append(f"{name}: no 'runner_commit'. That commit is what "
                       "records whether this baseline is built; None is an "
                       "answer, an absent field is not")
        if "per_seed_artifacts" not in spec:
            bad.append(f"{name}: no 'per_seed_artifacts'. Section 2.4(4) "
                       "requires each baseline to declare what it fits PER "
                       "PREFIX SEED; an empty tuple is an answer, an absent "
                       "field is not")
        elif not isinstance(spec["per_seed_artifacts"], tuple):
            bad.append(f"{name}: per_seed_artifacts is "
                       f"{type(spec['per_seed_artifacts']).__name__}, "
                       "expected a tuple of names")
    return bad


def per_seed_artifacts(name):
    """What `name` must produce once per prefix seed. Possibly empty."""
    return tuple(BASELINES[name].get("per_seed_artifacts") or ())


def artifact_seed_faults(name, produced, seeds):
    """Why `produced` does not answer this baseline's per-seed obligation.

    `produced` maps artifact name -> the seeds it was produced for. Returns
    faults rather than raising, and says [not declared] rather than passing
    when the registry has nothing to check against -- a silent pass here
    would be the same shape as the missing field itself.
    """
    want = per_seed_artifacts(name)
    if not want:
        return []
    bad = []
    for a in want:
        got = produced.get(a)
        if got is None:
            bad.append(f"{name}: no {a!r} was produced, and 2.4(4) fits it "
                       "per prefix seed")
        elif sorted(got) != sorted(seeds):
            bad.append(
                f"{name}: {a!r} exists for seeds {sorted(got)}, expected "
                f"{sorted(seeds)}. One artifact shared across three prefixes "
                "is a different experiment -- validation is drawn per seed "
                "so that each prefix gets its own adaptation, and the "
                "seed-to-seed spread carries that adaptation's noise")
    return bad
