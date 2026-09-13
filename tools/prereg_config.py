"""The preregistered design constants, in one place, with what follows from them.

WHY THIS EXISTS. Six numbers from section 2.1/2.2 were written out by hand in
six modules, and one of them is DERIVED from two others:

    N_VALIDATION = N_ELIGIBLE * VALIDATION_PER_CLASS      36 * 4 = 144

but it was a literal `144` in both `method_a_viability` and
`freeze_baseline_spec`. Change VALIDATION_PER_CLASS to 5 and those two keep
asserting 144 -- shape checks that pass while describing a run that no longer
exists. The user asked directly whether a new threshold or a new validation
size could be a one-number change; with six copies and one silent derivation,
the honest answer was no. It is one number now, and the derived ones cannot
go stale because they are computed.

⚠ NOT a JSON in results/. The original sketch put these in
`results/prereg_config.json`, hashed. Two things changed that: results/ lives
only on the server, so every fixture would have had to cope with the config
being absent; and since 14.0b-2 (t) the freeze records the repository COMMIT,
which already pins this file exactly. A module is in git, importable
everywhere, and pinned by the same commit the results carry -- the JSON would
have added a second source of truth and no evidence.

WHAT IS AND IS NOT HERE. Only numbers the preregistration fixes. The gamma
grid stays in `prototype_targets` (it is part of the intervention, not the
data design), and the statistical constants stay in `method_a_stats` next to
the tests that use them.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# section 2.1 -- the task and the splits
# --------------------------------------------------------------------------
MIN_TRAIN_PER_CLASS = 15
"""Deduplicated TRAIN rows a class needs to be eligible.

! THIS NUMBER DOES NOT INCLUDE THE DEMONSTRATIONS, and 14.0a is the record of
what that cost. It was read as "enough for validation plus every seed's
demos plus whatever comes next", and it is not: the thinnest eligible classes
had 16 unique rows against 4 validation + three seeds' K=5 draws, leaving an
expected 2.38 and a 1.9% chance of zero. Anything drawn from TRAIN beyond the
validation set and the prefixes has to justify its own budget here.
"""

VALIDATION_PER_CLASS = 4
"""Validation queries drawn per eligible class, per prefix seed (14.0a)."""

N_ELIGIBLE = 36
"""Eligible classes on TREC-fine: present in both splits with at least
MIN_TRAIN_PER_CLASS deduplicated train rows.

MEASURED, not chosen -- it is a property of the data under the threshold
above. Recorded here so shape checks have something to compare against, and
asserted against the manifest wherever one is loaded rather than trusted.
"""

# --------------------------------------------------------------------------
# section 2.2 -- the PCW test protocol
# --------------------------------------------------------------------------
N_TEST_PER_SEED = 250
REGISTERED_SEEDS = (42, 43, 44)
"""The three prefix seeds. Their role is stability, not estimation: three
seeds do not estimate a seed population, and nothing here may be extended at
run time."""

# --------------------------------------------------------------------------
# derived -- never write these out by hand
# --------------------------------------------------------------------------
N_VALIDATION = N_ELIGIBLE * VALIDATION_PER_CLASS
"""144 today. Two modules carried this as a literal; changing either input
left them asserting a shape no run produced."""

N_TEST_CELLS = N_TEST_PER_SEED * len(REGISTERED_SEEDS)
"""750 (query, seed) cells, NOT 750 independent observations -- the three
draws come from one pool and overlap by construction, which is why the
inference clusters on unique query_id (13.6.4)."""


# --------------------------------------------------------------------------
# preconditions
# --------------------------------------------------------------------------
def fold_balance_is_exact() -> bool:
    """Does the fold split come out exactly balanced?

    `method_a_viability.assign_folds` alternates the starting parity by class
    ordinal, so a class with an odd number of queries sends its extra query to
    alternating folds. That cancels exactly when the CLASS COUNT is even, and
    leaves a one-query imbalance when it is odd. 36 is even, so the split is
    exact today; this is a precondition of that claim, not a law.
    """
    return N_ELIGIBLE % 2 == 0


def config_faults() -> list:
    """Why this configuration could not produce the design it describes.

    RETURNS faults rather than raising: a config module that explodes on
    import takes every consumer with it, including the fixtures that would
    have reported the problem legibly.
    """
    bad = []
    if VALIDATION_PER_CLASS >= MIN_TRAIN_PER_CLASS:
        bad.append(
            f"VALIDATION_PER_CLASS={VALIDATION_PER_CLASS} is not below "
            f"MIN_TRAIN_PER_CLASS={MIN_TRAIN_PER_CLASS}: an eligible class "
            "would have nothing left to demonstrate with")
    if len(set(REGISTERED_SEEDS)) != len(REGISTERED_SEEDS):
        bad.append(f"REGISTERED_SEEDS {REGISTERED_SEEDS} repeats a seed")
    if N_VALIDATION != N_ELIGIBLE * VALIDATION_PER_CLASS:
        bad.append("N_VALIDATION is no longer the product it is defined as")
    for name, v in (("MIN_TRAIN_PER_CLASS", MIN_TRAIN_PER_CLASS),
                    ("VALIDATION_PER_CLASS", VALIDATION_PER_CLASS),
                    ("N_ELIGIBLE", N_ELIGIBLE),
                    ("N_TEST_PER_SEED", N_TEST_PER_SEED)):
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            bad.append(f"{name}={v!r} is not a positive int")
    return bad


def summary() -> str:
    return (f"eligible={N_ELIGIBLE} min_train={MIN_TRAIN_PER_CLASS} "
            f"validation={VALIDATION_PER_CLASS}/class => {N_VALIDATION} "
            f"per seed; test={N_TEST_PER_SEED} x {len(REGISTERED_SEEDS)} "
            f"seeds => {N_TEST_CELLS} cells; fold balance exact="
            f"{fold_balance_is_exact()}")
