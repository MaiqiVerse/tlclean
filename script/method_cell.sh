#!/usr/bin/env bash
# script/method_cell.sh  MTAG  [STEPS]
#
# THE SECTION-13.5 CONSTRUCTIONS ON ONE (MODEL, TASK) CELL, end to end:
#
#   manifest    the model-independent class space and the validation / test
#               draws (tools/build_query_manifest.py)
#   calib       the calibration prompts at every K a level needs, label-locked
#               to the K=KDISC seed-42 file, validation reserved out of the demo
#               pool, every prompt gated against the model's window
#   precheck    the two GPU pre-checks a model must pass before a cell is spent
#               on it: its two-path noise in this cell's dtype against the
#               receivers' 1-ulp gate, and the capture / DLA identities on its
#               architecture against the Llama-3.1-8B reference (REDISCOVERY 7.6)
#   labelspace  this model's frozen label token space
#   nesting     each level's K_base draw nests inside its K_full draw
#   natural     natural readouts at every K and the native increment totals
#   carriers    section 2.4(2) carrier discovery: the frozen full-validation
#               top-8 (and the fold sets) for THIS model on THIS task
#   tsla        the TSLA vectors of all three head classes at every K_full
#   fv          the Function Vectors (Todd et al., prereg 14.0b-25) of every
#               level, built from the level's extra demonstrations only;
#               FVHEADS=<n> overrides the paper's size rule
#   tv          the Task Vectors (Hendel et al., prereg 14.0b-25) of every
#               level at the five candidate layers; the layer is chosen on
#               validation (tools/select_tv_layer.py, after the level's
#               receiver) and the test read runs that layer alone
#   icv         the In-Context Vectors (Liu et al., prereg 14.0b-25) of every
#               level; the lambda is chosen on validation
#               (tools/select_icv_lambda.py) and the test read runs it alone
#   i2cl        the I2CL context vectors and the 4 L calibrated scalars (Li
#               et al., prereg 14.0b-25) of every level -- the costly one:
#               I2CL_EPOCHS (100) x |E_s| pseudo-queries per seed, about an
#               hour per level on an 8B model; SKIP=i2cl to leave it out
#   k0 / k2 / k10   the levels: a K_base receiver whose carriers read the K_full
#               memory (K_base = 0: the whole bank), with the TSLA families,
#               the FV, TV, ICV and I2CL arms beside it; the two increment
#               levels add the carriers' direct write and the absorption. A
#               level whose npz predates one of the cell's baseline sidecars
#               gains the missing arms in place (done_ok --needs-arm, then the
#               receiver's --append: the stored arms stay bit for bit)
#   ceiling     the carriers' direct write on the NATIVE K_full prompt (the
#               method's own upper bound, RESULTS 53.6); optional, GPU-heavy
#   summary     the tables (three levels side by side, alpha = 1 accuracy,
#               direct write vs model) -- zero GPU
#   test        the one-shot test_seed read of every level through the UNSAFE
#               wrapper (prereg 14.0b-23); opt-in, never in the default list
#
# Every step is skipped when its artifacts are complete (tools/method_status),
# so the same command re-submitted after an abort continues where it stopped;
# STEPS / SKIP / FROM choose what runs; every phase event lands in the cell's
# ledger with the job id.
#
# Range. Any model tag in tools/model_tags.py (or MODEL= / METHOD= overrides
# for an unregistered checkpoint), SelfExtend included -- the receivers' mask
# enters the attention module the same way. Any k-per-class task with a shared
# demonstration bank and a train split: trec_fine, banking77, clinc150,
# yelp_full, dbpedia14, yahoo_answers (their _per_class variants), and the
# shared-bank forms of the generated tasks -- monk_bank_r{1,2,3}_per_class,
# synthetic_linear_bank_per_class, synthetic_mlp_bank_per_class: one fixed
# hidden concept per task, a train bank and a disjoint test pool
# (tasks/shared_bank_task.py). The per-prompt generated tasks
# (synthetic_*_per_class, monk_per_class*) carry their demonstrations inside
# each query and are refused (tools/method_status.py --task-check); the
# rediscovery chain covers those. A level whose prompts overrun the model's
# window is skipped and the reason recorded, never silently truncated.
#
# Usage (server):
#   sbatch script/lsu1.sh bash script/method_cell.sh L31c36
#   TASK=banking77_per_class LEVELS="0:2 1:2 2:5" sbatch script/lsu1.sh bash script/method_cell.sh L31c36
#   SKIP="tsla ceiling" sbatch script/lsu1.sh bash script/method_cell.sh Q25c36
#   TOPN="30 64" sbatch ...   # carrier-count cuts run beside the registered top-8 (default 30 =
#                             # TSLA's head count, so selective top-30 and TSLA read the same
#                             # number of heads on the same queries); TOPN="" runs none
#   FROM=carriers sbatch ...                          # resume from a step
#   bash script/method_cell.sh L31c36 "k0 summary"     # a subset, in this order
#   DRY=1 bash script/method_cell.sh L31c36            # print every command, run nothing
#   TEST=1 sbatch ... bash script/method_cell.sh L31c36 test   # the test read, after validation
#   Smoke on a cached checkpoint (this host, no slurm):
#   MTAG=smoke MODEL=Qwen/Qwen2-7B-Instruct VPC=1 NQ=12 LIMIT=6 LEVELS="0:5 2:5" bash script/method_cell.sh smoke
# -E (errtrace): every tool runs inside the `run` function, and without it
# bash skips the ERR trap below for a failure inside a function -- the cell
# exited at the traceback with no [abort] line and no ledger abort event
# (the Monk-1 smoke of 2026-09-13; the same silence on every earlier abort).
set -eE
# The eager probes materialise float32 [heads, L, L] score matrices; on long-text tasks the
# caching allocator's fragmentation (8-14 GiB reserved but unallocated at the OOMs of jobs
# 844655 / 844658 / 844677) is what tipped a 40 GB card. Expandable segments remove it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MTAG_BASE="${1:?usage: method_cell.sh MTAG [STEPS]}"
ALL_STEPS="manifest calib precheck labelspace nesting natural carriers tsla fv tv icv i2cl k0 k2 k10 ceiling summary test"
DEFAULT_STEPS="manifest calib precheck labelspace nesting natural carriers tsla fv tv icv i2cl k0 k2 k10 summary"
STEPS="${2:-${DEFAULT_STEPS}}"
SKIP="${SKIP:-}"
if [ -n "${FROM:-}" ]; then
  case " ${ALL_STEPS} " in
    *" ${FROM} "*) STEPS="${FROM}${DEFAULT_STEPS#*${FROM}}" ;;
    *) echo "[abort] FROM=${FROM}: not one of ${ALL_STEPS}"; exit 1 ;;
  esac
fi
has () { case " ${SKIP} " in *" $1 "*) return 1 ;; esac
         case " ${STEPS} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

TASK="${TASK:-trec_fine_per_class}"
LEVELS="${LEVELS:-0:5 2:5 5:10}"      # base:full pairs; base 0 is the K=0 receiver
KDISC="${KDISC:-5}"                   # the K the carriers are discovered at (2.4(2))
NQ="${NQ:-250}"                       # test queries per seed = calibration prompts
VPC="${VPC:-4}"                       # validation queries per eligible class per seed
CLASSES="${CLASSES-tl,tr,random}"     # TSLA head classes; CLASSES="" = no TSLA arms (an explicit empty is honoured)
FVHEADS="${FVHEADS-}"                 # FV head count; empty = the paper's size rule (run_fv_increment.head_count_for)
I2CL_EPOCHS="${I2CL_EPOCHS-100}"      # I2CL calibration epochs (the upstream's 100); I2CL_BS pseudo-queries per step (8)
I2CL_BS="${I2CL_BS-8}"
# TOPN: carrier-count cuts. For each N every level runs once more with the
# first N heads of the bundle's own ranking (artifacts top${N}_*, test
# UNSAFE_top${N}_*); the registered top-8 arm is untouched. 30 = TSLA's
# floor(3 % of 32 x 32), so the selective top-30 arm and the TSLA arms in the
# same npz read the same head count on the same queries. The receivers mark
# these runs registered_arm=false: a cut is exploratory, and adding heads
# also adds earlier layers (run_k0_receiver --top-n help). TOPN="" = none.
TOPN="${TOPN-30}"
TOPN_ARG=(); [ -n "${TOPN}" ] && TOPN_ARG=(--top-n ${TOPN})   # the ceiling's cuts (analyze_direct_write_accuracy)
LIMIT="${LIMIT:-0}"                   # smoke: --limit on the GPU receivers / probes
DRY="${DRY:-0}"
TEST="${TEST:-0}"
SPEC_FREEZE="${SPEC_FREEZE:-results/baseline_spec_freeze_v2.json}"
SEEDS="42 43 44"                      # REGISTERED_SEEDS; not a knob

python tools/method_status.py --task-check "${TASK}" || exit 1
MODEL=$(python tools/model_tags.py "${MTAG_BASE}" --field model ${MODEL:+--model "${MODEL}"} ${METHOD:+--method "${METHOD}"}) || exit 1
SE=$(python tools/model_tags.py "${MTAG_BASE}" --field se ${MODEL:+--model "${MODEL}"} ${METHOD:+--method "${METHOD}"}) || exit 1
GATE=$(python tools/model_tags.py "${MTAG_BASE}" --field gate ${MODEL:+--model "${MODEL}"} ${METHOD:+--method "${METHOD}"}) || exit 1
# DTYPE: the tag's default (float32 for the Qwen tags -- their bf16 cache path
# disagrees with the monolithic one by up to 13 ulp, see tools/model_tags.py),
# overridable; passed to every method-line runner as --dtype.
DTYPE=$(python tools/model_tags.py "${MTAG_BASE}" --field dtype ${MODEL:+--model "${MODEL}"} ${METHOD:+--method "${METHOD}"} ${DTYPE:+--dtype "${DTYPE}"}) || exit 1
SE="${SE} --dtype ${DTYPE}"
METHOD_NAME=$(python tools/model_tags.py "${MTAG_BASE}" --field method ${MODEL:+--model "${MODEL}"} ${METHOD:+--method "${METHOD}"}) || exit 1   # the summary's paper rows name SelfExtend
# ATTN: the attention kernel every GPU tool loads the model with (tools/model_args
# --attn, spelled every time). eager = HF's [heads, L, L] float32 softmax, the
# kernel every registered run used; sdpa = torch's fused kernels, no score
# matrix, so the K=10 native prompts of the long-text tasks fit a 40 GB card
# (RESULTS 63.17). The precheck measures the two-path noise under it and
# cache_gate.json records it; a gate measured under another kernel is refused
# below. SelfExtend is eager only (models/selfExtend has its own attention).
ATTN="${ATTN:-eager}"
case "${ATTN}" in eager|sdpa) ;; *) echo "ATTN=${ATTN}: expected eager or sdpa"; exit 1;; esac
if [ "${METHOD_NAME}" = "selfextend" ] && [ "${ATTN}" != "eager" ]; then
  echo "ATTN=${ATTN} with SelfExtend: models/selfExtend runs its own eager attention; use ATTN=eager"; exit 1
fi
SE="${SE} --attn ${ATTN}"
ATTN_SFX=""; [ "${ATTN}" != "eager" ] && ATTN_SFX="_${ATTN}"   # eager keeps the artifact names every finished cell has
TASKTAG=$(python tools/rediscover_status.py --task-tag "${TASK}") || exit 1
LEVEL_LINES=$(python tools/method_status.py --levels "${LEVELS}") || exit 1
KS=$(python tools/method_status.py --ks "${LEVELS}" --kdisc "${KDISC}") || exit 1

MCELL="${MTAG_BASE}${TASKTAG}"
OUT="results/method/${MCELL}"
CALIB="data/method/${MTAG_BASE}/${TASK}"
QM="${OUT}/query_manifest.json"
LS="${OUT}/label_space_${MTAG_BASE}.json"
BUNDLE="${OUT}/carriers_full_validation.json"
FOLD="${OUT}/carriers_fold.json"
SCORES="${OUT}/carrier_scores.npz"
SRC="${CALIB}/calibration_${TASK}_K${KDISC}_seed42_uuid.jsonl"
LEDGER="${OUT}/ledger.jsonl"
# The checkout this job started from. Every step re-reads HEAD and refuses to
# go on if it moved: a `git pull` under a live job swaps the python tools
# between steps while bash keeps reading the old script, so the cell would be
# built from two versions of the code without any record of which step used
# which (job 841290 printed a pre-precheck step list at the top and commit
# 39d98dc from discover_carriers). Every finished step is skipped on
# re-submit, so refusing costs nothing but the step in flight.
HEAD0=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
mkdir -p "${OUT}" "${CALIB}"
LIM=(); [ "${LIMIT}" != "0" ] && LIM=(--limit "${LIMIT}")

note () {   # note STEP EVENT [DETAIL]   (a dry run writes nothing)
  [ "${DRY}" = "1" ] && return 0
  printf '{"utc":"%s","job":"%s","cell":"%s","task":"%s","step":"%s","event":"%s","detail":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_JOB_ID:-}" "${MCELL}" "${TASK}" \
    "$1" "$2" "${3:-}" >> "${LEDGER}"
}
CURRENT="setup"
trap 'rc=$?; note "${CURRENT}" abort "exit=${rc}"; echo "[abort] step ${CURRENT} exit ${rc}; re-submit the same command (or FROM=${CURRENT}) to resume"' ERR
step () {   # step NAME DESCRIPTION -- refuses if the checkout moved since the job started
  local now; now=$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
  if [ "${now}" != "${HEAD0}" ]; then
    note "$1" abort "checkout moved ${HEAD0} -> ${now}"
    echo "[abort] step ${1}: the checkout moved from ${HEAD0} to ${now} while this job was running."
    echo "        A git pull under a live job mixes tool versions between steps. Re-submit the same command:"
    echo "        every finished step is skipped, this one is redone on the new code."
    exit 1
  fi
  CURRENT="$1"; note "$1" start; echo; echo "── ${1}: ${2}"
}

# run CMD...: print, then execute unless DRY=1.
run () {
  printf '  $'; printf ' %q' "$@"; echo
  if [ "${DRY}" = "1" ]; then return 0; fi
  "$@"
}
done_ok () {   # done_ok PATH [--rows N | --limit L]
  if [ -e "$1" ]; then
    if python tools/method_status.py --check "$@" --fix > /dev/null; then return 0; fi
    echo "[rebuild] $1 was present but not complete; moved aside"
  fi
  return 1
}
level_fits () {   # level_fits KB KF -> 0 iff every prompt of both K fits this model's window
  [ "${DRY}" = "1" ] && return 0
  local files=()
  for K in $1 $2; do
    [ "${K}" = "0" ] && continue
    for S in ${SEEDS}; do files+=("${OUT}/length_K${K}_s${S}.json"); done
  done
  python tools/method_status.py --fits "${files[@]}"
}

echo "=============================================================="
echo "  method cell ${MCELL}: ${MODEL} ${SE}"
echo "  task ${TASK}   levels ${LEVELS}   K ${KS}   discovery K ${KDISC}"
echo "  n=${NQ}/seed, validation ${VPC}/class, TSLA classes '${CLASSES}', limit ${LIMIT}"
echo "  steps: ${STEPS}   skip: ${SKIP:-none}   dry: ${DRY}   code: ${HEAD0}"
echo "  job ${SLURM_JOB_ID:-none}   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================="
note cell start "${STEPS}"

# ---------------------------------------------------------------- manifest
if has manifest; then step manifest "the class space and the draws (CPU)"
  if done_ok "${QM}"; then echo "[skip] ${QM}"; else
    run python tools/build_query_manifest.py --task "${TASK}" --K "${KDISC}" \
        --demo-seeds 42,43,44 --n-queries "${NQ}" --validation-per-class "${VPC}" \
        --n-test-per-seed "${NQ}" --output "${QM}"
  fi
fi

# ---------------------------------------------------------------- calib
build_calibration () {   # build_calibration K SEED SRC
  local k="$1" seed="$2" src="$3"
  local out="${CALIB}/calibration_${TASK}_K${k}_seed${seed}_uuid.jsonl"
  if [ -f "${out}" ] && python tools/check_calibration_header.py \
         --path "${out}" --K "${k}" --seed "${seed}" --quiet; then
    echo "[skip] ${out}"
  else
    local extra=()
    if [ -n "${src}" ]; then
      extra+=(--reuse-labels "${src}")
      if [ "${k}" != "${KDISC}" ]; then extra+=(--allow-different-K); fi
    fi
    run python experiments/data_calibration.py --task "${TASK}" --K "${k}" \
        --seed "${seed}" --n-queries "${NQ}" --label-type single_token \
        --model "${MODEL}" --output "${CALIB}" --exclude-validation "${QM}" "${extra[@]}"
  fi
  # THE WINDOW, every prompt (the levels below read these records)
  local lj="${OUT}/length_K${k}_s${seed}.json"
  if [ ! -f "${lj}" ] || [ "${DRY}" = "1" ]; then
    run python tools/calibration_prompt_length.py --path "${out}" --model "${MODEL}" \
        ${GATE} --limit 0 --json-out "${lj}" || \
      echo "[window] K=${k} seed ${seed}: the prompts exceed the window; the levels using K=${k} will be skipped"
  fi
}
if has calib; then step calib "calibration prompts at K ${KS}, label-locked to K=${KDISC} seed 42 (CPU)"
  build_calibration "${KDISC}" 42 ""
  for K in ${KS}; do
    for S in ${SEEDS}; do
      if [ "${K}" = "${KDISC}" ] && [ "${S}" = "42" ]; then continue; fi
      build_calibration "${K}" "${S}" "${SRC}"
    done
  done
fi

# ---------------------------------------------------------------- precheck
# Two GPU pre-checks a model must pass before a cell is spent on it
# (REDISCOVERY 7.6): (1) the two-path noise -- one forward vs prefill +
# continue -- in THIS cell's dtype, against the receivers' own 1-ulp gate
# (tools/check_two_path_noise.py; Qwen2 fails it in bf16, passes in float32);
# (2) the capture / DLA identities on this architecture, read against the
# Llama-3.1-8B reference (tools/smoke_model_architecture.py). Minutes each. A
# failure aborts the cell; a pass leaves a .PASS marker so a re-submit skips
# it (the json alone would not do: the noise check writes it on failure too).
# Weights download themselves on first use when the node reaches the hub; on
# an offline compute node pre-fetch them on the login node
# (huggingface-cli download <model>).
if has precheck; then step precheck "two-path noise in ${DTYPE} under ${ATTN}, and the architecture identities (GPU, minutes)"
  TP="${OUT}/precheck_two_path_${DTYPE}${ATTN_SFX}.json"
  # The check is about the model (one forward vs prefill + continue) and
  # runs HF eager attention, whose float32 softmax is [heads, L, L]: on a
  # 77-class task the K=5 prompt is 7.1k tokens and that is 6 GiB per copy,
  # which took a 40 GB card down (job 843562). The smallest K of the cell
  # exercises the same two paths on the same label tokens at a fraction of
  # the length, so that is the prompt it reads.
  PK=$(echo "${KS}" | tr " " "\n" | sort -n | head -1)
  TPSRC="${CALIB}/calibration_${TASK}_K${PK}_seed42_uuid.jsonl"
  # Until 2026-09-13 this first check refused the dtype at 1 ulp ("run in
  # float32"). That predates prereg 14.0b-24, under which the receivers'
  # gate is the noise measured below at the cell's largest K; the fixed 1
  # here then only turned away cells the measured gate would carry (L31c36
  # x clinc150 in bf16: 3 ulp at K=1, job 845170, while its 4.8k-token K=2
  # prompt does not fit a 40 GB card in fp32, RESULTS 63.11). The check now
  # records the noise and REFUSES only what no gate covers: the two paths'
  # argmax differing on a live tail.
  if [ -f "${TP}.PASS" ]; then echo "[skip] ${TP} (passed)"; else
    run python tools/check_two_path_noise.py --model "${MODEL}" --calibration "${TPSRC}" ${SE} \
        --max-ulp 1000000 --require-argmax --json-out "${TP}"
    [ "${DRY}" = "1" ] || : > "${TP}.PASS"
  fi
  # The receivers hold the cache to the model's OWN two-path noise at the
  # length they run: Qwen3-14B bf16 is 1 ulp at K=1 and 2-3 ulp at K=5
  # (RESULTS 63.3), and a gate at 1 there measures the model, not the cache.
  # This measures the noise on the cell's largest K and writes cache_gate.json
  # (tools/cache_gate_from_noise: max(1, ceil(worst))), which the level and
  # test steps hand to the receivers. The gate only rises to a value this
  # measurement produced, never to one a failure suggested (prereg 14.0b-24).
  # Not a pass/fail: if the measurement cannot run (an OOM), the gate stays at 1.
  # Measured on the K_max prompt of EVERY registered seed, and the gate is the
  # worst of the three: one prompt (seed 42) gave clinc150 in bf16 a 2-ulp
  # gate and the test read's seed 43 then met a 3-ulp site on its own prompt
  # (job 845173, RESULTS 63.22) -- the noise varies by prompt, so the
  # measurement covers the prompts the receivers will actually read.
  PKMAX=$(echo "${KS}" | tr " " "\n" | sort -n | tail -1)
  CG="${OUT}/cache_gate.json"
  if [ -f "${CG}" ]; then echo "[skip] ${CG} ($(python tools/cache_gate_from_noise.py --read "${CG}" --field attn); checked against ATTN below)"; else
    TPMAXS=()
    for PSEED in 42 43 44; do
      TPMAX="${OUT}/precheck_two_path_${DTYPE}${ATTN_SFX}_K${PKMAX}_s${PSEED}.json"
      TPMAXS+=("${TPMAX}")
      if [ -f "${TPMAX}" ]; then echo "[skip] ${TPMAX}"; else
        run python tools/check_two_path_noise.py --model "${MODEL}" \
            --calibration "${CALIB}/calibration_${TASK}_K${PKMAX}_seed${PSEED}_uuid.jsonl" ${SE} \
            --max-ulp 1000000 --json-out "${TPMAX}" \
          || echo "[note] the K=${PKMAX} seed ${PSEED} two-path measurement did not run; that prompt counts as 1 ulp"
      fi
    done
    [ "${DRY}" = "1" ] || python tools/cache_gate_from_noise.py --noise-json "${TPMAXS[@]}" --K "${PKMAX}" \
        --dtype "${DTYPE}" --attn "${ATTN}" --out "${CG}"
  fi
  AR="${OUT}/precheck_architecture.txt"
  if [ -f "${AR}.PASS" ]; then echo "[skip] ${AR} (passed)"; else
    REF=$(python -c 'from tools.smoke_model_architecture import DEFAULT_MODELS; print(DEFAULT_MODELS[0])')
    MODELS=("${MODEL}"); [ "${MODEL}" != "${REF}" ] && MODELS=("${REF}" "${MODEL}")
    printf '  $ python tools/smoke_model_architecture.py --models'; printf ' %q' "${MODELS[@]}"; echo " > ${AR}"
    if [ "${DRY}" != "1" ]; then
      python tools/smoke_model_architecture.py --models "${MODELS[@]}" > "${AR}" 2>&1 || { cat "${AR}"; exit 1; }
      grep -A "${#MODELS[@]}" '^SUMMARY' "${AR}"
      : > "${AR}.PASS"
    fi
  fi
fi

# ---------------------------------------------------------------- cache gate
# The receivers' 13.5.4(2) bound: 1 ulp, or the model's own two-path noise at
# the cell's largest K when the precheck measured more (cache_gate.json above;
# read here in every invocation, so the test job sees it too).
GATE=()
if [ -f "${OUT}/cache_gate.json" ]; then
  GATE_ATTN=$(python tools/cache_gate_from_noise.py --read "${OUT}/cache_gate.json" --field attn)
  if [ "${GATE_ATTN}" != "${ATTN}" ]; then
    echo "${OUT}/cache_gate.json was measured under ${GATE_ATTN}; this run is ATTN=${ATTN}. The gate belongs to" \
         "(model, dtype, kernel): run with ATTN=${GATE_ATTN}, or move cache_gate.json aside so the precheck re-measures it."
    exit 1
  fi
  GATE_ULP=$(python tools/cache_gate_from_noise.py --read "${OUT}/cache_gate.json" --field max_ulp)
  GATE_SRC=$(python tools/cache_gate_from_noise.py --read "${OUT}/cache_gate.json" --field source)
  if [ "${GATE_ULP}" != "1" ]; then
    GATE=(--cache-gate-ulp "${GATE_ULP}" --cache-gate-source "${GATE_SRC}")
    echo "  cache gate ${GATE_ULP} ulp: ${GATE_SRC}"
  fi
fi

# ---------------------------------------------------------------- labelspace
if has labelspace; then step labelspace "this model's frozen label token space (CPU)"
  if done_ok "${LS}"; then echo "[skip] ${LS}"; else
    run python tools/build_label_space.py --uuid-jsonl "${SRC}" --query-manifest "${QM}" \
        --model "${MODEL}" --output "${LS}"
  fi
fi

# ---------------------------------------------------------------- nesting
if has nesting; then step nesting "each level's K_base draw nests inside its K_full draw (CPU)"
  while read -r KB KF; do
    [ "${KB}" = "0" ] && continue
    J="${OUT}/demo_nesting_K${KB}_in_K${KF}.json"
    if done_ok "${J}"; then echo "[skip] ${J}"; else
      run python tools/check_demo_nesting.py --query-manifest "${QM}" --task "${TASK}" \
          --k-small "${KB}" --k-large "${KF}" --json-out "${J}"
    fi
  done <<< "${LEVEL_LINES}"
fi

# ---------------------------------------------------------------- natural
if has natural; then step natural "natural readouts at every K and the native increment totals (GPU)"
  for K in ${KS}; do
    NAT="${OUT}/natural_K${K}/baseline_natural_readout_validation.npz"
    if done_ok "${NAT}"; then echo "[skip] ${NAT}"; else
      if ! level_fits "${K}" "${K}" > /dev/null; then echo "[skip] natural K=${K}: exceeds the window"; continue; fi
      run python tools/baselines/run_natural_readout.py --mode validation --model "${MODEL}" ${SE} \
          --task "${TASK}" --K "${K}" --query-manifest "${QM}" --label-space "${LS}" \
          --calibration-dir "${CALIB}" --out "${OUT}/natural_K${K}"
    fi
  done
  while read -r KB KF; do
    [ "${KB}" = "0" ] && continue
    A="${OUT}/natural_K${KB}/baseline_natural_readout_validation.npz"
    B="${OUT}/natural_K${KF}/baseline_natural_readout_validation.npz"
    J="${OUT}/natural_K${KB}_vs_K${KF}.json"
    if [ -f "${A}" ] && [ -f "${B}" ] || [ "${DRY}" = "1" ]; then
      run python tools/compare_natural_K.py --base "${A}" --more "${B}" --json-out "${J}"
    fi
  done <<< "${LEVEL_LINES}"
fi

# ---------------------------------------------------------------- carriers
if has carriers; then step carriers "section 2.4(2) carrier discovery on this model and task (GPU)"
  if done_ok "${BUNDLE}"; then echo "[skip] ${BUNDLE}"; else
    run python tools/discover_carriers.py --query-manifest "${QM}" --uuid-jsonl "${SRC}" \
        --label-space "${LS}" --model "${MODEL}" ${SE} --task "${TASK}" --K "${KDISC}" \
        --out-scores "${SCORES}" --out-fold "${FOLD}" --out-full "${BUNDLE}"
  fi
  # RESULTS 40 (zero GPU): how much the nine carrier sets -- three seeds x
  # three folds, beside the full-validation set -- agree, and along which axis
  run python tools/analyze_carrier_agreement.py --fold "${FOLD}" --full "${BUNDLE}" \
      --json-out "${OUT}/carrier_agreement.json"
fi

# ---------------------------------------------------------------- tsla vectors
VEC_OF () { echo "${OUT}/tsla_vectors_K${1}_validation_classes.json"; }
if has tsla && [ -n "${CLASSES}" ]; then step tsla "TSLA vectors (${CLASSES}) at every K_full, validation discovery rows (GPU)"
  for KF in $(echo "${LEVEL_LINES}" | awk '{print $2}' | sort -un); do
    VEC=$(VEC_OF "${KF}")
    if done_ok "${VEC}"; then echo "[skip] ${VEC}"; else
      run python tools/baselines/run_tsla_tl_on_icl.py --mode validation --model "${MODEL}" ${SE} \
          --task "${TASK}" --K "${KF}" --discovery-only --query-manifest "${QM}" \
          --label-space "${LS}" --calibration-dir "${CALIB}" --out "${OUT}/tsla_K${KF}" \
          --vectors-out "${VEC}"
    fi
  done
fi

# ---------------------------------------------------------------- fv vectors
# Function Vectors (Todd et al.) adapted to the increment setting, prereg
# 14.0b-25: one sidecar per level (K_base, K_full), built only from the extra
# demonstrations. The receivers add the `FV-K<full> a=<alpha>` arms when the
# sidecar exists, and a receiver npz that predates the sidecar is rerun
# (done_ok --needs-arm), so adding this step to a finished cell re-reads its
# levels with the new arms beside the old ones.
# FV_OF KB KF -> the level's sidecar (no trailing comment on the next line:
# tools/list_experiments.py reads one-line functions by their closing brace)
FV_OF () { echo "${OUT}/fv_vectors_K${2}_into_K${1}.json"; }
if has fv; then step fv "Function Vectors from each level's extra demonstrations (AtP screen, exact CIE; GPU)"
  while read -r KB KF; do
    if ! level_fits "${KB}" "${KF}"; then echo "[skip] fv K=${KB}->${KF}: the prompts exceed this model's window"; continue; fi
    VEC=$(FV_OF "${KB}" "${KF}")
    if done_ok "${VEC}"; then echo "[skip] ${VEC}"; else
      run python tools/baselines/run_fv_increment.py --query-manifest "${QM}" --label-space "${LS}" \
          --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} --task "${TASK}" \
          --K-base "${KB}" --K-full "${KF}" ${FVHEADS:+--n-heads ${FVHEADS}} "${LIM[@]}" --out "${VEC}"
    fi
  done <<< "${LEVEL_LINES}"
fi
fv_args () {   # fv_args KB KF -> --fv-vectors ... when the level's sidecar exists
  local vec; vec=$(FV_OF "$1" "$2")
  if [ -f "${vec}" ] || { [ "${DRY}" = "1" ] && has fv; }; then echo "--fv-vectors ${vec}"; fi
}

# ---------------------------------------------------------------- tv vectors
# Task Vectors (Hendel et al.) adapted to the increment setting, prereg
# 14.0b-25: one sidecar per level with theta at every candidate layer. The
# validation receivers run every candidate (plus the descriptive m5 family),
# select_tv_layer picks the layer by validation accuracy, and the test read
# runs that layer alone.
# TV_OF KB KF -> the level's sidecar; TVL_OF KB KF -> the layer chosen on validation
TV_OF () { echo "${OUT}/tv_vectors_K${2}_into_K${1}.json"; }
TVL_OF () { echo "${OUT}/tv_layer_K${2}_into_K${1}.json"; }
if has tv; then step tv "Task Vectors from each level's extra demonstrations at the candidate layers (GPU, a few forwards)"
  while read -r KB KF; do
    if ! level_fits "${KB}" "${KF}"; then echo "[skip] tv K=${KB}->${KF}: the prompts exceed this model's window"; continue; fi
    VEC=$(TV_OF "${KB}" "${KF}")
    if done_ok "${VEC}"; then echo "[skip] ${VEC}"; else
      run python tools/baselines/run_tv_increment.py --query-manifest "${QM}" --label-space "${LS}" \
          --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} --task "${TASK}" \
          --K-base "${KB}" --K-full "${KF}" --out "${VEC}"
    fi
  done <<< "${LEVEL_LINES}"
fi
tv_args () {   # tv_args KB KF SPLIT -> --tv-vectors ... (+ --tv-m5 on validation; --tv-layers <chosen> on test)
  local vec; vec=$(TV_OF "$1" "$2"); local sel; sel=$(TVL_OF "$1" "$2")
  if [ -f "${vec}" ] || { [ "${DRY}" = "1" ] && has tv; }; then
    if [ "$3" = "validation" ]; then echo "--tv-vectors ${vec} --tv-m5"
    elif [ -f "${sel}" ]; then echo "--tv-vectors ${vec} --tv-layers $(python tools/select_tv_layer.py --read "${sel}")"
    elif [ "${DRY}" = "1" ]; then echo "--tv-vectors ${vec} --tv-layers CHOSEN_ON_VALIDATION"
    fi
  fi
}
tv_need () {   # tv_need KB KF SPLIT -> the main-family TV arm a receiver npz must carry
  local vec; vec=$(TV_OF "$1" "$2"); local sel; sel=$(TVL_OF "$1" "$2"); local L="?"
  if [ "$3" = "validation" ]; then [ -f "${vec}" ] && L=$(python tools/select_tv_layer.py --first-layer "${vec}")
  else [ -f "${sel}" ] && L=$(python tools/select_tv_layer.py --read "${sel}"); fi
  echo "TV-K$2 L=${L} a=1"
}
tv_select () {   # tv_select KB KF SPLIT PREFIX READOUT -> choose the layer from the registered validation run's readout
  [ "$3" = "validation" ] && [ -z "$4" ] && [ -n "$(tv_args "$1" "$2" "$3")" ] || return 0
  run python tools/select_tv_layer.py --readout "$5" --K-full "$2" --json-out "$(TVL_OF "$1" "$2")"
}

# ---------------------------------------------------------------- icv vectors
# In-Context Vectors (Liu et al.) adapted to the increment setting, prereg
# 14.0b-25: one direction per level from the extra demonstrations' (x, xy)
# pairs. The validation receivers run the lambda grid (one hooked prefix
# cache per lambda), select_icv_lambda picks lambda by validation NLL, and
# the test read runs that lambda alone.
# ICV_OF KB KF -> the level's sidecar; ICVL_OF KB KF -> the lambda chosen on validation
ICV_OF () { echo "${OUT}/icv_vectors_K${2}_into_K${1}.json"; }
ICVL_OF () { echo "${OUT}/icv_lambda_K${2}_into_K${1}.json"; }
if has icv; then step icv "In-Context Vectors from each level's extra demonstrations (GPU, 2 short forwards per demonstration)"
  while read -r KB KF; do
    if ! level_fits "${KB}" "${KF}"; then echo "[skip] icv K=${KB}->${KF}: the prompts exceed this model's window"; continue; fi
    VEC=$(ICV_OF "${KB}" "${KF}")
    if done_ok "${VEC}"; then echo "[skip] ${VEC}"; else
      run python tools/baselines/run_icv_increment.py --query-manifest "${QM}" --label-space "${LS}" \
          --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} --task "${TASK}" \
          --K-base "${KB}" --K-full "${KF}" "${LIM[@]}" --out "${VEC}"
    fi
  done <<< "${LEVEL_LINES}"
fi
icv_args () {   # icv_args KB KF SPLIT -> --icv-vectors ... (the whole grid on validation; --icv-lambdas <chosen> on test)
  local vec; vec=$(ICV_OF "$1" "$2"); local sel; sel=$(ICVL_OF "$1" "$2")
  if [ -f "${vec}" ] || { [ "${DRY}" = "1" ] && has icv; }; then
    if [ "$3" = "validation" ]; then echo "--icv-vectors ${vec}"
    elif [ -f "${sel}" ]; then echo "--icv-vectors ${vec} --icv-lambdas $(python tools/select_icv_lambda.py --read "${sel}")"
    elif [ "${DRY}" = "1" ]; then echo "--icv-vectors ${vec} --icv-lambdas CHOSEN_ON_VALIDATION"
    fi
  fi
}
icv_need () {   # icv_need KB KF SPLIT -> the ICV arm a receiver npz must carry
  local vec; vec=$(ICV_OF "$1" "$2"); local sel; sel=$(ICVL_OF "$1" "$2"); local L="?"
  if [ "$3" = "validation" ]; then [ -f "${vec}" ] && L=$(python tools/select_icv_lambda.py --first-lambda "${vec}")
  else [ -f "${sel}" ] && L=$(python tools/select_icv_lambda.py --read "${sel}"); fi
  echo "ICV-K$2 a=${L}"
}
icv_select () {   # icv_select KB KF SPLIT PREFIX READOUT -> choose lambda from the registered validation run's readout
  [ "$3" = "validation" ] && [ -z "$4" ] && [ -n "$(icv_args "$1" "$2" "$3")" ] || return 0
  run python tools/select_icv_lambda.py --readout "$5" --K-full "$2" --json-out "$(ICVL_OF "$1" "$2")"
}

# ---------------------------------------------------------------- i2cl vectors
# I2CL (Li et al.) adapted to the increment setting, prereg 14.0b-25: context
# vectors from the extra demonstrations and 4 L scalars calibrated on them as
# pseudo-queries after the K_base prefix. No selection: the calibrated
# coefficients are the strengths, so `I2CL-K<full> a=1` is the arm on every
# split and `a=0` the gate.
# I2CL_OF KB KF -> the level's npz sidecar
I2CL_OF () { echo "${OUT}/i2cl_vectors_K${2}_into_K${1}.npz"; }
if has i2cl; then step i2cl "I2CL context vectors + ${I2CL_EPOCHS}-epoch noisy self-calibration per level (GPU, the costly baseline)"
  while read -r KB KF; do
    if ! level_fits "${KB}" "${KF}"; then echo "[skip] i2cl K=${KB}->${KF}: the prompts exceed this model's window"; continue; fi
    VEC=$(I2CL_OF "${KB}" "${KF}")
    if done_ok "${VEC}"; then echo "[skip] ${VEC}"; else
      run python tools/baselines/run_i2cl_increment.py --query-manifest "${QM}" --label-space "${LS}" \
          --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} --task "${TASK}" \
          --K-base "${KB}" --K-full "${KF}" --epochs "${I2CL_EPOCHS}" --grad-bs "${I2CL_BS}" "${LIM[@]}" --out "${VEC}"
    fi
  done <<< "${LEVEL_LINES}"
fi
i2cl_args () {   # i2cl_args KB KF -> --i2cl-vectors ... when the level's npz exists
  local vec; vec=$(I2CL_OF "$1" "$2")
  if [ -f "${vec}" ] || { [ "${DRY}" = "1" ] && has i2cl; }; then echo "--i2cl-vectors ${vec}"; fi
}

# ---------------------------------------------------------------- the levels
tsla_args () {   # tsla_args KF -> --tsla-vectors ... --tsla-classes ... when the sidecar exists
  local vec; vec=$(VEC_OF "$1")
  if [ -n "${CLASSES}" ] && { [ -f "${vec}" ] || [ "${DRY}" = "1" ]; }; then
    echo "--tsla-vectors ${vec} --tsla-classes ${CLASSES}"
  fi
}
run_level () {   # run_level KB KF SPLIT WRAP PREFIX FREEZE_ARGS...
  local KB="$1" KF="$2" SPLIT="$3" WRAP="$4" PREFIX="$5"; shift 5
  local FZ=("$@")
  if ! level_fits "${KB}" "${KF}"; then
    note "level_${KB}_${KF}" skipped "window"
    echo "[skip] level K=${KB}->${KF}: the prompts exceed this model's window (see ${OUT}/length_K*.json)"
    return 0
  fi
  local TS; TS=$(tsla_args "${KF}")
  local FVA; FVA=$(fv_args "${KB}" "${KF}")
  local TVA; TVA=$(tv_args "${KB}" "${KF}" "${SPLIT}")
  local ICVA; ICVA=$(icv_args "${KB}" "${KF}" "${SPLIT}")
  local I2A; I2A=$(i2cl_args "${KB}" "${KF}")
  local NEED=()
  [ -n "${FVA}" ] && NEED+=(--needs-arm "FV-K${KF} a=1")
  [ -n "${TVA}" ] && NEED+=(--needs-arm "$(tv_need "${KB}" "${KF}" "${SPLIT}")")
  [ -n "${ICVA}" ] && NEED+=(--needs-arm "$(icv_need "${KB}" "${KF}" "${SPLIT}")")
  [ -n "${I2A}" ] && NEED+=(--needs-arm "I2CL-K${KF} a=1")
  if [ "${KB}" = "0" ]; then
    local NPZ="${OUT}/${PREFIX}k0_receiver_K${KF}_${SPLIT}.npz"
    if done_ok "${NPZ}" --limit "${LIMIT}" "${NEED[@]}"; then echo "[skip] ${NPZ}"; else
      local APP=(); [ -f "${NPZ}" ] && APP=(--append)   # a file lacking arms keeps its arms, gains the missing ones
      run ${WRAP} python tools/run_k0_receiver.py "${GATE[@]}" --carrier-bundle "${BUNDLE}" --query-manifest "${QM}" \
          --label-space "${LS}" --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} \
          --task "${TASK}" --K "${KF}" ${TS} ${FVA} ${TVA} ${ICVA} ${I2A} "${LIM[@]}" "${FZ[@]}" "${APP[@]}" --out "${NPZ}"
    fi
    run python tools/analyze_k0_receiver.py --npz "${NPZ}" --json-out "${NPZ%.npz}_readout.json"
    tv_select "${KB}" "${KF}" "${SPLIT}" "${PREFIX}" "${NPZ%.npz}_readout.json"
    icv_select "${KB}" "${KF}" "${SPLIT}" "${PREFIX}" "${NPZ%.npz}_readout.json"
    return 0
  fi
  local INC="${OUT}/${PREFIX}k${KF}_increment_into_K${KB}_${SPLIT}.npz"
  local DIRECT="${OUT}/${PREFIX}carrier_direct_K${KB}base_${SPLIT}.npz"
  if done_ok "${INC}" --limit "${LIMIT}" "${NEED[@]}"; then echo "[skip] ${INC}"; else
    local APP=(); [ -f "${INC}" ] && APP=(--append)
    run ${WRAP} python tools/run_k10_increment.py "${GATE[@]}" --carrier-bundle "${BUNDLE}" --query-manifest "${QM}" \
        --label-space "${LS}" --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} \
        --task "${TASK}" --K-base "${KB}" --K-full "${KF}" ${TS} ${FVA} ${TVA} ${ICVA} ${I2A} "${LIM[@]}" "${FZ[@]}" "${APP[@]}" --out "${INC}"
  fi
  run python tools/analyze_k0_receiver.py --npz "${INC}" --json-out "${INC%.npz}_readout.json"
  tv_select "${KB}" "${KF}" "${SPLIT}" "${PREFIX}" "${INC%.npz}_readout.json"
  icv_select "${KB}" "${KF}" "${SPLIT}" "${PREFIX}" "${INC%.npz}_readout.json"
  if done_ok "${DIRECT}" --limit "${LIMIT}"; then echo "[skip] ${DIRECT}"; else
    run ${WRAP} python tools/probe_carrier_direct_response.py --carrier-bundle "${BUNDLE}" \
        --query-manifest "${QM}" --label-space "${LS}" --calibration-dir "${CALIB}" \
        --model "${MODEL}" ${SE} --task "${TASK}" --K-base "${KB}" --K-full "${KF}" \
        "${LIM[@]}" "${FZ[@]}" --out "${DIRECT}"
  fi
  run python tools/analyze_direct_write_accuracy.py --direct "${DIRECT}" --final "${INC}" \
      --json-out "${OUT}/${PREFIX}direct_write_accuracy_K${KB}base_${SPLIT}.json"
  run python tools/analyze_carrier_absorption.py --direct "${DIRECT}" --final "${INC}" \
      --arm-sel "selective TL K${KF} memory" --arm-base "K${KB}-offset natural" \
      --json-out "${OUT}/${PREFIX}carrier_absorption_K${KB}base_${SPLIT}.json"
}
level_step_name () { if [ "$1" = "0" ]; then echo k0; elif [ "$1" -lt "$2" ] && [ "$2" -le 5 ]; then echo k2; else echo k10; fi; }
while read -r KB KF; do
  NAME=$(level_step_name "${KB}" "${KF}")
  if has "${NAME}"; then
    step "${NAME}" "level K=${KB} receiver reading the K=${KF} memory, validation (GPU)"
    run_level "${KB}" "${KF}" validation "" ""
    for N in ${TOPN}; do
      echo "   top-${N} cut: the first ${N} heads of the bundle's ranking (TSLA-matched head count; artifacts top${N}_*)"
      run_level "${KB}" "${KF}" validation "" "top${N}_" --top-n "${N}"
    done
  fi
done <<< "${LEVEL_LINES}"

# ---------------------------------------------------------------- ceiling
if has ceiling; then step ceiling "the carriers' direct write on the native K_full prompt (GPU, all heads)"
  for KF in $(echo "${LEVEL_LINES}" | awk '{print $2}' | sort -un); do
    if ! level_fits "${KF}" "${KF}" > /dev/null; then echo "[skip] ceiling K=${KF}: exceeds the window"; continue; fi
    DLA="${OUT}/tl_class_attribution_K${KF}.npz"
    if done_ok "${DLA}" --limit "${LIMIT}"; then echo "[skip] ${DLA}"; else
      run python tools/probe_tl_class_attribution.py --query-manifest "${QM}" --label-space "${LS}" \
          --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} --task "${TASK}" --K "${KF}" \
          "${LIM[@]}" --out "${DLA}"
    fi
    run python tools/analyze_direct_write_accuracy.py --dla "${DLA}" --carrier-bundle "${BUNDLE}" \
        "${TOPN_ARG[@]}" --json-out "${OUT}/direct_write_ceiling_K${KF}.json"
  done
fi

# ---------------------------------------------------------------- summary
if has summary; then step summary "the tables (zero GPU)"
  run python tools/summarize_method_cell.py --cell "${OUT}" --levels "${LEVELS}" \
      --split validation --json-out "${OUT}/summary_validation.json" \
      --model "${MODEL}" --method "${METHOD_NAME}" --task "${TASK}" ${TOPN:+--topn ${TOPN%% *}}
  for N in ${TOPN}; do
    run python tools/summarize_method_cell.py --cell "${OUT}" --levels "${LEVELS}" \
        --split validation --variant "top${N}_" --json-out "${OUT}/summary_validation_top${N}.json"
  done
fi

# ---------------------------------------------------------------- test (opt-in)
if has test; then
  if [ "${TEST}" != "1" ]; then
    echo "[abort] the test read is opt-in: TEST=1 ... method_cell.sh ${MTAG_BASE} test"; exit 1
  fi
  # The test read uses the configuration frozen on validation: the cell's
  # manifest, label space, carrier bundle and, when CLASSES is set, each
  # K_full's TSLA sidecar. Job 841344 ran `test` on a cell whose validation
  # steps had not run and died inside the lock opener; say so up front.
  MISSING=()
  for f in "${QM}" "${LS}" "${BUNDLE}"; do [ -f "${f}" ] || MISSING+=("${f}"); done
  if [ -n "${CLASSES}" ]; then
    for KF in $(echo "${LEVEL_LINES}" | awk '{print $2}' | sort -un); do
      v=$(VEC_OF "${KF}"); [ -f "${v}" ] || MISSING+=("${v}")
    done
  fi
  if [ "${#MISSING[@]}" != "0" ] && [ "${DRY}" != "1" ]; then
    echo "[abort] the test read needs the validation cell first; missing:"; printf '    %s\n' "${MISSING[@]}"
    echo "        run   sbatch script/lsu1.sh bash script/method_cell.sh ${MTAG_BASE}   and only then   TEST=1 ... ${MTAG_BASE} test"
    exit 1
  fi
  step test "the one-shot test_seed read of every level (UNSAFE wrapper, prereg 14.0b-23)"
  UDIR="${OUT}/UNSAFE"; mkdir -p "${UDIR}"
  FREEZE="${UDIR}/UNSAFE_freeze_manifest.json"
  if [ ! -f "${SPEC_FREEZE}" ] && [ "${DRY}" != "1" ]; then
    echo "[abort] ${SPEC_FREEZE} not found: the lock opener binds the baseline spec freeze; SPEC_FREEZE=... to point at it"; exit 1
  fi
  if done_ok "${FREEZE}"; then echo "[skip] ${FREEZE}"; else
    run python tools/UNSAFE_open_test_lock.py --gamma-group 1.0 --query-manifest "${QM}" \
        --carriers "${BUNDLE}" --label-space "${LS}" --spec-freeze "${SPEC_FREEZE}" \
        --out-dir "${UDIR}" --i-am-bypassing-13-4-4
  fi
  WRAP="python tools/UNSAFE_run_test_forward.py --i-am-bypassing-13-4-4 --runner"
  # the wrapper takes the runner name after --runner and the runner's own
  # arguments after --; run_level supplies `python tools/<runner>.py`, so the
  # wrapper form is assembled here per level
  test_level () {
    local KB="$1" KF="$2" PREFIX="$3"; shift 3   # PREFIX "" = the registered arm; "top30_" with --top-n 30 = a cut
    if ! level_fits "${KB}" "${KF}"; then echo "[skip] test level K=${KB}->${KF}: window"; return 0; fi
    local TS; TS=$(tsla_args "${KF}")
    local FVA; FVA=$(fv_args "${KB}" "${KF}")
    local TVA; TVA=$(tv_args "${KB}" "${KF}" test_seed)
    local ICVA; ICVA=$(icv_args "${KB}" "${KF}" test_seed)
    local I2A; I2A=$(i2cl_args "${KB}" "${KF}")
    local NEED=()
    [ -n "${FVA}" ] && NEED+=(--needs-arm "FV-K${KF} a=1")
    [ -n "${TVA}" ] && NEED+=(--needs-arm "$(tv_need "${KB}" "${KF}" test_seed)")
    [ -n "${ICVA}" ] && NEED+=(--needs-arm "$(icv_need "${KB}" "${KF}" test_seed)")
    [ -n "${I2A}" ] && NEED+=(--needs-arm "I2CL-K${KF} a=1")
    [ -z "${I2A}" ] && echo "   [note] no I2CL sidecar for K=${KB}->${KF} ($(I2CL_OF "${KB}" "${KF}")): this read carries no I2CL arms; run the i2cl step on validation first"
    [ -z "${FVA}" ] && echo "   [note] no FV sidecar for K=${KB}->${KF} ($(FV_OF "${KB}" "${KF}")): this read carries no FV arms; run the fv step on validation first"
    if [ -z "${TVA}" ]; then
      if [ -f "$(TV_OF "${KB}" "${KF}")" ]; then echo "   [note] TV sidecar present but no layer chosen on validation ($(TVL_OF "${KB}" "${KF}")): this read carries no TV arms; rerun the validation level first"
      else echo "   [note] no TV sidecar for K=${KB}->${KF} ($(TV_OF "${KB}" "${KF}")): this read carries no TV arms; run the tv step on validation first"; fi
    fi
    if [ -z "${ICVA}" ]; then
      if [ -f "$(ICV_OF "${KB}" "${KF}")" ]; then echo "   [note] ICV sidecar present but no lambda chosen on validation ($(ICVL_OF "${KB}" "${KF}")): this read carries no ICV arms; rerun the validation level first"
      else echo "   [note] no ICV sidecar for K=${KB}->${KF} ($(ICV_OF "${KB}" "${KF}")): this read carries no ICV arms; run the icv step on validation first"; fi
    fi
    local FZ=(--split test_seed --freeze-manifest "${FREEZE}" "$@")
    if [ "${KB}" = "0" ]; then
      local NPZ="${UDIR}/UNSAFE_${PREFIX}k0_receiver_K${KF}_test_seed.npz"
      if done_ok "${NPZ}" --limit "${LIMIT}" "${NEED[@]}"; then echo "[skip] ${NPZ}"; else
      local APP=(); [ -f "${NPZ}" ] && APP=(--append)   # a file lacking arms keeps its arms, gains the missing ones
        run ${WRAP} k0_receiver -- "${GATE[@]}" --carrier-bundle "${BUNDLE}" --query-manifest "${QM}" \
            --label-space "${LS}" --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} \
            --task "${TASK}" --K "${KF}" ${TS} ${FVA} ${TVA} ${ICVA} ${I2A} "${LIM[@]}" "${FZ[@]}" "${APP[@]}" --out "${NPZ}"
      fi
      run python tools/analyze_k0_receiver.py --npz "${NPZ}" --json-out "${NPZ%.npz}_readout.json"
      return 0
    fi
    local INC="${UDIR}/UNSAFE_${PREFIX}k${KF}_increment_into_K${KB}_test_seed.npz"
    local DIRECT="${UDIR}/UNSAFE_${PREFIX}carrier_direct_K${KB}base_test_seed.npz"
    if done_ok "${INC}" --limit "${LIMIT}" "${NEED[@]}"; then echo "[skip] ${INC}"; else
    local APP=(); [ -f "${INC}" ] && APP=(--append)
      run ${WRAP} k10_increment -- "${GATE[@]}" --carrier-bundle "${BUNDLE}" --query-manifest "${QM}" \
          --label-space "${LS}" --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} \
          --task "${TASK}" --K-base "${KB}" --K-full "${KF}" ${TS} ${FVA} ${TVA} ${ICVA} ${I2A} "${LIM[@]}" "${FZ[@]}" "${APP[@]}" --out "${INC}"
    fi
    run python tools/analyze_k0_receiver.py --npz "${INC}" --json-out "${INC%.npz}_readout.json"
    if done_ok "${DIRECT}" --limit "${LIMIT}"; then echo "[skip] ${DIRECT}"; else
      run ${WRAP} carrier_direct -- --carrier-bundle "${BUNDLE}" --query-manifest "${QM}" \
          --label-space "${LS}" --calibration-dir "${CALIB}" --model "${MODEL}" ${SE} \
          --task "${TASK}" --K-base "${KB}" --K-full "${KF}" "${LIM[@]}" "${FZ[@]}" --out "${DIRECT}"
    fi
    run python tools/analyze_direct_write_accuracy.py --direct "${DIRECT}" --final "${INC}" \
        --json-out "${UDIR}/UNSAFE_${PREFIX}direct_write_accuracy_K${KB}base_test_seed.json"
    run python tools/analyze_carrier_absorption.py --direct "${DIRECT}" --final "${INC}" \
        --arm-sel "selective TL K${KF} memory" --arm-base "K${KB}-offset natural" \
        --json-out "${UDIR}/UNSAFE_${PREFIX}carrier_absorption_K${KB}base_test_seed.json"
  }
  while read -r KB KF; do
    test_level "${KB}" "${KF}" ""
    for N in ${TOPN}; do test_level "${KB}" "${KF}" "top${N}_" --top-n "${N}"; done
  done <<< "${LEVEL_LINES}"
  # the method's own upper bound on test: the carriers' direct write on the
  # native K_full prompt (all heads), when `ceiling` is among the steps
  if has ceiling; then
    for KF in $(echo "${LEVEL_LINES}" | awk '{print $2}' | sort -un); do
      if ! level_fits "${KF}" "${KF}" > /dev/null; then echo "[skip] test ceiling K=${KF}: window"; continue; fi
      DLA="${UDIR}/UNSAFE_tl_class_attribution_K${KF}_test_seed.npz"
      if done_ok "${DLA}" --limit "${LIMIT}"; then echo "[skip] ${DLA}"; else
        run ${WRAP} tl_class_attribution -- --query-manifest "${QM}" --label-space "${LS}" \
            --calibration-dir "${CALIB}" --carrier-bundle "${BUNDLE}" --model "${MODEL}" ${SE} \
            --task "${TASK}" --K "${KF}" "${LIM[@]}" --split test_seed \
            --freeze-manifest "${FREEZE}" --out "${DLA}"
      fi
      run python tools/analyze_direct_write_accuracy.py --dla "${DLA}" --carrier-bundle "${BUNDLE}" \
          "${TOPN_ARG[@]}" --json-out "${UDIR}/UNSAFE_direct_write_ceiling_K${KF}_test_seed.json"
    done
  fi
  run python tools/summarize_method_cell.py --cell "${OUT}" --levels "${LEVELS}" \
      --split test_seed --json-out "${OUT}/summary_test_seed.json" \
      --model "${MODEL}" --method "${METHOD_NAME}" --task "${TASK}" ${TOPN:+--topn ${TOPN%% *}}
  for N in ${TOPN}; do
    run python tools/summarize_method_cell.py --cell "${OUT}" --levels "${LEVELS}" \
        --split test_seed --variant "top${N}_" --json-out "${OUT}/summary_test_seed_top${N}.json"
  done
fi

note cell done "${STEPS}"
echo
echo "done: ${MCELL}  $(date -u +%Y-%m-%dT%H:%M:%SZ)   ledger ${LEDGER}"
