# Running a method cell

A *cell* is one (model, task) pair. `script/method_cell.sh` runs the whole
pipeline for it: the query manifest and calibration prompts, the two
pre-checks, the label space, the natural readouts, carrier discovery, the
TSLA / FV / TV / ICV / I2CL baselines, the receivers at every level, and the
summary. A second invocation with `TEST=1 ... "ceiling test"` reads the test
split once and adds the direct-write ceiling. Every step is skipped when its
artifacts are already on disk, so an interrupted run is resumed by repeating
the same command.

## Environment

Python 3.12, one CUDA card with at least 40 GB for the float32 cells (the
bfloat16 cells fit in 24 GB with `ATTN=sdpa`).

```
conda create -n icl python=3.12 && conda activate icl
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121   # pick the index matching the card's CUDA; torch 2.5 to 2.7 both work
pip install -r requirements.txt
huggingface-cli login                # meta-llama/* checkpoints are gated
export HF_HOME=/path/with/space      # model weights and the datasets cache (tasks download from the Hub on first use)
```

`requirements.txt` pins every package the code imports; `env/` holds the
full `pip freeze` of the two machines the results were produced on. No
flash-attention, triton or cuDNN setup is needed: models are loaded with
`attn_implementation` eager (default) or sdpa (`ATTN=sdpa`), both built into
torch.

`results/` and `data/` are created by the driver (`results/method/<cell>/`
for artifacts, `data/method/<tag>/<task>/` for calibration prompts); link
them to a large disk if you like. `results/baseline_spec_freeze_v2.json`
must be present for the test step (it is in the repository).

## The two commands

Llama-3.1-8B on banking77 (K = 1 receiver reading a K = 2 memory; float32):

```
mkdir -p logs && CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  TASK=banking77_per_class LEVELS="0:2 1:2" KDISC=2 DTYPE=float32 bash script/method_cell.sh L31c36 \
  && TASK=banking77_per_class LEVELS="0:2 1:2" KDISC=2 DTYPE=float32 TEST=1 bash script/method_cell.sh L31c36 "ceiling test"
' > logs/bk77_run.out 2>&1 &
```

Qwen3-8B-Base on TREC-fine (K = 0 and K = 2 receivers reading a K = 5 memory;
the tag's default is float32):

```
mkdir -p logs && CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  LEVELS="0:5 2:5" bash script/method_cell.sh Q3c36 \
  && LEVELS="0:5 2:5" TEST=1 bash script/method_cell.sh Q3c36 "ceiling test"
' > logs/Q3_run.out 2>&1 &
```

The second command of each pair starts only if the first finished cleanly.
Each pair takes about a day on one card; the I2CL calibration is the slow
step (about half an hour per seed per level).

Model tags (`tools/model_tags.py`): `L31c36` Llama-3.1-8B, `L2c36`
Llama-2-7B, `SEc36` Llama-2-7B + SelfExtend, `Q25c36` Qwen2.5-7B, `Q3c36`
Qwen3-8B-Base, `Q34c36` Qwen3-4B-Base, `Q314c36` Qwen3-14B-Base. Tasks:
`trec_fine_per_class` (default), `banking77_per_class`, `clinc150_per_class`,
`dbpedia14_per_class`, `yahoo_answers_per_class`, `yelp_full_per_class`.
Knobs: `LEVELS` (base:full pairs), `KDISC` (the K carriers are discovered
at), `DTYPE`, `ATTN`, `TOPN` (carrier-count cut beside the registered
top-8; default 30), `SKIP` / `FROM` (step selection), `DRY=1` (print every
command, run nothing).

## Reading the output

```
tail -f logs/bk77_run.out            # progress
grep -n '^── ' logs/bk77_run.out     # the steps, in order
```

The last section of each run (`4. the paper table's rows`) prints one LaTeX
row per level for two tables: A with the natural baseline, the full-prompt
reference, the direct-write upper bound, TSLA and ours; B with the four
adapted baselines beside them. The validation run's rows are in sample; the
rows to report are the test run's. The numbers behind them are in
`results/method/<cell>/summary_test_seed.json` and the per-level readout
files next to it.
