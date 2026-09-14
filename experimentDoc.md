# Running a method cell

## Environment

Python 3.12.13 and PyTorch 2.7.1 are fixed. The CUDA build of PyTorch is
free: install 2.7.1 from whichever wheel index matches the card (cu121,
cu124, cu126 all work).

```
conda create -n icl python=3.12.13 && conda activate icl
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126   # or /whl/cu121, /whl/cu124
huggingface-cli login                # meta-llama/* checkpoints are gated
export HF_HOME=/path/with/space      # model weights and the datasets cache
```

No flash-attention, triton or cuDNN setup is needed.

The remaining packages, exactly as pinned in `requirements.txt`
(`pip install -r requirements.txt` installs them):

```
transformers==4.52.3
tokenizers==0.21.4
safetensors==0.7.0
huggingface_hub==0.36.2
numpy==2.3.5
scipy==1.17.1
scikit-learn==1.8.0
datasets==4.7.0
lm_eval==0.4.11
```

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

## The generated tasks (no download)

The three synthetic datasets ship as shared-bank tasks -- one fixed hidden
concept per task, a training bank the demonstrations are drawn from and a
disjoint test pool: `synthetic_mlp_bank_per_class` and
`synthetic_linear_bank_per_class` (six classes each, generated in memory
from `function_seed=0`, 60 bank rows per class) and `monk_bank_r1_per_class`
(Monk-1, two classes, the UCI files under `tasks/monk/`, 62 bank rows per
class). Nothing is downloaded.

These tasks need more demonstrations than the text tasks before the model
learns anything from them: on Llama-3.1-8B the linear concept reads at
chance with 5 demonstrations per class and only at 30 % with 10, Monk-1 at
59 % with 10 (chance 50), so the default levels of the text tasks (`0:5 2:5
5:10`) leave nothing for a memory to carry. Run them at `10:20 20:40` with
the carriers discovered at K = 20, under a tag of your own with the model
named explicitly (any unregistered tag works when `MODEL=` and `DTYPE=` are
given; the tag keeps the cell's directories apart from a K = 5 run of the
same task). The prompts stay short (K = 40 is 3--6k tokens), so a pair takes
hours rather than a day.

```
mkdir -p logs && CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=synthetic_mlp_bank_per_class LEVELS="10:20 20:40" KDISC=20 bash script/method_cell.sh L31k40 \
  && MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=synthetic_mlp_bank_per_class LEVELS="10:20 20:40" KDISC=20 TEST=1 bash script/method_cell.sh L31k40 "ceiling test"
' > logs/synmlp_run.out 2>&1 &
```

The linear one:

```
mkdir -p logs && CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=synthetic_linear_bank_per_class LEVELS="10:20 20:40" KDISC=20 bash script/method_cell.sh L31k40 \
  && MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=synthetic_linear_bank_per_class LEVELS="10:20 20:40" KDISC=20 TEST=1 bash script/method_cell.sh L31k40 "ceiling test"
' > logs/synlin_run.out 2>&1 &
```

Monk-1 has two classes, so the default 4 validation queries per class
give a seed only 8; `VPC=12` gives 24, and the bank keeps 50 per class for
the demonstrations, enough for K = 40:

```
mkdir -p logs && CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=monk_bank_r1_per_class VPC=12 LEVELS="10:20 20:40" KDISC=20 bash script/method_cell.sh L31k40 \
  && MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=monk_bank_r1_per_class VPC=12 LEVELS="10:20 20:40" KDISC=20 TEST=1 bash script/method_cell.sh L31k40 "ceiling test"
' > logs/monk1_run.out 2>&1 &
```

The cells land in `results/method/L31k40_synmb`, `L31k40_synlb` and
`L31k40_monkb1`, their calibration prompts under `data/method/L31k40/`. The
levels are pairs `K_base:K_full` with nested draws, so any pair the bank
supports can be given the same way; `KDISC` is the K at which the carriers
are discovered (the text tasks use their first level's `K_full`).
