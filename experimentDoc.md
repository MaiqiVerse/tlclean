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
  && TASK=banking77_per_class LEVELS="0:2 1:2" KDISC=2 DTYPE=float32 SPEC_FREEZE=none TEST=1 bash script/method_cell.sh L31c36 "ceiling test"
' > logs/bk77_run.out 2>&1 &
```

Qwen3-8B-Base on TREC-fine (K = 0 and K = 2 receivers reading a K = 5 memory;
the tag's default is float32):

```
mkdir -p logs && CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  LEVELS="0:5 2:5" bash script/method_cell.sh Q3c36 \
  && LEVELS="0:5 2:5" SPEC_FREEZE=none TEST=1 bash script/method_cell.sh Q3c36 "ceiling test"
' > logs/Q3_run.out 2>&1 &
```

The second command of each pair starts only if the first finished cleanly.
Each pair takes about a day on one card; the I2CL calibration is the slow
step (about half an hour per seed per level).

`SPEC_FREEZE=none` in the test half: the test read binds a freeze manifest,
and in the full tree that manifest also registers a baseline spec freeze
whose validation reads documents this tree does not carry. With `none` the
lock opener records the literal instead of a file, the lock refuses it by
name, and the UNSAFE wrapper discards that one blocker and writes it into
the cell's `UNSAFE/` record -- the same thing it does with the spec-stage
blocker in the full tree. Everything else the lock checks (the query
manifest, carriers, label space and gammas are hashed when the lock is
opened and re-checked on every read) stays fatal. Without it the test half
stops at `[abort] results/baseline_spec_freeze_v2.json not found`.

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
  && MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=synthetic_mlp_bank_per_class LEVELS="10:20 20:40" KDISC=20 SPEC_FREEZE=none TEST=1 bash script/method_cell.sh L31k40 "ceiling test"
' > logs/synmlp_run.out 2>&1 &
```

The linear one:

```
mkdir -p logs && CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=synthetic_linear_bank_per_class LEVELS="10:20 20:40" KDISC=20 bash script/method_cell.sh L31k40 \
  && MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=synthetic_linear_bank_per_class LEVELS="10:20 20:40" KDISC=20 SPEC_FREEZE=none TEST=1 bash script/method_cell.sh L31k40 "ceiling test"
' > logs/synlin_run.out 2>&1 &
```

Monk-1 has two classes, so the default 4 validation queries per class
give a seed only 8; `VPC=12` gives 24, and the bank keeps 50 per class for
the demonstrations, enough for K = 40:

```
mkdir -p logs && CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=monk_bank_r1_per_class VPC=12 LEVELS="10:20 20:40" KDISC=20 bash script/method_cell.sh L31k40 \
  && MODEL=meta-llama/Llama-3.1-8B DTYPE=bfloat16 TASK=monk_bank_r1_per_class VPC=12 LEVELS="10:20 20:40" KDISC=20 SPEC_FREEZE=none TEST=1 bash script/method_cell.sh L31k40 "ceiling test"
' > logs/monk1_run.out 2>&1 &
```

The cells land in `results/method/L31k40_synmb`, `L31k40_synlb` and
`L31k40_monkb1`, their calibration prompts under `data/method/L31k40/`. The
levels are pairs `K_base:K_full` with nested draws, so any pair the bank
supports can be given the same way; `KDISC` is the K at which the carriers
are discovered (the text tasks use their first level's `K_full`).

## TREC-fine at K = 10 → 20 (Llama-3.1-8B; a 40 GB card and `ATTN=sdpa`)

The paper's TREC-fine rows stop at 5 → 10. The next level, a K = 10
receiver reading a K = 20 memory, is the same cell with one more pair and
nothing else changed: the carriers are discovered at K = 5 as for the other
rows (`KDISC` keeps its default, and the K = 5 calibration prompts are
built for that), the validation draw is 4 queries per class (144 per seed
over the 36 classes), the test draw 250 per seed.

```
mkdir -p logs && CUDA_VISIBLE_DEVICES=0 nohup bash -c '
  LEVELS="10:20" ATTN=sdpa bash script/method_cell.sh L31c36 \
  && LEVELS="10:20" ATTN=sdpa SPEC_FREEZE=none TEST=1 bash script/method_cell.sh L31c36 "ceiling test"
' > logs/trec_k20_run.out 2>&1 &
```

`ATTN=sdpa` is required, not a preference. The K = 20 prompt is about
12.3k tokens (36 classes × 20 demonstrations at ~17 tokens each; the K = 10
prompt is 6.1–6.2k, K = 5 is 3.0–3.1k), and HF's eager attention
materialises a [32 heads, T, T] score matrix per layer: one 12k-token
forward of this model under eager ran out of memory on a 47 GiB card (41.5
GiB allocated when it asked for 17.2 more), while the same forward under
the fused kernel peaks at 19.4 GiB. The cell's precheck measures the
kernel's two-path noise and sets the receivers' cache gate to it (about one
bfloat16 step under sdpa, as under eager).

Memory, measured on this model in bfloat16:

| what | GiB |
|---|---|
| weights | 15.0 |
| one K = 20 forward (12,000 tokens), `sdpa` | 19.4 (19.7 at 12,800) |
| the same forward under eager attention | out of memory on 47 GiB |
| a K = 20 prefix cache (128 KiB per token) | 1.5 |
| the receiver: one forward + the base cache + one steered copy at a time | ≈ 22–23 |

The steps above that are FV's attribution-patching backward passes through
its 6k-token extraction prompts (359 of the extra demonstrations each) and
the exact-CIE forwards; they are the peak of the chain, not the memory
reads. The reference point is the dbpedia cell in the paper, whose prompts
have the same lengths (11–12k tokens for its K = 10 memory, 5.7k for its FV
prompts): its whole chain and its test read ran on a 40 GB A100 under sdpa,
8 h 14 min and 7 h 38 min. Expect the same here: a 40 GB card runs the
pair, an 80 GB card is comfortable, a 24 GB card is not enough (the
one-forward peak alone is 19.4 GiB and the backward passes need more than
the remaining 5). The pair takes about a day on one card; the I2CL
calibration and the 750-query test reads are most of it. The summary at
the end of each half prints the table rows for `10 -> 20` in the paper's
format; a sdpa cell is marked as such beside the eager K ≤ 10 rows, as the
dbpedia and yahoo rows are.
