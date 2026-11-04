# AutoNet: Automated Architecture Search and Hyperparameter Tuning

> **Late upload.** This was my project for the Machine Learning class at
> Ahmedabad University (2021). The original repository was lost, so this is a
> reconstruction from the two surviving notebooks in `notebooks/` and my
> project notes, uploaded well after the course ended.

AutoNet combines differentiable architecture search (DARTS) with a weighted
random hyperparameter search:

- **Weighted random search** (`weighted_random.py`). Instead of a fixed grid,
  every parameter is sampled from a bracket around the two best values seen so
  far. The bracket is widened on both sides and each half is cut into a number
  of candidate points proportional to the *other* half's loss, so the search
  concentrates near the better value while still being able to leave the
  bracket and escape local optima.
- **Enhanced nodal operations and trainable edge weights** (`ops.py`,
  `model.py`). The DARTS op set is extended with 7x7 separable, factorised
  7x1/1x7 and squeeze-and-excitation gated separable convolutions, and every
  node carries trainable weights over its incoming edges that are optimised
  together with the op weights and used when the genotype is read off.
- **Custom anti-collapse regulariser** (`regularizer.py`). A penalty on the
  probability mass given to skip connections, an annealed entropy term that
  makes edges commit to one op, and an L2 term on the raw architecture weights
  keep the search from degenerating into identity cells. A `--max-skip` cap
  on the discretised genotype is available as a hard backstop.
- **Search loop and driver** (`search.py`, `tune.py`). `search.py` runs
  first-order DARTS on a train/validation split with the regulariser;
  `tune.py` wraps it in the weighted random search to tune the learning
  rates, weight decays and regulariser strengths.

## Layout

| File | Purpose |
| --- | --- |
| `weighted_random.py` | Weighted random search (`Param`, `WeightedRandomSearch`), runnable on a toy objective |
| `ops.py` | DARTS primitives plus the enhanced operations and `MixedOp` |
| `model.py` | Search cells, `SearchCNNController` with alpha/beta weights, genotype parsing |
| `regularizer.py` | `CollapseRegularizer` added to the architecture-step loss |
| `search.py` | DARTS search CLI; also exposes `run_search(**config)` |
| `tune.py` | Weighted random search over the DARTS hyperparameters |
| `notebooks/` | The two original notebooks that survived |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Architecture search on CIFAR-10 with the enhanced op set and the regulariser
(datasets download into `./data`):

```bash
python search.py --dataset cifar10 --epochs 50 --lam-skip 1.0 --lam-entropy 0.05 --max-skip 2
```

Weighted random search over the search hyperparameters, five short trials on a
10k-image subset:

```bash
python tune.py --iters 5 --epochs 5 --subset 10000
```

Quick CPU check with random data, and the search algorithm on its own:

```bash
python search.py --dataset fake --epochs 1 --subset 64 --batch-size 16 --layers 3 --init-channels 4
```

```bash
python weighted_random.py
```

`search.py` writes `genotype.json`, `history.json` and `best.pt` to `--out`
(default `runs/search`). `tune.py` writes one `trial_XX/` directory per trial
plus `record.json` and `best.json` to `runs/tune`. Supported datasets are
`cifar10`, `cifar100`, `fashion_mnist`, `mnist` and `fake`. Use
`--primitives darts --edge-weights false --lam-skip 0` to fall back to plain
DARTS for comparison.

## Notes on the originals

- `notebooks/Weighted_Random_CIFAR.ipynb` is the original weighted random
  search tuning a Keras CNN on CIFAR-10 (per-layer dropout rates, L2
  strengths and the learning rate). `weighted_random.py` is the same
  algorithm factored out so it can drive the DARTS search as well.
- `notebooks/darts_load.ipynb` loads a search checkpoint from the pt.darts
  code the original DARTS experiments were built on. The code here keeps the
  same structure (`SearchCNNController`, genotype parsing) but is
  self-contained.
- The accuracy numbers from the original course report were not re-run for
  this upload.
