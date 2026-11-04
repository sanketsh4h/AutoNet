"""AutoNet driver: weighted random search over the DARTS search hyperparameters.

Each trial runs a reduced-budget architecture search (``search.run_search``)
with the proposed learning rates, weight decays and regulariser strengths and
reports its best validation loss.  The weighted random search then proposes
the next trial from the widened, loss-weighted bracket around the two best
trials so far.  Results land in ``<out>/record.json`` and the best trial's
genotype in ``<out>/best.json``.
"""
import argparse
import json
import os

from search import run_search
from weighted_random import Param, WeightedRandomSearch

SPACE = {
    'w_lr':               Param(0.01, 0.1, log=True),
    'w_weight_decay':     Param(1e-4, 3e-3, log=True),
    'alpha_lr':           Param(1e-4, 1e-2, log=True),
    'alpha_weight_decay': Param(1e-4, 1e-2, log=True),
    'lam_skip':           Param(0.1, 3.0, log=True, bounds=(0.0, None)),
    'lam_entropy':        Param(1e-3, 1e-1, log=True, bounds=(0.0, None)),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--iters', type=int, default=5, help='search iterations after the two initial trials')
    p.add_argument('--epochs', type=int, default=5, help='search epochs per trial')
    p.add_argument('--subset', type=int, default=10000, help='training images used per trial (0 = all)')
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--data', default='./data')
    p.add_argument('--out', default='./runs/tune')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--layers', type=int, default=5)
    p.add_argument('--init-channels', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--primitives', default='autonet', choices=['darts', 'autonet'])
    p.add_argument('--widen', type=float, default=0.5, help='how far past the two bests the bracket extends')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fixed = dict(dataset=args.dataset, data=args.data, epochs=args.epochs, subset=args.subset, seed=args.seed,
                 layers=args.layers, init_channels=args.init_channels, batch_size=args.batch_size,
                 primitives=args.primitives, log_every=0)
    counter = {'trial': 0}

    def objective(cfg):
        out = os.path.join(args.out, f"trial_{counter['trial']:02d}")
        counter['trial'] += 1
        res = run_search(**fixed, **cfg, out=out)
        return res['val_loss'], {'val_acc': res['val_acc'], 'genotype': res['genotype'], 'dir': out}

    search = WeightedRandomSearch(SPACE, objective, n_iter=args.iters, seed=args.seed, widen=args.widen,
                                  log_path=os.path.join(args.out, 'record.json'))
    best = search.run()

    with open(os.path.join(args.out, 'best.json'), 'w') as f:
        json.dump(best, f, indent=2)
    print('\n[tune] best trial:', best['trial'], 'val_loss=%.4f val_acc=%.4f' % (best['loss'], best['val_acc']))
    print('[tune] hyperparameters:', json.dumps(best['config'], indent=2))
    print('[tune] genotype:', json.dumps(best['genotype']))


if __name__ == '__main__':
    main()
