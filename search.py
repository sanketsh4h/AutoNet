"""DARTS architecture search (first-order) with trainable edge weights and the
anti-collapse regulariser.

Run from the command line, or call ``run_search(**config)`` from ``tune.py``.
The training set is split in half: one half trains the network weights, the
other half trains the architecture weights.  After every epoch the current
genotype, validation loss/accuracy and the regulariser statistics are logged,
and the best genotype is written to ``<out>/genotype.json``.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, SubsetRandomSampler

from model import SearchCNNController, genotype_to_dict
from regularizer import CollapseRegularizer

DEFAULTS = dict(
    dataset='cifar10', data='./data', out='./runs/search', seed=0, device=None,
    epochs=50, batch_size=64, subset=0, workers=0, log_every=50,
    init_channels=16, layers=8, nodes=4, primitives='autonet', edge_weights=True,
    w_lr=0.025, w_lr_min=0.001, w_momentum=0.9, w_weight_decay=3e-4, grad_clip=5.0,
    alpha_lr=3e-4, alpha_weight_decay=1e-3,
    lam_skip=1.0, skip_target=-1.0, lam_entropy=0.0, lam_l2=0.0, warmup_epochs=0, max_skip=-1,
)


def get_dataset(name, root):
    """Return (train dataset, input channels, number of classes)."""
    name = name.lower()
    if name == 'fake':  # random images, for smoke tests without a download
        ds = torchvision.datasets.FakeData(size=512, image_size=(3, 32, 32), num_classes=10, transform=T.ToTensor())
        return ds, 3, 10
    if name in ('cifar10', 'cifar100'):
        if name == 'cifar10':
            cls, n_classes = torchvision.datasets.CIFAR10, 10
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        else:
            cls, n_classes = torchvision.datasets.CIFAR100, 100
            mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
        return cls(root, train=True, download=True, transform=tf), 3, n_classes
    if name in ('fashion_mnist', 'fashionmnist', 'mnist'):
        cls = torchvision.datasets.FashionMNIST if name != 'mnist' else torchvision.datasets.MNIST
        tf = T.Compose([T.RandomCrop(28, padding=2), T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        return cls(root, train=True, download=True, transform=tf), 1, 10
    raise ValueError(f'unknown dataset {name!r}')


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def run_search(**overrides):
    cfg = {**DEFAULTS, **overrides}
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    device = cfg['device'] or ('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['out'], exist_ok=True)

    # ---- data: half for weights, half for architecture -------------------
    dataset, C_in, n_classes = get_dataset(cfg['dataset'], cfg['data'])
    idx = np.random.RandomState(cfg['seed']).permutation(len(dataset))
    if cfg['subset']:
        idx = idx[:cfg['subset']]
    split = len(idx) // 2
    kw = dict(batch_size=cfg['batch_size'], num_workers=cfg['workers'], pin_memory=device == 'cuda')
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(idx[:split]), **kw)
    valid_loader = DataLoader(dataset, sampler=SubsetRandomSampler(idx[split:]), **kw)

    # ---- model, regulariser, optimisers ----------------------------------
    model = SearchCNNController(C_in, cfg['init_channels'], n_classes, cfg['layers'], cfg['nodes'],
                                cfg['primitives'], cfg['edge_weights']).to(device)
    reg = CollapseRegularizer(model.primitives, lam_skip=cfg['lam_skip'],
                              skip_target=None if cfg['skip_target'] < 0 else cfg['skip_target'],
                              lam_entropy=cfg['lam_entropy'], lam_l2=cfg['lam_l2'],
                              warmup_epochs=cfg['warmup_epochs'], total_epochs=cfg['epochs'])
    w_optim = torch.optim.SGD(model.weights(), cfg['w_lr'], momentum=cfg['w_momentum'],
                              weight_decay=cfg['w_weight_decay'])
    a_optim = torch.optim.Adam(model.arch_parameters(), cfg['alpha_lr'], betas=(0.5, 0.999),
                               weight_decay=cfg['alpha_weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, cfg['epochs'], eta_min=cfg['w_lr_min'])
    max_skip = None if cfg['max_skip'] < 0 else cfg['max_skip']

    print(f"[search] device={device} params={sum(p.numel() for p in model.weights()) / 1e6:.2f}M "
          f"ops={len(model.primitives)} train={split} valid={len(idx) - split}")

    best = {'val_loss': float('inf'), 'val_acc': 0.0, 'epoch': -1, 'genotype': None}
    history = []
    for epoch in range(cfg['epochs']):
        model.train()
        t0 = time.time()
        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
            trn_X, trn_y = trn_X.to(device), trn_y.to(device)
            val_X, val_y = val_X.to(device), val_y.to(device)

            # architecture step: validation loss + collapse regulariser
            a_optim.zero_grad()
            a_loss = model.loss(val_X, val_y) + reg(model.alphas(), epoch)
            a_loss.backward()
            a_optim.step()

            # weight step: training loss
            w_optim.zero_grad()
            loss = model.loss(trn_X, trn_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.weights(), cfg['grad_clip'])
            w_optim.step()

            if cfg['log_every'] and step % cfg['log_every'] == 0:
                print(f'  epoch {epoch} step {step} train_loss={loss.item():.4f} arch_loss={a_loss.item():.4f}')
        scheduler.step()

        val_loss, val_acc = evaluate(model, valid_loader, device)
        genotype = model.genotype(max_skip=max_skip)
        stats = reg.stats(model.alphas())
        record = {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc, **stats,
                  'genotype': genotype_to_dict(genotype), 'seconds': time.time() - t0}
        history.append(record)
        print(f"[search] epoch {epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"skip_share={stats['skip_share']:.3f} entropy={stats['entropy']:.3f} ({record['seconds']:.0f}s)")
        print(f'  genotype: {genotype}')

        if val_loss < best['val_loss']:
            best = {'val_loss': val_loss, 'val_acc': val_acc, 'epoch': epoch, 'genotype': genotype_to_dict(genotype)}
            with open(os.path.join(cfg['out'], 'genotype.json'), 'w') as f:
                json.dump(best, f, indent=2)
            torch.save({'state_dict': model.state_dict(), 'genotype': best['genotype'], 'config': cfg},
                       os.path.join(cfg['out'], 'best.pt'))
        with open(os.path.join(cfg['out'], 'history.json'), 'w') as f:
            json.dump({'config': cfg, 'history': history}, f, indent=2)

    print(f"[search] best epoch {best['epoch']}: val_loss={best['val_loss']:.4f} val_acc={best['val_acc']:.4f}")
    return {**best, 'history': history, 'config': cfg}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, val in DEFAULTS.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(val, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ('1', 'true', 'yes'), default=val, metavar='BOOL')
        else:
            p.add_argument(flag, type=type(val) if val is not None else str, default=val)
    return p


if __name__ == '__main__':
    run_search(**vars(build_parser().parse_args()))
