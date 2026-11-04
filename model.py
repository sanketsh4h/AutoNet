"""Search network for DARTS with trainable edge weights.

Every edge of a cell is a ``MixedOp`` weighted by architecture parameters
``alpha`` (one row per edge, one column per candidate op).  On top of the
original DARTS formulation, each node also carries trainable edge weights
``beta`` that decide how much every incoming edge contributes to the node's
sum.  Both ``alpha`` and ``beta`` are optimised on the validation set by the
architecture step and both are used when the final genotype is read off.
"""
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import PRIMITIVE_SETS, MixedOp, ReLUConvBN, FactorizedReduce

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class SearchCell(nn.Module):
    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction, primitives):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        if reduction_p:
            self.preproc0 = FactorizedReduce(C_pp, C, affine=False)
        else:
            self.preproc0 = ReLUConvBN(C_pp, C, 1, 1, 0, affine=False)
        self.preproc1 = ReLUConvBN(C_p, C, 1, 1, 0, affine=False)

        self.dag = nn.ModuleList()
        for i in range(n_nodes):
            edges = nn.ModuleList(MixedOp(C, 2 if reduction and j < 2 else 1, primitives) for j in range(2 + i))
            self.dag.append(edges)

    def forward(self, s0, s1, w_ops, w_edges):
        states = [self.preproc0(s0), self.preproc1(s1)]
        for edges, w_op, w_edge in zip(self.dag, w_ops, w_edges):
            states.append(sum(w_edge[j] * edges[j](states[j], w_op[j]) for j in range(len(edges))))
        return torch.cat(states[2:], dim=1)


class SearchCNN(nn.Module):
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes, primitives, stem_multiplier=3):
        super().__init__()
        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(C_in, C_cur, 3, 1, 1, bias=False), nn.BatchNorm2d(C_cur))

        C_pp, C_p, C_cur = C_cur, C_cur, C
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            reduction = i in (n_layers // 3, 2 * n_layers // 3)
            if reduction:
                C_cur *= 2
            self.cells.append(SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction, primitives))
            reduction_p = reduction
            C_pp, C_p = C_p, C_cur * n_nodes

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, w_normal, w_reduce, e_normal, e_reduce):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1, w_reduce, e_reduce)
            else:
                s0, s1 = s1, cell(s0, s1, w_normal, e_normal)
        return self.linear(self.gap(s1).flatten(1))


class SearchCNNController(nn.Module):
    """Holds the network, the architecture weights (alpha) and edge weights (beta)."""

    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, primitives='autonet',
                 edge_weights=True, stem_multiplier=3):
        super().__init__()
        self.primitives = PRIMITIVE_SETS[primitives] if isinstance(primitives, str) else list(primitives)
        self.n_nodes = n_nodes
        self.edge_weights = edge_weights
        n_ops = len(self.primitives)

        self.alpha_normal = nn.ParameterList(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)) for i in range(n_nodes))
        self.alpha_reduce = nn.ParameterList(nn.Parameter(1e-3 * torch.randn(i + 2, n_ops)) for i in range(n_nodes))
        self.beta_normal = nn.ParameterList(nn.Parameter(torch.zeros(i + 2)) for i in range(n_nodes))
        self.beta_reduce = nn.ParameterList(nn.Parameter(torch.zeros(i + 2)) for i in range(n_nodes))

        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, self.primitives, stem_multiplier)
        self.criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------------- weights
    def _edge_weights(self, betas):
        if not self.edge_weights:
            return [torch.ones_like(b) for b in betas]
        # softmax over a node's incoming edges, rescaled so the weights average 1
        return [F.softmax(b, dim=-1) * b.numel() for b in betas]

    def arch_weights(self):
        w_normal = [F.softmax(a, dim=-1) for a in self.alpha_normal]
        w_reduce = [F.softmax(a, dim=-1) for a in self.alpha_reduce]
        return w_normal, w_reduce, self._edge_weights(self.beta_normal), self._edge_weights(self.beta_reduce)

    def forward(self, x):
        return self.net(x, *self.arch_weights())

    def loss(self, X, y):
        return self.criterion(self(X), y)

    def weights(self):
        return self.net.parameters()

    def alphas(self):
        return list(self.alpha_normal) + list(self.alpha_reduce)

    def betas(self):
        return list(self.beta_normal) + list(self.beta_reduce)

    def arch_parameters(self):
        return self.alphas() + (self.betas() if self.edge_weights else [])

    # -------------------------------------------------------------- genotype
    def genotype(self, k=2, max_skip=None):
        gene_normal = parse(self.alpha_normal, self.beta_normal, self.primitives, k, max_skip, self.edge_weights)
        gene_reduce = parse(self.alpha_reduce, self.beta_reduce, self.primitives, k, max_skip, self.edge_weights)
        concat = list(range(2, 2 + self.n_nodes))
        return Genotype(gene_normal, concat, gene_reduce, concat)

    def print_alphas(self):
        for name, params in (('normal', self.alpha_normal), ('reduce', self.alpha_reduce)):
            print(f'# alpha - {name}')
            for a in params:
                print(F.softmax(a, dim=-1).detach().cpu().numpy().round(3))


def parse(alphas, betas, primitives, k=2, max_skip=None, use_beta=True):
    """Discretise one cell type.

    For every node keep the ``k`` strongest incoming edges, each with its best
    non-'none' op, where an edge's strength is its best op probability times
    the node's softmaxed edge weight.  If ``max_skip`` is given, any
    skip-connections beyond that count are replaced by the next-best op on the
    same edge (weakest skips first), which caps the collapse mode where cells
    degenerate into identity mappings.
    """
    none_idx = primitives.index('none') if 'none' in primitives else None
    skip_idx = primitives.index('skip_connect') if 'skip_connect' in primitives else None
    gene, strengths = [], []
    with torch.no_grad():
        for a, b in zip(alphas, betas):
            p = F.softmax(a, dim=-1).clone()
            if none_idx is not None:
                p[:, none_idx] = 0
            if use_beta:
                p = p * F.softmax(b, dim=-1)[:, None]
            edge_best, op_idx = p.max(dim=1)
            top = edge_best.topk(min(k, len(edge_best))).indices.tolist()
            gene.append([(primitives[int(op_idx[e])], int(e)) for e in top])
            strengths.append(p)

    if max_skip is not None and skip_idx is not None:
        skips = [(float(strengths[i][e, skip_idx]), i, n)
                 for i, node in enumerate(gene) for n, (op, e) in enumerate(node) if op == 'skip_connect']
        for _, i, n in sorted(skips)[:max(0, len(skips) - max_skip)]:
            e = gene[i][n][1]
            p = strengths[i][e].clone()
            p[skip_idx] = 0
            gene[i][n] = (primitives[int(p.argmax())], e)
    return gene


def genotype_to_dict(g: Genotype) -> dict:
    return {'normal': [list(x) for x in g.normal], 'normal_concat': list(g.normal_concat),
            'reduce': [list(x) for x in g.reduce], 'reduce_concat': list(g.reduce_concat)}
