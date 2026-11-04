"""Custom regulariser on the architecture weights.

Plain DARTS tends to collapse: as the search goes on the softmaxed alphas pile
up on parameter-free ops (above all ``skip_connect``) because they are the
easiest way to lower the bilevel validation loss, and the discretised network
then trains to a much worse accuracy than the supernet suggested.  This
regulariser is added to the architecture-step loss and has three terms:

skip     hinge penalty on the average probability mass given to skip
         connections (optionally other parameter-free ops) above a target
         share, switched on after ``warmup_epochs``;
entropy  mean per-edge entropy of the op distribution, ramped in linearly
         after warm-up so edges are pushed to commit to one op by the end of
         the search, which shrinks the gap between supernet and genotype;
l2       squared norm of the raw alphas, which keeps the softmax from
         saturating early.

All weights are ordinary hyperparameters and are among the ones ``tune.py``
searches over.
"""
import torch
import torch.nn.functional as F


class CollapseRegularizer:
    def __init__(self, primitives, lam_skip=1.0, skip_target=None, lam_entropy=0.0, lam_l2=0.0,
                 warmup_epochs=0, total_epochs=1, penalize=('skip_connect',)):
        self.idx = [primitives.index(p) for p in penalize if p in primitives]
        self.skip_target = (1.0 / len(primitives)) if skip_target is None else float(skip_target)
        self.lam_skip = lam_skip
        self.lam_entropy = lam_entropy
        self.lam_l2 = lam_l2
        self.warmup_epochs = warmup_epochs
        self.total_epochs = max(total_epochs, 1)

    def gate(self, epoch):
        """0 during warm-up, 1 afterwards."""
        return 1.0 if epoch is None or epoch >= self.warmup_epochs else 0.0

    def ramp(self, epoch):
        """0 during warm-up, then linear to 1 at the last epoch."""
        if epoch is None:
            return 1.0
        span = max(self.total_epochs - self.warmup_epochs, 1)
        return min(1.0, max(0.0, (epoch + 1 - self.warmup_epochs) / span))

    @staticmethod
    def _probs(alphas):
        return torch.cat([F.softmax(a, dim=-1) for a in alphas], dim=0)  # (n_edges, n_ops)

    def __call__(self, alphas, epoch=None):
        probs = self._probs(alphas)
        skip_share = probs[:, self.idx].sum(dim=1).mean() if self.idx else probs.new_zeros(())
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        l2 = sum((a ** 2).sum() for a in alphas)
        return (self.gate(epoch) * self.lam_skip * F.relu(skip_share - self.skip_target)
                + self.ramp(epoch) * self.lam_entropy * entropy
                + self.lam_l2 * l2)

    @torch.no_grad()
    def stats(self, alphas):
        probs = self._probs(alphas)
        skip_share = probs[:, self.idx].sum(dim=1).mean().item() if self.idx else 0.0
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
        return {'skip_share': skip_share, 'entropy': entropy}
