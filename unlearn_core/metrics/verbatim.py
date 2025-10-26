from typing import List, Dict
from .base import BaseMetric

class VerbatimMetric(BaseMetric):
    def compute(self, rows, prefix_suffix_fn, max_length=512, batch_size=2, show_progress=False):
        prefixes, suffixes = [], []
        for r in rows:
            p, s = prefix_suffix_fn(r)
            prefixes.append(p); suffixes.append(s)
        vals = self.conditional_suffix_logprob(
            prefixes, suffixes,
            max_length=max_length, batch_size=batch_size,
            show_progress=show_progress, desc="Verbatim"
        )
        return {
            "verbatim_logprob_mean": float(vals.mean().item()),
            "verbatim_logprob_std": float(vals.std().item()) if len(vals)>1 else 0.0
        }


