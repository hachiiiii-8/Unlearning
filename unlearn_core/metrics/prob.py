# 统一概率口径：整条答案的 per-token mean logP
from typing import List, Dict
import torch
from torch.nn import functional as F
from .base import BaseMetric

class ProbMetric(BaseMetric):
    @torch.no_grad()
    def answer_mean_logprob(self, rows: List[Dict], prompt_fn, ref_key="answer",
                            max_length=768, batch_size=2, show_progress=False, desc="Calculating loss"):
        from tqdm import tqdm
        vals = []
        it = range(0, len(rows), batch_size)
        if show_progress: 
            it = tqdm(it, total=(len(rows)+batch_size-1)//batch_size, desc=desc, dynamic_ncols=True, mininterval=0.2)
        for i in it:
            chunk = rows[i:i+batch_size]
            texts = [prompt_fn(r).rstrip() + " " + r.get(ref_key,"") for r in chunk]
            enc = self.tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.model.device)
            out = self.model(**enc, use_cache=False)
            logits = out.logits[:, :-1, :]
            labels = enc["input_ids"][:, 1:]
            logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            mask = enc["attention_mask"][:, 1:].float()
            mean_logp = (logp * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            vals.extend(mean_logp.detach().cpu().tolist())
            del out, logits, labels, logp, mask, enc
            torch.cuda.empty_cache()
        return vals

    @staticmethod
    def normalise_per_row(scores: List[float]) -> List[float]:
        # 简单 min-max；也可做 z-score
        if not scores: return []
        lo, hi = min(scores), max(scores)
        if hi - lo < 1e-8: 
            return [0.5]*len(scores)
        return [(s - lo) / (hi - lo) for s in scores]
