from typing import List, Dict, Optional
from dataclasses import dataclass
from rouge_score import rouge_scorer
from .base import BaseMetric, DecodingConfig
from tqdm import tqdm
import csv

_sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def rouge_l(a: str, b: str) -> float:
    return _sc.score(b, a)["rougeL"].fmeasure

def em(pred: str, ref: str, norm_fn) -> float:
    return float(norm_fn(pred) == norm_fn(ref))

def f1(pred: str, ref: str, norm_fn) -> float:
    p = norm_fn(pred).split()
    r = norm_fn(ref).split()
    if not p and not r: return 1.0
    if not p or not r:  return 0.0
    from collections import Counter
    cp, cr = Counter(p), Counter(r)
    common = sum(min(cp[t], cr[t]) for t in cp)
    if common == 0: return 0.0
    prec = common / len(p)
    rec  = common / len(r)
    return 2*prec*rec/(prec+rec+1e-8)

class QAMetric(BaseMetric):
    def compute(self, rows, prompt_fn, ref_key="answer",
                cfg: DecodingConfig = DecodingConfig(), k_samples:int=1,
                show_progress: bool = False):
        ems, f1s, rls = [], [], []
        it = range(len(rows))
        if show_progress:
            it = tqdm(it, desc="Calculating text similarity", total=len(rows),
                      mininterval=0.2, dynamic_ncols=True)
        for i in it:
            r = rows[i]
            prompt = prompt_fn(r)
            best, best_rl = "", -1.0
            for _ in range(k_samples):
                pred = self.generate([prompt], cfg)[0]
                rl = rouge_l(pred, r.get(ref_key, ""))
                if rl > best_rl: best_rl, best = rl, pred
            ref = r.get(ref_key, "")
            ems.append(em(best, ref, self.normalize))
            f1s.append(f1(best, ref, self.normalize))
            rls.append(rouge_l(best, ref))
            if show_progress:
                n = i + 1
                it.set_postfix(EM=f"{sum(ems)/n:.3f}", F1=f"{sum(f1s)/n:.3f}", rL=f"{sum(rls)/n:.3f}")
        n = max(len(rows), 1)
        return {"qa_em": float(sum(ems)/n), "qa_f1": float(sum(f1s)/n), "qa_rougeL": float(sum(rls)/n)}

