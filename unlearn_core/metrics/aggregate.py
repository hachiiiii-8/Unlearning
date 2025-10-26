from typing import Dict, List

def orient(value: float, kind: str) -> float:
    if kind == "higher_better": return value
    if kind == "lower_better":  return -value
    if kind == "auroc_to_mid":  return 1.0 - 2.0*abs(value - 0.5)
    raise ValueError(kind)

def mean_rank(rows: List[Dict[str, float]], metric_kinds: Dict[str, str]) -> List[Dict]:
    import numpy as np
    metrics = list(metric_kinds.keys())
    arr = []
    for r in rows:
        arr.append([orient(r.get(m, float("nan")), metric_kinds[m]) for m in metrics])
    arr = np.array(arr, dtype=float)

    ranks = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        order = np.argsort(-col)  # desc
        rk = [0]*len(rows)
        for pos, idx in enumerate(order):
            rk[idx] = pos + 1
        ranks.append(rk)
    ranks = list(zip(*ranks))
    out = []
    for i, r in enumerate(rows):
        mr = sum(ranks[i]) / len(metrics)
        out.append({**r, "mean_rank": mr})
    return sorted(out, key=lambda x: x["mean_rank"])
