import os, json, hashlib, random
from typing import Iterable, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

DEFAULT_SEED = 20250930

def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def save_jsonl(rows: Iterable[Dict], path: str | os.PathLike) -> None:
    path = Path(path); ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_jsonl(path: str | os.PathLike) -> List[Dict]:
    return [json.loads(x) for x in Path(path).read_text("utf-8").splitlines()]

@dataclass
class SplitManifest:
    bench: str
    forget_ratio: float
    group_key: str
    forget_groups: List[str]
    retain_groups: List[str]
    seed: int = DEFAULT_SEED
    note: str = "groupwise split: shuffle groups with stable hash + seed"

    def save(self, path: str | os.PathLike):
        path = Path(path); ensure_dir(path.parent)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

def groupwise_forget_retain(
    items: List[Dict], group_key: str, forget_ratio: float, seed: int = DEFAULT_SEED, bench_name: str="UNKNOWN"
) -> Tuple[List[Dict], List[Dict], SplitManifest]:
    groups = {}
    for r in items:
        g = str(r[group_key])
        groups.setdefault(g, []).append(r)

    group_ids = sorted(groups.keys(), key=lambda x: (stable_hash(x), x))
    rnd = random.Random(seed)
    rnd.shuffle(group_ids)

    k_forget = max(1, int(round(len(group_ids) * forget_ratio)))
    forget_g = set(group_ids[:k_forget])
    retain_g = set(group_ids[k_forget:])

    forget_rows = [r for g in forget_g for r in groups[g]]
    retain_rows = [r for g in retain_g for r in groups[g]]

    mani = SplitManifest(
        bench=bench_name,
        forget_ratio=forget_ratio,
        group_key=group_key,
        forget_groups=sorted(list(forget_g)),
        retain_groups=sorted(list(retain_g)),
        seed=seed,
    )
    return forget_rows, retain_rows, mani
