# uncensored/unlearning/unlearn_core/cli/prepare_dataset.py
# -*- coding: utf-8 -*-
"""
一键下载并整理 Open-Unlearning 评测所需数据：
- TOFU: locuslab/TOFU -> data/splits/tofu_f10_{forget,retain,holdout}.jsonl
- MUSE: muse-bench/MUSE-Books / MUSE-News (verbmem 默认) -> data/muse/<subset>_<cfg>_{forget,retain}.jsonl
- WMDP: cais/wmdp (MCQ) -> data/wmdp/test_mcq.jsonl  {question, options[], answer, category}

用法示例：
  python -m unlearn_core.cli.prepare_dataset --all
  python -m unlearn_core.cli.prepare_dataset --tofu --muse_subsets MUSE-Books MUSE-News --muse_cfgs verbmem --wmdp
  python -m unlearn_core.cli.prepare_dataset --tofu --limit 200 --force

要求：
  pip install datasets>=2.20 huggingface_hub>=0.23
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Iterable, Any

# 依赖检查与安装提示
def _ensure_deps():
    try:
        import datasets  # noqa: F401
        import huggingface_hub  # noqa: F401
    except Exception as e:
        print("[prepare_dataset] Missing deps: datasets / huggingface_hub", file=sys.stderr)
        print("  pip install -U datasets huggingface_hub", file=sys.stderr)
        raise

_ensure_deps()
from datasets import load_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # 到仓库根：.../uncensored/unlearning
DATA_DIR = ROOT / "data"
SPLIT_DIR = DATA_DIR / "tofu"
MUSE_DIR = DATA_DIR / "muse"
WMDP_DIR = DATA_DIR / "wmdp"

def save_jsonl(rows: Iterable[Dict[str, Any]], out_path: Path, force: bool = False) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[prepare_dataset] Exists, skip: {out_path}")
        return sum(1 for _ in out_path.open("r", encoding="utf-8"))
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"[prepare_dataset] Wrote {n} rows -> {out_path}")
    return n

# ------------------------------ TOFU ------------------------------
from datasets import get_dataset_config_names, get_dataset_split_names

from datasets import get_dataset_config_names

def download_tofu(limit: int | None = None, force: bool = False) -> Dict[str, Path]:
    """
    locuslab/TOFU：不同集合用 config 区分，而不是 split。
    - full        -> 全量
    - forget10    -> 遗忘10%
    - retain90    -> 对应保留90%（有的卡可能命名 retain10；我们做回退）
    参考 HF 卡片: 可用 forget sets: forget01/05/10；retain sets 对应可用。 
    """
    print("[prepare_dataset] === TOFU: locuslab/TOFU (config-based) ===")

    out = {}
    def dump_config(cfg: str, out_path: Path):
        try:
            ds = load_dataset("locuslab/TOFU", cfg, split="train")
        except Exception as e:
            return False
        if limit: ds = ds.select(range(min(len(ds), limit)))
        rows = [{k: r.get(k) for k in ds.column_names} for r in ds]
        save_jsonl(rows, out_path, force=force)
        return True

    # full -> holdout(评测集的替代；很多工作直接用 full 的一个子集做评测)
    holdout_ok = dump_config("full", SPLIT_DIR / "tofu_f10_holdout.jsonl")
    if not holdout_ok:
        print("[prepare_dataset][TOFU] WARN: config 'full' not found.")

    # forget10
    forget_ok = dump_config("forget10", SPLIT_DIR / "tofu_f10_forget.jsonl")
    if not forget_ok:
        # 退而求其次：forget05 or forget01
        for alt in ["forget05", "forget01"]:
            if dump_config(alt, SPLIT_DIR / "tofu_f10_forget.jsonl"):
                print(f"[prepare_dataset][TOFU] fallback to {alt} for forget set.")
                forget_ok = True
                break
        if not forget_ok:
            print("[prepare_dataset][TOFU] WARN: no forget config (forget10/05/01) found.")

    # retain90
    retain_ok = dump_config("retain90", SPLIT_DIR / "tofu_f10_retain.jsonl")
    if not retain_ok:
        # 有些版本把“retain 对应 forget10”命名成 retain10
        if dump_config("retain10", SPLIT_DIR / "tofu_f10_retain.jsonl"):
            print("[prepare_dataset][TOFU] fallback to retain10 as retain set.")
            retain_ok = True
    if not retain_ok:
        print("[prepare_dataset][TOFU] WARN: no retain config (retain90/retain10) found.")

    return out


from datasets import get_dataset_split_names

def download_muse(subsets: List[str], cfgs: List[str], limit: int | None = None, force: bool = False) -> Dict[str, List[Path]]:
    print(f"[prepare_dataset] === MUSE: muse-bench/{subsets} cfg={cfgs} (auto-discovery) ===")
    results: Dict[str, List[Path]] = {}
    subsets = subsets or ["MUSE-Books", "MUSE-News"]
    cfgs = cfgs or ["verbmem"]

    WANT = {
        "forget":  ["forget_qa", "forget"],
        "retain":  ["retain_qa", "retain2", "retain1", "retain"],
        "holdout": ["holdout_qa", "holdout"],
    }

    for subset in subsets:
        for cfg in cfgs:
            paths = []
            try:
                avail = get_dataset_split_names(f"muse-bench/{subset}", cfg)
            except Exception as e:
                print(f"[prepare_dataset][MUSE] Skip {subset}:{cfg} -> {e}")
                continue
            avail_low = [s.lower() for s in avail]

            chosen = {}
            for kind, cands in WANT.items():
                for c in cands:
                    if c.lower() in avail_low:
                        chosen[kind] = avail[avail_low.index(c.lower())]
                        break

            for kind in ["forget", "retain", "holdout"]:
                sp = chosen.get(kind)
                if not sp:
                    print(f"[prepare_dataset][MUSE] WARN: {subset}:{cfg} has no split for '{kind}' (available={avail})")
                    continue
                try:
                    ds = load_dataset(f"muse-bench/{subset}", cfg, split=sp)
                except Exception as e:
                    print(f"[prepare_dataset][MUSE] Skip {subset}:{cfg}:{sp} -> {e}")
                    continue
                if limit:
                    ds = ds.select(range(min(len(ds), limit)))
                rows = [{k: r.get(k) for k in ds.column_names} for r in ds]
                out_path = MUSE_DIR / f"{subset}_{cfg}_{kind}.jsonl"
                save_jsonl(rows, out_path, force=force)
                paths.append(out_path)

            results[f"{subset}:{cfg}"] = paths
    return results


# ------------------------------ WMDP ------------------------------
from datasets import get_dataset_split_names

def download_wmdp(split_prefer: List[str] | None = None, force: bool = False) -> Path | None:
    """
    cais/wmdp 需要指定 config：['wmdp-bio','wmdp-chem','wmdp-cyber']。
    我们逐个加载，优先使用 test，否则 validation，再不行 train。
    统一字段：question/options(answer texts)/answer(index)/category。
    合并三个 config 到 data/wmdp/test_mcq.jsonl
    """
    print("[prepare_dataset] === WMDP: cais/wmdp (configs: bio/chem/cyber) ===")
    split_order = split_prefer or ["test", "validation", "dev", "train"]
    configs = ["wmdp-bio", "wmdp-chem", "wmdp-cyber"]

    all_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        try:
            ds = load_dataset("cais/wmdp", cfg)
        except Exception as e:
            print(f"[prepare_dataset][WMDP] ERROR loading {cfg}: {e}")
            continue
        split = next((s for s in split_order if s in ds), None)
        if not split:
            print(f"[prepare_dataset][WMDP] WARN: no preferred split in {cfg}, available={list(ds.keys())}")
            split = list(ds.keys())[0]
        dset = ds[split]
        for r in dset:
            q = r.get("question", r.get("prompt", ""))
            opts = r.get("choices") or r.get("options") or r.get("answers")
            if isinstance(opts, dict) and "text" in opts:
                opts = opts["text"]
            if not isinstance(opts, (list, tuple)):
                continue
            ans_idx = r.get("answer_index", r.get("label", r.get("answer", None)))
            if isinstance(ans_idx, str):
                ABC = {"A":0,"B":1,"C":2,"D":3}
                if ans_idx in ABC:
                    ans_idx = ABC[ans_idx]
                else:
                    try:
                        ans_idx = list(map(str, opts)).index(ans_idx)
                    except Exception:
                        ans_idx = None
            if not isinstance(ans_idx, int):
                continue
            cat = r.get("category", r.get("domain", cfg))
            all_rows.append({
                "question": str(q),
                "options": list(map(str, opts)),
                "answer": int(ans_idx),
                "category": str(cat),
            })

    if not all_rows:
        print("[prepare_dataset][WMDP] ERROR: gathered 0 rows.")
        return None

    out_path = WMDP_DIR / "test_mcq.jsonl"
    save_jsonl(all_rows, out_path, force=force)
    return out_path


# ------------------------------ CLI ------------------------------
def main():
    p = argparse.ArgumentParser(description="Download & prepare benchmarks for Open-Unlearning-style evaluation")
    p.add_argument("--all", action="store_true", help="Download TOFU + MUSE(Books/News:verbmem) + WMDP")
    p.add_argument("--tofu", action="store_true")
    p.add_argument("--muse", action="store_true")
    p.add_argument("--wmdp", action="store_true")
    p.add_argument("--muse_subsets", nargs="*", default=["MUSE-Books", "MUSE-News"])
    p.add_argument("--muse_cfgs", nargs="*", default=["verbmem"], help="e.g., verbmem raw scal sust")
    p.add_argument("--limit", type=int, default=None, help="limit rows per split (debug)")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    args = p.parse_args()

    # 目录
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    MUSE_DIR.mkdir(parents=True, exist_ok=True)
    WMDP_DIR.mkdir(parents=True, exist_ok=True)

    do_tofu = args.all or args.tofu
    do_muse = args.all or args.muse
    do_wmdp = args.all or args.wmdp

    if not any([do_tofu, do_muse, do_wmdp]):
        print("Nothing to do. Use --all or specify --tofu/--muse/--wmdp")
        return

    if do_tofu:
        download_tofu(limit=args.limit, force=args.force)

    if do_muse:
        download_muse(subsets=args.muse_subsets, cfgs=args.muse_cfgs, limit=args.limit, force=args.force)

    if do_wmdp:
        download_wmdp(force=args.force)

    print("[prepare_dataset] Done.")

if __name__ == "__main__":
    main()
