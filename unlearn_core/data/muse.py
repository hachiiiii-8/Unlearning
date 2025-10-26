# unlearn_core/data/muse.py
from typing import Tuple, Dict
from datasets import load_dataset
from .builder import QABuilder
from .schemas import get_schema
from .common import DEFAULT_SEED, SplitManifest, save_jsonl, ensure_dir

def _rows_official(ds, prompt_key: str, answer_key: str | None, group_value: str):
    rows = []
    for i, r in enumerate(ds):
        rows.append({
            "prompt": r[prompt_key],
            "answer": (r.get(answer_key) if answer_key else None),
            "group": group_value,           # 关键：官方 split 下不再需要真实 group 字段
            "uid": str(r.get("id", i)),     # 没有 id 就用索引
            "meta": {}
        })
    return rows

def build_muse_books(
    forget_ratio: float,
    out_dir: str,
    split: str = "train",
    seed: int = DEFAULT_SEED,
    cfg: str = "knowmem",
    mode: str = "official",        # official: 直接使用 forget_qa / retain_qa
) -> Tuple[Dict[str, str], dict]:
    schema = get_schema("muse_books")
    tag = f"muse_books_{cfg}_f{int(round(forget_ratio*100))}"

    if mode == "official":
        ds_f = load_dataset("muse-bench/MUSE-Books", name=cfg, split="forget_qa")
        ds_r = load_dataset("muse-bench/MUSE-Books", name=cfg, split="retain_qa")

        rows_f = _rows_official(ds_f, schema.prompt_key, schema.answer_key, group_value="__official_forget__")
        rows_r = _rows_official(ds_r, schema.prompt_key, schema.answer_key, group_value="__official_retain__")

        ensure_dir(out_dir)
        fp = f"{out_dir}/{tag}_forget.jsonl"
        rp = f"{out_dir}/{tag}_retain.jsonl"
        save_jsonl(rows_f, fp); save_jsonl(rows_r, rp)

        mani = SplitManifest(
            bench=f"muse_books:{cfg}",
            forget_ratio=forget_ratio,
            group_key="__none__",          # 没有真实分组字段
            forget_groups=["__official__"],
            retain_groups=["__official__"],
            seed=seed,
            note="official MUSE split (forget_qa / retain_qa); no group column in samples"
        ).__dict__
        return {"forget": fp, "retain": rp}, mani

    # custom: 用支持可分割的数据（例如 cfg=raw 且 split=train），再做组级切分
    qb = QABuilder("muse-bench/MUSE-Books", split=split, name=cfg).load().to_rows(schema)
    forget, retain, mani = qb.split_groupwise(forget_ratio, "group", seed, bench_name=f"muse_books:{cfg}")
    paths = qb.export(forget, retain, out_dir, tag=tag)
    return paths, mani

def build_muse_news(
    forget_ratio: float,
    out_dir: str,
    split: str = "train",
    seed: int = DEFAULT_SEED,
    cfg: str = "knowmem",
    mode: str = "official",
) -> Tuple[Dict[str, str], dict]:
    schema = get_schema("muse_news")
    tag = f"muse_news_{cfg}_f{int(round(forget_ratio*100))}"

    if mode == "official":
        ds_f = load_dataset("muse-bench/MUSE-News", name=cfg, split="forget_qa")
        ds_r = load_dataset("muse-bench/MUSE-News", name=cfg, split="retain_qa")

        rows_f = _rows_official(ds_f, schema.prompt_key, schema.answer_key, group_value="__official_forget__")
        rows_r = _rows_official(ds_r, schema.prompt_key, schema.answer_key, group_value="__official_retain__")

        ensure_dir(out_dir)
        fp = f"{out_dir}/{tag}_forget.jsonl"
        rp = f"{out_dir}/{tag}_retain.jsonl"
        save_jsonl(rows_f, fp); save_jsonl(rows_r, rp)

        mani = SplitManifest(
            bench=f"muse_news:{cfg}",
            forget_ratio=forget_ratio,
            group_key="__none__",
            forget_groups=["__official__"],
            retain_groups=["__official__"],
            seed=seed,
            note="official MUSE split (forget_qa / retain_qa); no group column in samples"
        ).__dict__
        return {"forget": fp, "retain": rp}, mani

    qb = QABuilder("muse-bench/MUSE-News", split=split, name=cfg).load().to_rows(schema)
    forget, retain, mani = qb.split_groupwise(forget_ratio, "group", seed, bench_name=f"muse_news:{cfg}")
    paths = qb.export(forget, retain, out_dir, tag=tag)
    return paths, mani
