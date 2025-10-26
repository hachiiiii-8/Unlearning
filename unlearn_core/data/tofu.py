from typing import Tuple, Dict, Optional
from .builder import QABuilder
from .schemas import get_schema
from .common import DEFAULT_SEED, SplitManifest, save_jsonl, ensure_dir

def _name_from_ratio(fr: float) -> str:
    pct = int(round(fr * 100))
    return f"forget{pct}"  # 例如0.10 -> "forget10"

def _retain_name_from_ratio(fr: float) -> str:
    pct = 100 - int(round(fr * 100))
    return f"retain{pct}"

def _holdout_name_from_ratio(fr: float) -> Optional[str]:
    # 官方只提供 10% 版本的 holdout：holdout10
    if round(fr, 2) == 0.10:
        return "holdout10"
    return None

def build_tofu(
    forget_ratio: float,
    out_dir: str,
    split: str = "train",
    seed: int = DEFAULT_SEED,
    official: bool = True,
    include_holdout: bool = False,   # <== 新增：是否一并导出官方 holdout
) -> Tuple[Dict[str, str], dict]:
    schema = get_schema("tofu")
    tag = f"tofu_f{int(round(forget_ratio*100))}"

    if official:
        # 直接使用HF官方切分
        qb_forget = (
            QABuilder("locuslab/TOFU", name=_name_from_ratio(forget_ratio), split=split)
            .load().to_rows(schema)
        )
        qb_retain = (
            QABuilder("locuslab/TOFU", name=_retain_name_from_ratio(forget_ratio), split=split)
            .load().to_rows(schema)
        )
        ensure_dir(out_dir)
        fpath = f"{out_dir}/{tag}_forget.jsonl"
        rpath = f"{out_dir}/{tag}_retain.jsonl"
        save_jsonl(qb_forget.rows, fpath)
        save_jsonl(qb_retain.rows, rpath)

        paths = {"forget": fpath, "retain": rpath}

        # 可选：导出官方 holdout
        if include_holdout:
            hname = _holdout_name_from_ratio(forget_ratio)
            if hname is None:
                raise ValueError("TOFU 官方 holdout 仅支持 forget_ratio=0.10（即 holdout10）")
            qb_holdout = (
                QABuilder("locuslab/TOFU", name=hname, split=split)
                .load().to_rows(schema)
            )
            hpath = f"{out_dir}/{tag}_holdout.jsonl"
            save_jsonl(qb_holdout.rows, hpath)
            paths["holdout"] = hpath

        # 构造manifest（记录使用官方切分）
        mani = SplitManifest(
            bench="tofu",
            forget_ratio=forget_ratio,
            group_key="author_id",
            forget_groups=["__official__"],
            retain_groups=["__official__"],
            seed=seed,
            note="official HF split (forget/retain" + ("/holdout" if include_holdout else "") + ")",
        ).__dict__
        return paths, mani

    # 自定义组级切分（按 author 分组）
    qb = QABuilder("locuslab/TOFU", split=split).load().to_rows(schema)
    forget, retain, mani = qb.split_groupwise(forget_ratio, "group", seed, bench_name="tofu")
    paths = qb.export(forget, retain, out_dir, tag)
    # 自定义切分不自动生成 holdout；如需要，你可以在这里再做一个基于 retain 的二次切分。
    return paths, mani

