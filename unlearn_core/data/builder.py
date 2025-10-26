from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset as HFDataset
from .schemas import QASchema
from .common import groupwise_forget_retain, save_jsonl, ensure_dir, DEFAULT_SEED

class QABuilder:
    """
    通用构建器：HF加载 -> 标准化 -> 组级划分 -> 导出
    导出的每行字段固定：prompt / answer / group / uid / meta
    """
    def __init__(self, hf_path: str, subset: Optional[str] = None, split: str = "train", name: Optional[str]=None):
        self.hf_path = hf_path
        self.subset = subset
        self.split = split
        self.name = name
        self.raw: Optional[HFDataset] = None
        self.rows: Optional[List[Dict]] = None

    def load(self) -> "QABuilder":
        if self.name is not None and self.subset is not None:
            ds = load_dataset(self.hf_path, self.subset, name=self.name, split=self.split)  # 罕见情况
        elif self.name is not None:
            ds = load_dataset(self.hf_path, name=self.name, split=self.split)
        elif self.subset is not None:
            ds = load_dataset(self.hf_path, self.subset, split=self.split)
        else:
            ds = load_dataset(self.hf_path, split=self.split)
        self.raw = ds
        return self

    def to_rows(self, schema: QASchema) -> "QABuilder":
        assert self.raw is not None, "Call .load() first"
        rows = []
        for i, r in enumerate(self.raw):
            uid = r.get(schema.uid_key) if schema.uid_key else str(i)
            prompt = r[schema.prompt_key]
            answer = r.get(schema.answer_key) if schema.answer_key else None
            group = r.get(schema.group_key, "UNKNOWN")
            meta = {k: r[k] for k in r.keys() if k not in {schema.prompt_key, schema.answer_key, schema.group_key, schema.uid_key}}
            rows.append({
                "prompt": prompt,
                "answer": answer,
                "group": str(group),
                "uid": str(uid),
                "meta": meta,
            })
        self.rows = rows
        return self

    def split_groupwise(
        self, forget_ratio: float, group_key: str = "group", seed: int = DEFAULT_SEED, bench_name: str="UNKNOWN"
    ) -> Tuple[List[Dict], List[Dict], dict]:
        assert self.rows is not None, "Call .to_rows() first"
        forget_rows, retain_rows, mani = groupwise_forget_retain(
            self.rows, group_key, forget_ratio, seed, bench_name
        )
        return forget_rows, retain_rows, vars(mani)

    def export(self, forget: List[Dict], retain: List[Dict], out_dir: str, tag: str) -> Dict[str, str]:
        ensure_dir(out_dir)
        fpath = f"{out_dir}/{tag}_forget.jsonl"
        rpath = f"{out_dir}/{tag}_retain.jsonl"
        save_jsonl(forget, fpath)
        save_jsonl(retain, rpath)
        return {"forget": fpath, "retain": rpath}
