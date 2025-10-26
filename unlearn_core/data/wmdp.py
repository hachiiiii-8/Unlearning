from typing import Tuple, Dict, Optional
from datasets import load_dataset
from .builder import QABuilder
from .schemas import get_schema
from .common import DEFAULT_SEED, save_jsonl, ensure_dir

def build_wmdp_qas(forget_ratio: float, out_dir: str, split: str="test", seed: int=DEFAULT_SEED) -> Tuple[Dict[str,str], dict]:
    """
    结构化问答版（HF:cais/wmdp），常用评测划分是 test
    """
    schema = get_schema("wmdp")
    qb = QABuilder("cais/wmdp", split=split).load().to_rows(schema)
    forget, retain, mani = qb.split_groupwise(forget_ratio, "group", seed, bench_name="wmdp_qas")
    paths = qb.export(forget, retain, out_dir, tag=f"wmdp_qas_f{int(round(forget_ratio*100))}")
    return paths, mani

def build_wmdp_text_corpus(data_files: str, out_dir: str, tag: str="wmdp_corpus") -> Dict[str,str]:
    """
    预训练语料版：本地 .jsonl （{"text": "..."}），用作语言建模风格的遗忘
    """
    ds = load_dataset("text", data_files=data_files, split="train")
    rows = [{"prompt": r["text"], "answer": None, "group": "corpus", "uid": str(i), "meta": {}} for i, r in enumerate(ds)]
    ensure_dir(out_dir)
    outp = f"{out_dir}/{tag}.jsonl"
    save_jsonl(rows, outp)
    return {"corpus": outp}
