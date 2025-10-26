from typing import List, Dict, Optional
from transformers import AutoTokenizer

def load_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def format_prompt(prompt: str, answer: Optional[str]=None) -> str:
    base = (prompt.rstrip() + "\nAnswer:").strip()
    if answer is not None and answer != "":
        return base + " " + answer.strip()
    return base

def make_collate(tokenizer, max_length: int = 1024):
    def _fn(batch: List[Dict]):
        texts = [format_prompt(b["prompt"], b.get("answer")) for b in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        labels = enc["input_ids"].clone()
        return {**enc, "labels": labels}
    return _fn
