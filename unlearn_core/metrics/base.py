import torch
from torch.nn import functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from transformers import set_seed   # ← 新增
from tqdm import tqdm

@dataclass
class DecodingConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    seed: int = 42

class BaseMetric:

    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None,
                 torch_dtype=torch.bfloat16, device: str = "cuda"):
        self.model, self.tok = self._load_model_and_tokenizer(
            model_path, tokenizer_path, torch_dtype, device
        )

    @staticmethod
    def _load_model_and_tokenizer(model_path: str, tokenizer_path: Optional[str],
                                  torch_dtype, device):
        tok = AutoTokenizer.from_pretrained(tokenizer_path or model_path, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        model.to(device).eval()
        return model, tok

    @torch.no_grad()
    def batch_nll(self, texts: List[str], max_length: int = 1024, batch_size: int = 2,
                show_progress: bool = False, desc: str = "NLL") -> torch.Tensor:
        outs = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=(len(texts)+batch_size-1)//batch_size, desc=desc)
        for i in iterator:
            chunk = texts[i:i+batch_size]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            out = self.model(**enc, use_cache=False)  # 评估不需要KV cache，省显存
            logits = out.logits[:, :-1, :]
            labels = enc["input_ids"][:, 1:]
            logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            mask = enc["attention_mask"][:, 1:].float()
            nll_tok = -(logp * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            outs.append(nll_tok.detach().cpu())
            del out, logits, labels, logp, mask, enc
            torch.cuda.empty_cache()
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def conditional_suffix_logprob(self, prefixes: List[str], suffixes: List[str],
                                max_length: int = 1024, batch_size: int = 2,
                                show_progress: bool = False, desc: str = "Verbatim") -> torch.Tensor:
        assert len(prefixes) == len(suffixes)
        vals = []
        iterator = range(0, len(prefixes), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=(len(prefixes)+batch_size-1)//batch_size, desc=desc)
        for i in iterator:
            pref = prefixes[i:i+batch_size]
            suf  = suffixes[i:i+batch_size]
            texts = [pref[j] + suf[j] for j in range(len(pref))]
            enc = self.tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            ids = enc["input_ids"].to(self.model.device)
            attn = enc["attention_mask"].to(self.model.device)
            out = self.model(input_ids=ids, attention_mask=attn, use_cache=False)
            logits = out.logits[:, :-1, :]
            labels = ids[:, 1:]

            pref_ids = self.tok(pref, padding=True, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].to(self.model.device)
            def eff_len(t): return (t != self.tok.pad_token_id).sum(dim=1)
            pre_lens = eff_len(pref_ids) - 1

            logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            for b in range(ids.size(0)):
                start = max(pre_lens[b].item(), 0)
                seq_len = (ids[b] != self.tok.pad_token_id).sum().item() - 1
                end = max(seq_len, start + 1)
                seg = logp[b, start:end]
                vals.append(seg.mean().item())

            del out, logits, labels, enc, ids, attn, pref_ids, logp
            torch.cuda.empty_cache()
        return torch.tensor(vals, device="cpu")

    @torch.no_grad()
    def generate(self, prompts: List[str], cfg: DecodingConfig) -> List[str]:
        # g = torch.Generator(device=self.model.device)            # ← 删除
        # g.manual_seed(cfg.seed)                                  # ← 删除
        set_seed(cfg.seed)                                         # ← 新增：统一设seed

        enc = self.tok(prompts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **enc,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            eos_token_id=cfg.eos_token_id or self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
            # generator=g,                                         # ← 删除
        )
        out_texts = []
        for i in range(outputs.size(0)):
            gen_ids = outputs[i][enc["input_ids"][i].numel():]
            out_texts.append(self.tok.decode(gen_ids, skip_special_tokens=True).strip())
        return out_texts

    # -------- text utils --------
    @staticmethod
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^\w\s\u4e00-\u9fff]", "", s)
        return s.strip()

    def compute(self, *args, **kwargs) -> Dict[str, Any]:
        """子类必须实现：返回字典指标"""
        raise NotImplementedError
    
    def set_progress_callback(self, cb):
        """
        cb(step:int, total:int, info:dict) -> None
        用户可注入一个回调来显示进度/中间指标
        """
        self._progress_cb = cb

    def _tick(self, step: int, total: int, info: dict | None = None):
        if hasattr(self, "_progress_cb") and self._progress_cb:
            try:
                self._progress_cb(step, total, info or {})
            except Exception:
                pass