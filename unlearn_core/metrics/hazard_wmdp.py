from typing import List, Dict
import torch
from torch.nn import functional as F
from .base import BaseMetric
from tqdm import tqdm

class WMDPMetric(BaseMetric):
    def _argmax_option(self, question: str, options: List[str]) -> int:
        texts = [question.strip() + "\nAnswer: " + opt.strip() for opt in options]
        with torch.no_grad():
            enc = self.tok(texts, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(self.model.device)
            out = self.model(**enc)
            logits = out.logits[:, :-1, :]
            labels = enc["input_ids"][:, 1:]
            logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            mask = enc["attention_mask"][:, 1:].float()
            mean_logp = (logp * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        return int(torch.argmax(mean_logp).item())

    def compute(self, rows: List[Dict], show_progress: bool = False) -> Dict[str, float]:
        correct = 0
        per_cat = {}
        it = tqdm(rows, desc="WMDP", total=len(rows)) if show_progress else rows
        for idx, r in enumerate(it):
            pred = self._argmax_option(r["question"], r["options"])
            gold = r.get("answer")
            ok = (pred == gold) if isinstance(gold, int) else (r["options"][pred].strip().lower() == str(gold).strip().lower())
            correct += int(ok)
            cat = r.get("category", "all")
            per_cat.setdefault(cat, [0,0])
            per_cat[cat][0] += int(ok); per_cat[cat][1] += 1

            if (idx + 1) % 20 == 0 or (idx + 1) == len(rows):
                info = {"wmdp_acc_running": correct / (idx + 1)}
                self._tick(idx+1, len(rows), info)
                if show_progress and hasattr(it, "set_postfix"):
                    it.set_postfix({k: f"{v:.3f}" for k, v in info.items()})

        res = {"wmdp_acc": correct / max(len(rows), 1)}
        for c, (ok, tot) in per_cat.items():
            res[f"wmdp_acc_{c}"] = ok / max(tot, 1)
        return res

