import zlib
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F
from .base import BaseMetric

def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

class MIAMetric(BaseMetric):
    """
    实现三种常用 MIA：Loss、Zlib、MinK ；返回 AUROC
    """

    def _texts_to_nll(self, texts: List[str], max_length: int = 768, batch_size: int = 2,
                      show_progress: bool = False, desc: str = "MIA NLL") -> np.ndarray:
        return self.batch_nll(
            texts, max_length=max_length, batch_size=batch_size,
            show_progress=show_progress, desc=desc
        ).detach().cpu().numpy()

    @staticmethod
    def _zlib_score(t: str) -> int:
        # 可压缩性分数：越短越“模板化”
        if not isinstance(t, str):
            t = str(t)
        return -len(zlib.compress(t.encode("utf-8", errors="ignore")))

    @torch.no_grad()
    def _mink_scores(self, texts: List[str], k: int = 20, max_length: int = 768, batch_size: int = 2,
                     show_progress: bool = False, desc: str = "MIA MinK") -> np.ndarray:
        outs = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=(len(texts)+batch_size-1)//batch_size, desc=desc)
        for i in iterator:
            chunk = texts[i:i+batch_size]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.model.device)
            out = self.model(**enc, use_cache=False)
            probs = F.softmax(out.logits[:, :-1, :], dim=-1)
            vals, _ = torch.topk(probs, k, dim=-1, largest=False)   # 取最小的k个概率
            mask = enc["attention_mask"][:, 1:].unsqueeze(-1).float()
            mk = (vals * mask).sum(dim=(1,2)) / (mask.sum(dim=(1,2)) + 1e-8)
            outs.append(mk.detach().cpu())
            del out, probs, vals, mask, enc
            torch.cuda.empty_cache()
        return torch.cat(outs, dim=0).numpy()

    def compute(self, member_texts: List[str], nonmember_texts: List[str],
                k: int = 20, max_length: int = 768, batch_size: int = 2,
                show_progress: bool = False) -> Dict[str, float]:
        # Loss AUROC
        nll_m = self._texts_to_nll(member_texts, max_length, batch_size, show_progress, "MIA NLL (members)")
        nll_nm= self._texts_to_nll(nonmember_texts, max_length, batch_size, show_progress, "MIA NLL (nonmembers)")
        y = np.array([1]*len(nll_m) + [0]*len(nll_nm))
        s_loss = -np.concatenate([nll_m, nll_nm])   # NLL 越低越像成员 → 分数取负
        auc_loss = float(roc_auc_score(y, s_loss))

        # Zlib AUROC
        s_zlib = np.array(
            [*(self._zlib_score(t) for t in member_texts),
             *(self._zlib_score(t) for t in nonmember_texts)]
        )
        auc_zlib = float(roc_auc_score(y, s_zlib))

        # MinK AUROC
        mk_m = self._mink_scores(member_texts, k=k, max_length=max_length, batch_size=batch_size,
                                 show_progress=show_progress, desc="MIA MinK (members)")
        mk_nm= self._mink_scores(nonmember_texts, k=k, max_length=max_length, batch_size=batch_size,
                                 show_progress=show_progress, desc="MIA MinK (nonmembers)")
        s_mink = np.concatenate([-mk_m, -mk_nm])    # MinK 越小越尖锐 → 成员 → 取负当分数
        auc_mink = float(roc_auc_score(y, s_mink))

        return {
            "mia_loss_auroc": auc_loss,
            "mia_zlib_auroc": auc_zlib,
            "mia_mink_auroc": auc_mink
        }



