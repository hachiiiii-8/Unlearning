from typing import List, Dict
from .prob import ProbMetric

class TruthRatioMetric(ProbMetric):
    def compute_pair(self, rows: List[Dict], prompt_fn, true_key="answer", neg_key="neg_answer",
                     max_length=768, batch_size=2, show_progress=False, tag="Truth_Ratio") -> dict:
        # 要求每行有 answer & neg_answer（或自己预处理构造）
        pairs = []
        for r in rows:
            pairs.append((prompt_fn(r), r.get(true_key,""), r.get(neg_key,"")))
        # 扁平化做 logP
        flat = [p + " " + a for p,a,_ in pairs] + [p + " " + n for p,_,n in pairs]
        vals = self.answer_mean_logprob(
            [{"prompt": f, "answer": ""} for f in [{"prompt": ""}]*0],  # 不用这个签名，直接改内部
            prompt_fn=lambda x: x["prompt"], ref_key="answer",
            # 为简化，直接改 answer_mean_logprob 接受 texts 列表更好；这里给思路
        )
        # 实际实现时你可以直接写一个 answer_mean_logprob_texts(texts: List[str])，更简单。
        return {f"{tag}": 0.0}  # 占位：见下方更实用的实现提示
