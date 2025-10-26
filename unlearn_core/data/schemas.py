from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class QASchema:
    """将不同数据集字段映射为统一字段"""
    prompt_key: str
    answer_key: Optional[str]
    group_key: str
    uid_key: Optional[str] = None

def get_schema(bench: Literal["tofu","muse_books","muse_news","wmdp"], subset: Optional[str]=None) -> QASchema:
    if bench == "tofu":
        # HF: locuslab/TOFU
        return QASchema(
            prompt_key="question",
            answer_key="answer",
            group_key="author_id",  
            uid_key="id",
        )
    if bench == "muse_books":
        # HF: muse-bench/MUSE-Books
        return QASchema(
            prompt_key="question",
            answer_key="answer",
            group_key="book_id",
            uid_key="id",
        )
    if bench == "muse_news":
        # HF: muse-bench/MUSE-News
        return QASchema(
            prompt_key="question",
            answer_key="answer",
            group_key="article_id",
        )
    if bench == "wmdp":
        # HF: cais/wmdp 
        return QASchema(
            prompt_key="question",
            answer_key="answer",   
            group_key="category",
            uid_key="id",
        )
    raise ValueError(f"Unknown bench: {bench}")
