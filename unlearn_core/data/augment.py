# unlearn_core/data/augment.py
import random, re

def paraphrase_prompt(p: str) -> str:
    # 极简替代：规则/同义词（建议后续换成小模型改写）
    p = p.replace("Explain", "Describe").replace("What is", "Define")
    return p

def perturb_prompt(p: str, prob=0.08) -> str:
    # 轻度字符/标点/空白扰动；保持语义大体不变
    out = []
    for ch in p:
        if random.random() < prob:
            choice = random.choice(["drop","space","punct"])
            if choice == "drop":  continue
            if choice == "space": out.append(" "); continue
            if choice == "punct": out.append(random.choice([",",".",";","?","!"])); continue
        out.append(ch)
    # 清理多空格
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()

def make_para_row(r: dict) -> dict:
    return {**r, "prompt": paraphrase_prompt(r["question"])}

def make_pert_row(r: dict) -> dict:
    return {**r, "prompt": perturb_prompt(r["question"])}
