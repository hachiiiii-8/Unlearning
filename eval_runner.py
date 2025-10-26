from unlearn_core.metrics.base import DecodingConfig
from unlearn_core.metrics.verbatim import VerbatimMetric
from unlearn_core.metrics.qa import QAMetric
from unlearn_core.metrics.mia import MIAMetric
from unlearn_core.metrics.hazard_wmdp import WMDPMetric
from unlearn_core.metrics.utility_harness import run_harness
from unlearn_core.metrics.report import build_report, save_report
from unlearn_core.data.augment import make_para_row, make_pert_row
from unlearn_core.metrics.prob import ProbMetric
from unlearn_core.metrics.extraction import ExtractionMetric
from unlearn_core.metrics.privacy_summary import privleak_score

from pathlib import Path
import json, argparse

def read_wmdp_jsonl(p):
    if not p:
        return []
    lines = Path(p).read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines]

def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines()]

def plain_prefix_suffix(r):
    prefix = r["question"].rstrip() + "\nAnswer: "
    suffix = r.get("answer", "")
    return prefix, suffix

def plain_prompt(r):
    return r["question"].rstrip() + "\nAnswer:"

def make_chat_prefix_suffix(tok, r):
    # chat 模板：更适合 Instruct 模型
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": r["question"].rstrip()}
    ]
    prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    suffix = r.get("answer","")
    return prefix, suffix

def make_chat_prompt(tok, r):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": r["question"].rstrip()}
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokenizer_dir", default=None)
    ap.add_argument("--tofu_eval_jsonl", required=True)    # holdout
    ap.add_argument("--members_jsonl", required=True)       # forget
    ap.add_argument("--nonmembers_jsonl", required=True)    # retain
    ap.add_argument("--out_json", default="out/tofu_eval_report.json")
    ap.add_argument("--wmdp_jsonl", default=None)
    ap.add_argument("--harness_tasks", default="")
    ap.add_argument("--harness_limit", type=int, default=None)
    ap.add_argument("--harness_batch", type=int, default=None)
    ap.add_argument("--eval_bs", type=int, default=2)
    ap.add_argument("--eval_maxlen", type=int, default=512)
    ap.add_argument("--show_progress", action="store_true")
    ap.add_argument("--save_intermediate", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true",
                    help="Use tokenizer.apply_chat_template for prompts (recommended for Instruct models)")
    args = ap.parse_args()

    # ---- load rows ----
    eval_rows  = read_jsonl(args.tofu_eval_jsonl)[:200]
    forget_rows= read_jsonl(args.members_jsonl)[:128]
    retain_rows= read_jsonl(args.nonmembers_jsonl)[:128]

    # ---- PARA/PERT ----
    forget_para = [make_para_row(r) for r in forget_rows]
    forget_pert = [make_pert_row(r) for r in forget_rows]
    retain_para = [make_para_row(r) for r in retain_rows]
    retain_pert = [make_pert_row(r) for r in retain_rows]

    # ---- intermediate dir ----
    intermediate_dir = Path("out/intermediate")
    if args.save_intermediate:
        intermediate_dir.mkdir(parents=True, exist_ok=True)

    # ==== Verbatim ====
    v = VerbatimMetric(args.model_dir, tokenizer_path=args.tokenizer_dir)
    if args.use_chat_template:
        tofu_prefix_suffix = lambda r: make_chat_prefix_suffix(v.tok, r)
    else:
        tofu_prefix_suffix = plain_prefix_suffix

    v_metrics = v.compute(
        eval_rows, tofu_prefix_suffix,
        max_length=args.eval_maxlen, batch_size=args.eval_bs, show_progress=args.show_progress
    )
    if args.save_intermediate:
        (intermediate_dir / "verbatim.json").write_text(json.dumps(v_metrics, indent=2, ensure_ascii=False), "utf-8")

    # ==== QA ====
    qa = QAMetric(args.model_dir, tokenizer_path=args.tokenizer_dir)
    if args.use_chat_template:
        tofu_prompt = lambda r: make_chat_prompt(qa.tok, r)
    else:
        tofu_prompt = plain_prompt

    cfg = DecodingConfig(max_new_tokens=64, temperature=0.7, top_p=0.9, do_sample=True, seed=123)
    qa_metrics = qa.compute(
        eval_rows, tofu_prompt, ref_key="answer", cfg=cfg, k_samples=3, show_progress=args.show_progress
    )
    if args.save_intermediate:
        (intermediate_dir / "qa.json").write_text(json.dumps(qa_metrics, indent=2, ensure_ascii=False), "utf-8")

    # ==== MIA ====
    mia = MIAMetric(args.model_dir, tokenizer_path=args.tokenizer_dir)
    if args.use_chat_template:
        def to_texts(rows):
            return [make_chat_prompt(v.tok, r) + r.get("answer","") for r in rows]
    else:
        def to_texts(rows):
            return [r["question"].rstrip() + "\nAnswer: " + (r.get("answer","")) for r in rows]

    mia_metrics = mia.compute(
        to_texts(forget_rows), to_texts(retain_rows),
        k=20, max_length=args.eval_maxlen, batch_size=args.eval_bs, show_progress=args.show_progress
    )
    if args.save_intermediate:
        (intermediate_dir / "mia.json").write_text(json.dumps(mia_metrics, indent=2, ensure_ascii=False), "utf-8")

    # ==== Prob-based (Q_A_Prob / PARA / PERT / normalised) ====
    prob = ProbMetric(args.model_dir, tokenizer_path=args.tokenizer_dir)
    if args.use_chat_template:
        pf = lambda r: make_chat_prompt(prob.tok, r)
    else:
        pf = plain_prompt

    fg_probs      = prob.answer_mean_logprob(forget_rows, pf, "answer",
                        max_length=args.eval_maxlen, batch_size=args.eval_bs,
                        show_progress=args.show_progress, desc="Calculating loss")
    fg_para_probs = prob.answer_mean_logprob(forget_para, pf, "answer",
                        max_length=args.eval_maxlen, batch_size=args.eval_bs,
                        show_progress=args.show_progress, desc="Calculating loss (PARA)")
    fg_pert_probs = prob.answer_mean_logprob(forget_pert, pf, "answer",
                        max_length=args.eval_maxlen, batch_size=args.eval_bs,
                        show_progress=args.show_progress, desc="Calculating loss (PERT)")
    # 可选：retain 也算一份
    # rt_probs = prob.answer_mean_logprob(retain_rows, pf, "answer", ...)

    # ==== Extraction Strength ====
    es = ExtractionMetric(args.model_dir, tokenizer_path=args.tokenizer_dir)
    es_metrics = es.compute(
        forget_rows, ref_key="answer", k_samples=3, tau=0.7,
        cfg=DecodingConfig(max_new_tokens=96, temperature=0.7, top_p=0.9),
        show_progress=args.show_progress
    )

    # ==== Hazard (WMDP) ====
    hazard_metrics = {}
    if args.wmdp_jsonl:
        wmdp_rows = read_wmdp_jsonl(args.wmdp_jsonl)
        if wmdp_rows:
            wmdp = WMDPMetric(args.model_dir, tokenizer_path=args.tokenizer_dir)
            hazard_metrics = wmdp.compute(wmdp_rows)
            if args.save_intermediate:
                (intermediate_dir / "wmdp.json").write_text(json.dumps(hazard_metrics, indent=2, ensure_ascii=False), "utf-8")

    # ==== Utility (harness) ====
    utility_metrics = {}
    if args.harness_tasks:
        tasks = [t.strip() for t in args.harness_tasks.split(",") if t.strip()]
        if tasks:
            try:
                utility_metrics = run_harness(
                    args.model_dir, tasks=tasks,
                    limit=args.harness_limit, batch_size=args.harness_batch
                )
            except Exception as e:
                utility_metrics = {"error": str(e)}
            if args.save_intermediate:
                (intermediate_dir / "utility.json").write_text(json.dumps(utility_metrics, indent=2, ensure_ascii=False), "utf-8")

    # ==== 组装 metrics（最后一步再建字典，避免覆盖） ====
    forget_panel = {
        **v_metrics,
        **qa_metrics,
        "forget_Q_A_Prob":           float(sum(fg_probs)/max(len(fg_probs),1)),
        "forget_Q_A_PARA_Prob":      float(sum(fg_para_probs)/max(len(fg_para_probs),1)),
        "forget_Q_A_PERT_Prob":      float(sum(fg_pert_probs)/max(len(fg_pert_probs),1)),
        **es_metrics,
    }

    privacy_panel = {
        **mia_metrics,
        **privleak_score(mia_metrics),   # {"privleak": ...}
    }

    metrics = {
        "forget":  forget_panel,
        "privacy": privacy_panel,
        "hazard":  hazard_metrics,
        "utility": utility_metrics,
        # 你也可以加 "ra"/"wf" 面板，这里先略
    }

    # ==== 保存报告 ====
    rep = build_report(
        model_id=args.model_dir, base_arch="Llama-3.2-1B-Instruct",
        method="GA/BASELINE", forget_ratio=0.10,
        decoding_cfg=cfg.__dict__, metrics=metrics, retain_ref=None
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    save_report(rep, args.out_json)
    print(f"Saved report to {args.out_json}")



""" 
python eval_runner.py \
  --model_dir ../ckpts/Llama-3.2-1B-Instruct \
  --tofu_eval_jsonl data/tofu/tofu_f10_holdout.jsonl \
  --members_jsonl   data/tofu/tofu_f10_forget.jsonl \
  --nonmembers_jsonl data/tofu/tofu_f10_retain.jsonl \
  --wmdp_jsonl data/wmdp/test_mcq.jsonl \
  --harness_tasks mmlu,hellaswag \
  --harness_limit 100 \
  --harness_batch 4 \
  --eval_bs 32 \
  --eval_maxlen 768 \
  --show_progress \
  --save_intermediate \
  --use_chat_template \
  --out_json out/tofu_full_eval.full.json

"""



