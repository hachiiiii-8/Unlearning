from typing import List, Dict, Optional

def run_harness(model_path: str, tasks: List[str], limit: Optional[int] = None,
                batch_size: Optional[int] = None) -> Dict[str, float]:
    try:
        from lm_eval import evaluator, tasks as lm_tasks
    except Exception as e:
        raise RuntimeError("Install lm-evaluation-harness to use run_harness()") from e

    lm_tasks.initialize_tasks()
    results = evaluator.simple_evaluate(
        model="hf-causal",
        model_args=f"pretrained={model_path},dtype=bfloat16",
        tasks=",".join(tasks),
        batch_size=batch_size or 4,
        limit=limit,
    )
    out = {}
    for t, v in results["results"].items():
        if "acc" in v: out[t] = v["acc"]
        elif "acc_norm" in v: out[t] = v["acc_norm"]
        elif "exact_match" in v: out[t] = v["exact_match"]
        else: out[t] = list(v.values())[0]
    return out
