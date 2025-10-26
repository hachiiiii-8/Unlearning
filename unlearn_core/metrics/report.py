import json, time, platform
from typing import Dict, Any

def build_report(model_id: str, base_arch: str, method: str, forget_ratio: float,
                 decoding_cfg: Dict[str, Any], metrics: Dict[str, Any], retain_ref: str|None=None):
    return {
        "model_id": model_id,
        "base_arch": base_arch,
        "unlearn_method": method,
        "forget_ratio": forget_ratio,
        "decoding_cfg": decoding_cfg,
        "metrics": metrics,
        "retain_ref": retain_ref,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": {"python": platform.python_version(), "system": platform.platform()},
        "version": {"impl": "unlearn_core.metrics.base_v1"}
    }

def save_report(report: Dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

