"""Where things live.

Defaults are repo-relative, so a fresh clone works with no configuration. Override any of
them with an environment variable when your corpora sit elsewhere:

    LAYA_KO_ROOT        repo root                    (default: parent of scripts/)
    LAYA_KO_DATA        caches + built training set  (default: $ROOT/data)
    LAYA_KO_MODELS      checkpoints + ONNX builds    (default: $ROOT/models)
    LAYA_KO_EVAL_DIR    kmmlu/ mmlu/ jcommonsenseqa/ (default: $DATA/eval_datasets)
    LAYA_KO_AIHUB_DIR   AI-Hub jsonl corpora         (default: $DATA/korean_datasets)

`LAYA_KO_EVAL_DIR` and `LAYA_KO_AIHUB_DIR` point at data this repository does not
redistribute. Benchmarks that need them are skipped when they are absent; KLUE,
typed-decisions and Kev decision-v2 are downloaded at run time and need nothing here.
"""
import os

ROOT = os.environ.get(
    "LAYA_KO_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA = os.environ.get("LAYA_KO_DATA", os.path.join(ROOT, "data"))
MODELS = os.environ.get("LAYA_KO_MODELS", os.path.join(ROOT, "models"))
EVAL_DIR = os.environ.get("LAYA_KO_EVAL_DIR", os.path.join(DATA, "eval_datasets"))
AIHUB_DIR = os.environ.get("LAYA_KO_AIHUB_DIR", os.path.join(DATA, "korean_datasets"))


def data(*parts):
    return os.path.join(DATA, *parts)


def models(*parts):
    return os.path.join(MODELS, *parts)


def ensure(path):
    """Create the directory holding `path` and return `path`."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    return path
