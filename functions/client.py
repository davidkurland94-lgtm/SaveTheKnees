"""HTTP client for the Save the Knees API -- the notebook's remote mode.

Mirrors the local one-liners (predict_study, predict_report, final_table,
verdict_sheet, ...) but fetches everything from a running API, so a teammate
with NO dataset, NO caches and NO checkpoints can run models_pipe.ipynb end to
end against Niko's machine (or any deployment):

    set STK_API=https://<tunnel-or-host>      # then open the notebook

Standard library only (urllib): the client must work in any environment with
pandas, which requirements-dev.txt already guarantees.
"""
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from functions.env import load_env

load_env()   # .env at the repo root; the shell environment wins over the file


def _api_base():
    """The API root, from STK_API (shell env or .env). Never hardcoded."""
    base = os.environ.get("STK_API", "").rstrip("/")
    if not base:
        raise RuntimeError("STK_API is not set -- export it or add "
                           "'STK_API=https://<api-url>' to the repo's .env "
                           "(see .env.example)")
    return base


def _get(path, base=None, timeout=300, **params):
    url = f"{base or _api_base()}{path}"
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(path, body, base=None, timeout=300):
    req = urllib.request.Request(f"{base or _api_base()}{path}",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _table(payload):
    df = pd.DataFrame(payload["rows"])
    return df.set_index("label") if "label" in df.columns else df


# --- the notebook one-liners, remote flavour ------------------------------

def health(base=None):
    return _get("/health", base)


def predict_study(study_uid, model="sagittal", base=None):
    """uid -> {model_status, label: prob} via GET /studies/{uid}/predict."""
    r = _get(f"/studies/{study_uid}/predict", base, model=model)
    return {"model_status": "trained", **r["predictions"]}


def predict_report(study_uid, base=None):
    """uid -> label probabilities from the report model, via the study's
    translated report (GET /studies/{uid}/report?lang=en -> POST /predict/report)."""
    text = _get(f"/studies/{study_uid}/report", base, lang="en")["report"]
    return _post_json("/predict/report", {"text": text}, base)["predictions"]


def predict_report_text(text, base=None):
    return _post_json("/predict/report", {"text": text}, base)["predictions"]


def final_table(base=None):
    return _table(_get("/report/table", base))


def image_model_vs_llm(base=None):
    return _table(_get("/report/compare/image", base))


def report_model_vs_llm(base=None):
    return _table(_get("/report/compare/report", base))


def david_vs_best_reader(base=None):
    return _table(_get("/report/compare/showdown", base))


def verdict_sheet(top=20, base=None):
    return pd.DataFrame(_get("/report/verdicts", base, top=top)["rows"])


def model_summary(name, base=None):
    """Print a trained model's Keras summary fetched from the server."""
    r = _get(f"/models/{name}/summary", base)
    print(f"{name} -- {r['parameters']:,} parameters (served remotely)")
    print(r["summary"])
