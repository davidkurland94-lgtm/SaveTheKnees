"""The evaluation artifacts over HTTP -- what models_pipe.ipynb needs so a
teammate WITHOUT the dataset or the checkpoints can still run every cell.

Everything here is a thin JSON view over models/evaluate_labels.py and
models/report_quality.py, which read small CSVs under data/meta/. Where those
files are not mounted the endpoints answer 503, never a traceback.
"""

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(tags=["report"])


def _records(table):
    """DataFrame (label index) -> JSON-safe list of rows."""
    out = table.reset_index()
    # NaN is not JSON, and pandas keeps NaN in float columns even after
    # where(..., None); go through object dtype so None actually lands.
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict(orient="records")


def _guarded(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"evaluation artifact missing on the server: {exc}") from exc


@router.get("/report/table")
def report_table():
    """The full sheet: every reader vs the 58 gold studies (final_table)."""
    from models.evaluate_labels import final_table
    return {"rows": _records(_guarded(final_table))}


@router.get("/report/compare/{which}")
def report_compare(which: str):
    """image = David vs LLM · report = Kevin vs LLM · showdown = David vs best reader."""
    from models import evaluate_labels as ev
    fn = {"image": ev.image_model_vs_llm, "report": ev.report_model_vs_llm,
          "showdown": ev.david_vs_best_reader}.get(which)
    if fn is None:
        raise HTTPException(404, "which must be image | report | showdown")
    table = _guarded(fn)
    if table is None:
        raise HTTPException(503, "image model gold predictions not available yet")
    return {"rows": _records(table)}


@router.get("/report/verdicts")
def report_verdicts(top: int = Query(20, ge=1, le=500)):
    """Worst reports first: what the report said vs what the images think."""
    from models import report_quality as rq
    if not rq.OUT.exists():
        _guarded(rq.run, top=0)              # score the corpus once, quietly
    return {"rows": _guarded(rq.verdict_sheet, top).to_dict(orient="records")}


@router.get("/models/{name}/summary")
def model_summary(name: str):
    """Keras layer table + parameter count of one trained model, as text."""
    from functions.predict import _load_study_models, STUDY_MODELS
    if name == "report":
        from models.report_model import load_report_predictor
        try:
            model, _ = load_report_predictor()
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc
    elif name in STUDY_MODELS:
        models = _load_study_models(name)
        if not models:
            raise HTTPException(503, f"no checkpoint for '{name}' on the server")
        model = models[0]
    else:
        raise HTTPException(404, "name must be sagittal | multiplane | fusion | report")
    lines = []
    model.summary(line_length=100, print_fn=lines.append)
    return {"name": name, "parameters": int(model.count_params()), "summary": "\n".join(lines)}
