"""One import for models_pipe.ipynb, two backends.

    from functions.pipe import predict_study, final_table, ...

LOCAL  (default)   -- the functions run here, against data/ and models/ on disk.
REMOTE (STK_API set) -- the same names call a running API over HTTP, so the
                      notebook works on a laptop that has neither the dataset
                      nor the checkpoints:

    set STK_API=https://<tunnel-or-host>     (Windows)
    export STK_API=https://<tunnel-or-host>  (mac / linux)

Every cell in the notebook is written once; this module decides where it runs.
"""
import os

from functions.env import load_env

load_env()   # .env can set STK_API too; the shell environment wins
STK_API = os.environ.get("STK_API", "").rstrip("/")
REMOTE = bool(STK_API)
MODE = f"remote -> {STK_API}" if REMOTE else "local"

if REMOTE:
    from functions.client import (                      # noqa: F401
        predict_study, predict_report, predict_report_text,
        final_table, image_model_vs_llm, report_model_vs_llm,
        david_vs_best_reader, verdict_sheet, model_summary,
    )
    from functions.client import _get as _remote_get

    def styled(table):
        from models.evaluate_labels import styled as _styled   # pure pandas, no data
        return _styled(table)

    def plot_readers():
        from models.evaluate_labels import plot_readers as _plot
        return _plot(table=final_table())

    def run_verdicts(top=10):
        """Remote: the certificates were computed server-side; show the sheet."""
        return verdict_sheet(top)

else:
    from functions.predict import predict_study                          # noqa: F401
    from models.report_model import predict_report, predict_report_text  # noqa: F401
    from models.evaluate_labels import (                                 # noqa: F401
        final_table, image_model_vs_llm, report_model_vs_llm,
        david_vs_best_reader, styled, plot_readers,
    )
    from models.report_quality import verdict_sheet                      # noqa: F401

    def run_verdicts(top=10):
        from models.report_quality import run
        run(top=0)                 # calibrate + validate + score; prints certificates
        return verdict_sheet(top)

    def model_summary(name):
        """Print a trained model's Keras summary from the local checkpoint."""
        import keras
        from functions.predict import REPO_ROOT, STUDY_MODELS
        path = REPO_ROOT / "models" / ("report_model.keras" if name == "report"
                                       else STUDY_MODELS[name][0])
        m = keras.saving.load_model(path, compile=False)
        print(f"{name} -- {m.count_params():,} parameters")
        m.summary(line_length=100)
