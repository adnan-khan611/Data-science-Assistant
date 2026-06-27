from pathlib import Path
import shutil
import threading
import traceback
import uuid

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename

from utils.data_loader import allowed_file, build_preview_payload, load_dataset
from utils.pipeline import MODEL_CATALOG, PLOT_OPTIONS, run_analysis


# Application setup
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["OUTPUT_FOLDER"] = str(OUTPUT_DIR)
app.config["MODEL_FOLDER"] = str(MODEL_DIR)
app.secret_key = "data-pilot-ai-secret-key"

for directory in (UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# In-memory job tracker for progress polling
JOB_STORE = {}
JOB_LOCK = threading.Lock()


def update_job(job_id, **payload):
    """Safely update the status of a background job."""
    with JOB_LOCK:
        JOB_STORE.setdefault(job_id, {})
        JOB_STORE[job_id].update(payload)


def get_job(job_id):
    """Read a job record from memory."""
    with JOB_LOCK:
        return JOB_STORE.get(job_id)


def save_uploaded_file(file_storage, job_id):
    """Save an uploaded dataset with a unique name."""
    filename = secure_filename(file_storage.filename)
    saved_name = f"{job_id}_{filename}"
    save_path = UPLOAD_DIR / saved_name
    file_storage.save(save_path)
    return save_path


def background_analysis_job(job_id, file_path, options):
    """Run the full automated data science workflow in a background thread."""
    try:
        output_path = OUTPUT_DIR / job_id
        output_path.mkdir(parents=True, exist_ok=True)

        def report_progress(percent, stage, detail=""):
            update_job(
                job_id,
                status="running",
                progress=percent,
                stage=stage,
                detail=detail,
            )

        report_progress(5, "Job created", "Dataset saved and queued for analysis.")

        result = run_analysis(
            dataset_path=file_path,
            problem_type=options["problem_type"],
            target_column=options.get("target_column"),
            model_name=options["model_name"],
            selected_plots=options["selected_plots"],
            output_dir=output_path,
            model_dir=MODEL_DIR,
            progress_callback=report_progress,
            templates_dir=BASE_DIR / "templates",
        )

        result["result_url"] = f"/results/{job_id}"
        result["download_url"] = f"/download/{job_id}"
        result["report_url"] = f"/artifacts/{job_id}/report.html"
        result["source_dataset_path"] = str(file_path)

        update_job(
            job_id,
            status="completed",
            progress=100,
            stage="Completed",
            detail="Analysis finished successfully.",
            results=result,
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            progress=100,
            stage="Failed",
            detail=str(exc),
            error=str(exc),
            traceback=traceback.format_exc(),
        )


@app.route("/")
def index():
    """Render the home page."""
    return render_template(
        "index.html",
        model_catalog=MODEL_CATALOG,
        plot_options=PLOT_OPTIONS,
    )


@app.route("/preview", methods=["POST"])
def preview_dataset():
    """Preview an uploaded dataset before running the analysis."""
    uploaded_file = request.files.get("dataset")

    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"success": False, "message": "Please choose a dataset file first."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"success": False, "message": "Only CSV and Excel files are supported."}), 400

    temp_id = f"preview_{uuid.uuid4().hex}"
    temp_path = save_uploaded_file(uploaded_file, temp_id)

    try:
        dataframe = load_dataset(temp_path)
        preview_payload = build_preview_payload(dataframe)
        return jsonify({"success": True, **preview_payload})
    except Exception as exc:
        return jsonify({"success": False, "message": f"Unable to preview dataset: {exc}"}), 400
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.route("/start-analysis", methods=["POST"])
def start_analysis():
    """Validate the request and launch the background analysis thread."""
    problem_type = request.form.get("problem_type", "").strip().lower()
    model_name = request.form.get("model_name", "").strip()
    selected_plots = request.form.getlist("plots")
    target_column = request.form.get("target_column", "").strip() or None
    existing_file_path = request.form.get("existing_file_path", "").strip()

    if problem_type not in MODEL_CATALOG:
        return jsonify({"success": False, "message": "Please select a valid problem type."}), 400

    valid_models = MODEL_CATALOG[problem_type]
    if model_name not in valid_models:
        return jsonify({"success": False, "message": "Please choose a model that matches the selected problem type."}), 400

    if problem_type in {"classification", "regression"} and not target_column:
        return jsonify({"success": False, "message": "Target column is required for classification and regression."}), 400

    if target_column == "__none__":
        target_column = None

    job_id = uuid.uuid4().hex

    if existing_file_path:
        dataset_path = Path(existing_file_path)
        if not dataset_path.exists():
            return jsonify({"success": False, "message": "The original dataset file could not be found for retraining."}), 400
    else:
        uploaded_file = request.files.get("dataset")
        if not uploaded_file or uploaded_file.filename == "":
            return jsonify({"success": False, "message": "Please upload a dataset file."}), 400
        if not allowed_file(uploaded_file.filename):
            return jsonify({"success": False, "message": "Only CSV and Excel files are supported."}), 400
        dataset_path = save_uploaded_file(uploaded_file, job_id)

    update_job(
        job_id,
        status="queued",
        progress=0,
        stage="Queued",
        detail="Waiting for the worker thread to start.",
        result_url=url_for("results", job_id=job_id),
    )

    worker = threading.Thread(
        target=background_analysis_job,
        args=(
            job_id,
            dataset_path,
            {
                "problem_type": problem_type,
                "target_column": target_column,
                "model_name": model_name,
                "selected_plots": selected_plots,
            },
        ),
        daemon=True,
    )
    worker.start()

    return jsonify(
        {
            "success": True,
            "job_id": job_id,
            "progress_url": url_for("job_progress", job_id=job_id),
            "result_url": url_for("results", job_id=job_id),
        }
    )


@app.route("/progress/<job_id>")
def job_progress(job_id):
    """Return the live progress of a running analysis job."""
    job = get_job(job_id)
    if not job:
        return jsonify({"success": False, "message": "Job not found."}), 404
    return jsonify({"success": True, **job})


@app.route("/results/<job_id>")
def results(job_id):
    """Render the results dashboard for a completed job."""
    job = get_job(job_id)
    if not job:
        return render_template("results.html", job_id=job_id, state="missing")

    if job.get("status") == "completed":
        return render_template(
            "results.html",
            job_id=job_id,
            state="completed",
            job=job,
            results=job["results"],
            model_catalog=MODEL_CATALOG,
            plot_options=PLOT_OPTIONS,
        )

    if job.get("status") == "failed":
        return render_template("results.html", job_id=job_id, state="failed", job=job)

    return render_template("results.html", job_id=job_id, state="running", job=job)


@app.route("/download/<job_id>")
def download_bundle(job_id):
    """Download the ZIP bundle created for a completed job."""
    job = get_job(job_id)
    if not job or job.get("status") != "completed":
        return jsonify({"success": False, "message": "The ZIP bundle is not ready yet."}), 404

    zip_path = Path(job["results"]["zip_path"])
    if not zip_path.exists():
        return jsonify({"success": False, "message": "ZIP file was not found."}), 404

    return send_file(zip_path, as_attachment=True)


@app.route("/artifacts/<job_id>/<path:filename>")
def artifact_file(job_id, filename):
    """Serve generated plots and report files from a job directory."""
    job_output_dir = OUTPUT_DIR / job_id
    return send_from_directory(job_output_dir, filename)


@app.route("/cleanup/<job_id>", methods=["POST"])
def cleanup_job(job_id):
    """Optional cleanup endpoint to remove generated job artifacts."""
    job = get_job(job_id)
    if not job:
        return jsonify({"success": False, "message": "Job not found."}), 404

    output_path = OUTPUT_DIR / job_id
    if output_path.exists():
        shutil.rmtree(output_path, ignore_errors=True)

    return jsonify({"success": True, "message": "Job artifacts removed from outputs folder."})


if __name__ == "__main__":
    app.run(debug=True)
