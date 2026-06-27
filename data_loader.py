from pathlib import Path

import pandas as pd


ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}


def allowed_file(filename):
    """Check whether the uploaded file has a supported extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataset(file_path):
    """Load a CSV or Excel dataset into a pandas DataFrame."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)
    if extension in {".xls", ".xlsx"}:
        return pd.read_excel(path)

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def build_preview_payload(dataframe, preview_rows=5):
    """Prepare a JSON-safe preview payload for the UI."""
    preview_html = dataframe.head(preview_rows).to_html(
        classes="table table-striped table-hover align-middle mb-0",
        index=False,
        border=0,
    )

    return {
        "shape": {"rows": int(dataframe.shape[0]), "columns": int(dataframe.shape[1])},
        "columns": dataframe.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "missing_values": dataframe.isna().sum().to_dict(),
        "preview_html": preview_html,
    }
