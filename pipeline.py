from pathlib import Path
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils.insights import generate_decision_suggestions
from utils.data_loader import load_dataset
from utils.reporter import build_report
from utils.visualizer import generate_correlation_heatmap, generate_selected_plots


MODEL_CATALOG = {
    "classification": [
        "Logistic Regression",
        "Random Forest",
        "Decision Tree",
        "KNN",
        "SVM",
    ],
    "regression": [
        "Linear Regression",
        "Random Forest",
        "Decision Tree",
        "KNN",
        "SVM",
    ],
    "clustering": [
        "KMeans",
    ],
}

PLOT_OPTIONS = [
    {"value": "histogram", "label": "Histogram"},
    {"value": "box", "label": "Box Plot"},
    {"value": "scatter", "label": "Scatter Plot"},
    {"value": "pair", "label": "Pair Plot"},
    {"value": "heatmap", "label": "Heatmap"},
]


def problem_statement(problem_type, target_column):
    """Generate a simple problem definition from the selected workflow."""
    if problem_type == "classification":
        return f"Predict the category of '{target_column}' using the uploaded dataset."
    if problem_type == "regression":
        return f"Predict the numeric value of '{target_column}' using the uploaded dataset."
    return "Discover natural groups inside the uploaded dataset without using a target column."


def _build_preprocessor(X):
    """Create a preprocessing pipeline for numeric and categorical features."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def _create_estimator(problem_type, model_name, sample_count=None):
    """Map the chosen UI model name to a scikit-learn estimator."""
    if problem_type == "classification":
        estimator_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(probability=True),
        }
    elif problem_type == "regression":
        estimator_map = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "KNN": KNeighborsRegressor(),
            "SVM": SVR(),
        }
    else:
        cluster_count = 3 if sample_count is None else max(2, min(3, int(sample_count)))
        estimator_map = {
            "KMeans": KMeans(n_clusters=cluster_count, n_init=10, random_state=42),
        }

    return estimator_map[model_name]


def _extract_feature_names(preprocessor, feature_frame):
    """Return transformed feature names after one-hot encoding."""
    return preprocessor.get_feature_names_out(feature_frame.columns).tolist()


def _extract_top_features(trained_model, feature_names):
    """Extract the most influential features when the estimator supports it."""
    estimator = trained_model.named_steps.get("model", trained_model)

    if hasattr(estimator, "feature_importances_"):
        weights = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coefficients = estimator.coef_
        weights = np.mean(np.abs(coefficients), axis=0) if np.ndim(coefficients) > 1 else np.abs(coefficients)
    else:
        return []

    ranking = sorted(
        (
            {"feature": feature_names[index], "importance": float(weight)}
            for index, weight in enumerate(weights)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranking[:5]


def _plot_confusion_matrix(y_true, y_pred, output_path):
    """Save a confusion matrix heatmap for classification jobs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=14, weight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_cluster_distribution(cluster_labels, output_path):
    """Save a simple bar chart showing cluster membership."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    counts = pd.Series(cluster_labels).value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="crest")
    plt.title("Cluster Distribution", fontsize=14, weight="bold")
    plt.xlabel("Cluster")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def _summarize_cleaning(original_df, cleaned_df, target_column=None):
    """Build a short data-cleaning summary."""
    duplicates_removed = int(original_df.duplicated().sum())
    missing_values_handled = int(original_df.isna().sum().sum())

    return {
        "original_rows": int(original_df.shape[0]),
        "original_columns": int(original_df.shape[1]),
        "cleaned_rows": int(cleaned_df.shape[0]),
        "cleaned_columns": int(cleaned_df.shape[1]),
        "duplicates_removed": duplicates_removed,
        "missing_values_handled": missing_values_handled,
        "target_column": target_column or "Not used",
    }


def _prepare_supervised_data(dataframe, target_column, problem_type):
    """Separate features and target and encode the target when necessary."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    working_df = dataframe.copy().drop_duplicates()
    working_df = working_df.dropna(subset=[target_column]).copy()

    X = working_df.drop(columns=[target_column])
    y = working_df[target_column].copy()

    target_encoder = None

    if problem_type == "classification":
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=working_df.index, name=target_column)
        if y.nunique() < 2:
            raise ValueError("Classification needs at least two target classes after cleaning.")
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid_rows = y.notna()
        X = X.loc[valid_rows].copy()
        working_df = working_df.loc[valid_rows].copy()
        y = y.loc[valid_rows].copy()
        if y.empty:
            raise ValueError("The selected target column must contain numeric values for regression.")

    return working_df, X, y, target_encoder


def _prepare_clustering_data(dataframe):
    """Prepare feature data for clustering problems."""
    working_df = dataframe.copy().drop_duplicates()
    return working_df, working_df.copy()


def run_analysis(
    dataset_path,
    problem_type,
    target_column,
    model_name,
    selected_plots,
    output_dir,
    model_dir,
    progress_callback,
    templates_dir,
):
    """Execute the full end-to-end automated data science lifecycle."""
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    progress_callback(10, "Loading dataset", "Reading the uploaded file into pandas.")
    original_df = load_dataset(dataset_path)

    if original_df.empty:
        raise ValueError("The uploaded dataset is empty.")

    progress_callback(18, "Defining problem", "Creating the workflow objective from the selected analysis type.")
    problem_definition = problem_statement(problem_type, target_column)

    progress_callback(30, "Cleaning data", "Handling duplicates, missing values, and categorical fields.")
    if problem_type in {"classification", "regression"}:
        cleaned_df, feature_frame, target_series, target_encoder = _prepare_supervised_data(
            original_df, target_column, problem_type
        )
    else:
        cleaned_df, feature_frame = _prepare_clustering_data(original_df)
        target_series = None
        target_encoder = None

    if cleaned_df.empty:
        raise ValueError("No usable rows remained after cleaning. Please review missing values and target column quality.")

    if feature_frame.shape[1] == 0:
        raise ValueError("No feature columns are available after removing the target column.")

    cleaning_summary = _summarize_cleaning(original_df, cleaned_df, target_column if problem_type != "clustering" else None)

    progress_callback(42, "Exploring data", "Generating summary statistics and automatic heatmap.")
    summary_stats_html = cleaned_df.describe(include="all").fillna("").round(3).to_html(
        classes="table table-striped table-sm align-middle",
        border=0,
    )

    auto_plot_paths = []
    heatmap_path = generate_correlation_heatmap(cleaned_df, plots_dir / "correlation_heatmap.png")
    if heatmap_path:
        auto_plot_paths.append({"title": "Correlation Heatmap", "relative_path": f"plots/{heatmap_path.name}"})

    progress_callback(55, "Generating plots", "Creating the visualizations selected in the dashboard.")
    selected_plot_paths = generate_selected_plots(cleaned_df, selected_plots, plots_dir, target_column=target_column)

    progress_callback(68, "Building model", "Preparing the preprocessing and machine learning pipeline.")
    preprocessor, numeric_features, categorical_features = _build_preprocessor(feature_frame)
    estimator = _create_estimator(problem_type, model_name, sample_count=len(feature_frame))
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    metrics = {}
    top_features = []
    evaluation_plots = []
    evaluation_notes = []

    if problem_type in {"classification", "regression"}:
        if len(feature_frame) < 4:
            raise ValueError("Please upload a dataset with at least 4 usable rows for supervised learning.")

        stratify = target_series if problem_type == "classification" and target_series.nunique() > 1 else None
        test_size = 0.2 if len(feature_frame) >= 10 else 0.5

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                feature_frame,
                target_series,
                test_size=test_size,
                random_state=42,
                stratify=stratify,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                feature_frame,
                target_series,
                test_size=test_size,
                random_state=42,
            )

        progress_callback(78, "Training model", "Fitting the selected model on the training split.")
        model_pipeline.fit(X_train, y_train)

        progress_callback(86, "Evaluating model", "Scoring the fitted model on the test split.")
        predictions = model_pipeline.predict(X_test)
        transformed_feature_names = _extract_feature_names(preprocessor, feature_frame)
        top_features = _extract_top_features(model_pipeline, transformed_feature_names)

        if problem_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, predictions, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            }

            confusion_path = plots_dir / "confusion_matrix.png"
            _plot_confusion_matrix(y_test, predictions, confusion_path)
            evaluation_plots.append({"title": "Confusion Matrix", "relative_path": f"plots/{confusion_path.name}"})

            if target_encoder is not None:
                metrics["classes"] = target_encoder.classes_.tolist()
        else:
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "r2_score": float(r2_score(y_test, predictions)),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            }

        evaluation_notes.append("The dataset was split into training and testing subsets automatically.")
    else:
        if len(feature_frame) < 2:
            raise ValueError("Clustering needs at least two usable rows after cleaning.")

        progress_callback(78, "Training model", "Running unsupervised clustering on the prepared features.")
        cluster_labels = model_pipeline.fit_predict(feature_frame)
        cleaned_df = cleaned_df.copy()
        cleaned_df["cluster_label"] = cluster_labels

        progress_callback(86, "Evaluating model", "Reviewing clustering quality with segmentation metrics.")
        transformed_features = model_pipeline.named_steps["preprocessor"].transform(feature_frame)
        unique_clusters = len(np.unique(cluster_labels))
        if 1 < unique_clusters < len(feature_frame):
            silhouette = float(silhouette_score(transformed_features, cluster_labels))
        else:
            silhouette = None

        metrics = {
            "cluster_count": int(unique_clusters),
            "silhouette_score": silhouette,
            "inertia": float(model_pipeline.named_steps["model"].inertia_),
            "rows_clustered": int(len(feature_frame)),
        }

        cluster_plot = plots_dir / "cluster_distribution.png"
        _plot_cluster_distribution(cluster_labels, cluster_plot)
        evaluation_plots.append({"title": "Cluster Distribution", "relative_path": f"plots/{cluster_plot.name}"})
        evaluation_notes.append("Train/test split is skipped for clustering because the task is unsupervised.")

    progress_callback(92, "Saving artifacts", "Writing the cleaned data, model, and metadata to disk.")
    cleaned_dataset_path = output_dir / "cleaned_dataset.csv"
    cleaned_df.to_csv(cleaned_dataset_path, index=False)

    model_artifact = {
        "model": model_pipeline,
        "problem_type": problem_type,
        "model_name": model_name,
        "target_column": target_column,
        "feature_columns": feature_frame.columns.tolist(),
        "top_features": top_features,
    }

    model_path = model_dir / f"{Path(dataset_path).stem}_{problem_type}_{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model_artifact, model_path)
    shutil.copy2(model_path, output_dir / "trained_model.pkl")

    insights = generate_decision_suggestions(problem_type, model_name, metrics, cleaning_summary, top_features)

    result_payload = {
        "dataset_name": Path(dataset_path).name,
        "problem_type": problem_type.title(),
        "problem_definition": problem_definition,
        "model_name": model_name,
        "target_column": target_column or "Not required",
        "cleaning_summary": cleaning_summary,
        "summary_stats_html": summary_stats_html,
        "metrics": metrics,
        "top_features": top_features,
        "selected_plots": selected_plot_paths,
        "auto_plots": auto_plot_paths,
        "evaluation_plots": evaluation_plots,
        "evaluation_notes": evaluation_notes,
        "insights": insights,
        "cleaned_dataset_filename": cleaned_dataset_path.name,
        "model_filename": "trained_model.pkl",
        "selected_plot_keys": selected_plots,
    }

    progress_callback(97, "Building report", "Rendering the final HTML report and ZIP package.")
    report_path = build_report(result_payload, templates_dir, output_dir)

    summary_json_path = output_dir / "analysis_summary.json"
    summary_json_path.write_text(json.dumps(result_payload, indent=2, default=str), encoding="utf-8")

    zip_base = output_dir.parent / f"{output_dir.name}_bundle"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=output_dir)

    result_payload["report_path"] = str(report_path)
    result_payload["zip_path"] = zip_path
    result_payload["cleaned_preview_html"] = cleaned_df.head(10).to_html(
        classes="table table-striped table-hover align-middle mb-0",
        index=False,
        border=0,
    )

    return result_payload
