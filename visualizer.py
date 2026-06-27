from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")


def _save_current_plot(output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_correlation_heatmap(dataframe, output_path):
    """Create a correlation heatmap from numeric columns."""
    numeric_df = dataframe.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None

    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="crest", fmt=".2f")
    plt.title("Correlation Heatmap", fontsize=14, weight="bold")
    _save_current_plot(output_path)
    return output_path


def generate_selected_plots(dataframe, selected_plots, output_dir, target_column=None):
    """Generate user-selected plots and return metadata for the results page."""
    plot_metadata = []
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()

    if "histogram" in selected_plots and numeric_columns:
        output_path = output_dir / "histogram.png"
        dataframe[numeric_columns[:6]].hist(figsize=(12, 8), bins=20, color="#0e7490", edgecolor="white")
        plt.suptitle("Histogram Overview", fontsize=14, weight="bold")
        _save_current_plot(output_path)
        plot_metadata.append({"title": "Histogram", "relative_path": f"plots/{output_path.name}"})

    if "box" in selected_plots and numeric_columns:
        output_path = output_dir / "box_plot.png"
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=dataframe[numeric_columns[:6]], palette="Set2")
        plt.xticks(rotation=25, ha="right")
        plt.title("Box Plot Overview", fontsize=14, weight="bold")
        _save_current_plot(output_path)
        plot_metadata.append({"title": "Box Plot", "relative_path": f"plots/{output_path.name}"})

    if "scatter" in selected_plots and len(numeric_columns) >= 2:
        output_path = output_dir / "scatter_plot.png"
        plt.figure(figsize=(9, 6))
        x_axis = numeric_columns[0]
        y_axis = numeric_columns[1]
        if target_column and target_column in dataframe.columns and dataframe[target_column].nunique() <= 10:
            sns.scatterplot(data=dataframe, x=x_axis, y=y_axis, hue=target_column, palette="tab10")
        else:
            sns.scatterplot(data=dataframe, x=x_axis, y=y_axis, color="#f97316")
        plt.title(f"Scatter Plot: {x_axis} vs {y_axis}", fontsize=14, weight="bold")
        _save_current_plot(output_path)
        plot_metadata.append({"title": "Scatter Plot", "relative_path": f"plots/{output_path.name}"})

    if "pair" in selected_plots and len(numeric_columns) >= 2:
        output_path = output_dir / "pair_plot.png"
        pair_columns = numeric_columns[: min(4, len(numeric_columns))]
        plot_df = dataframe[pair_columns].copy()
        if len(plot_df) >= 2:
            if target_column and target_column in dataframe.columns and dataframe[target_column].nunique() <= 10:
                plot_df[target_column] = dataframe[target_column]
                pair_plot = sns.pairplot(plot_df.sample(min(len(plot_df), 200), random_state=42), hue=target_column)
            else:
                pair_plot = sns.pairplot(plot_df.sample(min(len(plot_df), 200), random_state=42))
            pair_plot.fig.suptitle("Pair Plot", y=1.02, fontsize=14, weight="bold")
            pair_plot.savefig(output_path, dpi=180, bbox_inches="tight")
            plt.close("all")
            plot_metadata.append({"title": "Pair Plot", "relative_path": f"plots/{output_path.name}"})

    if "heatmap" in selected_plots:
        output_path = output_dir / "heatmap.png"
        saved_plot = generate_correlation_heatmap(dataframe, output_path)
        if saved_plot:
            plot_metadata.append({"title": "Heatmap", "relative_path": f"plots/{saved_plot.name}"})

    return plot_metadata
