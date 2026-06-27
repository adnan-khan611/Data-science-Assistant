def _format_metric(value):
    """Format metrics consistently for rule-based recommendations."""
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def generate_decision_suggestions(problem_type, model_name, metrics, cleaning_summary, top_features):
    """Create beginner-friendly business insights and recommended next actions."""
    insights = []
    recommendations = []

    duplicates_removed = cleaning_summary.get("duplicates_removed", 0)
    missing_handled = cleaning_summary.get("missing_values_handled", 0)

    if duplicates_removed > 0:
        insights.append(
            f"The dataset contained {duplicates_removed} duplicate rows, which were removed before training."
        )

    if missing_handled > 0:
        insights.append(
            f"The pipeline handled {missing_handled} missing values automatically to keep the analysis stable."
        )

    if problem_type == "classification":
        accuracy = metrics.get("accuracy")
        f1_score = metrics.get("f1_score")

        if accuracy is not None and accuracy >= 0.9:
            insights.append("The classifier is performing strongly and can support confident day-to-day predictions.")
        elif accuracy is not None and accuracy >= 0.75:
            insights.append("The classifier shows useful predictive power and is suitable for analyst-assisted decisions.")
        else:
            insights.append("The classifier needs improvement before being used for high-confidence decisions.")

        recommendations.append(
            f"Track both accuracy ({_format_metric(accuracy)}) and F1-score ({_format_metric(f1_score)}) in future retraining cycles."
        )

    elif problem_type == "regression":
        r2_score = metrics.get("r2_score")
        rmse = metrics.get("rmse")

        if r2_score is not None and r2_score >= 0.85:
            insights.append("The regression model explains most of the outcome variation, which is a strong sign of business fit.")
        elif r2_score is not None and r2_score >= 0.6:
            insights.append("The regression model captures a meaningful signal, but there is room to improve forecast precision.")
        else:
            insights.append("The regression model is still noisy, so predictions should be reviewed carefully before business use.")

        recommendations.append(
            f"Use RMSE ({_format_metric(rmse)}) as the main operational error benchmark for future model comparisons."
        )

    else:
        silhouette = metrics.get("silhouette_score")
        cluster_count = metrics.get("cluster_count")

        if silhouette is not None and silhouette >= 0.5:
            insights.append("The clustering pattern is reasonably well separated, suggesting distinct groups exist in the data.")
        else:
            insights.append("The clustering structure is weak, which may indicate overlapping segments or limited feature signal.")

        recommendations.append(
            f"Review the discovered {cluster_count} clusters with business teams and validate whether they match real operational segments."
        )

    if top_features:
        top_feature_names = ", ".join(feature["feature"] for feature in top_features[:3])
        insights.append(f"The most influential drivers identified by the {model_name} model are: {top_feature_names}.")
        recommendations.append("Use the top drivers to guide feature monitoring, policy design, or data collection priorities.")

    recommendations.append("Retrain the pipeline with a different model to compare performance before locking in a final choice.")
    recommendations.append("If results are business-critical, add domain-specific feature engineering and validation rules next.")

    return {
        "insights": insights,
        "recommendations": recommendations,
    }
