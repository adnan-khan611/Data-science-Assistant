# Data Pilot AI

Data Pilot AI is a Flask-based Auto Data Science web application that lets a user upload a dataset, choose a machine learning task, run an automated pipeline, and download a complete bundle of outputs.

## Features

- Upload CSV and Excel datasets
- Preview the dataset before training
- Select problem type:
  - Classification
  - Regression
  - Clustering
- Run automated data cleaning:
  - Duplicate removal
  - Missing value handling
  - Categorical encoding
- Generate exploratory analysis:
  - Summary statistics
  - Correlation heatmap
- Train machine learning models:
  - Linear Regression
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - KNN
  - SVM
  - KMeans for clustering
- Evaluate the trained model:
  - Accuracy, precision, recall, F1-score for classification
  - RMSE, MAE, R2-score for regression
  - Silhouette score and inertia for clustering
  - Confusion matrix for classification
- Generate visualizations:
  - Histogram
  - Box Plot
  - Scatter Plot
  - Pair Plot
  - Heatmap
- Save the trained model as a `.pkl` file
- Build an HTML report automatically
- Create a downloadable ZIP bundle containing:
  - Cleaned dataset
  - Trained model
  - Plots
  - Report
- Suggest business insights and next-step recommendations
- Retrain with another model from the results page

## Project Structure

```text
Data pilot AI/
|-- app.py
|-- requirements.txt
|-- README.md
|-- models/
|-- outputs/
|-- uploads/
|-- static/
|   |-- css/
|   |   `-- style.css
|   `-- js/
|       `-- main.js
|-- templates/
|   |-- base.html
|   |-- index.html
|   |-- results.html
|   `-- report_template.html
`-- utils/
    |-- __init__.py
    |-- data_loader.py
    |-- insights.py
    |-- pipeline.py
    |-- reporter.py
    `-- visualizer.py
```

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python app.py
```

Open the browser at:

```text
http://127.0.0.1:5000
```

## How to Use

1. Upload a CSV or Excel dataset.
2. Click `Preview Dataset`.
3. Choose the problem type.
4. Select the target column if needed.
5. Pick the machine learning model.
6. Select one or more plot types.
7. Click `Start Auto Analysis`.
8. Wait for the progress bar to complete.
9. Review the metrics, plots, cleaned data preview, and recommendations.
10. Download the ZIP bundle or retrain with a different model.

## Notes

- For classification and regression, the target column is required.
- For clustering, the app uses `KMeans`.
- The report is generated as HTML to keep the project lightweight and easy to run.
- Generated job artifacts are stored in the `outputs/` folder.
- Uploaded datasets are stored in the `uploads/` folder for retraining support.

## Suggested Improvements

- Add user authentication for multi-user deployments
- Add feature selection and hyperparameter tuning
- Persist jobs in a database instead of in-memory storage
- Export reports to PDF
- Add SHAP or model explainability visualizations
