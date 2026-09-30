# Using GitHub Actions for Model Training and Versioning

This repository demonstrates how to use GitHub Actions to automate the process of training a machine learning model, storing the model, and versioning it. This allows you to easily update and improve your model in a collaborative environment.

Watch the tutorial video for this lab at [Github action Lab2](https://youtu.be/cj5sXIMZUjQ)

## My Changes

### Dataset: Breast Cancer Wisconsin (replaces synthetic data)
- `train_model.py` now loads scikit-learn's bundled [Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) dataset: 569 tumours, 30 numeric features, 212 malignant / 357 benign. It ships with scikit-learn, so the CI runner needs no download.
- **Positive class = malignant.** scikit-learn encodes `0 = malignant, 1 = benign`, and `f1_score` scores label `1` by default. The labels are flipped (`y = 1 - y`) right after loading, so every metric reports how well the model catches malignant tumours, which is the error that matters most in screening.

### Evaluation: proper hold-out test set
- **Problem in the original:** `evaluate_model.py` generated "new" data with `make_classification` at a different `n_samples`. Its random stream depends on `n_samples`, so the column order and cluster covariances came out different. The model was scored on a *different distribution* (F1 ≈ 0.54, i.e. noise). When the two sample sizes happened to match, the evaluation data was identical to the training data (F1 ≈ 1.0).
- **Fix:** `train_model.py` makes a stratified 70/30 split (`random_state=42`) and pickles it to `data/`. `evaluate_model.py` loads `data/X_test.pickle` and `data/y_test.pickle`. Both scripts run in the same job, so the files are still there for the evaluation step.
- MLflow now logs training metrics as `Train Accuracy` / `Train F1 Score`, so they aren't mistaken for test metrics.

**Result** (local run, 171 held-out tumours, 64 malignant):

| Metric | Value |
|---|---|
| F1 (malignant) | 0.951 |
| Malignant recall | 0.906 |
| Missed malignant (false negatives) | 6 |
| False alarms (false positives) | 0 |

### Pipeline and workflow fixes
| Issue | Fix |
|---|---|
| Unpinned `mlflow` resolves to 3.x on modern Python, which rejects the `./mlruns` file store and crashes training | Tracking URI set to `sqlite:///mlflow.db`; all versions pinned in `requirements.txt` |
| Python 3.9 is end-of-life | CI uses Python `'3.12'` (quoted: unquoted `3.10` would be read by YAML as `3.1`) with `actions/setup-python@v5` |
| `GITHUB_TOKEN` is read-only by default, so the final `git push` fails with 403 | Added `permissions: contents: write` |
| Automated commits were attributed to the instructor's account | Commits as `github-actions[bot]` |
| Workflow retrained on every push to `main`, even for unrelated labs | `paths:` filter limits it to Lab 2 source, requirements, and the workflow file |
| Push rejected if `main` moved during training | `git pull --rebase origin main` before `git push` |

The workflow that actually runs is `.github/workflows/github_lab2_model_calibration_on_push.yml`. The copy in `Labs/Github_Labs/Lab2/workflows/` is kept identical for reference.

### Running locally
```bash
conda create -n mlops-lab2 python=3.12 -y && conda activate mlops-lab2
pip install -r Labs/Github_Labs/Lab2/requirements.txt
ts=$(date '+%Y%m%d%H%M%S')
python Labs/Github_Labs/Lab2/src/train_model.py --timestamp "$ts"
python Labs/Github_Labs/Lab2/src/evaluate_model.py --timestamp "$ts"
cat "${ts}_metrics.json"
```
Run from the repo root, as CI does. The artifacts it creates (`data/`, `mlflow.db`, the model file) are gitignored.


## Prerequisites

- [GitHub](https://github.com) account
- Basic knowledge of Python and machine learning
- Git command-line tool (optional)

## Getting Started

1. **Fork this Repository**: Click the "Fork" button at the top right of this [repository](https://github.com/raminmohammadi/MLOps/) to create your own copy.
3. **Clone Your Repository**:
   ```bash
   git clone https://github.com/your-username/your-forked-repo.git
   cd your-forked-repo

   ```
4. GitHub account
5. Basic knowledge of Python and machine learning
6. Git command-line tool (optional)

# Running the Workflow
## Customize Model Training
1. Modify the `train_model.py` script in the `src/` directory according to your dataset and model requirements. The original script generated synthetic data for demonstration purposes; this version uses the Breast Cancer Wisconsin dataset (see [My Changes](#my-changes)).

## Push Your Changes:
1. Commit your changes and push them to your forked repository.

## GitHub Actions Workflow:
1. Once you push changes to the main branch, the GitHub Actions workflow will be triggered automatically.

## View Workflow Progress:
1. You can track the progress of the workflow by going to the "Actions" tab in your GitHub repository.

## Retrieve the Trained Model:
1. After the workflow completes successfully, the trained model will be stored in the `models/` directory.

# Model Evaluation
The model evaluation is performed automatically within the GitHub Actions workflow. The evaluation results (e.g., F1 Score) are stored in the `metrics/` directory.

# Versioning the Model
Each time you run the workflow, a new version of the model is created and stored. You can access and use these models for your projects.

# GitHub Actions Workflow Details
The workflow consists of the following steps:

- Generate and Store Timestamp: A timestamp is generated and stored in a file for versioning.
- Model Training: The `train_model.py` script is executed, which trains a random forest classifier on synthetic data and stores the model in the `models/` directory.
- Model Evaluation: The `evaluate_model.py` script is executed to evaluate the model's F1 Score on synthetic data, and the results are stored in the `metrics/` directory.
- Store and Version the New Model: The trained model is moved to the `models/` directory with a timestamp-based version.
- Commit and Push Changes: The metrics and updated model are committed to the repository, allowing you to track changes.

# Model Calibration Workflow
## Overview
The `model_calibration_on_push.yml` workflow is a part of the automation process for machine learning model calibration within this repository. It is essential for ensuring that the model's predictions are accurate and well-calibrated, a critical step in machine learning applications.

## Workflow Purpose
This workflow's primary purpose is to calibrate a trained machine learning model after each push to the main branch of the repository. Calibration is a crucial step to align model predictions with reality, particularly when dealing with classification tasks. In simple terms, calibration ensures that a model's predicted probabilities match the actual likelihood of an event happening.

## Workflow Execution
Let's break down how this workflow operates step by step:

### Step 1: Trigger on Push to Main Branch
- This workflow is automatically initiated when changes are pushed to the main branch of the repository. It ensures that the model remains calibrated and up-to-date with the latest data and adjustments.

### Step 2: Prepare Environment
- The workflow begins by setting up a Python environment and installing the necessary Python libraries and dependencies. This is crucial to ensure that the model calibration process can be executed without any issues.

### Step 3: Load Trained Model
- The trained machine learning model, which has been previously saved in the `models/` directory, is loaded into memory. This model should be the most recent version, as trained by the `train_model.py` script.

### Step 4: Calibrate Model Probabilities
- In this step, the model's predicted probabilities are calibrated. Calibration methods, such as Platt scaling or isotonic regression, are applied. These methods adjust the model's predicted probabilities to match the actual likelihood of an event occurring. This calibration step is critical for reliable decision-making based on the model's predictions.

### Step 5: Save Calibrated Model
- The calibrated model is saved back to the `models/` directory. It is given a distinct identifier to differentiate it from the original, uncalibrated models. This ensures that both the original model and the calibrated model are available for comparison and use.

### Step 6: Commit and Push Changes
- This final step involves committing the calibrated model and any other relevant files to the repository. It is essential to keep track of the changes made during the calibration process and to store the calibrated model in the repository for future applications and reference.

# Customization
The `model_calibration_on_push.yml` workflow can be customized to align with your specific project requirements. You can modify calibration methods, the directory where the calibrated model is saved, or any other aspects of the calibration process to meet your project's unique needs.

# Integration with Model Training
This workflow is designed to work seamlessly with the main model training workflow, `model_retraining_on_push.yml`. In the initial workflow, the model is trained, and in this workflow, the calibrated model is generated. The calibrated model can then be used in applications where precise, well-calibrated probabilities are essential.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments
- This project uses GitHub Actions for continuous integration and deployment.
- Model training and evaluation are powered by Python and scikit-learn.

# Questions or Issues
If you have any questions or encounter issues while using this GitHub Actions workflow, please open an issue in the Issues section of your repository.

