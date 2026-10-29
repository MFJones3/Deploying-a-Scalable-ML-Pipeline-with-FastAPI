
# Census Income Prediction — Scalable ML Pipeline with FastAPI

## Project Overview

This project implements a machine learning pipeline that predicts whether an individual's annual income is greater than $50,000 using census demographic and employment data.

The project includes model training, evaluation, a REST API, automated tests, and continuous integration.

## Model

- Algorithm: RandomForestClassifier
- Features: Census demographic, education, occupation, and employment attributes
- Target: Annual income >$50K versus <=$50K
- Training/testing: Stratified 80/20 split
- Random seed: 42

### Test Set Performance

| Metric | Score |
|---|---|
| Precision | 0.7327 |
| Recall | 0.6397 |
| F1 | 0.6830 |

Performance across categorical feature slices is recorded in `slice_output.txt`.

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate fastapi
```

Alternatively, install dependencies using:

```bash
pip install -r requirements.txt
```

## Model Training

Run the training script:

```bash
python train_model.py
```

The script trains the model, evaluates its performance, and saves the trained model and encoder in the `model/` directory.

## Run the FastAPI Application

Start the local API:

```bash
uvicorn main:app --reload
```

The application runs at:

http://127.0.0.1:8000

Interactive API documentation:

http://127.0.0.1:8000/docs

### API Endpoints

**GET /**

Returns a welcome message.

**POST /data/**

Accepts census features as JSON and returns a predicted income category:

- `>50K`
- `<=50K`

## Local API Testing

With the API running, execute:

```bash
python local_api.py
```

This sends GET and POST requests and prints the response codes and returned results.

## Unit Tests

Run:

```bash
pytest -v test_ml.py
```

Three automated tests validate:

1. Inference returns one prediction per input record.
2. Classification metrics are calculated correctly.
3. Saving and loading a trained model preserves predictions.

## Code Quality

Run:

```bash
flake8 . --count --exit-zero --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --max-complexity=10 --max-line-length=127 --statistics
```

## Continuous Integration

GitHub Actions automatically runs linting and unit tests on pushes to the `main` branch using Python 3.10.

Successful CI execution is documented in `screenshots/continuous_integration.png`.

## Repository Evidence

The `screenshots/` directory contains:

- `unit_test.png`
- `local_api.png`
- `continuous_integration.png`

## Limitations

Model performance may vary across demographic and employment subgroups. The repository includes categorical slice-level evaluation results to support inspection of these differences.

This model is an educational demonstration and should not be used to make consequential decisions about individuals.

