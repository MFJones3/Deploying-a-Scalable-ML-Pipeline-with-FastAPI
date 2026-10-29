# Model Card — Census Income Classification

## Model Details

This project uses a Random Forest classifier to predict whether an individual's annual income exceeds $50,000 based on demographic, educational, and employment-related features from the Census Income dataset.

The model was developed using Python 3.10 and scikit-learn. Categorical features are transformed through one-hot encoding, and the fitted encoder is saved alongside the trained model to ensure consistent preprocessing during inference.

The Random Forest classifier uses 100 estimators and a random seed of 42 for reproducibility. The trained model and categorical encoder are saved as serialized files in the `model/` directory.

## Intended Use

The model is intended as an educational demonstration of supervised machine learning, classification evaluation, and deployment using FastAPI.

It allows users to submit Census-style demographic and employment data through a REST API and receive a prediction indicating whether an individual's annual income is above or at most $50,000.

The model is not intended for employment screening, credit decisions, insurance eligibility, or other consequential decisions about individuals.

## Training Data

The model was trained using the Census Income dataset provided with the Udacity project starter repository.

The dataset contains demographic and employment-related attributes, including age, education, occupation, marital status, working hours, and other categorical and numerical features.

The data was divided into training and testing subsets using a stratified 80/20 split. Approximately 80% of the observations were used for training, while the remaining 20% were reserved for evaluation.

Categorical variables were encoded using the preprocessing functions supplied in `ml/data.py`. The income label was transformed into a binary classification target.

## Evaluation Data

The model was evaluated using the held-out 20% test subset. The test data was not used to train the classifier.

The same categorical encoder fitted during training was applied to the test data to maintain consistent feature representations.

Additional evaluations were performed across individual values of categorical features. These subgroup results are recorded in `slice_output.txt`.

## Metrics

Model performance was evaluated using precision, recall, and the F1 score for the positive income class (`>50K`).

| Metric | Test Score |
|---|---:|
| Precision | 0.7327 |
| Recall | 0.6397 |
| F1 Score | 0.6830 |

A precision of 0.7327 indicates that approximately 73.27% of individuals predicted to earn more than $50,000 actually belonged to that class in the test data.

A recall of 0.6397 indicates that the model correctly identified approximately 63.97% of individuals who actually belonged to the higher-income class.

The F1 score of 0.6830 summarizes the balance between precision and recall.

Performance was also evaluated across categorical slices to identify differences between subgroups. These measurements should be interpreted alongside subgroup sample sizes, particularly where relatively few observations are available.

## Ethical Considerations

The dataset includes attributes related to demographic characteristics and socioeconomic circumstances. Models trained on these features may reproduce historical biases, correlations, or inequalities present in the underlying data.

Differences in predictive performance across demographic groups may create fairness concerns if the model is used to make decisions affecting individuals.

An income prediction is a statistical estimate, not a measure of personal worth, ability, or potential. Predictions should not be used to justify discriminatory treatment or automate consequential decisions.

The model does not establish causal relationships between demographic attributes and income.

## Caveats and Recommendations

The model is based on historical Census Income data and may not generalize reliably to contemporary populations, different geographic regions, or substantially different economic conditions.

The income threshold is fixed at $50,000 and is not adjusted for inflation.

The Random Forest model may make errors, particularly when identifying higher-income individuals, as reflected in its recall of approximately 0.64.

Performance estimates for small demographic slices can be unstable and should not be interpreted as strong evidence of subgroup-level accuracy.

Future improvements could include hyperparameter optimization, cross-validation, calibration, more detailed fairness assessments, and evaluation on newer, independently collected data.

The deployed API should be treated as a demonstration system rather than a production service for consequential decisions.
