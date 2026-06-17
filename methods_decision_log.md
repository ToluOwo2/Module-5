# Methods decision log

This file summarises key methodological and reproducibility decisions for the BRFSS 2022 self-reported MI/CHD machine-learning analysis.

## Dataset access

The BRFSS 2022 dataset was downloaded programmatically using `kagglehub` from a Kaggle mirror of the CDC public-use BRFSS dataset. The raw dataset was not stored in the repository because it is large and externally available. The notebook verifies file shape and checksum before preprocessing.

## Outcome definition

The target outcome was `_MICHD`, the CDC-derived BRFSS indicator for self-reported myocardial infarction or coronary heart disease. The target was validated against its component survey items, `CVDINFR4` and `CVDCRHD4`, before modelling.

## Analytical framing

The analysis was framed as survey-based classification of reported MI/CHD status, not clinical diagnosis or prospective cardiovascular risk prediction. This framing was used because BRFSS captures self-reported diagnosis rather than clinically adjudicated disease.

## Leakage control

Variables defining or auditing the target were excluded from model predictors. These included `_MICHD`, `CVDINFR4` and `CVDCRHD4`. `CVDSTRK3` was also excluded as a conservative clinical-overlap control. The train-test split was performed before preprocessing objects were fitted.

## Missingness and preprocessing

BRFSS special codes were handled using variable-specific rules rather than global replacement. High-missingness variables, administrative variables, identifiers, survey-design variables, contextual variables and no-variation variables were excluded. Continuous/count-style variables were median-imputed and scaled for scale-sensitive models. Categorical/survey-code variables were mode-imputed and one-hot encoded.

## Class imbalance

Synthetic oversampling was avoided to reduce leakage risk and preserve the survey-response structure. Class imbalance was handled using model weighting where supported. Logistic regression used `class_weight="balanced"`, random forest used `class_weight="balanced_subsample"` and XGBoost used `scale_pos_weight`. The MLP was retained as an unweighted neural-network comparator because the scikit-learn implementation used did not support class weighting in the same way.

## Model comparison

Four models were compared: logistic regression, random forest, XGBoost and multilayer perceptron. Logistic regression was used as the interpretable baseline, while random forest, XGBoost and MLP represented increasingly complex comparators.

## Evaluation strategy

Evaluation prioritised ROC-AUC and PR-AUC, with PR-AUC emphasised because the positive class was imbalanced. Secondary evaluation included precision, sensitivity, specificity, F1-score, threshold analysis, Brier score, calibration, runtime, cross-validation, repeated-seed stability, paired testing, bootstrap uncertainty, feature importance, permutation importance, SHAP and subgroup error analysis.

## Reproducibility

The notebook uses fixed random seeds, pipeline-based preprocessing and exported package requirements. Pinned package versions are provided in `requirements_pinned.txt`. The notebook generates an `outputs/` folder when run, but generated outputs are not required to be stored permanently because they can be reproduced from the notebook.

## Interpretation boundary

The final model should be interpreted as a survey-based classifier of reported MI/CHD status. It should not be interpreted as a clinical diagnostic model, individual cardiovascular risk calculator or deployable decision-support system.
