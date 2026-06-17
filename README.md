# Module 5: Predicting Self-Reported MI/CHD from BRFSS 2022

## Project overview

This repository contains the reproducible Google Colab workflow for a supervised machine-learning analysis of self-reported myocardial infarction or coronary heart disease (MI/CHD) status using the 2022 Behavioral Risk Factor Surveillance System (BRFSS) dataset.

The project does not simply ask which model has the highest ROC-AUC. Instead, it evaluates whether increasingly complex tabular machine-learning models provide a practically meaningful improvement over an interpretable logistic regression baseline once class imbalance, threshold behaviour, calibration, runtime, stability, interpretability and subgroup error patterns are considered.

The final model should be interpreted as a survey-based classifier of reported MI/CHD status. It is not a clinical diagnostic model, individual cardiovascular risk calculator or deployable decision-support system.

---

## Research question

To what extent do increasingly complex machine-learning models improve classification of self-reported MI/CHD from BRFSS 2022 survey data, and how do performance gains compare with trade-offs in interpretability, calibration, threshold behaviour, runtime, stability and subgroup error behaviour?

---

## Repository structure

The notebook and report use the same base filename. They are distinguished by file extension.

```text
.
├── README.md
├── HDS_ML_TOLUWALOPEOWOEYE_2606.ipynb
├── HDS_ML_TOLUWALOPEOWOEYE_2606.docx
├── HDS_ML_TOLUWALOPEOWOEYE_2606.pdf
├── requirements.txt
├── requirements_pinned.txt
└── methods_decision_log.md
```

The notebook generates an `outputs/` folder when run. Generated outputs are not required to be stored permanently in the repository because they can be reproduced from the notebook.

---

## Dataset

**Dataset:** Behavioral Risk Factor Surveillance System 2022
**Source:** Kaggle mirror of BRFSS 2022
**Kaggle identifier:** `ariaxiong/behavioral-risk-factor-surveillance-system-2022`
**Dataset link:** https://www.kaggle.com/datasets/ariaxiong/behavioral-risk-factor-surveillance-system-2022

The raw dataset is not stored in this repository. The notebook downloads it programmatically using `kagglehub`.

| Item                                |                        Value |
| ----------------------------------- | ---------------------------: |
| Raw dataset size                    | 445,132 rows × 326 variables |
| Analytical sample                   |          440,111 respondents |
| Positive self-reported MI/CHD cases |                       39,751 |
| Positive prevalence                 |                        9.03% |
| Retained raw predictors             |                          120 |
| Encoded model features              |                          500 |
| Held-out test set                   |           88,023 respondents |
| Held-out test positive cases        |                        7,950 |

---

## Target variable

The binary target was based on `_MICHD`, the CDC-derived BRFSS indicator for self-reported myocardial infarction, angina or coronary heart disease.

| Target value | Meaning                 |
| ------------ | ----------------------- |
| 1            | Self-reported MI/CHD    |
| 0            | No self-reported MI/CHD |

The target was audited against the underlying BRFSS cardiovascular variables `CVDINFR4` and `CVDCRHD4`. The reconstructed target matched `_MICHD` for all 440,111 comparable rows, confirming coding consistency.

Important limitation: the outcome is self-reported and cross-sectional. The model predicts reported MI/CHD status, not clinically adjudicated disease or future cardiovascular events.

---

## Analytical workflow

The notebook follows the workflow used in the report:

1. Project setup and reproducibility controls
2. Package-version and environment audit
3. Programmatic BRFSS 2022 download using `kagglehub`
4. Dataset shape and checksum verification
5. Target definition and validation
6. Exploratory data audit
7. BRFSS-specific missing-code handling
8. Continuous/count-variable protection
9. Leakage and feature-exclusion controls
10. Stratified train-test split before preprocessing
11. Pipeline-based imputation, scaling and one-hot encoding
12. Logistic regression baseline
13. Random forest comparator
14. XGBoost comparator
15. MLP neural-network comparator
16. Default-threshold model comparison
17. Threshold analysis
18. Calibration analysis
19. Cross-validation and repeated-seed stability checks
20. Paired statistical testing and bootstrap uncertainty
21. Feature importance and SHAP interpretation
22. Exploratory subgroup error analysis
23. Export of report-supporting outputs

Preprocessing was fitted inside model pipelines after the train-test split to reduce leakage risk.

---

## Data cleaning and leakage control

BRFSS uses variable-specific survey codes for refusal, uncertainty and inapplicability. The notebook therefore applies BRFSS-specific cleaning rather than a single global missing-code replacement rule.

Variables were excluded if they were:

* direct target or target-component variables;
* conservative clinical-overlap leakage controls;
* administrative, identifier or survey-design variables;
* raw measurement-entry variables where cleaner derived variables were available;
* contextual variables not appropriate for the supervised classification task;
* variables with more than 50% missingness;
* constant or no-variation variables.

The final model matrix contained 120 retained raw predictors and 500 encoded model features.

---

## Feature engineering and preprocessing

Continuous/count-style variables were median-imputed and standardised where required. Categorical and survey-code variables were mode-imputed and one-hot encoded using rare-category grouping.

| Model type          | Preprocessing                |
| ------------------- | ---------------------------- |
| Logistic regression | Scaled pipeline              |
| MLP                 | Scaled pipeline              |
| Random forest       | Unscaled tree-based pipeline |
| XGBoost             | Unscaled tree-based pipeline |

---

## Models compared

| Model                 | Role                           |
| --------------------- | ------------------------------ |
| Logistic regression   | Interpretable baseline         |
| Random forest         | Non-linear ensemble comparator |
| XGBoost               | Main complex tabular model     |
| Multilayer perceptron | Neural-network comparator      |

Class imbalance was handled through model weighting where supported. Synthetic oversampling was avoided in the primary workflow to preserve survey-response structure and reduce leakage risk.

---

## Evaluation strategy

Because the positive MI/CHD class represented approximately 9.03% of the analytical sample, accuracy was not prioritised.

The analysis used:

* ROC-AUC;
* PR-AUC;
* precision;
* sensitivity/recall;
* specificity;
* F1-score;
* Brier score;
* confusion matrices;
* runtime;
* threshold analysis;
* quantile calibration curves;
* cross-validation;
* repeated stratified train-test splits;
* paired tests with Holm correction;
* paired bootstrap uncertainty;
* permutation importance;
* SHAP;
* subgroup error analysis.

PR-AUC was prioritised because the outcome was imbalanced.

---

## Main held-out results

Held-out test-set performance at the default 0.50 threshold:

| Model               | ROC-AUC | PR-AUC | Precision | Sensitivity | Specificity | F1-score |  Brier |  Runtime |
| ------------------- | ------: | -----: | --------: | ----------: | ----------: | -------: | -----: | -------: |
| Logistic regression |  0.8512 | 0.3668 |    0.2399 |      0.7951 |      0.7499 |   0.3686 | 0.1630 |  785.51s |
| Random forest       |  0.8499 | 0.3609 |    0.2441 |      0.7886 |      0.7576 |   0.3728 | 0.1523 | 1449.36s |
| XGBoost             |  0.8542 | 0.3701 |    0.2379 |      0.8131 |      0.7415 |   0.3682 | 0.1620 |   48.12s |
| MLP                 |  0.8525 | 0.3679 |    0.5506 |      0.1020 |      0.9917 |   0.1721 | 0.0676 |  129.92s |

XGBoost achieved the strongest held-out discrimination, but its improvement over logistic regression was small:

| Comparison                            | Difference |
| ------------------------------------- | ---------: |
| XGBoost - logistic regression ROC-AUC |    +0.0030 |
| XGBoost - logistic regression PR-AUC  |    +0.0033 |

Bootstrap analysis supported a small ROC-AUC advantage, but the PR-AUC confidence interval crossed zero. The main conclusion is that XGBoost was a marginally stronger classifier, not a clearly superior or clinically deployable risk model.

---

## Threshold, calibration and runtime findings

Model usefulness depended strongly on threshold choice.

Key findings:

* The MLP had competitive ROC-AUC and PR-AUC but very low default-threshold sensitivity.
* Lowering the MLP threshold from 0.50 to 0.10 increased sensitivity from 0.1020 to 0.8216 at the cost of precision.
* Logistic regression, random forest and XGBoost substantially overpredicted MI/CHD status in the highest calibration bin.
* For XGBoost, the highest-bin mean predicted probability was 0.8520, while the observed event rate was 0.3870.
* XGBoost trained substantially faster than random forest while achieving stronger discrimination.

Predicted probabilities should therefore be interpreted as classification scores within this survey sample, not as reliable clinical risk estimates without recalibration.

---

## Stability and uncertainty

Stability and uncertainty analyses included:

* 3-fold stratified cross-validation on a 120,000-row training sample;
* five repeated stratified 80/20 splits;
* paired t-tests with Holm correction;
* 1,000 paired bootstrap resamples of held-out predictions.

Key uncertainty findings:

| Comparison                    | Metric  | Result                            |
| ----------------------------- | ------- | --------------------------------- |
| XGBoost - logistic regression | ROC-AUC | +0.0025; Holm-adjusted p = 0.002  |
| XGBoost - logistic regression | PR-AUC  | +0.0052; Holm-adjusted p = 0.021  |
| Bootstrap, 1,000 resamples    | ROC-AUC | +0.0031; 95% CI 0.0022 to 0.0040  |
| Bootstrap, 1,000 resamples    | PR-AUC  | +0.0032; 95% CI -0.0003 to 0.0066 |

The PR-AUC bootstrap interval crossed zero, reinforcing that the imbalance-aware gain over logistic regression was uncertain and practically small.

---

## Interpretability and subgroup analysis

Interpretability analyses included built-in tree importance, permutation importance and SHAP for XGBoost.

Important interpretation points:

* Feature attributions reflected clinical-history, self-rated health, demographic, healthcare-access and survey-response variables.
* Feature importance was interpreted as fitted model behaviour, not causal evidence.
* SHAP and permutation importance did not rank all variables consistently.
* Sparse one-hot encoded categories were interpreted cautiously.

Subgroup analysis was performed as an exploratory error audit using XGBoost predictions on the held-out test set. It examined available BRFSS demographic and socioeconomic variables, including age, sex, race/ethnicity, education and income.

Across income groups:

* false-negative rates ranged from approximately 10% to 38%;
* false-positive rates ranged from approximately 10% to 52%.

These findings indicate that aggregate ROC-AUC and PR-AUC did not fully describe error behaviour across subgroups. They should not be interpreted as definitive evidence of directional socioeconomic bias.

---

## How to run in Google Colab

1. Open `HDS_ML_TOLUWALOPEOWOEYE_2606.ipynb` in Google Colab.
2. Select a standard CPU runtime.
3. Run all cells from the top.
4. The notebook downloads the BRFSS 2022 dataset using `kagglehub`.
5. The notebook generates an `outputs/` folder during execution.

Expected runtime: approximately **2-3 hours on a standard Google Colab CPU runtime**, depending on runtime availability and whether optional interpretability/stability sections are rerun.

---

## Requirements and environment

Core packages are listed in `requirements.txt`.

Minimum required packages:

```text
pandas
numpy
scikit-learn
xgboost
scipy
matplotlib
shap
kagglehub
jupyter
```

Pinned package versions from the final Colab runtime are listed in `requirements_pinned.txt`.

---

## Outputs

The notebook generates an `outputs/` folder when run. Generated outputs are not required to be stored permanently in the repository because they can be reproduced from the notebook.

When the notebook is executed, the `outputs/` folder may include exported tables, figures, environment-version files, pinned requirements and report-supporting CSVs. Important reproducibility outputs include:

```text
outputs/00_environment_versions.csv
outputs/requirements_pinned.txt
```

The exported outputs are intended to support traceability between the notebook and the written report. If the `outputs/` folder is not present in the repository, it can be regenerated by running the notebook from top to bottom.

---

## Reproducibility notes

Key reproducibility controls include:

* programmatic dataset download;
* dataset shape validation;
* SHA256 checksum verification;
* fixed random seeds;
* stratified train-test split;
* preprocessing inside model pipelines;
* leakage-variable exclusions before modelling;
* package-version export;
* pinned package requirements;
* methods decision log.

Primary model comparison used the full held-out test set. Reduced samples were used only for computationally expensive secondary analyses, including tuning, cross-validation, permutation importance and SHAP.

---

## Methods decision log

The repository includes `methods_decision_log.md`, which documents major analytical decisions and their rationale. This supports reproducibility and transparency for choices such as:

* dataset selection;
* target definition;
* BRFSS-specific missing-code handling;
* leakage-variable exclusion;
* high-missingness exclusions;
* model selection;
* class-imbalance handling;
* threshold analysis;
* calibration assessment;
* interpretability methods;
* subgroup error analysis;
* non-deployment interpretation.

---

## Limitations

Key limitations include:

* self-reported MI/CHD rather than clinically adjudicated disease;
* cross-sectional design rather than incident cardiovascular risk prediction;
* no external validation on another survey year or clinical cohort;
* no survey-weighted population-level performance estimation;
* reduced samples for computationally expensive secondary analyses;
* feature attribution is predictive, not causal;
* subgroup analysis is exploratory and not a formal fairness assessment;
* predicted probabilities are not calibrated clinical risk estimates;
* no clinical deployment claim.

---

## Ethical and fairness considerations

Because the target is self-reported, predictions may reflect healthcare access, diagnostic opportunity, recall, health literacy and survey-response behaviour as well as underlying cardiovascular morbidity.

Subgroup analysis was included as an exploratory fairness-relevant error audit across available BRFSS demographic and socioeconomic variables. Further validation, recalibration and formal fairness evaluation would be required before any public-health or clinical use.

No attempt was made to identify individuals. BRFSS is a public, de-identified survey dataset.

---

## Use of generative AI

Generative AI was used to support code structuring, debugging, explanation of machine-learning concepts, workflow checking and drafting support. All outputs were reviewed, edited and validated by the author. The analysis, interpretation and final submitted work remain the author’s responsibility.

---

## Report

The full academic report is provided as:

```text
HDS_ML_TOLUWALOPEOWOEYE_2606.pdf
```

A Word version may also be included as:

```text
HDS_ML_TOLUWALOPEOWOEYE_2606.docx
```


