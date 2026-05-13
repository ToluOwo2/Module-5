# Module-5
# Predicting Self-Reported MI/CHD from BRFSS 2022

## Project overview

This project uses the 2022 Behavioral Risk Factor Surveillance System (BRFSS) dataset to predict self-reported myocardial infarction or coronary heart disease. The task is framed as a supervised binary classification problem.

The critical focus is not simply whether machine learning can classify self-reported MI/CHD. Instead, the project evaluates whether increasingly complex models provide meaningful improvement over an interpretable logistic regression baseline once class imbalance, threshold behaviour, calibration, runtime, interpretability and subgroup performance are considered.

The model should be interpreted as a population-level, survey-based risk stratification analysis. It is not a clinical diagnostic model.

## Research question

To what extent do increasingly complex machine learning models improve prediction of self-reported myocardial infarction or coronary heart disease from BRFSS 2022 survey data, and how do performance gains compare with trade-offs in interpretability, threshold behaviour, calibration, runtime and fairness?

## Critical angle

Many applied machine-learning projects select the model with the highest headline ROC-AUC. This project takes a more critical approach. It asks whether model complexity produces practically meaningful improvement in an imbalanced healthcare classification task.

The analysis therefore considers:

- whether XGBoost, Random Forest and MLP models materially outperform logistic regression;
- whether ROC-AUC alone is sufficient for judging model usefulness;
- how sensitivity, precision, specificity and F1-score change with threshold choice;
- whether more complex models justify reduced interpretability or higher computational cost;
- whether model performance appears uneven across available demographic subgroups;
- whether predicted probabilities are suitable for risk interpretation.

## Dataset

Dataset: Behavioral Risk Factor Surveillance System 2022  
Source: Kaggle  
Kaggle identifier: `ariaxiong/behavioral-risk-factor-surveillance-system-2022`

Raw dataset size:

- 445,132 rows
- 326 variables

Analytical sample:

- 440,111 rows after excluding missing target labels
- 120 retained raw predictors after leakage, missingness and feature-exclusion checks
- 500 encoded model features after preprocessing

The raw BRFSS dataset is not included in this repository. The notebook downloads the dataset programmatically using `kagglehub`.

## Target variable

The binary target was derived from `_MICHD`, the CDC-calculated indicator of self-reported myocardial infarction or coronary heart disease.

Target definition:

- `1` = self-reported MI/CHD
- `0` = no self-reported MI/CHD

The target was validated against the underlying cardiovascular variables used to construct MI/CHD status. Direct target or leakage variables were excluded before modelling.

Important limitation: the outcome is self-reported and survey-based. It is not clinically adjudicated heart disease. The model therefore predicts reported MI/CHD status rather than confirmed clinical disease.

## Methods summary

The workflow includes:

1. Project setup and reproducibility checks
2. Dataset loading and validation
3. Initial data audit
4. Target definition and validation
5. BRFSS-specific missing-code cleaning
6. Continuous-variable protection
7. Leakage control and feature selection
8. Stratified train/test split
9. Feature engineering and preprocessing
10. Logistic regression baseline
11. Random Forest model
12. XGBoost model
13. MLP neural-network comparator
14. Model comparison
15. Threshold analysis
16. Model stability check
17. Calibration analysis
18. Feature importance and interpretability
19. Subgroup/fairness analysis
20. Final reproducibility summary

## Feature engineering and preprocessing

Feature preparation included:

- BRFSS-specific missing-code handling
- protection of continuous or count-style variables from inappropriate missing-code conversion
- removal of direct target/leakage variables
- removal of administrative and survey-design variables
- removal of variables with more than 50% missingness
- removal of constant/no-variation variables
- separation of continuous/count-style features from categorical/code features
- median imputation for continuous/count-style features
- most-frequent imputation for categorical/code features
- one-hot encoding of categorical/code variables
- scaling for scale-sensitive models
- unscaled preprocessing for tree-based models

Preprocessing was fitted inside model pipelines after the train/test split to reduce data leakage.

## Models compared

The following supervised classification models were compared:

- Logistic Regression
- Random Forest
- XGBoost
- Multilayer Perceptron neural network

These models were selected to represent increasing model complexity:

- Logistic Regression: interpretable linear baseline
- Random Forest: non-linear bagging ensemble
- XGBoost: gradient boosting model for tabular prediction
- MLP: neural-network comparator for encoded tabular features

More specialised deep-learning architectures such as CNNs, RNNs, transformers or autoencoders were not used because BRFSS is structured tabular survey data rather than imaging, sequential, text or representation-learning data.

## Evaluation metrics

Because the positive MI/CHD class represented approximately 9.03% of the analytical sample, model evaluation was not based on accuracy alone.

The following metrics were used:

- ROC-AUC
- PR-AUC
- Precision
- Recall / sensitivity
- Specificity
- F1-score
- Brier score
- Confusion matrix
- Training time
- Threshold analysis
- Calibration curves
- Subgroup performance

## Main findings

XGBoost achieved the strongest default-threshold performance overall, with the highest ROC-AUC, PR-AUC and sensitivity, while also having the shortest training time.

However, improvements over logistic regression were modest. This suggests that model complexity provided incremental rather than transformative gains for this BRFSS tabular prediction task.

Random Forest achieved similar predictive performance but required substantially longer runtime. The MLP achieved similar discrimination but was highly conservative at the default 0.50 threshold, requiring threshold adjustment to improve sensitivity and F1-score.

The main conclusion is that model usefulness depends not only on ROC-AUC, but also on threshold behaviour, calibration, interpretability, fairness and computational cost.

## Report

The full academic report is provided as `report.pdf`.

The report contains the background, methods, results, critical discussion, limitations, ethics/fairness considerations and references. The notebook provides the reproducible technical workflow supporting the report.

## How to run the notebook

1. Open `final_notebook.ipynb` in Google Colab.
2. Run all cells from the top.
3. The notebook downloads the BRFSS 2022 dataset using `kagglehub`.
4. Outputs are saved to the `outputs/` directory.

Expected runtime: approximately 30–60 minutes on Google Colab CPU, depending on runtime availability.

## Requirements

Core Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- kagglehub

These are listed in `requirements.txt`.

A minimal `requirements.txt` should contain:

```text
pandas
numpy
matplotlib
scikit-learn
xgboost
kagglehub
```

## Repository structure

```text
.
├── README.md
├── report.pdf
├── final_notebook.ipynb
├── requirements.txt
└── outputs/
    ├── model comparison tables
    ├── threshold analysis outputs
    ├── calibration outputs
    ├── feature importance outputs
    ├── subgroup/fairness outputs
    ├── plots
    └── summary files
```

## Reproducibility

The notebook is designed to run from a clean Google Colab runtime.

Key reproducibility steps include:

- programmatic dataset download
- dataset shape validation
- target validation
- fixed random seed
- stratified train/test split
- leakage-variable exclusion
- pipeline-based preprocessing
- model outputs saved to CSV
- plots saved to PNG
- final summary outputs saved automatically

Preprocessing is fitted inside model pipelines after the train/test split to reduce data leakage.

## Limitations

Key limitations include:

- The outcome is self-reported MI/CHD, not clinically confirmed disease.
- The dataset is cross-sectional, so the model predicts reported disease status rather than future incident disease.
- The positive class is imbalanced, limiting precision.
- Feature importance is predictive, not causal.
- The model was evaluated on a held-out BRFSS test set but was not externally validated on another dataset or survey year.
- Subgroup analysis is an error-pattern audit, not proof of fairness.
- Predicted probabilities should be interpreted as survey-based reported-risk estimates, not clinical diagnosis probabilities.
- The model is not intended for clinical diagnosis or direct deployment without further validation.

## Ethical and fairness considerations

Because the target is self-reported, predictions may reflect healthcare access, diagnosis opportunity, recall, health literacy and survey response patterns, not only underlying disease status.

The subgroup analysis is included to examine whether error patterns vary across available BRFSS demographic groups. It should not be interpreted as proof that the model is fair. Further validation would be required before any operational or public-health deployment.

## AI use declaration

Generative AI was used to support code structuring, debugging, explanation of machine-learning concepts and drafting of written sections. All outputs were reviewed, edited and validated by the author. The analysis, interpretation and final submitted work remain the author’s responsibility.
