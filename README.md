# Jamboree Graduate Admission Analysis

## Project Description

This project analyzes factors influencing graduate school admissions using data from Jamboree (`Jamboree_Admission.csv`). It performs Exploratory Data Analysis (EDA), builds Linear Regression models (OLS, Ridge, Lasso) to predict the 'Chance of Admit', tests model assumptions, and provides actionable insights.

The primary goal is to understand which factors (GRE score, TOEFL score, CGPA, University Rating, SOP/LOR strength, Research experience) are most important for admission and how they interact.

A detailed explanation of the methodology, findings, and recommendations can be found in `documentation.md`.

## Files in this Repository

*   `Jamboree_Admission.csv`: The raw dataset containing applicant information.
*   `jamboree_analysis.py`: The main Python script performing the analysis and modeling.
*   `documentation.md`: Detailed documentation of the project, methodology, findings, and recommendations.
*   `README.md`: This file.
*   **Generated Plots (after running the script):**
    *   `univariate_plots.png`
    *   `bivariate_plots_scatter_box.png`
    *   `correlation_matrix.png`
    *   `boxplots_outliers.png`
    *   `residuals_vs_fitted.png`
    *   `residuals_normality.png`

## Requirements

The analysis requires Python 3 and the following libraries:

*   pandas
*   numpy
*   matplotlib
*   seaborn
*   statsmodels
*   scikit-learn

You can install these dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

## How to Run

1.  Ensure you have Python 3 and the required libraries installed.
2.  Make sure the `Jamboree_Admission.csv` file is in the same directory as the script.
3.  Open a terminal or command prompt in the project directory.
4.  Execute the Python script:
    ```bash
    python jamboree_analysis.py
    ```
5.  The script will print the analysis steps, model summaries, assumption test results, performance metrics, and insights to the console.
6.  It will also save the generated plots as PNG files in the same directory.

## Key Findings Summary

*   GRE Score, TOEFL Score, and CGPA are highly correlated with the Chance of Admit.
*   LOR strength and Research experience also show significant positive influence.
*   Multicollinearity exists among academic scores (GRE, TOEFL, CGPA).
*   While a strictly VIF-reduced OLS model is simpler, Ridge and Lasso models (using all features) provide better predictive accuracy (R² ≈ 0.82 vs. ≈ 0.68 for reduced OLS).
*   Linear regression assumptions (Homoscedasticity, Normality) are partially violated, suggesting potential benefits from exploring non-linear models.

Refer to `documentation.md` for a full breakdown of the results and recommendations.
