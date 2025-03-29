# Jamboree Graduate Admission Analysis - Documentation

## 1. Project Overview

**Objective:** This project aims to analyze the factors influencing graduate school admissions for students, particularly from an Indian perspective, using data provided by Jamboree. The analysis helps understand the importance of various factors (GRE scores, TOEFL scores, GPA, etc.) and their interrelationships. Additionally, it involves building predictive models (Linear Regression) to estimate a student's chance of admission.

**Context:** Jamboree helps students prepare for standardized tests (GMAT, GRE, SAT) and apply to top colleges abroad. This analysis supports their feature that estimates admission probability for IVY league colleges.

**Dataset:** `Jamboree_Admission.csv`

**Methodology:**
*   Exploratory Data Analysis (EDA) using non-graphical and graphical methods.
*   Data Preprocessing (handling identifiers, duplicates, outliers - though none were removed).
*   Linear Regression Modeling using `statsmodels` (OLS) and `scikit-learn` (Ridge, Lasso).
*   Rigorous testing of Linear Regression assumptions (Multicollinearity via VIF, Residual Mean, Linearity, Homoscedasticity, Normality).
*   Model Evaluation using standard metrics (MAE, RMSE, R², Adjusted R²).
*   Derivation of Actionable Insights and Recommendations.

## 2. Data Description

The dataset (`Jamboree_Admission.csv`) contains information about graduate applicants:

*   **Serial No.:** Unique row identifier (Dropped during preprocessing).
*   **GRE Score:** Score out of 340.
*   **TOEFL Score:** Score out of 120.
*   **University Rating:** Rating of the target university (1 to 5).
*   **SOP:** Strength of Statement of Purpose (1 to 5).
*   **LOR:** Strength of Letter of Recommendation (1 to 5).
*   **CGPA:** Undergraduate GPA (out of 10).
*   **Research:** Research experience (0 for No, 1 for Yes).
*   **Chance of Admit:** Probability of admission (ranging from 0 to 1) - Target Variable.

**Initial Data Characteristics:**
*   500 entries, 9 columns initially.
*   No missing values detected.
*   Data types are appropriate (int64, float64).

## 3. Exploratory Data Analysis (EDA)

**Objective:** Understand data distributions, relationships between variables, and identify potential issues like outliers or multicollinearity.

**Steps:**
*   **Univariate Analysis:** Histograms for continuous variables (GRE, TOEFL, CGPA, SOP, LOR, Chance of Admit) and count plots for categorical/ordinal variables (University Rating, Research).
    *   *Findings:* Scores (GRE, TOEFL, CGPA) show roughly normal distributions. Most applicants have research experience. University ratings 2, 3, and 4 are most common.
*   **Bivariate Analysis:** Scatter plots (GRE, TOEFL, CGPA vs. Chance of Admit) and box plots (University Rating, Research vs. Chance of Admit).
    *   *Findings:* Clear positive linear trends observed between Chance of Admit and GRE, TOEFL, CGPA. Higher University Rating and having Research experience correlate positively with admission chances.
*   **Correlation Matrix:** Heatmap showing correlations between all numerical variables.
    *   *Findings:* Confirmed strong positive correlations between Chance of Admit and GRE, TOEFL, CGPA, LOR, SOP, University Rating. Also revealed high correlations *among* GRE, TOEFL, and CGPA, indicating potential multicollinearity.

**Generated Plots:**
*   `univariate_plots.png`
*   `bivariate_plots_scatter_box.png`
*   `correlation_matrix.png`
*   `boxplots_outliers.png` (Used for visual outlier check - minor outliers noted but not removed).

## 4. Data Preprocessing

*   **Identifier Removal:** 'Serial No.' column dropped.
*   **Column Renaming:** Columns renamed for easier access (e.g., 'GRE Score' to 'GRE_Score').
*   **Duplicate Check:** No duplicate rows found.
*   **Outlier Treatment:** Visual inspection via boxplots showed some potential outliers (e.g., low LOR), but they were deemed plausible within the context and not removed.
*   **Train-Test Split:** Data split into 80% training and 20% testing sets (`random_state=42`).

## 5. Model Building and Evaluation

Three types of linear regression models were built:

**a) Ordinary Least Squares (OLS) with VIF Reduction & Robust Errors:**
*   **Initial Model:** Showed high R² (~0.82) but significant multicollinearity (high VIFs) and failed homoscedasticity/normality tests.
*   **VIF Reduction:** Features were iteratively removed until all remaining features had VIF <= 5. This resulted in dropping `GRE_Score`, `CGPA`, `SOP`, `LOR`, and `University_Rating`.
*   **Final OLS Model:**
    *   Predictors: `TOEFL_Score`, `Research`.
    *   Fitted using `statsmodels.OLS` with heteroscedasticity-consistent standard errors (`cov_type='HC3'`) due to failed Breusch-Pagan test.
    *   Performance (Test Set): R² ≈ 0.68, MAE ≈ 0.062, RMSE ≈ 0.080.
    *   Both predictors were highly significant (p < 0.001).
    *   *Trade-off:* Strictly addressed multicollinearity but significantly reduced predictive power compared to the initial model.

**b) Ridge Regression:**
*   Applied to the original, scaled feature set.
*   Handles multicollinearity through L2 regularization (`alpha=1.0`).
*   Performance (Test Set): R² ≈ 0.82, MAE ≈ 0.043, RMSE ≈ 0.061.
*   Retained predictive power similar to the initial OLS model while mitigating multicollinearity effects.

**c) Lasso Regression:**
*   Applied to the original, scaled feature set.
*   Handles multicollinearity and performs feature selection through L1 regularization (`alpha=0.001`).
*   Performance (Test Set): R² ≈ 0.82, MAE ≈ 0.043, RMSE ≈ 0.061.
*   Did not zero out any coefficients with the chosen alpha, indicating all original features contributed somewhat. Performance similar to Ridge and initial OLS.

## 6. Assumption Testing (Final OLS Model)

*   **Multicollinearity:** Resolved by VIF reduction (VIFs for `TOEFL_Score` and `Research` were low).
*   **Mean of Residuals:** Confirmed to be effectively zero.
*   **Linearity:** Residuals vs. Fitted plot (`residuals_vs_fitted.png`) showed mostly random scatter around zero, suggesting the linearity assumption is reasonably met for the reduced model.
*   **Homoscedasticity:** Breusch-Pagan test remained significant (p < 0.05), indicating heteroscedasticity persists. This was addressed by using robust standard errors (HC3) for inference.
*   **Normality of Residuals:** Residual histogram and Q-Q plot (`residuals_normality.png`) looked roughly normal, but statistical tests (Omnibus, Jarque-Bera) in the model summary indicated significant deviations from normality. This assumption remains violated.

## 7. Key Findings and Insights

*   Academic scores (`GRE_Score`, `TOEFL_Score`, `CGPA`) are strongly correlated with admission chances.
*   Other factors like `LOR` strength, `SOP` strength, `University_Rating`, and `Research` experience also positively influence admission probability.
*   Significant multicollinearity exists between GRE, TOEFL, and CGPA.
*   Strictly removing multicollinearity (VIF < 5) leads to a simpler OLS model (using only `TOEFL_Score` and `Research`) but sacrifices considerable predictive accuracy (R² drops from ~0.82 to ~0.68).
*   Ridge and Lasso regression models, which handle multicollinearity internally, maintain higher predictive accuracy (R² ≈ 0.82) using the full feature set.
*   Linear regression assumptions of homoscedasticity and normality of residuals are violated, suggesting that while the R² is high for some models, inference might be affected (partially addressed with robust errors) and non-linear models could potentially offer improvements.

## 8. Recommendations

*   **For Students:** Focus on maximizing all key areas: GRE/TOEFL scores, CGPA, LOR/SOP quality, and gaining research experience.
*   **For Jamboree:**
    *   Use the insights to guide student preparation strategies.
    *   For the website prediction feature, consider using the **Ridge or Lasso model** due to their superior predictive accuracy while managing multicollinearity, rather than the heavily reduced OLS model.
    *   Acknowledge model limitations (violated assumptions) and consider exploring non-linear models (e.g., Gradient Boosting, Random Forest) for potentially even better prediction accuracy, although interpretability might decrease.
    *   Consider collecting more granular data (specific universities, programs) for future model enhancements.

## 9. Conclusion

The analysis successfully identified key factors influencing graduate admissions and highlighted the strong predictive power of the available features (R² up to ~0.82). While the standard OLS assumptions were not fully met, using robust standard errors and comparing with regularized models (Ridge/Lasso) provides a reliable basis for understanding relationships and making predictions. Ridge or Lasso regression appears most suitable for a practical prediction tool based on this dataset.
