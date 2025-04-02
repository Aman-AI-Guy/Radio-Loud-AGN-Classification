# Project Title: Classification of Radio-Loud Active Galactic Nuclei using Multi-Survey Astronomical Data

## 1. Introduction

This project undertakes the analysis and classification of radio-loud Active Galactic Nuclei (AGN) using a rich dataset compiled from major astronomical surveys: the Sloan Digital Sky Survey (SDSS), the NRAO VLA Sky Survey (NVSS), and the Faint Images of the Radio Sky at Twenty-Centimeters (FIRST) survey. The dataset comprises 18,286 radio galaxies, each characterized by attributes such as sky coordinates (Right Ascension, Declination), redshift (`z`), radio flux densities from NVSS (`SNVSS`) and FIRST (`SFIRST`), and the offset between optical and radio positions. The central objective is to leverage machine learning techniques to explore this dataset and accurately classify galaxies, specifically distinguishing between radio-loud AGN (coded as `A=1`) and star-forming galaxies (`A=0`) based on their observed properties.

## 2. Dataset

The dataset contains information for 18,286 radio galaxies. Key features utilized in this analysis include:

*   **SDSS Identifiers:** `Plate`, `MJD`, `Fiber` (Spectroscopic observation identifiers).
*   **Sky Coordinates:** `RAHour`, `DDecl` (J2000 coordinates).
*   **Galaxy Properties:**
    *   `z`: Redshift (spectroscopic, indicating distance and cosmic expansion).
    *   `SNVSS`: Integrated 1.4GHz radio flux density from NVSS (in Jy).
    *   `SFIRST`: Integrated 1.4GHz radio flux density from FIRST (in Jy, with missing values coded as 0).
    *   `Offset`: Angular separation between the optical galaxy and the FIRST radio source (in arcseconds).
*   **Classification Flags:**
    *   `Radio_Class`: A numerical code (1-4) indicating the matching quality between NVSS and FIRST sources.
    *   `A`: The primary target variable for classification (1 for radio-loud AGN, 0 for star-forming galaxy).
*   *(Note: Flags `M`, `L`, and `H` were present in the original data but removed during initial loading for this specific analysis).*

## 3. Methodology

The project follows a structured machine learning workflow:

*   **Data Loading and Preparation:** The dataset (`table1.dat`) was loaded using `Pandas`, assigning appropriate column names. Initial data cleaning involved dropping columns (`M`, `L`, `H`) not central to the primary classification task (`A`). Numerical conversion was applied where necessary.
*   **Exploratory Data Analysis (EDA):**
    *   Initial inspection using `.info()` and `.describe()` to understand data types, missing values (implicitly handled by `SFIRST=0`), and statistical summaries.
    *   Visualization of feature distributions using histograms (`sns.distplot`) and the target variable balance (`sns.countplot`).
    *   Correlation analysis using a heatmap (`sns.heatmap`) to identify relationships between features.
*   **Feature Importance Analysis:** An `XGBoost` classifier was trained on the full feature set to evaluate the relative importance of each feature in predicting the target class `A`. The results were visualized to identify the most influential predictors.
*   **Model Development and Comparison (Feature Selection):**
    *   Two modeling approaches were compared using `XGBoost` within Scikit-learn `Pipelines` (including `StandardScaler`):
        1.  **Full Model:** Utilized all available features (`Plate`, `MJD`, `Fiber`, `RAHour`, `DDecl`, `z`, `SNVSS`, `SFIRST`, `Offset`, `Radio_Class`).
        2.  **Restricted Model:** Utilized only the features identified as most important (`z`, `SNVSS`, `SFIRST`, `Offset`, `Radio_Class`).
    *   Performance was evaluated on a held-out test set (20% split, stratified) using metrics like Accuracy, Classification Report, Confusion Matrix, Log Loss, and ROC-AUC score. This comparison assessed whether a simpler model could achieve comparable performance.
*   **Model Development and Comparison (Algorithm Selection):**
    *   Using the more focused **Restricted Model** feature set, several standard classification algorithms were trained and evaluated:
        *   Logistic Regression
        *   K-Nearest Neighbors (KNN)
        *   Decision Tree
        *   Random Forest
        *   XGBoost
        *   Support Vector Machine (SVM)
    *   Each classifier was implemented within a Scikit-learn `Pipeline` including `StandardScaler`. Performance was compared using the same metrics (Accuracy, Classification Report, Confusion Matrix, Log Loss, ROC-AUC) to determine the most effective algorithm for this specific classification task on the reduced feature space.

## 4. Tools and Libraries

The analysis was conducted using Python 3 and relies on the following core libraries:

*   **Data Manipulation:** `Pandas`, `NumPy`
*   **Visualization:** `Matplotlib`, `Seaborn`
*   **Machine Learning:** `Scikit-learn` (for preprocessing, pipelines, model selection, metrics, and various classifiers), `XGBoost` (for the gradient boosting classifier).

## 5. Objectives

*   To explore the characteristics of radio galaxies in the combined SDSS-NVSS-FIRST dataset.
*   To identify the key observational features that distinguish radio-loud AGN from star-forming galaxies.
*   To build, train, and evaluate machine learning models for classifying these galaxy types.
*   To assess the impact of feature selection on model performance and complexity.
*   To compare the effectiveness of different classification algorithms on this astronomical dataset.

## 6. Expected Outcomes

This project will yield:

*   Insights into the data distributions and correlations relevant to radio galaxy classification.
*   A ranked list of feature importances for distinguishing radio-loud AGN.
*   A quantitative comparison between models using all features versus a reduced set of important features.
*   A comparative performance analysis of various classification algorithms, identifying the most suitable model(s) for this task based on standard evaluation metrics.
