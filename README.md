# Cerebral Palsy (CP) Risk Factors Analysis

This repository contains a comprehensive machine learning analysis aimed at identifying risk factors associated with Cerebral Palsy (CP) by comparing them to non-CP cases. The project involves data preprocessing, feature selection, model training using Random Forest, XGBoost, and SVM classifiers, and extensive evaluation of model performance.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Instructions](#instructions)
- [Output](#output)
- [Notes](#notes)
- [License](#license)


## Overview

The main goal of this project is to identify and analyze the risk factors associated with CP by comparing CP cases to non-CP cases. The analysis uses various machine learning models to determine the most significant features contributing to the condition and evaluates their performance in classification tasks.

## Features

- **Data Loading and Preprocessing:**
  - Loads data from an Excel file ([CCPF11DFLL.xlsx](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/main/CCPF11DFLL.xlsx)).
  - Separates features and target variables (`CP/Non-CP`).
  - Performs statistical tests to identify significant features (t-test for numerical features, chi-squared test for categorical features).

- **Model Training and Evaluation:**
  - Utilizes three models:
    - Random Forest Classifier
    - XGBoost Classifier
    - Linear SVM (Support Vector Machine)
  - Constructs pipelines for each model incorporating preprocessing steps (imputation and scaling).
  - Trains models on significant features.
  - Evaluates models using cross-validation to assess mean accuracy and standard deviation.
  - Generates ROC curves and calculates AUC-ROC scores for model comparison.

- **Visualization:**
  - Plots feature importances for top features identified by each model.
  - Creates Venn diagrams to illustrate common significant features among models.
  - Visualizes ROC curves for model performance comparison.

## Requirements

Ensure you have the following Python libraries installed:
- pandas
- matplotlib
- numpy
- scikit-learn
- xgboost
- seaborn
- scipy
- matplotlib-venn

You can install these libraries using pip:
```sh
pip install pandas matplotlib numpy scikit-learn xgboost seaborn scipy matplotlib-venn
```

## Instructions

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis.git
   cd Cerebral-Palsy-CP-Risk-Factors-Analysis
   ```

2. **Navigate to the Relevant Branch:**
   ```sh
   git checkout Machine-Learning-Modeling
   ```

3. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Code:**
   - Execute the main script `RF_XGB_SVM_CODE.py` to run the entire pipeline:
     ```sh
     python RF_XGB_SVM_CODE.py
     ```

## Output

The code will generate several plots illustrating feature importances, Venn diagrams, ROC curves, and cross-validation results. The outputs will be saved in the current directory and can be viewed directly via the following links:

- [Top Features with Mean Differences](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/published_top_features_with_mean_diff.png)
- [Merged Common Features](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/merged_common_features.png)
- [Venn Diagram of Top Features](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/venn_diagram_top_features.png)
- [Cross-Validation Results](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/cross_validation_results.png)
- [AUC-ROC Curve](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/auc_roc_curve.png)



### 1. Top Features with Mean Differences
- **Image:** [Top Features with Mean Differences](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/published_top_features_with_mean_diff.png)

![published_top_features_with_mean_diff](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/caf26734-578d-4988-b7dd-c065e5261cb0)

- **Description:** This plot displays the top 10 most important features identified by each of the three models (Random Forest, XGBoost, and SVM). Each feature's importance is color-coded based on the mean difference between CP and Non-CP groups. This helps to visualize not only which features are important, but also how they differ between the two groups.

### 2. Merged Common Features
- **Image:** [Merged Common Features](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/merged_common_features.png)
![merged_common_features](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/3e88f14f-50cf-4116-bf27-3cd7b416db9f)

- **Description:** This bar plot shows the common top features across all three models (Random Forest, XGBoost, and SVM). It highlights the importance of each feature as determined by each model, providing insight into which features are consistently important for distinguishing between CP and Non-CP cases.

### 3. Venn Diagram of Top Features
- **Image:** [Venn Diagram of Top Features](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/venn_diagram_top_features.png)
![sdwd](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/1ec1c561-925a-4fc2-95a8-d90ea0c5df0e)

- **Description:** This Venn diagram illustrates the overlap of the top features identified by the three models. It helps to visualize which features are unique to each model and which are common across multiple models. This is useful for understanding the agreement between different models on feature importance.

### 4. Cross-Validation Results
- **Image:** [Cross-Validation Results](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/cross_validation_results.png)

![cross_validation_results](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/688c7fe0-9a17-4429-80e2-55f9dda72096)

- **Description:** This bar plot shows the mean accuracy and standard deviation of each model (Random Forest, XGBoost, and SVM) based on cross-validation. Cross-validation is used to evaluate the performance of the models and ensure they generalize well to unseen data. The plot provides a comparison of the models' performance.

### 5. AUC-ROC Curve
- **Image:** [AUC-ROC Curve](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Machine-Learning-Modeling/auc_roc_curve.png)

![auc_roc_curve](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/5b4f4aef-4d41-45f2-b8b6-7827547c1fa2)

- **Description:** The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. The Area Under the Curve (AUC) provides a single measure of overall model performance. This plot includes the ROC curves for Random Forest, XGBoost, and SVM, along with their AUC scores, allowing for comparison of their classification abilities.

These outputs collectively provide a comprehensive view of the feature importance, model performance, and consistency across different machine learning models in the context of identifying risk factors for Cerebral Palsy.

## Notes

- **Python Version:** Python 3.x is recommended.
- **Data Format:** Ensure your data follows the structure expected by the code (features and target variable).
- **Adjustments:** Modify parameters in `RF_XGB_SVM_CODE.py` or pipeline configurations (`Pipeline` and `ColumnTransformer`) as per your specific requirements.
- **Feedback:** For any issues or improvements, feel free to raise an issue or submit a pull request.


## License

This project is licensed under the MIT License. See the [LICENSE file](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/LICENSE) for more details.  The MIT License is a permissive free software license that allows you to freely use, modify, and distribute this software. It encourages collaborative development by permitting contributions under the same license, ensuring that the software remains open and accessible to the community.

### Contributing

Contributions to this groundbreaking project are highly valued. We encourage researchers, clinicians, and stakeholders interested in pediatric neurology and precision medicine to collaborate and contribute expertise, insights, and resources. By working together, we can accelerate advancements in CP research and translate findings into impactful clinical practices.

### Acknowledgments

- This project is made possible through the generous support of the funding provided by [NPRP (National Priorities Research Program)](https://www.hbku.edu.qa/en/research/sro/nprp-NPRP-S-14th-cycle-awarded-projects) under project ID NPRP14S-0319-210075. We extend our sincere gratitude to [Dr. Mohammad Farhan](mailto:mohammadfarhan@hbku.edu.qa) for his leadership and vision in spearheading this initiative. Special thanks to the College of Health and Life Sciences of ([Hamad Bin Khalifa University](https://www.hbku.edu.qa/en/home)) for their commitment to advancing healthcare through innovative research.

### Contact Information

For inquiries, collaborations, or further information about this project, please contact Lead Principal Investigator (LPI) [Dr. Mohammad Farhan](mailto:mohammadfarhan@hbku.edu.qa). Your engagement is pivotal in achieving our shared goal of transforming care for children affected by cerebral palsy. For questions, suggestions, or feedback, please reach out to [Foysal Ahammad](mailto:foah48505@hbku.edu.qa) a PhD student of the project. Your insights and contributions play a crucial role in our mission to advance precision medicine for pediatric cerebral palsy. Together, we can make a meaningful difference in the lives of affected children and their families.
