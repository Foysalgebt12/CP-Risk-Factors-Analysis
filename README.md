
 ### Project Name: Cerebral Palsy CP Risk Factors Analysis 



---

## A. Overview 

The Python script conducts logistic regression analysis (LRA) on a dataset ([CCPF11DFLL.xlsx](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/main/CCPF11DFLL.xlsx) to identify significant predictors of Cerebral Palsy (CP). The goal is to explore the relationship between various predictors and the likelihood of CP occurrence.

## B. Requirements

Ensure you have the following installed before running the code:

-Python 3.x
-pandas
-numpy
-matplotlib
-statsmodels
-scikit-learn

You can install the required Python libraries using `pip`:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

## C. Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis.git
   cd Cerebral-Palsy-CP-Risk-Factors-Analysis
   ```

2. **Download Dataset:**
      The dataset ([CCPF11DFLL.xlsx](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/main/CCPF11DFLL.xlsx)  utilized in the development and evaluation of the logistic regression analysis, serves 
    as the cornerstone of this study on cerebral palsy (CP) risk factors analysis. This Excel file contains structured data encompassing various clinical features and outcomes related to CP and non-CP cases. The Datset used during the LRA is ([CCPF11DFLL.xlsx](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/main/CCPF11DFLL.xlsx))  can be download from here. 
 
## D.  Usage

1. **Load and Preprocess Data:**

   Update the `filepath` variable in [Logistic-Regression-Analysis](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/tree/Logistic-Regression-Analysis) with the path to your dataset.

2. **Run the Script:**

   Execute the Python script to preprocess data,  and generate evaluation metrics and plots:

   ```bash
   python logistic_regression_analysis.py
   ```

## E.  Results
The script generates the following outputs, each offering critical insights into the performance and interpretability of the LRA model used to analyze cerebral palsy (CP) risk factors:

### **1. Univariate Logistic Regression Results Plot:**

This plot ([univariate_logistic_regression_results.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Logistic-Regression-Analysis/multivariate_logistic_regression_results.png) shows the odds ratios (with 95% confidence intervals) for each predictor variable in the dataset based on univariate logistic regression analysis. Each point represents the odds ratio, with error bars indicating the lower and upper bounds of the confidence interval. The vertical dashed line at x=1 indicates the reference line where the odds ratio equals 1 (no effect). Significant predictors (where the p-value < 0.05) are marked with a star (*).

![univariate_logistic_regression_results](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/2a0946ad-44ec-40ee-a175-2f54a50add5c)



### **2.Multivariate Logistic Regression Results Plot:**
This plot [multivariate_logistic_regression_results.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Logistic-Regression-Analysis/multivariate_logistic_regression_results.png) displays the odds ratios (with 95% confidence intervals) for each predictor variable in the dataset based on multivariate logistic regression analysis. Similar to the univariate plot, each point represents the odds ratio, and error bars indicate the confidence intervals. The vertical dashed line at x=1 denotes the reference line where the odds ratio equals 1 (no effect). Significant predictors (where the p-value < 0.05) are marked with a star (*).
![multivariate_logistic_regression_results](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/5e404bcb-65df-4ceb-8e2c-433dc37b5ce9)


### **3. Excel File with Logistic Regression Results**

This Excel file ([logistic_regression_results.xlsx](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Logistic-Regression-Analysis/logistic_regression_results.xlsx)) contains two sheets:

**3.1. Sheet 1 (Univariate_Results):** Contains the results of univariate logistic regression, including odds ratio, 95% confidence intervals (lower and upper bounds), and p-values for each predictor variable.
**3.4. Sheet 2 (Multivariate_Results):** Contains the results of multivariate logistic regression, similarly including odds ratio, 95% confidence intervals (lower and upper bounds), and p-values for each predictor variable.

### **4. AUC-ROC Curve Plot:**

This plot [auc_roc_curve.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Logistic-Regression-Analysis/auc_roc_curve.png) illustrates the Receiver Operating Characteristic (ROC) curve for the logistic regression model. It shows the trade-off between the True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity) across different threshold values. The Area Under the Curve (AUC) value is included in the legend, indicating the overall performance of the model in distinguishing between the two classes (CP and Non-CP).

![auc_roc_curve](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/b8fcb626-ddfd-460a-9769-b227035bb296)


## License

This project is licensed under the MIT License. See the [LICENSE file](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/LICENSE) for more details.  The MIT License is a permissive free software license that allows you to freely use, modify, and distribute this software. It encourages collaborative development by permitting contributions under the same license, ensuring that the software remains open and accessible to the community.

### Contributing

Contributions to this groundbreaking project are highly valued. We encourage researchers, clinicians, and stakeholders interested in pediatric neurology and precision medicine to collaborate and contribute expertise, insights, and resources. By working together, we can accelerate advancements in CP research and translate findings into impactful clinical practices.

### Acknowledgments

- This project is made possible through the generous support of the funding provided by [NPRP (National Priorities Research Program)](https://www.hbku.edu.qa/en/research/sro/nprp-NPRP-S-14th-cycle-awarded-projects) under project ID NPRP14S-0319-210075. We extend our sincere gratitude to [Dr. Mohammad Farhan](mailto:mohammadfarhan@hbku.edu.qa) for his leadership and vision in spearheading this initiative. Special thanks to the College of Health and Life Sciences of ([Hamad Bin Khalifa University](https://www.hbku.edu.qa/en/home)) for their commitment to advancing healthcare through innovative research.

### Contact Information

For inquiries, collaborations, or further information about this project, please contact Lead Principal Investigator (LPI) [Dr. Mohammad Farhan](mailto:mohammadfarhan@hbku.edu.qa). Your engagement is pivotal in achieving our shared goal of transforming care for children affected by cerebral palsy. For questions, suggestions, or feedback, please reach out to [Foysal Ahammad](mailto:foah48505@hbku.edu.qa) a PhD student of the project. Your insights and contributions play a crucial role in our mission to advance precision medicine for pediatric cerebral palsy. Together, we can make a meaningful difference in the lives of affected children and their families.
