
 ### Project Name: Cerebral Palsy CP Risk Factors Analysis 



---

## A. Overview 

This project utilizes a Feedforward Neural Network (FNN) to analyze risk factors associated with Cerebral Palsy (CP) based on medical data. The model predicts whether a child will develop CP or not based on various demographic and health-related features. The FNN is a fundamental type of artificial neural network where information flows in one direction—from input nodes through hidden layers to output nodes. This architecture is characterized by its simplicity and effectiveness in learning mappings from input data to output predictions without feedback loops or cycles in its structure. 

![FNN](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/6065df04-db71-4c60-b4dd-513d5636cb36)

In an FNN, each layer of nodes processes inputs from the previous layer, applying weighted transformations and activation functions to produce outputs that serve as inputs to the subsequent layer. This process allows the network to learn complex patterns and relationships within the data, making it suitable for various tasks such as classification, regression, and pattern recognition.

## B. Requirements

Ensure you have the following installed before running the code:

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- SHAP (SHapley Additive exPlanations)

You can install the required Python libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap
```

## C. Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis.git
   cd Cerebral-Palsy-CP-Risk-Factors-Analysis
   ```

2. **Download Dataset:**

The dataset ([CCPF11DFL.xlsx](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Data-Preparation/CCPF11DFL.xlsx)), utilized in the development and evaluation of the Feedforward Neural Network (FFN) model, serves as the cornerstone of this study on cerebral palsy (CP) risk factors analysis. This Excel file contains structured data encompassing various clinical features and outcomes related to CP and non-CP cases. The Datset used during the FFN is ([CCPF11DFL.xlsx](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Data-Preparation/CCPF11DFL.xlsx)) can be download from here. 
 
## D.  Usage

1. **Load and Preprocess Data:**

   Update the `filepath` variable in `Feedforward_Neural_Network_(FNN).py` with the path to your dataset.

2. **Run the Script:**

   Execute the Python script to preprocess data, train the FNN model, and generate evaluation metrics and plots:

   ```bash
   python Feedforward_Neural_Network_(FNN).py
   ```

## E.  Results
The script generates the following outputs, each offering critical insights into the performance and interpretability of the Feedforward Neural Network (FFN) model used to analyze cerebral palsy (CP) risk factors:

### **1. Model Training and Validation Curves:**

The training and validation curves provide insights into the performance and behavior of the Feedforward Neural Network (FNN) model during its training phase. These curves are essential for understanding how well the model learns from the training data and generalizes to unseen validation data. Here, [training_validation_curves_stylish.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/training_validation_curves_stylish.png): Shows accuracy and loss trends during model training and validation. 

![training_validation_curves_stylish](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/9578a0a6-aaee-4898-8895-d22d9ab4362d)


**1. 1. Accuracy Trend**

The accuracy trend depicted in the training_validation_curves_stylish.png plot illustrates how the model's accuracy evolves over epochs during training and validation phases. Accuracy is a metric that measures the proportion of correctly predicted outcomes (both CP and non-CP) compared to the total number of predictions made.

**1.1.1.** Train Accuracy: This line shows the accuracy of predictions on the training dataset across different epochs. It indicates how well the model fits the training data over time.

**1.1.2.** Validation Accuracy: This line represents the accuracy on a separate validation dataset that the model hasn't seen during training. It serves as a proxy for how well the model generalizes to new, unseen data.

A widening gap between training and validation accuracies could indicate overfitting, where the model performs well on training data but fails to generalize to new data. Conversely, if both accuracies increase and stabilize together, it suggests that the model is learning effectively without overfitting.

**1.2. Loss Trend**

The loss trend in the same plot  [training_validation_curves_stylish.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/training_validation_curves_stylish.png) shows how the model's loss function evolves over epochs during training and validation.

****1.2.1. Train Loss: This curve illustrates the loss (typically binary cross-entropy in this case) calculated on the training dataset as the model iterates through epochs. Lower values indicate better fit of the model to the training data.

**1.2.2.** Validation Loss: This curve shows the loss on the validation dataset. It provides an indication of how well the model is generalizing to new data. A decreasing validation loss alongside training loss typically indicates effective learning and generalization.

**1.3. Interpretation**

**1.3.1**. Ideal Behavior: Ideally, both accuracy curves should increase and plateau, while loss curves should decrease and plateau, indicating that the model is learning well and not overfitting.

**1.3.2.** Overfitting: If the validation accuracy stagnates or decreases while the training accuracy continues to improve, or if the validation loss starts increasing, it suggests potential overfitting. Overfitting occurs when the model learns noise and details from the training data that do not generalize well to new data.


### **2. Confusion Matrix:**
  The  [confusion_matrix_stylish.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/confusion_matrix_stylish.png) visualizes the performance of the Feedforward Neural Network (FNN) model in predicting cases of Cerebral Palsy (CP) versus non-CP based on the evaluation of test data.

**2.1. Understanding the Confusion Matrix**
A confusion matrix is a table that summarizes the performance of a classification model. It compares the predicted labels from the model against the actual labels in the test dataset. The matrix is structured as follows:

- **True Positive (TP):** Instances where the model correctly predicts CP cases.
- **True Negative (TN):** Instances where the model correctly predicts non-CP cases.
- **False Positive (FP):** Instances where the model incorrectly predicts CP when the actual label is non-CP (Type I error).
- **False Negative (FN):** Instances where the model incorrectly predicts non-CP when the actual label is CP (Type II error).

**2.2. Interpretation of the Confusion Matrix**

The confusion matrix provides several key metrics:

- **Accuracy:** Overall accuracy of the model, calculated as \((TP + TN) / (TP + TN + FP + FN)\). It represents the proportion of correctly classified instances out of total instances.
  
- **Precision (Positive Predictive Value):** Precision measures the proportion of true CP cases among all instances predicted as CP, calculated as \(TP / (TP + FP)\). It indicates how confident the model is when predicting CP.

- **Recall (Sensitivity or True Positive Rate):** Recall measures the proportion of true CP cases that were correctly identified by the model, calculated as \(TP / (TP + FN)\). It indicates the model's ability to capture all actual CP cases.

- **Specificity (True Negative Rate):** Specificity measures the proportion of true non-CP cases that were correctly identified by the model, calculated as \(TN / (TN + FP)\).

![confusion_matrix_stylish](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/e95acded-3196-430f-bd36-39a4230b8bca)

**2.3. Visual Representation**
The  [confusion_matrix_stylish.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/confusion_matrix_stylish.png)   plot visualizes the confusion matrix using a heatmap. Here's what it typically shows:

- Axes: The x-axis represents predicted labels ('Non-CP' and 'CP'), while the y-axis represents actual labels ('Non-CP' and 'CP').
- Cells: Each cell in the heatmap contains the count of instances corresponding to a particular combination of predicted and actual labels.
- Color Gradient: Colors in the heatmap (often shades of blue in this case) indicate the intensity or frequency of instances in each cell. Darker shades usually represent higher counts.

**2.4. Importance**

The confusion matrix provides a more detailed understanding of model performance beyond simple accuracy. It helps in identifying which types of errors (false positives or false negatives) the model tends to make more frequently. This information is crucial for refining the model, adjusting thresholds, or prioritizing different evaluation metrics based on the specific application's requirements.

### **3. ROC Curve:**

The ROC curve, illustrated in [roc_curve_stylish.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/roc_curve_stylish.png), is a fundamental tool for evaluating the performance of binary classification models, such as the neural network used in this study to analyze cerebral palsy (CP) risk factors. This curve plots the true positive rate (sensitivity) against the false positive rate (1-specificity) across various thresholds.

**3.1. Interpretation:**
- **True Positive Rate (Sensitivity)**: This metric measures the proportion of actual positive cases (CP) correctly identified by the model.
- **False Positive Rate (1-Specificity)**: This metric indicates the proportion of actual negative cases (non-CP) incorrectly classified as positive.

**3.2.  ROC-AUC Score:**
The Area Under the Curve (AUC) of the ROC curve provides a single numerical value to assess the model's discriminatory ability:
- A high ROC-AUC score, close to 1.0, suggests that the model effectively distinguishes between CP and non-CP cases across different thresholds.
- A random classifier would have an AUC of 0.5, indicating no discrimination ability.

![roc_curve_stylish](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/bf6e5bad-d1ae-4ce7-9b9c-fbcb6abe2fdc)


In this study, the ROC curve and its associated AUC score [roc_curve_stylish.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/roc_curve_stylish.png) provide critical insights into the neural network's capability to differentiate between CP and non-CP cases. The curve's shape and the magnitude of the AUC score validate the model's effectiveness and reliability in clinical prediction tasks related to cerebral palsy risk factors.

### 4. SHAP Summary Plot:

The SHAP (SHapley Additive exPlanations) summary plot, accessible via [shap_summary_plot.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/shap_summary_plot.png), is a critical visualization tool used in this study to interpret the impact of features on the predictions made by the neural network model regarding cerebral palsy (CP) risk factors.

**4.1 Interpretation:**

**4.1.1. SHAP Values:** 
These values quantify the contribution of each feature to the model's output for individual predictions. They indicate whether a feature has a positive or negative impact on the prediction of CP.
  
**4.1.2. Features Analysis:**
The plot highlights which features significantly influence the model's decision-making process when distinguishing between CP and non-CP cases. Features such as "Gestation Age of baby (weeks)" and "Mother Age" are examples of variables that may exert substantial influence on the model's predictions.

**4.2. Visual Representation:**
- Each feature is represented by a vertical bar, where the horizontal position indicates the magnitude of the SHAP value, and the color represents the feature value (red for high values, blue for low values).
- The plot's vertical arrangement shows which features are most influential overall, providing insights into the relative importance of different variables in predicting CP.


![shap_summary_plot png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/assets/65643273/12054389-5d2d-41c8-83cf-1f1120f1fa49)


**4.3. Practical Insights:**
- **Clinical Relevance**: Clinicians and researchers can utilize these insights to better understand the underlying factors contributing to CP risk, potentially informing preventive strategies or targeted interventions.
- **Model Transparency**: The SHAP summary plot enhances the transparency of the neural network model by elucidating the rationale behind its predictions, fostering trust and facilitating model validation.

In summary, [shap_summary_plot.png](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/shap_summary_plot.png) serves as a pivotal visual aid in this study, elucidating the intricate relationship between input features and model predictions regarding CP risk factors. It underscores the model's interpretability and its potential utility in clinical decision-making and further research endeavors.


## License

This project is licensed under the MIT License. See the [LICENSE file](https://github.com/Foysalgebt12/Cerebral-Palsy-CP-Risk-Factors-Analysis/blob/Feedforward-Neural-Network-(FNN)/LICENSE) for more details.  The MIT License is a permissive free software license that allows you to freely use, modify, and distribute this software. It encourages collaborative development by permitting contributions under the same license, ensuring that the software remains open and accessible to the community.

### Contributing

Contributions to this groundbreaking project are highly valued. We encourage researchers, clinicians, and stakeholders interested in pediatric neurology and precision medicine to collaborate and contribute expertise, insights, and resources. By working together, we can accelerate advancements in CP research and translate findings into impactful clinical practices.

### Acknowledgments

- This project is made possible through the generous support of the funding provided by [NPRP (National Priorities Research Program)](https://www.hbku.edu.qa/en/research/sro/nprp-NPRP-S-14th-cycle-awarded-projects) under project ID NPRP14S-0319-210075. We extend our sincere gratitude to [Dr. Mohammad Farhan](mailto:mohammadfarhan@hbku.edu.qa) for his leadership and vision in spearheading this initiative. Special thanks to the College of Health and Life Sciences of ([Hamad Bin Khalifa University](https://www.hbku.edu.qa/en/home)) for their commitment to advancing healthcare through innovative research.

### Contact Information

For inquiries, collaborations, or further information about this project, please contact Lead Principal Investigator (LPI) [Dr. Mohammad Farhan](mailto:mohammadfarhan@hbku.edu.qa). Your engagement is pivotal in achieving our shared goal of transforming care for children affected by cerebral palsy. For questions, suggestions, or feedback, please reach out to [Foysal Ahammad](mailto:foah48505@hbku.edu.qa) a PhD student of the project. Your insights and contributions play a crucial role in our mission to advance precision medicine for pediatric cerebral palsy. Together, we can make a meaningful difference in the lives of affected children and their families.
