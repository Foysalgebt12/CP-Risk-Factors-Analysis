# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

# Load the Dataset
file_path = "D:\\LP\\RANDOMFOREST\\CCPF11DFLL.xlsx"
data = pd.read_excel(file_path)

# Encode the Target Variable
data['CP/Non-CP'] = data['CP/Non-CP'].map({'CP': 1, 'Non-CP': 0})

# Encode Categorical Variables
label_encoders = {}
categorical_cols = data.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Drop Rows with Missing Values
data = data.dropna()

# Define the Target Variable and Predictors
X = data.drop(columns=['CP/Non-CP'])
y = data['CP/Non-CP']

# Add a Constant to the Predictors for the Intercept Term
X = sm.add_constant(X)

# Univariate Logistic Regression
univariate_results = {}

for col in X.columns:
    if col == 'const':
        continue
    logit_model = sm.Logit(y, X[['const', col]])
    result = logit_model.fit(disp=False)
    univariate_results[col] = {
        'Odds Ratio': np.exp(result.params[col]),
        '95% CI Lower': np.exp(result.conf_int().loc[col][0]),
        '95% CI Upper': np.exp(result.conf_int().loc[col][1]),
        'p-value': result.pvalues[col]
    }

univariate_results_df = pd.DataFrame(univariate_results).T

# Display Univariate Logistic Regression Results
print("Univariate Logistic Regression Results:")
print(univariate_results_df)
print()

# Multivariate Logistic Regression
logit_model = sm.Logit(y, X.drop(columns=['const']))
result = logit_model.fit(disp=False)

multivariate_results = {
    'Odds Ratio': np.exp(result.params),
    '95% CI Lower': np.exp(result.conf_int()[0]),
    '95% CI Upper': np.exp(result.conf_int()[1]),
    'p-value': result.pvalues
}

multivariate_results_df = pd.DataFrame(multivariate_results)

# Display Multivariate Logistic Regression Results
print("Multivariate Logistic Regression Results:")
print(multivariate_results_df)
print()

# Identify Significant Predictors
alpha = 0.05
univariate_results_df['Significant'] = univariate_results_df['p-value'] < alpha
multivariate_results_df['Significant'] = multivariate_results_df['p-value'] < alpha

# Save Results to Excel
with pd.ExcelWriter('logistic_regression_results.xlsx') as writer:
    univariate_results_df.to_excel(writer, sheet_name='Univariate_Results')
    multivariate_results_df.to_excel(writer, sheet_name='Multivariate_Results')

# Create the Combined Forest Plot Function
def create_combined_figure(univariate_results_df, multivariate_results_df, filename):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Univariate Plot
    axs[0].errorbar(univariate_results_df['Odds Ratio'], np.arange(len(univariate_results_df)),
                    xerr=[univariate_results_df['Odds Ratio'] - univariate_results_df['95% CI Lower'],
                          univariate_results_df['95% CI Upper'] - univariate_results_df['Odds Ratio']],
                    fmt='o', color='#1f77b4', markersize=8, capsize=5, elinewidth=2, label='Univariate')
    axs[0].set_yticks(np.arange(len(univariate_results_df)))
    axs[0].set_yticklabels(univariate_results_df.index)
    axs[0].set_xlabel('Odds Ratio')
    axs[0].set_title('Univariate Logistic Regression Results', fontsize=14)
    axs[0].axvline(x=1, color='gray', linestyle='--')
    axs[0].legend(loc='upper right')
    axs[0].set_xscale('log')

    for i, sig in enumerate(univariate_results_df['Significant']):
        if sig:
            axs[0].plot(univariate_results_df.loc[univariate_results_df.index[i], 'Odds Ratio'], i,
                        marker='*', markersize=12, color='black')

    # Multivariate Plot
    axs[1].errorbar(multivariate_results_df['Odds Ratio'], np.arange(len(multivariate_results_df)),
                    xerr=[multivariate_results_df['Odds Ratio'] - multivariate_results_df['95% CI Lower'],
                          multivariate_results_df['95% CI Upper'] - multivariate_results_df['Odds Ratio']],
                    fmt='o', color='#d62728', markersize=8, capsize=5, elinewidth=2, label='Multivariate')
    axs[1].set_yticks(np.arange(len(multivariate_results_df)))
    axs[1].set_yticklabels(multivariate_results_df.index)
    axs[1].set_xlabel('Odds Ratio')
    axs[1].set_title('Multivariate Logistic Regression Results', fontsize=14)
    axs[1].axvline(x=1, color='gray', linestyle='--')
    axs[1].legend(loc='upper right')
    axs[1].set_xscale('log')

    for i, sig in enumerate(multivariate_results_df['Significant']):
        if sig:
            axs[1].plot(multivariate_results_df.loc[multivariate_results_df.index[i], 'Odds Ratio'], i,
                        marker='*', markersize=12, color='black')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Call the function to create combined forest plot
create_combined_figure(univariate_results_df, multivariate_results_df, 'combined_forest_plot_logistic_regression_results.png')

# Calculate AUC-ROC Curve
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_probs = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot AUC-ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('auc_roc_curve.png', dpi=300)
plt.show()
