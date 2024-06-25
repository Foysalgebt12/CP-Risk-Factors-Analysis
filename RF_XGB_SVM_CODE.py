# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
from matplotlib_venn import venn3  # Import venn3 for 3-way Venn diagram

# Step 2: Load the data
file_path = r'D:\LP\RANDOMFOREST\CCPF11DFLL.xlsx'
df = pd.read_excel(file_path)

# Step 3: Separate features and target variable
X = df.drop('CP/Non-CP', axis=1)
y = df['CP/Non-CP'].map({'CP': 1, 'Non-CP': 0})

# Step 4: Separate numerical and categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Step 5: Calculate p-values for numerical features using t-test
p_values_num = []
for col in numerical_cols:
    cp_group = X[y == 1][col]
    non_cp_group = X[y == 0][col]
    t_stat, p_val = ttest_ind(cp_group, non_cp_group, nan_policy='omit')
    p_values_num.append(p_val)

# Step 6: Calculate p-values for categorical features using chi-squared test
p_values_cat = []
for col in categorical_cols:
    contingency_table = pd.crosstab(X[col], y)
    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    p_values_cat.append(p_val)

# Step 7: Combine p-values and feature names
p_values = pd.Series(p_values_num + p_values_cat, index=numerical_cols.tolist() + categorical_cols.tolist())

# Step 8: Select significant features (p-value < 0.05)
significant_features = p_values[p_values < 0.05].index

# Step 9: Filter the dataset to include only significant features
X_significant = X[significant_features]

# Step 10: Preprocessing for significant features
categorical_cols_sig = X_significant.select_dtypes(include=['object']).columns
numerical_cols_sig = X_significant.select_dtypes(include=['float64', 'int64']).columns

# Step 11: Define transformers for preprocessing
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols_sig),
    ('cat', categorical_transformer, categorical_cols_sig)
])

# Step 12: Define models
rf_model = RandomForestClassifier(random_state=42)
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf_model)])

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])

svm_model = LinearSVC(dual=False)
svm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', svm_model)])

# Step 13: Train models
rf_pipeline.fit(X_significant, y)
xgb_pipeline.fit(X_significant, y)
svm_pipeline.fit(X_significant, y)

# Step 14: Get feature importances
rf_feature_importances = rf_pipeline.named_steps['model'].feature_importances_
xgb_feature_importances = xgb_pipeline.named_steps['model'].feature_importances_
svm_coefficients = svm_pipeline.named_steps['model'].coef_[0]

# Step 15: Get feature names after one-hot encoding
encoded_cat_cols = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols_sig)
feature_names = np.concatenate([numerical_cols_sig, encoded_cat_cols])

# Step 16: Get top 10 important features for Random Forest, XGBoost, and SVM
rf_top_features_indices = np.argsort(rf_feature_importances)[::-1][:10]
xgb_top_features_indices = np.argsort(xgb_feature_importances)[::-1][:10]
svm_top_features_indices = np.argsort(np.abs(svm_coefficients))[::-1][:10]

# Step 17: Extract feature names for top features
rf_top_features = feature_names[rf_top_features_indices]
xgb_top_features = feature_names[xgb_top_features_indices]
svm_top_features = feature_names[svm_top_features_indices]

# Step 18: Recalculate mean differences for one-hot encoded features
preprocessed_data = preprocessor.fit_transform(X_significant)
preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)
mean_diff = preprocessed_df.groupby(y).mean().diff().iloc[-1]

# Step 19: Set plotting parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'figure.figsize': (18, 8),
    'figure.dpi': 300,
    'figure.facecolor': 'white',
    'lines.linewidth': 2,
    'axes.edgecolor': 'black',
    'axes.spines.right': False,
    'axes.spines.top': False
})

# Step 20: Create subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 8))

# Step 21: Define function to plot feature importances
def plot_feature_importance(ax, top_features, importances, mean_diff, title):
    colors = ['#D95319' if mean_diff[feature] > 0 else '#0072BD' for feature in top_features]
    bars = ax.barh(range(len(top_features)), importances, align='center', color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    for bar in bars:
        bar.set_edgecolor('black')

# Step 22: Plot feature importances for Random Forest, XGBoost, and SVM
plot_feature_importance(axs[0], rf_top_features, rf_feature_importances[rf_top_features_indices], mean_diff, 'Top 10 Features - Random Forest')
plot_feature_importance(axs[1], xgb_top_features, xgb_feature_importances[xgb_top_features_indices], mean_diff, 'Top 10 Features - XGBoost')
plot_feature_importance(axs[2], svm_top_features, np.abs(svm_coefficients[svm_top_features_indices]), mean_diff, 'Top 10 Features - SVM')

# Step 23: Adjust layout
plt.tight_layout()

# Step 24: Save the figure
plt.savefig('published_top_features_with_mean_diff.png', bbox_inches='tight')

# Step 25: Show the plot
plt.show()

# Step 26: Get common top features among all models
common_features = set(rf_top_features) & set(xgb_top_features) & set(svm_top_features)

# Step 27: Get indices of common features
rf_common_indices = [np.where(rf_top_features == feature)[0][0] for feature in common_features]
xgb_common_indices = [np.where(xgb_top_features == feature)[0][0] for feature in common_features]
svm_common_indices = [np.where(svm_top_features == feature)[0][0] for feature in common_features]

# Step 28: Extract importances for common features
rf_common_importances = rf_feature_importances[rf_common_indices]
xgb_common_importances = xgb_feature_importances[xgb_common_indices]
svm_common_importances = np.abs(svm_coefficients[svm_common_indices])

# Step 29: Merge all common features and importances into a DataFrame
common_features_df = pd.DataFrame({
    'Feature': list(common_features),
    'Random Forest': rf_common_importances,
    'XGBoost': xgb_common_importances,
    'SVM': svm_common_importances
})

# Step 30: Melt the DataFrame to plot
common_features_melted = pd.melt(common_features_df, id_vars=['Feature'], var_name='Model', value_name='Importance')

# Step 31: Plot merged features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', hue='Model', data=common_features_melted, palette=['#D95319', '#0072BD', '#EDB120'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top Common Features Across Random Forest, XGBoost, and SVM')
plt.tight_layout()
plt.savefig('merged_common_features.png', bbox_inches='tight')
plt.show()

# Step 32: Import the venn3 function from matplotlib_venn
from matplotlib_venn import venn3

# Step 33: Calculate the set intersections
rf_features_set = set(rf_top_features)
xgb_features_set = set(xgb_top_features)
svm_features_set = set(svm_top_features)

# Step 34: Create subsets for the Venn diagram
venn_labels = {
    '100': rf_features_set - xgb_features_set - svm_features_set,
    '010': xgb_features_set - rf_features_set - svm_features_set,
    '110': rf_features_set & xgb_features_set - svm_features_set,
    '001': svm_features_set - rf_features_set - xgb_features_set,
    '101': rf_features_set & svm_features_set - xgb_features_set,
    '011': xgb_features_set & svm_features_set - rf_features_set,
    '111': rf_features_set & xgb_features_set & svm_features_set
}

# Step 35: Plot Venn diagram with specified colors for each model
plt.figure(figsize=(8, 8))
venn_diagram = venn3(subsets=(
    len(venn_labels['100']),
    len(venn_labels['010']),
    len(venn_labels['110']),
    len(venn_labels['001']),
    len(venn_labels['101']),
    len(venn_labels['011']),
    len(venn_labels['111'])),
    set_labels=('Random Forest', 'XGBoost', 'SVM'),
    set_colors=('#D95319', '#0072BD', '#EDB120'))

# Customize title
plt.title('Intersection of Top Features Among Models')
plt.savefig('venn_diagram_top_features.png', bbox_inches='tight')
plt.show()

# Step 36: Cross-validate Random Forest, XGBoost, and SVM
rf_cv_scores = cross_val_score(rf_pipeline, X_significant, y, cv=5)
xgb_cv_scores = cross_val_score(xgb_pipeline, X_significant, y, cv=5)
svm_cv_scores = cross_val_score(svm_pipeline, X_significant, y, cv=5)

# Step 37: Display cross-validation results
print("Random Forest Cross-Validation Scores:", rf_cv_scores)
print("Random Forest Mean Accuracy:", np.mean(rf_cv_scores))
print("XGBoost Cross-Validation Scores:", xgb_cv_scores)
print("XGBoost Mean Accuracy:", np.mean(xgb_cv_scores))
print("SVM Cross-Validation Scores:", svm_cv_scores)
print("SVM Mean Accuracy:", np.mean(svm_cv_scores))

# Step 38: Plot cross-validation results
models = ['Random Forest', 'XGBoost', 'SVM']
mean_scores = [np.mean(rf_cv_scores), np.mean(xgb_cv_scores), np.mean(svm_cv_scores)]
std_scores = [np.std(rf_cv_scores), np.std(xgb_cv_scores), np.std(svm_cv_scores)]

plt.figure(figsize=(8, 6))
plt.bar(models, mean_scores, yerr=std_scores, color=['#D95319', '#0072BD', '#EDB120'], capsize=5)
plt.xlabel('Models')
plt.ylabel('Mean Accuracy')
plt.title('Cross-Validation Mean Accuracy with Standard Deviation')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()

# Save the figure
plt.savefig('cross_validation_results.png', bbox_inches='tight')

# Show plot
plt.show()

# Step 39: Predict probabilities for each model
rf_probs = rf_pipeline.predict_proba(X_significant)[:, 1]
xgb_probs = xgb_pipeline.predict_proba(X_significant)[:, 1]
svm_probs = svm_pipeline.decision_function(X_significant)

# Step 40: Calculate AUC-ROC score for each model
rf_auc = roc_auc_score(y, rf_probs)
xgb_auc = roc_auc_score(y, xgb_probs)
svm_auc = roc_auc_score(y, svm_probs)

# Step 41: Plot ROC curves
plt.figure(figsize=(8, 6))

# Random Forest ROC curve
rf_fpr, rf_tpr, _ = roc_curve(y, rf_probs)
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})', color='#D95319')

# XGBoost ROC curve
xgb_fpr, xgb_tpr, _ = roc_curve(y, xgb_probs)
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.2f})', color='#0072BD')

# SVM ROC curve
svm_fpr, svm_tpr, _ = roc_curve(y, svm_probs)
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})', color='#EDB120')

# Plot ROC curve for random guessing (baseline)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('auc_roc_curve.png', bbox_inches='tight')

# Show plot
plt.show()
