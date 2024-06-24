import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns

# Function to load and preprocess data
def load_and_preprocess_data(filepath):
    data = pd.read_excel(filepath)

    numerical_features = ['Number of Prgenancy', 'Mother Age', 'Gestation Age of baby (weeks)',
                          'Number of Childerens', 'Birth Weight']
    categorical_features = ['Gender', 'Gravidity', 'Parity', 'Mother Health', 'Geastation Status',
                            'Term', 'Delivery Mode', 'BWC', 'Admit to NICU', 'Past Surgeries', 'G tube',
                            'Consanguine Marriage', 'Physcial Therapy']

    # Imputation for handling missing values
    imputer = SimpleImputer(strategy='mean')
    data[numerical_features] = imputer.fit_transform(data[numerical_features])

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ])

    X = data.drop('CP/Non-CP', axis=1)
    y = (data['CP/Non-CP'] == 'CP').astype(int)  # Convert to binary labels (0 or 1)
    X_preprocessed = preprocessor.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    feature_names = numerical_features + preprocessor.named_transformers_['cat'].get_feature_names_out().tolist()

    return X_train, X_test, y_train, y_test, feature_names

# Build the model
def build_model(input_shape):
    model = Sequential([
        Flatten(input_shape=(input_shape,)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load and preprocess data
    filepath = "D:\\LP\\RANDOMFOREST\\CCPF11DFL.xlsx"
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(filepath)

    # Build the model
    model = build_model(X_train.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping], batch_size=64)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot training & validation accuracy and loss values
    plt.figure(figsize=(16, 8))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2, marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2, marker='o')
    plt.title('Model Accuracy', fontsize=20, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=14)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2, marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2, marker='o')
    plt.title('Model Loss', fontsize=20, fontweight='bold')
    plt.ylabel('Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', fontsize=14)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('training_validation_curves_stylish.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Make predictions on the test data
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()

    # Compute and print classification report
    print(classification_report(y_test, y_pred))

    # Compute and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-CP', 'CP'], yticklabels=['Non-CP', 'CP'],
                cbar=False, annot_kws={"size": 16, "weight": "bold"}, linewidths=.5, linecolor='black')
    plt.xlabel('Predicted', fontsize=18, fontweight='bold')
    plt.ylabel('Actual', fontsize=18, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold', rotation=0)
    plt.grid(False)
    plt.savefig('confusion_matrix_stylish.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Calculate and print ROC-AUC score
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    print("ROC-AUC Score:", roc_auc)

    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=20, fontweight='bold')
    plt.legend(loc='lower right', fontsize=14)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.savefig('roc_curve_stylish.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Calculate SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot', fontsize=16)
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300)
    plt.show()
