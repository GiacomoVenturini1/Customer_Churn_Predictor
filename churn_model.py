import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    roc_curve, 
    auc
)

# Set plotting style
sns.set_theme(style="whitegrid")

# 1. Generate realistic synthetic data where features actually correlate with Churn
np.random.seed(42)
data_size = 1500

tenure = np.random.randint(1, 72, size=data_size)
monthly_charges = np.random.uniform(20, 120, size=data_size)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=data_size, p=[0.5, 0.3, 0.2])

# Create a logit score to determine churn logically
logit = (monthly_charges * 0.04) - (tenure * 0.08) + (contract == 'Month-to-month') * 1.5 - 1.0
probability = 1 / (1 + np.exp(-logit))
churn = np.random.binomial(1, probability)

data = pd.DataFrame({
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'Contract': contract,
    'Churn': churn
})

# 2. Separate features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# 3. Preprocess Categorical and Numerical Data
numeric_features = ['tenure', 'MonthlyCharges']
categorical_features = ['Contract']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Transform data and train the Logistic Regression Model
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = LogisticRegression()
model.fit(X_train_processed, y_train)

# 6. Predict and Evaluate
y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Generate and Save Evaluation Plots
# Ensure the images directory exists
os.makedirs('images', exist_ok=True)

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Churned'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix')
plt.grid(False)
plt.savefig('images/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('images/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Feature Importance
cat_encoder = preprocessor.named_transformers_['cat']
encoded_cat_features = list(cat_encoder.get_feature_names_out(['Contract']))
feature_names = numeric_features + encoded_cat_features
coefficients = model.coef_[0]

plt.figure(figsize=(8, 5))
sns.barplot(x=coefficients, y=feature_names, hue=feature_names, palette='coolwarm')
plt.title('Logistic Regression Feature Coefficients (Importance)')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nEvaluation plots successfully generated and saved to the 'images/' folder!")