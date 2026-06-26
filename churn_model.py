import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate realistic synthetic data where features actually correlate with Churn
np.random.seed(42)
data_size = 1500

tenure = np.random.randint(1, 72, size=data_size)
monthly_charges = np.random.uniform(20, 120, size=data_size)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=data_size, p=[0.5, 0.3, 0.2])

# Create a logit score to determine churn logically
# High charges, low tenure, and month-to-month contracts increase churn probability
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
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))