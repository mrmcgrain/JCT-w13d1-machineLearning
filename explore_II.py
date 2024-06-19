import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt 
# Load the dataset
data = pd.read_csv('employ_attr.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display the shape of the dataset
print("\nShape of the dataset:", data.shape)

# Display information about the dataset
print("\nInformation about the dataset:")
print(data.info())

# Display missing values in the dataset
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Encode categorical variables
le = LabelEncoder()
for column in data.select_dtypes(include=['object']).columns:
    data[column] = le.fit_transform(data[column])

# Split the dataset into features (X) and target (y)
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Evaluate Logistic Regression model
print("\nLogistic Regression - Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("Logistic Regression - Precision:", precision_score(y_test, y_pred_logreg))
print("Logistic Regression - Recall:", recall_score(y_test, y_pred_logreg))
print("Logistic Regression - F1-score:", f1_score(y_test, y_pred_logreg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))

# Plot confusion matrix for Logistic Regression
plt.figure(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test, y_pred_logreg), annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Training Random Forest model with GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=42)
clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
clf.fit(X_train, y_train)
best_rf = clf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Evaluate Random Forest model
print("\nRandom Forest - Best Parameters:", clf.best_params_)
print("Random Forest - Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest - Precision:", precision_score(y_test, y_pred_rf))
print("Random Forest - Recall:", recall_score(y_test, y_pred_rf))
print("Random Forest - F1-score:", f1_score(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Plot confusion matrix for Random Forest
plt.figure(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()