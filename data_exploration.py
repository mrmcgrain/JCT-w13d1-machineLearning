import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV


def main():

    df = pd.read_csv('employ_attr.csv')

     # Explore the dataset
    print("First few rows of the dataset:")
    print(df.head())
    print(f"Shape of the dataset: {df.shape}")
    print("Information about the dataset:")
    print(df.info())
    print("Missing values in the dataset:")
    print(df.isnull().sum())

    # Handle missing values
    df.dropna(inplace=True)

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Attrition')  # Target variable should not be encoded here
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Split the dataset
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Binary encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Preprocessing for numerical data: scaling
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data: one-hot encoding
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the model pipelines
    logistic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', LogisticRegression())])

    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier())])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training and evaluation - Logistic Regression
    print("Training Logistic Regression model...")
    logistic_pipeline.fit(X_train, y_train)
    y_pred = logistic_pipeline.predict(X_test)
    evaluate_model(y_test, y_pred, "Logistic Regression")

    # Model training and evaluation - Random Forest with GridSearchCV
    print("Training Random Forest model with GridSearchCV...")
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    evaluate_model(y_test, y_pred_rf, "Random Forest")

    # Feature importances from the Random Forest model
    print("Feature importances from the Random Forest model:")
    feature_importances = best_rf.named_steps['classifier'].feature_importances_
    # Get feature names after one-hot encoding
    feature_names = numerical_cols + list(best_rf.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols))
    feature_importance_series = pd.Series(feature_importances, index=feature_names)
    feature_importance_series.sort_values(ascending=False, inplace=True)
    print(feature_importance_series)

def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model_name} - Accuracy: {accuracy}")
    print(f"{model_name} - Precision: {precision}")
    print(f"{model_name} - Recall: {recall}")
    print(f"{model_name} - F1-score: {f1}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()

