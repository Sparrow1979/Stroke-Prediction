import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from typing import List
from pandas import DataFrame


def evaluate_models(data: pd.DataFrame, categorical_columns: list, numeric_columns: list) -> pd.DataFrame:
    """
    Evaluate different machine learning models using the provided data and column lists.

    Parameters:
        data (DataFrame): The input data containing features and target.
        categorical_columns (List[str]): List of categorical column names.
        numeric_columns (List[str]): List of numeric column names.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation metrics for each model.

    Note:
        This function assumes that the target column is named 'stroke' and it's binary.

    Example:
        evaluate_models(data, ['gender', 'work_type'], ['age', 'avg_glucose_level'])
    """
    data['stroke'] = data['stroke'].map({'Yes': 1, 'No': 0})
    data['stroke'] = data['stroke'].fillna(0)

    X = data.drop(['stroke'], axis=1)
    y = data['stroke']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69)

    numeric_features = numeric_columns
    categorical_features = categorical_columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    models = [
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('SVM', SVC()),
        ('XGBoost', XGBClassifier())
    ]

    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for name, model in models:
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        model_names.append(name)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    results_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1_scores
    })
    data['stroke'] = data['stroke'].map({1: 'Yes', 0: "No"})
    return results_df


def encode_categorical(data, cat_columns):
    encoded_data = data.copy()

    for col in cat_columns:
        unique_values = data[col].unique()

        if 'Yes' in unique_values and 'No' in unique_values:
            mapping = {'Yes': 1, 'No': 0}
        else:
            mapping = {value: idx for idx, value in enumerate(unique_values)}

        encoded_data[col] = encoded_data[col].map(mapping)

    return encoded_data
