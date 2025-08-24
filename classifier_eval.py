# Now that we have the synthetic samples and the training data balanced we create a function to classify the data and evaluate the model with classification metrics
## Use a bunch of classification algorithms to classify the data and evaluate the model with classification metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from imblearn.metrics import specificity_score
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classification_model(X_train, y_train, X_test, y_test, k_folds = 3, hardness_metric=None, list_classifiers=None, random_state=42):

    classifiers = [
            LogisticRegression(random_state=random_state, solver='liblinear'),
            RandomForestClassifier(random_state=random_state),
            # GaussianNB(),
            SVC(probability=True, random_state=random_state),
            KNeighborsClassifier(n_neighbors=5)
        ]        
    results = []
    # Train classifier
    for clf in classifiers:
        print(f"Training {clf.__class__.__name__}...")
        # Use cross-validation to get predictions
        # If the classifier supports cross-validation, use it
        if hasattr(clf, "fit"):
            # Fit the classifier
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf.fit(X_train, y_train)
        else:
            # If it doesn't support fit, use cross_val_predict
            y_pred = cross_val_predict(clf, X_train, y_train, cv=k_folds)
            # Fit the classifier on the entire training set
            print(f"Using cross-validation for {clf.__class__.__name__}...")
            clf = clf.fit(X_train, y_train)

    
            
        # Make predictions
        y_pred = clf.predict(X_test)
            
        # Print classification report and confusion matrix
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        # Get Table of classification metrics: accuracy, precision, recall, f1-score, specificity, balanced accuracy
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        specificity = specificity_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        metrics = {
            'Classifier': clf.__class__.__name__,
            'cv_k_folds': k_folds,
            'Random State': random_state,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Specificity': specificity,
            'Balanced Accuracy': balanced_accuracy
        }

        results.append(metrics)
    return results

