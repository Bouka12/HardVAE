from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path_processed, test_size = 0.2, random_state=None):
    # Read the data from a CSV file
    df = pd.read_csv(path_processed, sep=',')
    
    # Separate features and labels
    df_base = df.iloc[:, :-1]  # Features
    df_labels = df.iloc[:, -1].values  # Labels
    # Split data into train and test sets using stratified random sampling
    X_train, X_test,y_train, y_test = train_test_split(
        df_base.values,
        df_labels,
        test_size=test_size,
        stratify=df_labels,
        random_state=random_state
    )
    
    # Fit the StandardScaler on training data and transform both train and test sets
    standardizer = StandardScaler()
    X_train = standardizer.fit_transform(X_train)  # Fit and transform training data
    X_test = standardizer.transform(X_test)       # Transform test data only
    
    # Identify majority and minority classes in the training set
    label_counts = pd.Series(y_train).value_counts()
    #print(label_counts)
    minority_label = label_counts.idxmin()
    majority_label = label_counts.idxmax()

    # Calculate imbalance ratio
    minority_count = label_counts[minority_label]
    majority_count = label_counts[majority_label]
    imbalance_ratio = minority_count / majority_count
    
    # Separate the training data into majority and minority classes
    minority_data = X_train[y_train == minority_label]
    majority_data = X_train[y_train == majority_label]
    
    # Print imbalance ratio
    #print(f"Imbalance Ratio (IR): {imbalance_ratio:.2f} (Majority: {majority_count}, Minority: {minority_count})")
    #print(f"type of minority data: {type(minority_data)}")
    # Return train/test sets, scaler, and separated majority/minority data
    return X_train, y_train, X_test, y_test, standardizer, majority_data, minority_data, imbalance_ratio
