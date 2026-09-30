# from sklearn.datasets import fetch_rcv1
import mlflow, datetime, os, pickle
# import sklearn
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import sys
from sklearn.ensemble import RandomForestClassifier
import argparse

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    
    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")
    
    # Wisconsin Diagnostic Breast Cancer: 569 tumours, 30 numeric features.
    # sklearn encodes the target as 0 = malignant, 1 = benign.
    X, y = load_breast_cancer(return_X_y=True)

    # f1_score() scores label 1 by default, and missing a malignant tumour is the costly error
    y = 1 - y  # flip labels so that malignant = 1, benign = 0

    os.makedirs('data', exist_ok=True)

    # Hold out a stratified test set; evaluate_model.py loads data/X_test.pickle and data/y_test.pickle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    with open('data/X_test.pickle', 'wb') as f:
        pickle.dump(X_test, f)
    with open('data/y_test.pickle', 'wb') as f:
        pickle.dump(y_test, f)
    with open('data/X_train.pickle', 'wb') as f:
        pickle.dump(X_train, f)
    with open('data/y_train.pickle', 'wb') as f:
        pickle.dump(y_train, f)


    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    dataset_name = "Breast Cancer Wisconsin"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"    
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id,
                        run_name= f"{dataset_name}"):
        
        params = {
                    "dataset_name": dataset_name,
                    "number of datapoint": X_train.shape[0],
                    "number of dimensions": X_train.shape[1]}

        mlflow.log_params(params)


        forest = RandomForestClassifier(random_state=0)
        forest.fit(X_train, y_train)

        y_predict = forest.predict(X_train)
        mlflow.log_metrics({'Train Accuracy': accuracy_score(y_train, y_predict),
                            'Train F1 Score': f1_score(y_train, y_predict)})
        
        if not os.path.exists('models/'): 
            # then create it.
            os.makedirs("models/")
            
        # After retraining the model
        model_version = f'model_{timestamp}'  # Use a timestamp as the version
        model_filename = f'{model_version}_dt_model.joblib'
        dump(forest, model_filename)
                    

