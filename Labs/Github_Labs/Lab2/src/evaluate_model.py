import pickle, os, json
from sklearn.metrics import f1_score
import joblib, glob, sys
import argparse

sys.path.insert(0, os.path.abspath('..'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    args = parser.parse_args()
    
    # Access the timestamp
    timestamp = args.timestamp
    try:
        model_version = f'model_{timestamp}_dt_model'  # Use a timestamp as the version
        model = joblib.load(f'{model_version}.joblib')
    except:
        raise ValueError('Failed to catching the latest model')
        
    try:
        # Held-out split written by train_model.py earlier in the same job
        with open('data/X_test.pickle', 'rb') as f:
            X = pickle.load(f)
        with open('data/y_test.pickle', 'rb') as f:
            y = pickle.load(f)
    except FileNotFoundError as e:
        raise ValueError('Failed to load the held-out test set; run train_model.py first') from e
    
    y_predict = model.predict(X)
    metrics = {"F1_Score":f1_score(y, y_predict)}
    
    # Save metrics to a JSON file

    if not os.path.exists('metrics/'): 
        # then create it.
        os.makedirs("metrics/")
        
    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
               
    
