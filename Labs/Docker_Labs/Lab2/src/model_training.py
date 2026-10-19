import json
import os
import sys

import joblib
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # Fixed seed so the accuracy gate gives the same verdict on every build
    tf.keras.utils.set_random_seed(42)

    # Minimum test accuracy; set with `docker build --build-arg MIN_ACCURACY=...`
    min_accuracy = float(os.environ.get('MIN_ACCURACY', '0.9'))

    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # Build a simple TensorFlow model

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 50 epochs reached only 0.8667 test accuracy and failed the gate; 200 reaches 0.9667
    model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f} (minimum required: {min_accuracy:.4f})")

    # Quality gate: a non-zero exit fails this RUN step, so `docker build` stops
    # here and no serving image is produced from a weak model
    if test_accuracy < min_accuracy:
        print("Accuracy gate FAILED: model is below the minimum, aborting the build")
        sys.exit(1)

    # The serving stage needs the scaler too, or it would feed raw values
    # to a model that was trained on standardized ones
    model.save('my_model.keras')
    joblib.dump(sc, 'scaler.joblib')
    with open('metrics.json', 'w') as f:
        json.dump({'test_accuracy': test_accuracy, 'min_accuracy': min_accuracy}, f, indent=4)
    print("Model was trained and saved")
