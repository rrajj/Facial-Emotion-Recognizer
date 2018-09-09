import pandas as pd

def load_datasets():
    print("[*] Loading Datasets.")
    X_train = pd.read_csv("datasets/X_train.csv", header = None).values/255.0
    y_train = pd.read_csv("datasets/y_train.csv", header = None)
    X_test = pd.read_csv("datasets/X_test.csv", header = None).values/255.0
    y_test = pd.read_csv("datasets/y_test.csv", header = None)
    print("[*] Dataset Loaded.")
    return X_train, y_train, X_test, y_test

