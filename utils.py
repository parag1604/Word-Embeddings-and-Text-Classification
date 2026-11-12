"""
Loading and cleaning AG News, shared by notebooks 2-4.

The CSV files have no header and three columns:
class index (1-4) | title | description
"""

import os

import pandas as pd


def clean_data(X: str):
    # remove special characters, newlines, numbers, extra spaces, and non-ascii characters
    X = X.replace('\\n', ' ').replace('\\t', ' ').replace('\\', ' ')
    X = X.encode('ascii', errors='ignore').decode()
    X = X.lower()
    X = ''.join([i for i in X if i.isalpha() or i == ' '])
    X = ' '.join(X.split())
    return X


def read_split(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found: download AG News first (see README.md, "Data").')
    data = pd.read_csv(path, header=None)
    X, y = data[1] + ' ' + data[2], data[0]
    # labels 1..4 in the file -> 0..3
    return [clean_data(x) for x in X.values], y.values - 1


def get_data(data_dir: str = 'data/ag_news_csv'):
    X_train, y_train = read_split(os.path.join(data_dir, 'train.csv'))
    X_test, y_test = read_split(os.path.join(data_dir, 'test.csv'))
    return X_train, y_train, X_test, y_test
