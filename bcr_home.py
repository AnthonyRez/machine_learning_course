import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logreg = LogisticRegression()
rf = RandomForestClassifier()

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df = pd.read_csv('indians-diabetes.csv', names=names)

def accuracy(y_test, y_pred):
    return 1 - sum(abs(y_test - y_pred)/len(y_test))

def balanced_classified_score(y_test, y_pred):
    class1_test = [i for i in y_test if i == 0]
    class1_pred = getClassifiedArray(y_test, y_pred, 0)
    class2_test = [i for i in y_test if i > 0]
    class2_pred = getClassifiedArray(y_test, y_pred, 1)
    return 0.5 * (accuracy(class1_test, class1_pred) + accuracy(class2_test, class2_pred))


def getClassifiedArray(y_test, y_pred, value):
    arr = []
    for idx, val in enumerate(y_test):
        if val == value:
            arr.append(y_pred[idx])
    return arr

def norm_arr(arr):
    mean = arr.mean()
    std = arr.std()

    normalized = (arr - mean) / std
    return normalized


def norm_df(df):
    result = df.copy()

    for feature in df.columns:
        result[feature] = norm_arr(result[feature])

    return result

def CV_home(df, classifier, nfold, norm=True):
    acc = []
    for i in range(nfold):
        y = df['class']
        train, test = stratified_split(y)

        if norm:
            X_train = norm_df(df.iloc[train, 0:8])
            X_test = norm_df(df.iloc[test, 0:8])
        else:
            X_train = df.iloc[train, 0:8]
            X_test = df.iloc[test, 0:8]

        y_train = y[train]
        y_test = y[test]

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        acc.append(balanced_classified_score(y_test, y_pred))

    return acc
