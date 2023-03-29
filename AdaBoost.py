import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Decision stump used as weak classifier
class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        # Iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()

            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = np.floor(n_samples * sum(misclassified)) / n_samples

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha
            EPS = np.finfo(float).eps
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # print(w, y * clf.predict(X), np.exp(-np.single(y * clf.predict(X))))
            w *= np.exp(-np.single(clf.alpha * y * clf.predict(X)))
            # Normalize to one
            w /= np.sum(w)

            # Save classifier
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
names = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names',
                    sep=':', skiprows=range(0, 54), header=None)[7:11]

col_names = list(names[0])
col_names.append('Iris')
df.columns = col_names

# Convert classes in target variable to {-1, 1}
df.loc[df.Iris == 'Iris-setosa', 'Iris'] = 1
df.loc[df.Iris == 'Iris-versicolor', 'Iris'] = -1
df.loc[df.Iris == 'Iris-virginica', 'Iris'] = -1
print(df)

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='Iris').values, df.Iris.values,
                                                    train_size=int(2 * len(df) / 3), random_state=2)

# Fit model
ab = AdaBoost()
ab.fit(X_train, y_train)

# Predict on test set
y_pred = ab.predict(X_test)
print(y_pred)
print("Accuracy:", np.sum(y_test == y_pred) / len(y_test))
