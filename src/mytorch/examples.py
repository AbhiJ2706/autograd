import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from mytorch.nn import Dense, Sigmoid, Softmax
from mytorch.optimizer import DefaultOptimizer
from mytorch.tensor import Tensor
import mytorch.functional as F


class LogisticRegressor:
    def __init__(self):
        self.W = Tensor([0, 0, 0, 0])
        self.b = Tensor(0)
    
    def fit(self, X_train, y_train, t=100, lr=0.01):
        with DefaultOptimizer(self.W, self.b, lr=lr) as optim:
            for _ in tqdm(range(t)):
                optim.zero_grad()
                for ex, ey in zip(X_train, y_train):
                    pred = F.sigmoid(self.W @ ex + self.b)
                    loss = F.cross_entropy(pred, ey)
                    loss.backward()
                    optim.step()
                    loss.zero_grad()
    
    def predict(self, X_test):
        preds = []
        for ex in X_test:
            pred = F.sigmoid(self.W @ ex + self.b)
            preds.append(pred.val)
        return np.where(np.array(preds) > 0.5, 1, -1)
    

class SimpleANN:
    def __init__(self):
        self.l1 = Dense(4, 5)
        self.a1 = Sigmoid()
        self.l2 = Dense(5, 3)
        self.a2 = Softmax()
    
    def forward(self, x):
        x = self.a1(self.l1(x))
        x = self.a2(self.l2(x))
        return x

    def fit(self, X_train, y_train, t=100, lr=0.01):
        with DefaultOptimizer(*self.l1.parameters(), *self.l2.parameters(), lr=lr) as optim:
            for _ in tqdm(range(t)):
                optim.zero_grad()
                for ex, ey in zip(X_train, y_train):
                    pred = self.forward(ex)
                    loss = F.cross_entropy(pred, ey)
                    loss.backward()
                    optim.step()
    
    def predict(self, X_test):
        preds = []
        for ex in X_test:
            x = self.forward(ex)
            preds.append(np.argmax(x.val))
        return np.array(preds)


def logistic_regression():
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])
    df["target"] = df["target"].apply(lambda x: 1 if x == 0 else -1)

    X_train, X_test, y_train, y_test = train_test_split(df, df["target"], test_size=0.33, random_state=42)
    del X_train["target"]
    del X_test["target"]

    for col in X_train.columns:
        X_train[col] = StandardScaler().fit_transform(X_train[col].to_numpy().reshape((-1, 1)))
    for col in X_test.columns:
        X_test[col] = StandardScaler().fit_transform(X_test[col].to_numpy().reshape((-1, 1)))
    
    model = LogisticRegressor()
    model.fit(Tensor(X_train.to_numpy()), Tensor(y_train.to_numpy()))
    test_result = model.predict(Tensor(X_test.to_numpy()))
    
    print("accuracy score:", accuracy_score(test_result, y_test.to_numpy()))
    print("confusion matrix:\n", confusion_matrix(test_result, y_test.to_numpy()))
    print("ROC AUC score:", roc_auc_score(test_result, y_test.to_numpy())) 


def simple_ann():
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                        columns= iris['feature_names'] + ['target'])
    df["target"] = df["target"].apply(
        lambda x: np.array([1, 0, 0]) if x == 0 else np.array([0, 1, 0]) if x == 1 else np.array([0, 0, 1])
    )

    X_train, X_test, y_train, y_test = train_test_split(df, df["target"], test_size=0.33, random_state=42)
    del X_train["target"]
    del X_test["target"]

    for col in X_train.columns:
        X_train[col] = StandardScaler().fit_transform(X_train[col].to_numpy().reshape((-1, 1)))
    for col in X_test.columns:
        X_test[col] = StandardScaler().fit_transform(X_test[col].to_numpy().reshape((-1, 1)))

    y_train = np.array(list(map(lambda x: x.tolist(), y_train)))
    
    model = SimpleANN()
    model.fit(Tensor(X_train.to_numpy()), Tensor(y_train), lr=0.001)
    test_result = model.predict(Tensor(X_test.to_numpy()))
    
    print("accuracy score:", accuracy_score(test_result, y_test.apply(np.argmax).to_numpy()))
    print("confusion matrix:\n", confusion_matrix(test_result, y_test.apply(np.argmax).to_numpy()))


if __name__ == "__main__":
    logistic_regression()
    simple_ann()