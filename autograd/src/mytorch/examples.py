import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from mytorch.operation import DotProduct
from mytorch.tensor import Tensor
import mytorch.functional as F


class LogisticRegressor:
    def __init__(self):
        self.W = Tensor([0, 0, 0, 0])
        self.b = Tensor(0)
    
    def fit(self, X_train, y_train, t=100, lr=0.01):
        for _ in tqdm(range(t)):
            W_grad = np.zeros_like(self.W.val)
            b_grad = np.zeros_like(self.b.val)
            for ex, ey in zip(X_train, y_train):
                op = DotProduct()
                pred = F.sigmoid(op.forward(self.W, ex) + self.b)
                loss = F.cross_entropy(pred, ey)
                loss.backward()
                W_grad += lr * self.W.grad
                b_grad += lr * self.b.grad.reshape((1,))
                loss.zero_grad()
            self.W -= W_grad / 100
            self.b -= b_grad / 100
    
    def predict(self, X_test):
        preds = []
        for ex in X_test:
            op = DotProduct()
            pred = F.sigmoid(op.forward(self.W, ex) + self.b)
            preds.append(pred.val)
        return np.where(np.array(preds) > 0.5, 1, -1)


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


if __name__ == "__main__":
    logistic_regression()