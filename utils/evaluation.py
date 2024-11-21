# GeMax: Learning Graph Representation via Graph Entropy Maximization
""" Evaluate unsupervised embedding using a variety of basic classifiers. """
""" Credit: https://github.com/fanyun-sun/InfoGraph """
""" Some revise of the print format for our GeMax Project"""
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import torch.nn as nn
import torch

class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.fc(x)

def evaluate_embedding(embeddings, labels, search=True, device="cpu"):
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)

    logreg_accuracy, logreg_std = logistic_classify(x, y, device)
    print("Avg. LogReg", logreg_accuracy)

    svc_accuracy, svc_std = svc_classify(x, y, search)
    print("Avg. svc", svc_accuracy)

    return max(logreg_accuracy, svc_accuracy), logreg_std if logreg_accuracy > svc_accuracy else svc_std

def logistic_classify(x, y, device="cpu"):
    num_classes = np.unique(y).shape[0]
    num_features = x.shape[1]

    accuracies = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = LogisticRegression(num_features, num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()

        x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        for _ in range(100):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            outputs = model(x_test)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test).sum().item()
            accuracies.append(correct / y_test.size(0))

    return np.mean(accuracies), np.std(accuracies)

def svc_classify(x, y, search=True):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if search:
            parameters = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
            svc = SVC()
            model = GridSearchCV(svc, parameters, cv=5, scoring="accuracy", n_jobs=-1)
        else:
            model = SVC(C=10)

        model.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, model.predict(x_test)))

    return np.mean(accuracies), np.std(accuracies)