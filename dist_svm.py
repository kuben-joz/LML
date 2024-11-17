import torch.distributed as dist
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import random

WORLD_SIZE = int(os.environ["WORLD_SIZE"])
RANK = int(os.environ["RANK"])
dist.init_process_group("gloo", rank=RANK, world_size=WORLD_SIZE)


def merge_histories(histories):
    n_models = len(histories)
    n_epochs = len(histories[0])
    print(n_models, n_epochs)
    merged_history = []
    for epoch in range(n_epochs):
        epoch_data = {"w": [], "b": [], "loss": 0.0}

        total_loss = 0.0
        for history in histories:
            epoch_data["w"].append(history[epoch]["w"])
            epoch_data["b"].append(history[epoch]["b"])
            total_loss += history[epoch]["loss"]

        epoch_data["loss"] = total_loss / n_models
        merged_history.append(epoch_data)

    return merged_history

class LinearSVM(nn.Module):
    def __init__(self, size, C=1.0):
        super().__init__()
        self.fc = nn.Linear(size, 1)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        self.criterion = SVMLoss(C=C)

    def forward(self, x):
        return self.fc(x)

    def predict(self, X):
        with torch.no_grad():
            outputs = self.forward(X).squeeze(1)
            predictions = torch.sign(outputs)
            return predictions

    def fit(self, optimizer, X, y, epochs=10):
        history = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X).squeeze(1)
            loss = self.criterion(outputs, y, self.fc.weight)
            loss.backward()
            optimizer.step()

            # Store weights, bias, and loss for visualization
            history.append({
                'w': [self.fc.weight.detach().numpy().flatten()],
                'b': [self.fc.bias.detach().item()],
                'loss': loss.item()
            })

        return history

class SVMLoss(nn.modules.Module):
    def __init__(self, C=1.0):
        super().__init__()
        self.C = C

    def forward(self, outputs, labels, weights):
            '''TODO Excercise 3'''
            ll = 1 - (labels * outputs)
            ll[ll < 0] = 0
            ll = ll.sum()
            return self.C * ll + 0.5 * torch.sum(weights**2)

class DistBagging(torch.nn.Module):
    def __init__(self, base_model_factory, n_estimators=10, sample_size=1.0):
        super().__init__()
        self.base_model_factory = base_model_factory
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.models = torch.nn.ModuleList([self._initialize_model() for _ in range(n_estimators)])

    def _initialize_model(self):
        return self.base_model_factory()

    def forward(self, X):
        scores = []
        for model in self.models:
            score = model(X)
            scores.append(score)

        scores = torch.stack(scores)  # Shape: (n_estimators, n_samples)
        average_score = scores.mean(dim=0)

        # Convert the average score to class predictions (e.g., -1 or 1)
        predictions = torch.sign(average_score)
        return predictions

    def predict(self, X):
        predictions = []
        for model in self.models:
            preds = model.predict(X)
            predictions.append(preds)
        predictions = torch.stack(predictions)
        if RANK == 0:
            gather_list = [torch.zeros_like(predictions) for _ in range(WORLD_SIZE)]
        else:
            gather_list = None

        dist.gather(predictions, gather_list=gather_list)
        
        if RANK == 0:
            predictions = torch.cat(gather_list)
            majority_preds = torch.mode(predictions, dim=0)[0]
            return majority_preds
        else:
            return None

    def fit(self, X, y, optimizer_class, epochs=100):
        histories = []
        for i, model in enumerate(self.models):
            X_sample, y_sample = self._bootstrap_sample(X, y, self.sample_size)
            optimizer = optimizer_class(model.parameters())
            single_history = model.fit(optimizer, X=X_sample, y=y_sample, epochs=epochs)
            histories.append(single_history)

        return merge_histories(histories)

    def _bootstrap_sample(self, X, y, sample_size):
        num = max(1,round(len(y) * self.sample_size))
        choice = torch.randint(len(y), (num,))
        return X[choice], y[choice]

def model_fun():
  return LinearSVM(size=2, C=2)

def optimizer_fun(params):
  return optim.SGD(params, lr=0.03)

def evaluate_model(model, X_test, y_test):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model.predict(X_test)
        if RANK==0:
            correct = (outputs == y_test).sum().item()
            total = y_test.size(0)
            accuracy = correct / total
            return accuracy
        else:
            return None


n_samples=200
noise=0.1
factor=0.2
random_state=42
out_type = torch.float32
test_prop = 0.2
test_size = round(n_samples*test_prop)


if RANK==0:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train_circle = torch.tensor(X_train, dtype=out_type)
    y_train_circle = torch.tensor(y_train, dtype=out_type)
    X_test_circle = torch.tensor(X_test, dtype=out_type)
    y_test_circle = torch.tensor(y_test, dtype=out_type)
else:
    X_train_circle = torch.zeros((n_samples-test_size, 2), dtype=out_type)
    y_train_circle = torch.zeros((n_samples-test_size,), dtype=out_type)
    X_test_circle = torch.zeros((test_size,2), dtype=out_type)
    y_test_circle = torch.zeros((test_size,), dtype=out_type)

dist.broadcast(X_train_circle, 0)
dist.broadcast(y_train_circle, 0)
dist.broadcast(X_test_circle, 0)
dist.broadcast(y_test_circle, 0)

epochs = 100
if RANK == 0:
    single_model = model_fun()
    single_optimizer = optimizer_fun(single_model.parameters())
    history_single = single_model.fit(single_optimizer, X=X_train_circle, y=y_train_circle, epochs=epochs)

bagging_svm = DistBagging(model_fun, n_estimators=5, sample_size=0.1)
history_bagging = bagging_svm.fit(X_train_circle, y_train_circle, optimizer_fun, epochs=epochs)

if RANK == 0:
    svm_accuracy = evaluate_model(single_model, X_test_circle, y_test_circle)
bagging_svm_accuracy = evaluate_model(bagging_svm, X_test_circle, y_test_circle)

#Plot all of the margins of bagging ensemble.
#Only for the brave ones!
#plot_svm(X_train_circle, y_train_circle, epochs-1, history_bagging)

if RANK == 0:
    print(f"LinearSVM Accuracy: {svm_accuracy * 100:.2f}%")
    print(f"Bagging SVM Accuracy: {bagging_svm_accuracy * 100:.2f}%")
