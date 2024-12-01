import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from load_datasets import load_iris_dataset, load_wine_dataset, load_abalone_dataset
from NaiveBayes import NaiveBayes
from Knn import Knn


def accuracy(predictions: list, values: list):
    return np.sum(predictions == values) / len(values)


def precision(predictions: list, values: list):
    true_positives = np.sum((predictions == 1) & (values == 1))
    false_positives = np.sum((predictions == 1) & (values == 0))
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0


def recall(predictions: list, values: list):
    true_positives = np.sum((predictions == 1) & (values == 1))
    false_negatives = np.sum((predictions == 0) & (values == 1))
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0


def f1_score(predictions: list, values: list):
    p = precision(predictions, values)
    r = recall(predictions, values)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0


def confusion_matrix(predictions: list, values: list):
    unique_values = np.unique(values)
    matrix = np.zeros((len(unique_values), len(unique_values)))
    for true_label, pred_label in zip(values, predictions):
        matrix[true_label, pred_label] += 1
    return matrix


def cross_validation_knn(train: np.ndarray, labels: np.ndarray, k_values: list, distances: list, folds: int) -> dict:
    """
    Cross validation from scratch for KNN algorithm.
    """
    fold_size = len(train) // folds
    results = {}
    best_k = None
    best_distance = None
    best_accuracy = 0

    for k in k_values:
        for distance in distances:
            accuracies = []
            for fold in range(folds):
                start = fold * fold_size
                end = start + fold_size
                train_fold = np.concatenate([train[:start], train[end:]])
                labels_fold = np.concatenate([labels[:start], labels[end:]])
                validation_fold = train[start:end]
                validation_labels = labels[start:end]
                predictions = []

                knn = Knn(k=k, distance_metric=distance)
                knn.train(train_fold, labels_fold)

                accuracy = knn.evaluate(validation_fold, validation_labels)
                accuracies.append(accuracy)

            mean_accuracy = np.mean(accuracies)
            results[(k, distance)] = mean_accuracy

            if mean_accuracy > best_accuracy:
                best_accuracy = mean_accuracy
                best_k = k
                best_distance = distance

    return results, best_k, best_distance, best_accuracy


def print_prediction_summary(predictions: list, values: list):
    print(f"\tConfusion matrix:\n{confusion_matrix(predictions, values)}")

    for i in range(np.unique(values).shape[0]):
        print(f"\n\tClass {i}")
        print(f"\t\tAccuracy: {accuracy(predictions == i, values == i)}")
        print(f"\t\tPrecision: {precision(predictions == i, values == i)}")
        print(f"\t\tRecall: {recall(predictions == i, values == i)}")
        print(f"\t\tF1-score: {f1_score(predictions == i, values == i)}")


train_ratio = 0.7
train_iris, train_labels_iris, test_iris, test_labels_iris = load_iris_dataset(
    train_ratio)
train_wine, train_labels_wine, test_wine, test_labels_wine = load_wine_dataset(
    train_ratio)
train_abalone, train_labels_abalone, test_abalone, test_labels_abalone = load_abalone_dataset(
    train_ratio)

# Naive Bayes
nb_iris = NaiveBayes()
nb_abalone = NaiveBayes()
nb_wine = NaiveBayes()

nb_iris.train(train_iris, train_labels_iris)
nb_abalone.train(train_abalone, train_labels_abalone)
nb_wine.train(train_wine, train_labels_wine)

nb_iris.train(train_iris, train_labels_iris)
nb_abalone.train(train_abalone, train_labels_abalone)
nb_wine.train(train_wine, train_labels_wine)

print("\n\u001b[31;1mTrain Naive Bayes:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
# Display the confusion matric and the accuracy, precision, recall and f1-score for each classes of each datasets
predictions = np.array([nb_iris.predict(x) for x in train_iris])
print_prediction_summary(predictions, train_labels_iris)

print("\n\u001b[32;1mAbalone:\u001b[0m")
predictions = np.array([nb_abalone.predict(x) for x in train_abalone])
print_prediction_summary(predictions, train_labels_abalone)

print("\n\u001b[32;1mWine:\u001b[0m")
predictions = np.array([nb_wine.predict(x) for x in train_wine])
print_prediction_summary(predictions, train_labels_wine)

print("\n\u001b[31;1mTest Naive Bayes:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
predictions = np.array([nb_iris.predict(x) for x in test_iris])
print_prediction_summary(predictions, test_labels_iris)

print("\n\u001b[32;1mAbalone:\u001b[0m")
predictions = np.array([nb_abalone.predict(x) for x in test_abalone])
print_prediction_summary(predictions, test_labels_abalone)

print("\n\u001b[32;1mWine:\u001b[0m")
predictions = np.array([nb_wine.predict(x) for x in test_wine])
print_prediction_summary(predictions, test_labels_wine)

# KNN
k_values = [2, 3, 5, 7, 9]
distances = ["euclidean", "manhattan", "chebyshev", "minkowski"]
folds = 3

results_iris, best_k_iris, best_distance_iris, best_accuracy_iris = cross_validation_knn(
    train_iris, train_labels_iris, k_values, distances, folds)

print("\n\u001b[31;1mTrain KNN:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
print(f"\tBest k: {best_k_iris}")
print(f"\tBest distance: {best_distance_iris}")
print(f"\tBest accuracy: {best_accuracy_iris}")
print("\tResults:")
for key, value in results_iris.items():
    print(f"\t\tK: {key[0]}, Distance: {key[1]}, Accuracy: {value}")

results_abalone, best_k_abalone, best_distance_abalone, best_accuracy_abalone = cross_validation_knn(
    train_abalone, train_labels_abalone, k_values, distances, folds)

print("\n\u001b[32;1mAbalone:\u001b[0m")
print(f"\tBest k: {best_k_abalone}")
print(f"\tBest distance: {best_distance_abalone}")
print(f"\tBest accuracy: {best_accuracy_abalone}")
print("\tResults:")
for key, value in results_abalone.items():
    print(f"\t\tK: {key[0]}, Distance: {key[1]}, Accuracy: {value}")

results_wine, best_k_wine, best_distance_wine, best_accuracy_wine = cross_validation_knn(
    train_wine, train_labels_wine, k_values, distances, folds)

print("\n\u001b[32;1mWine:\u001b[0m")
print(f"\tBest k: {best_k_wine}")
print(f"\tBest distance: {best_distance_wine}")
print(f"\tBest accuracy: {best_accuracy_wine}")
print("\tResults:")
for key, value in results_wine.items():
    print(f"\t\tK: {key[0]}, Distance: {key[1]}, Accuracy: {value}")

knn_iris = Knn(k=best_k_iris, distance_metric=best_distance_iris)
knn_abalone = Knn(k=best_k_abalone, distance_metric=best_distance_abalone)
knn_wine = Knn(k=best_k_wine, distance_metric=best_distance_wine)

knn_iris.train(train_iris, train_labels_iris)
knn_abalone.train(train_abalone, train_labels_abalone)
knn_wine.train(train_wine, train_labels_wine)

print("\n\u001b[31;1mTrain KNN:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
predictions = np.array([knn_iris.predict(x) for x in train_iris])
print_prediction_summary(predictions, train_labels_iris)

print("\n\u001b[32;1mAbalone:\u001b[0m")
predictions = np.array([knn_abalone.predict(x) for x in train_abalone])
print_prediction_summary(predictions, train_labels_abalone)

print("\n\u001b[32;1mWine:\u001b[0m")
predictions = np.array([knn_wine.predict(x) for x in train_wine])
print_prediction_summary(predictions, train_labels_wine)

print("\n\u001b[31;1mTest KNN:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
predictions = np.array([knn_iris.predict(x) for x in test_iris])
print_prediction_summary(predictions, test_labels_iris)

print("\n\u001b[32;1mAbalone:\u001b[0m")
predictions = np.array([knn_abalone.predict(x) for x in test_abalone])
print_prediction_summary(predictions, test_labels_abalone)

print("\n\u001b[32;1mWine:\u001b[0m")
predictions = np.array([knn_wine.predict(x) for x in test_wine])
print_prediction_summary(predictions, test_labels_wine)

# Scikit-learn
gnb_iris_sklearn = GaussianNB()
gnb_abalone_sklearn = GaussianNB()
gnb_wine_sklearn = GaussianNB()

gnb_iris_sklearn.fit(train_iris, train_labels_iris)
gnb_abalone_sklearn.fit(train_abalone, train_labels_abalone)
gnb_wine_sklearn.fit(train_wine, train_labels_wine)

print("\n\u001b[31;1mTrain Scikit-learn (Naive Bayes):\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
predictions = gnb_iris_sklearn.predict(train_iris)
print_prediction_summary(predictions, train_labels_iris)

print("\n\u001b[32;1mAbalone:\u001b[0m")
predictions = gnb_abalone_sklearn.predict(train_abalone)
print_prediction_summary(predictions, train_labels_abalone)

print("\n\u001b[32;1mWine:\u001b[0m")
predictions = gnb_wine_sklearn.predict(train_wine)
print_prediction_summary(predictions, train_labels_wine)

print("\n\u001b[31;1mTest Scikit-learn (Naive Bayes):\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
predictions = gnb_iris_sklearn.predict(test_iris)
print_prediction_summary(predictions, test_labels_iris)

print("\n\u001b[32;1mAbalone:\u001b[0m")
predictions = gnb_abalone_sklearn.predict(test_abalone)
print_prediction_summary(predictions, test_labels_abalone)

print("\n\u001b[32;1mWine:\u001b[0m")
predictions = gnb_wine_sklearn.predict(test_wine)
print_prediction_summary(predictions, test_labels_wine)

knn_iris_sklearn = KNeighborsClassifier(
    n_neighbors=best_k_iris, metric=best_distance_iris)
knn_abalone_sklearn = KNeighborsClassifier(
    n_neighbors=best_k_abalone, metric=best_distance_abalone)
knn_wine_sklearn = KNeighborsClassifier(
    n_neighbors=best_k_wine, metric=best_distance_wine)

knn_iris_sklearn.fit(train_iris, train_labels_iris)
knn_abalone_sklearn.fit(train_abalone, train_labels_abalone)
knn_wine_sklearn.fit(train_wine, train_labels_wine)

print("\n\u001b[31;1mTrain Scikit-learn (KNN):\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
predictions = knn_iris_sklearn.predict(train_iris)
print_prediction_summary(predictions, train_labels_iris)

print("\n\u001b[32;1mAbalone:\u001b[0m")
predictions = knn_abalone_sklearn.predict(train_abalone)
print_prediction_summary(predictions, train_labels_abalone)

print("\n\u001b[32;1mWine:\u001b[0m")
predictions = knn_wine_sklearn.predict(train_wine)
print_prediction_summary(predictions, train_labels_wine)

print("\n\u001b[31;1mTest Scikit-learn (KNN):\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
predictions = knn_iris_sklearn.predict(test_iris)
print_prediction_summary(predictions, test_labels_iris)

print("\n\u001b[32;1mAbalone:\u001b[0m")
predictions = knn_abalone_sklearn.predict(test_abalone)
print_prediction_summary(predictions, test_labels_abalone)

print("\n\u001b[32;1mWine:\u001b[0m")
predictions = knn_wine_sklearn.predict(test_wine)
print_prediction_summary(predictions, test_labels_wine)

# Comparison between the two implementations
print("\n\u001b[31;1mComparison:\u001b[0m")
print("\u001b[33;1mNaive Bayes VS KNN:\u001b[0m")

print("\u001b[32;1mIris:\u001b[0m")
print(f"\tNaive Bayes: {nb_iris.evaluate(test_iris, test_labels_iris)}")
print(f"\tKNN: {knn_iris.evaluate(test_iris, test_labels_iris)}")
print(
    f"\tScikit-learn (Naive Bayes): {gnb_iris_sklearn.score(test_iris, test_labels_iris)}")
print(
    f"\tScikit-learn (KNN): {knn_iris_sklearn.score(test_iris, test_labels_iris)}")

print("\u001b[32;1mAbalone:\u001b[0m")
print(f"\tNaive Bayes: {nb_abalone.evaluate(
    test_abalone, test_labels_abalone)}")
print(f"\tKNN: {knn_abalone.evaluate(test_abalone, test_labels_abalone)}")
print(f"\tScikit-learn (Naive Bayes): {
      gnb_abalone_sklearn.score(test_abalone, test_labels_abalone)}")
print(
    f"\tScikit-learn (KNN): {knn_abalone_sklearn.score(test_abalone, test_labels_abalone)}")

print("\u001b[32;1mWine:\u001b[0m")
print(f"\tNaive Bayes: {nb_wine.evaluate(test_wine, test_labels_wine)}")
print(f"\tKNN: {knn_wine.evaluate(test_wine, test_labels_wine)}")
print(
    f"\tScikit-learn (Naive Bayes): {gnb_wine_sklearn.score(test_wine, test_labels_wine)}")
print(
    f"\tScikit-learn (KNN): {knn_wine_sklearn.score(test_wine, test_labels_wine)}")

# Time comparison
start_time_iris_nb = time.time()
nb_iris.evaluate(test_iris, test_labels_iris)
end_time_iris_nb = time.time()

start_time_abalone_nb = time.time()
nb_abalone.evaluate(test_abalone, test_labels_abalone)
end_time_abalone_nb = time.time()

start_time_wine_nb = time.time()
nb_wine.evaluate(test_wine, test_labels_wine)
end_time_wine_nb = time.time()

start_time_iris_knn = time.time()
knn_iris.evaluate(test_iris, test_labels_iris)
end_time_iris_knn = time.time()

start_time_abalone_knn = time.time()
knn_abalone.evaluate(test_abalone, test_labels_abalone)
end_time_abalone_knn = time.time()

start_time_wine_knn = time.time()
knn_wine.evaluate(test_wine, test_labels_wine)
end_time_wine_knn = time.time()

time_iris_nb_ms = (end_time_iris_nb - start_time_iris_nb) * 1000
time_abalone_nb_ms = (end_time_abalone_nb - start_time_abalone_nb) * 1000
time_wine_nb_ms = (end_time_wine_nb - start_time_wine_nb) * 1000
time_iris_knn_ms = (end_time_iris_knn - start_time_iris_knn) * 1000
time_abalone_knn_ms = (end_time_abalone_knn - start_time_abalone_knn) * 1000
time_wine_knn_ms = (end_time_wine_knn - start_time_wine_knn) * 1000

print("\n\u001b[31;1mTime comparison:\u001b[0m")
print("\u001b[32;1mIris:\u001b[0m")
print(f"\tNaive Bayes: {time_iris_nb_ms} ms")
print(f"\tKNN: {time_iris_knn_ms} ms")

print("\u001b[32;1mAbalone:\u001b[0m")
print(f"\tNaive Bayes: {time_abalone_nb_ms} ms")
print(f"\tKNN: {time_abalone_knn_ms} ms")

print("\u001b[32;1mWine:\u001b[0m")
print(f"\tNaive Bayes: {time_wine_nb_ms} ms")
print(f"\tKNN: {time_wine_knn_ms} ms")
