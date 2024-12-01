import time
import numpy as np
import sys
from load_datasets import load_iris_dataset, load_wine_dataset, load_abalone_dataset
from NaiveBayes import NaiveBayes
from sklearn.naive_bayes import GaussianNB
from Knn import Knn  # importer la classe du Knn
# importer d'autres fichiers et classes si vous en avez développés
from classifieur import Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def precision(model: Classifier, X, y):
    predictions = np.array([model.predict(x.reshape(1, -1))
                           for x in X])  # Reshape x to 2D
    true_positive = np.sum((predictions == 1) & (y == 1))
    false_positive = np.sum((predictions == 1) & (y != 1))
    return true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0


def recall(model: Classifier, X, y):
    predictions = np.array([model.predict(x.reshape(1, -1))
                           for x in X])  # Reshape x to 2D
    true_positive = np.sum((predictions == 1) & (y == 1))
    false_negative = np.sum((predictions != 1) & (y == 1))
    return true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0


def f1_score(model: Classifier, X, y):
    prec = precision(model, X, y)
    rec = recall(model, X, y)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0


def confusion_matrix(model: Classifier, X, y):
    predictions = np.array([model.predict(x.reshape(1, -1))
                           for x in X])  # Reshape x to 2D
    unique_labels = np.unique(y)
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for true_label, pred_label in zip(y, predictions):
        matrix[true_label, pred_label] += 1
    return matrix


def cross_validation_knn(train, train_labels, k_values, distance_metrics, num_folds=5):
    fold_size = len(train) // num_folds
    best_k = None
    best_distance_metric = None
    best_accuracy = 0

    print("Cross validation:")
    for k in k_values:
        for distance_metric in distance_metrics:
            accuracies = []
            for fold in range(num_folds):
                # Split the data into training and validation sets
                validation_start = fold * fold_size
                validation_end = validation_start + fold_size
                validation_data = train[validation_start:validation_end]
                validation_labels = train_labels[validation_start:validation_end]
                training_data = np.concatenate(
                    (train[:validation_start], train[validation_end:]), axis=0)
                training_labels = np.concatenate(
                    (train_labels[:validation_start], train_labels[validation_end:]), axis=0)

                # Train the model
                knn = Knn(k=k, distance_metric=distance_metric)
                knn.train(training_data, training_labels)

                # Evaluate the model
                accuracy = knn.evaluate(validation_data, validation_labels)
                accuracies.append(accuracy)

            # Compute the average accuracy
            average_accuracy = np.mean(accuracies)
            print(f'k: {k}, distance_metric: {
                  distance_metric}, accuracy: {average_accuracy:.4f}')
            # Update the best hyperparameters
            if average_accuracy > best_accuracy:
                best_accuracy = average_accuracy
                best_k = k
                best_distance_metric = distance_metric

    return best_k, best_distance_metric, best_accuracy


"""
C'est le fichier main duquel nous allons tout lancer
Vous allez dire en commentaire c'est quoi les parametres que vous avez utilises
En gros, vous allez :
1- Initialiser votre classifieur avec ses parametres
2- Charger les datasets
3- Entrainer votre classifieur
4- Le tester
"""

# Initialisez vos paramètres
train_ratio = 0.7

# Charger/lire les datasets
train_iris, train_labels_iris, test_iris, test_labels_iris = load_iris_dataset(
    train_ratio)
train_wine, train_labels_wine, test_wine, test_labels_wine = load_wine_dataset(
    train_ratio)
train_abalone, train_labels_abalone, test_abalone, test_labels_abalone = load_abalone_dataset(
    train_ratio)


# Initialisez/instanciez vos classifieurs avec leurs paramètres
naive_bayes_iris = NaiveBayes()
naive_bayes_abalone = NaiveBayes()
naive_bayes_wine = NaiveBayes()

# Scikit-learn Naive Bayes classifiers
sklearn_nb_iris = GaussianNB()
sklearn_nb_abalone = GaussianNB()
sklearn_nb_wine = GaussianNB()

# Entrainez votre classifieur Naive Bayes sur chaque dataset séparément
naive_bayes_iris.train(train_iris, train_labels_iris)
naive_bayes_abalone.train(train_abalone, train_labels_abalone)
naive_bayes_wine.train(train_wine, train_labels_wine)

# Entrainez les classifieurs scikit-learn
sklearn_nb_iris.fit(train_iris, train_labels_iris)
sklearn_nb_abalone.fit(train_abalone, train_labels_abalone)
sklearn_nb_wine.fit(train_wine, train_labels_wine)

"""
Apres avoir fait l'entrainement, évaluez votre modele sur 
les donnees d'entrainement.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la precision (precision)
    - le rappel (recall)
    - le F1-score
"""

# Evaluer Naive Bayes sur chaque dataset d'entrainement

# Evaluation on Training Data (Naive Bayes)
print("\n\u001b[31;1mTrain Naive Bayes:\u001b[0m")
print(f'\u001b[34mIris:\u001b[0m\n\tAccuracy: {naive_bayes_iris.evaluate(train_iris, train_labels_iris):.4f}\n\tPrecision: {precision(naive_bayes_iris, train_iris, train_labels_iris):.4f}\n\tRecall: {recall(
    naive_bayes_iris, train_iris, train_labels_iris):.4f}\n\tF1-score: {f1_score(naive_bayes_iris, train_iris, train_labels_iris):.4f}\n\tConfusion Matrix: \n{confusion_matrix(naive_bayes_iris, train_iris, train_labels_iris)}\n')
print(f'\u001b[35mWine:\u001b[0m\n\tAccuracy: {naive_bayes_wine.evaluate(train_wine, train_labels_wine):.4f}\n\tPrecision: {precision(naive_bayes_wine, train_wine, train_labels_wine):.4f}\n\tRecall: {recall(
    naive_bayes_wine, train_wine, train_labels_wine):.4f}\n\tF1-score: {f1_score(naive_bayes_wine, train_wine, train_labels_wine):.4f}\n\tConfusion Matrix: \n{confusion_matrix(naive_bayes_wine, train_wine, train_labels_wine)}\n')
print(f'\u001b[33mAbalone:\u001b[0m\n\tAccuracy: {naive_bayes_abalone.evaluate(train_abalone, train_labels_abalone):.4f}\n\tPrecision: {precision(naive_bayes_abalone, train_abalone, train_labels_abalone):.4f}\n\tRecall: {recall(
    naive_bayes_abalone, train_abalone, train_labels_abalone):.4f}\n\tF1-score: {f1_score(naive_bayes_abalone, train_abalone, train_labels_abalone):.4f}\n\tConfusion Matrix: \n{confusion_matrix(naive_bayes_abalone, train_abalone, train_labels_abalone)}\n')

# Sklearn Training Data Evaluation
sklearn_nb_predict_iris = sklearn_nb_iris.predict(train_iris)
sklearn_nb_predict_wine = sklearn_nb_wine.predict(train_wine)
sklearn_nb_predict_abalone = sklearn_nb_abalone.predict(train_abalone)
print("\n\u001b[31;1mTrain Naive Bayes (sklearn):\u001b[0m")
print(f'\u001b[34mIris:\u001b[0m\n\tAccuracy: {sklearn_nb_iris.score(train_iris, train_labels_iris):.4f}\n\tPrecision: {metrics.precision_score(train_labels_iris, sklearn_nb_predict_iris, average="micro"):.4f}\n\tRecall: {metrics.recall_score(
    train_labels_iris, sklearn_nb_predict_iris, average="micro"):.4f}\n\tF1-score: {metrics.f1_score(train_labels_iris, sklearn_nb_predict_iris, average="micro"):.4f}\n\tConfusion Matrix: \n{metrics.confusion_matrix(train_labels_iris, sklearn_nb_predict_iris)}\n')
print(f'\u001b[35mWine:\u001b[0m\n\tAccuracy: {sklearn_nb_wine.score(train_wine, train_labels_wine):.4f}\n\tPrecision: {metrics.precision_score(train_labels_wine, sklearn_nb_predict_wine, average="micro"):.4f}\n\tRecall: {metrics.recall_score(
    train_labels_wine, sklearn_nb_predict_wine, average="micro"):.4f}\n\tF1-score: {metrics.f1_score(train_labels_wine, sklearn_nb_predict_wine, average="micro"):.4f}\n\tConfusion Matrix: \n{metrics.confusion_matrix(train_labels_wine, sklearn_nb_predict_wine)}\n')
print(f'\u001b[33mAbalone:\u001b[0m\n\tAccuracy: {sklearn_nb_abalone.score(train_abalone, train_labels_abalone):.4f}\n\tPrecision: {metrics.precision_score(train_labels_abalone, sklearn_nb_predict_abalone, average="micro"):.4f}\n\tRecall: {metrics.recall_score(
    train_labels_abalone, sklearn_nb_predict_abalone, average="micro"):.4f}\n\tF1-score: {metrics.f1_score(train_labels_abalone, sklearn_nb_predict_abalone, average="micro"):.4f}\n\tConfusion Matrix: \n{metrics.confusion_matrix(train_labels_abalone, sklearn_nb_predict_abalone)}\n')

# Tester votre classifieur

"""
Finalement, évaluez votre modele sur les donnees de test.
IMPORTANT : 
    Vous devez afficher ici avec la commande print() de python,
    - la matrice de confusion (confusion matrix)
    - l'accuracy
    - la precision (precision)
    - le rappel (recall)
    - le F1-score
"""
# Evaluation on Testing Data
print("\n\u001b[32;1mTest Naive Bayes:\u001b[0m")
print(f'\u001b[34mIris:\u001b[0m\n\tAccuracy: {naive_bayes_iris.evaluate(test_iris, test_labels_iris):.4f}\n\tPrecision: {precision(naive_bayes_iris, test_iris, test_labels_iris):.4f}\n\tRecall: {recall(
    naive_bayes_iris, test_iris, test_labels_iris):.4f}\n\tF1-score: {f1_score(naive_bayes_iris, test_iris, test_labels_iris):.4f}\n\tConfusion Matrix: \n{confusion_matrix(naive_bayes_iris, test_iris, test_labels_iris)}\n')
print(f'\u001b[35mWine:\u001b[0m\n\tAccuracy: {naive_bayes_wine.evaluate(test_wine, test_labels_wine):.4f}\n\tPrecision: {precision(naive_bayes_wine, test_wine, test_labels_wine):.4f}\n\tRecall: {recall(
    naive_bayes_wine, test_wine, test_labels_wine):.4f}\n\tF1-score: {f1_score(naive_bayes_wine, test_wine, test_labels_wine):.4f}\n\tConfusion Matrix: \n{confusion_matrix(naive_bayes_wine, test_wine, test_labels_wine)}\n')
print(f'\u001b[33mAbalone:\u001b[0m\n\tAccuracy: {naive_bayes_abalone.evaluate(test_abalone, test_labels_abalone):.4f}\n\tPrecision: {precision(naive_bayes_abalone, test_abalone, test_labels_abalone):.4f}\n\tRecall: {recall(
    naive_bayes_abalone, test_abalone, test_labels_abalone):.4f}\n\tF1-score: {f1_score(naive_bayes_abalone, test_abalone, test_labels_abalone):.4f}\n\tConfusion Matrix: \n{confusion_matrix(naive_bayes_abalone, test_abalone, test_labels_abalone)}\n')

# Sklearn Testing
sklearn_nb_predict_iris = sklearn_nb_iris.predict(test_iris)
sklearn_nb_predict_wine = sklearn_nb_wine.predict(test_wine)
sklearn_nb_predict_abalone = sklearn_nb_abalone.predict(test_abalone)
print("\n\u001b[32;1mTest Naive Bayes (sklearn):\u001b[0m")
print(f'\u001b[34mIris:\u001b[0m\n\tAccuracy: {sklearn_nb_iris.score(test_iris, test_labels_iris):.4f}\n\tPrecision: {metrics.precision_score(test_labels_iris, sklearn_nb_predict_iris, average="micro"):.4f}\n\tRecall: {metrics.recall_score(
    test_labels_iris, sklearn_nb_predict_iris, average="micro"):.4f}\n\tF1-score: {metrics.f1_score(test_labels_iris, sklearn_nb_predict_iris, average="micro"):.4f}\n\tConfusion Matrix: \n{metrics.confusion_matrix(test_labels_iris, sklearn_nb_predict_iris)}\n')
print(f'\u001b[35mWine:\u001b[0m\n\tAccuracy: {sklearn_nb_wine.score(test_wine, test_labels_wine):.4f}\n\tPrecision: {metrics.precision_score(test_labels_wine, sklearn_nb_predict_wine, average="micro"):.4f}\n\tRecall: {metrics.recall_score(
    test_labels_wine, sklearn_nb_predict_wine, average="micro"):.4f}\n\tF1-score: {metrics.f1_score(test_labels_wine, sklearn_nb_predict_wine, average="micro"):.4f}\n\tConfusion Matrix: \n{metrics.confusion_matrix(test_labels_wine, sklearn_nb_predict_wine)}\n')
print(f'\u001b[33mAbalone:\u001b[0m\n\tAccuracy: {sklearn_nb_abalone.score(test_abalone, test_labels_abalone):.4f}\n\tPrecision: {metrics.precision_score(test_labels_abalone, sklearn_nb_predict_abalone, average="micro"):.4f}\n\tRecall: {metrics.recall_score(
    test_labels_abalone, sklearn_nb_predict_abalone, average="micro"):.4f}\n\tF1-score: {metrics.f1_score(test_labels_abalone, sklearn_nb_predict_abalone, average="micro"):.4f}\n\tConfusion Matrix: \n{metrics.confusion_matrix(test_labels_abalone, sklearn_nb_predict_abalone)}\n')

# KNN

# Initialisez vos paramètres
k_values = [2, 3, 5, 7, 9]
distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

best_k_iris, best_distance_metric_iris, best_accuracy_iris = None, None, 0

# Selection des meilleurs hyperparamètres
best_k_iris, best_distance_metric_iris, best_accuracy_iris = cross_validation_knn(
    train_iris, train_labels_iris, k_values, distance_metrics)
print(f'\u001b[34mIris:\u001b[0m\n\tBest k: {best_k_iris}, \n\tBest distance metric: {
      best_distance_metric_iris}, \n\tBest accuracy: {best_accuracy_iris}\n')
best_k_wine, best_distance_metric_wine, best_accuracy_wine = cross_validation_knn(
    train_wine, train_labels_wine, k_values, distance_metrics)
print(f'\u001b[35mWine:\u001b[0m\n\tBest k: {best_k_wine}, \n\tBest distance metric: {
      best_distance_metric_wine}, \n\tBest accuracy: {best_accuracy_wine}\n')
best_k_abalone, best_distance_metric_abalone, best_accuracy_abalone = cross_validation_knn(
    train_abalone, train_labels_abalone, k_values, distance_metrics)
print(f'\u001b[33mAbalone:\u001b[0m\n\tBest k: {best_k_abalone}, \n\tBest distance metric: {
      best_distance_metric_abalone}, \n\tBest accuracy: {best_accuracy_abalone}\n')

# Initialisez/instanciez vos classifieurs avec leurs paramètres
knn_iris = Knn(k=best_k_iris, distance_metric=best_distance_metric_iris)
knn_wine = Knn(k=best_k_wine, distance_metric=best_distance_metric_wine)
knn_abalone = Knn(k=best_k_abalone,
                  distance_metric=best_distance_metric_abalone)

# sklearn classifieur pour comparer
sklearn_knn_iris = KNeighborsClassifier(
    n_neighbors=best_k_iris, metric=best_distance_metric_iris)
sklearn_knn_wine = KNeighborsClassifier(
    n_neighbors=best_k_wine, metric=best_distance_metric_wine)
sklearn_knn_abalone = KNeighborsClassifier(
    n_neighbors=best_k_abalone, metric=best_distance_metric_abalone)

# Entrainez votre classifieur
knn_iris.train(train_iris, train_labels_iris)
knn_wine.train(train_wine, train_labels_wine)
knn_abalone.train(train_abalone, train_labels_abalone)

# sklearn training
sklearn_knn_iris.fit(train_iris, train_labels_iris)
sklearn_knn_wine.fit(train_wine, train_labels_wine)
sklearn_knn_abalone.fit(train_abalone, train_labels_abalone)

print("\n\u001b[31;1mTrain:\u001b[0m")
print(f'\u001b[34mIris:\u001b[0m\n\tAccuracy: {knn_iris.evaluate(train_iris, train_labels_iris)}\n\tPrécision: {precision(knn_iris, train_iris, train_labels_iris)}\n\tRappel: {recall(
    knn_iris, train_iris, train_labels_iris)}\n\tF1-score: {f1_score(knn_iris, train_iris, train_labels_iris)}\n\tMatrice de confusion: \n{confusion_matrix(knn_iris, train_iris, train_labels_iris)}\n')
print(f'\u001b[35mWine:\u001b[0m\n\tAccuracy: {knn_wine.evaluate(train_wine, train_labels_wine)}\n\tPrécision: {precision(knn_wine, train_wine, train_labels_wine)}\n\tRappel: {recall(
    knn_wine, train_wine, train_labels_wine)}\n\tF1-score: {f1_score(knn_wine, train_wine, train_labels_wine)}\n\tMatrice de confusion: \n{confusion_matrix(knn_wine, train_wine, train_labels_wine)}\n')
print(f'\u001b[33mAbalone:\u001b[0m\n\tAccuracy: {knn_abalone.evaluate(train_abalone, train_labels_abalone)}\n\tPrécision: {precision(knn_abalone, train_abalone, train_labels_abalone)}\n\tRappel: {recall(
    knn_abalone, train_abalone, train_labels_abalone)}\n\tF1-score: {f1_score(knn_abalone, train_abalone, train_labels_abalone)}\n\tMatrice de confusion: \n{confusion_matrix(knn_abalone, train_abalone, train_labels_abalone)}\n')

# sklearn train results
sklearn_knn_predict_iris = sklearn_knn_iris.predict(train_iris)
sklearn_knn_predict_wine = sklearn_knn_wine.predict(train_wine)
sklearn_knn_predict_abalone = sklearn_knn_abalone.predict(train_abalone)

print("\n\u001b[31;1mTrain (sklearn):\u001b[0m")
print(f'\u001b[34mIris:\u001b[0m\n\tAccuracy: {sklearn_knn_iris.score(train_iris, train_labels_iris)}\n\tPrécision: {metrics.precision_score(train_labels_iris, sklearn_knn_predict_iris, average="micro")}\n\tRappel: {metrics.recall_score(
    train_labels_iris, sklearn_knn_predict_iris, average="micro")}\n\tF1-score: {metrics.f1_score(train_labels_iris, sklearn_knn_predict_iris, average="micro")}\n\tMatrice de confusion: \n{metrics.confusion_matrix(train_labels_iris, sklearn_knn_predict_iris)}\n')
print(f'\u001b[35mWine:\u001b[0m\n\tAccuracy: {sklearn_knn_wine.score(train_wine, train_labels_wine)}\n\tPrécision: {metrics.precision_score(train_labels_wine, sklearn_knn_predict_wine, average="micro")}\n\tRappel: {metrics.recall_score(
    train_labels_wine, sklearn_knn_predict_wine, average="micro")}\n\tF1-score: {metrics.f1_score(train_labels_wine, sklearn_knn_predict_wine, average="micro")}\n\tMatrice de confusion: \n{metrics.confusion_matrix(train_labels_wine, sklearn_knn_predict_wine)}\n')
print(f'\u001b[33mAbalone:\u001b[0m\n\tAccuracy: {sklearn_knn_abalone.score(train_abalone, train_labels_abalone)}\n\tPrécision: {metrics.precision_score(train_labels_abalone, sklearn_knn_predict_abalone, average="micro")}\n\tRappel: {metrics.recall_score(
    train_labels_abalone, sklearn_knn_predict_abalone, average="micro")}\n\tF1-score: {metrics.f1_score(train_labels_abalone, sklearn_knn_predict_abalone, average="micro")}\n\tMatrice de confusion: \n{metrics.confusion_matrix(train_labels_abalone, sklearn_knn_predict_abalone)}\n')

# knn test results
print("\n\u001b[32;1mTest:\u001b[0m")
print(f'\u001b[34mIris:\u001b[0m\n\tAccuracy: {knn_iris.evaluate(test_iris, test_labels_iris)}\n\tPrécision: {precision(knn_iris, test_iris, test_labels_iris)}\n\tRappel: {recall(
    knn_iris, test_iris, test_labels_iris)}\n\tF1-score: {f1_score(knn_iris, test_iris, test_labels_iris)}\n\tMatrice de confusion: \n{confusion_matrix(knn_iris, test_iris, test_labels_iris)}\n')
print(f'\u001b[35mWine:\u001b[0m\n\tAccuracy: {knn_wine.evaluate(test_wine, test_labels_wine)}\n\tPrécision: {precision(knn_wine, test_wine, test_labels_wine)}\n\tRappel: {recall(
    knn_wine, test_wine, test_labels_wine)}\n\tF1-score: {f1_score(knn_wine, test_wine, test_labels_wine)}\n\tMatrice de confusion: \n{confusion_matrix(knn_wine, test_wine, test_labels_wine)}\n')
print(f'\u001b[33mAbalone:\u001b[0m\n\tAccuracy: {knn_abalone.evaluate(test_abalone, test_labels_abalone)}\n\tPrécision: {precision(knn_abalone, test_abalone, test_labels_abalone)}\n\tRappel: {recall(
    knn_abalone, test_abalone, test_labels_abalone)}\n\tF1-score: {f1_score(knn_abalone, test_abalone, test_labels_abalone)}\n\tMatrice de confusion: \n{confusion_matrix(knn_abalone, test_abalone, test_labels_abalone)}\n')

# sklearn test results
sklearn_knn_predict_iris = sklearn_knn_iris.predict(test_iris)
sklearn_knn_predict_wine = sklearn_knn_wine.predict(test_wine)
sklearn_knn_predict_abalone = sklearn_knn_abalone.predict(test_abalone)

print("\n\u001b[32;1mTest (sklearn):\u001b[0m")
print(f'\u001b[34mIris:\u001b[0m\n\tAccuracy: {sklearn_knn_iris.score(test_iris, test_labels_iris)}\n\tPrécision: {metrics.precision_score(test_labels_iris, sklearn_knn_predict_iris, average="micro")}\n\tRappel: {
    metrics.recall_score(test_labels_iris, sklearn_knn_predict_iris, average="micro")}\n\tF1-score: {metrics.f1_score(test_labels_iris, sklearn_knn_predict_iris, average="micro")}\n\tMatrice de confusion: \n{metrics.confusion_matrix(test_labels_iris, sklearn_knn_predict_iris)}\n')
print(f'\u001b[35mWine:\u001b[0m\n\tAccuracy: {sklearn_knn_wine.score(test_wine, test_labels_wine)}\n\tPrécision: {metrics.precision_score(test_labels_wine, sklearn_knn_predict_wine, average="micro")}\n\tRappel: {
    metrics.recall_score(test_labels_wine, sklearn_knn_predict_wine, average="micro")}\n\tF1-score: {metrics.f1_score(test_labels_wine, sklearn_knn_predict_wine, average="micro")}\n\tMatrice de confusion: \n{metrics.confusion_matrix(test_labels_wine, sklearn_knn_predict_wine)}\n')
print(f'\u001b[33mAbalone:\u001b[0m\n\tAccuracy: {sklearn_knn_abalone.score(test_abalone, test_labels_abalone)}\n\tPrécision: {metrics.precision_score(test_labels_abalone, sklearn_knn_predict_abalone, average="micro")}\n\tRappel: {
    metrics.recall_score(test_labels_abalone, sklearn_knn_predict_abalone, average="micro")}\n\tF1-score: {metrics.f1_score(test_labels_abalone, sklearn_knn_predict_abalone, average="micro")}\n\tMatrice de confusion: \n{metrics.confusion_matrix(test_labels_abalone, sklearn_knn_predict_abalone)}\n')

# Temps d'exécution
print("\nTemps d'exécution moyen pour 100 prédictions: ")
temps_depart_iris = time.time()
for i in range(100):
    naive_bayes_iris.predict(test_iris)
temps_fin_iris = time.time() - temps_depart_iris
temps_depart_wine = time.time()
for i in range(100):
    naive_bayes_wine.predict(test_wine)
temps_fin_wine = time.time() - temps_depart_wine
temps_depart_abalone = time.time()
for i in range(100):
    naive_bayes_abalone.predict(test_abalone)
temps_fin_abalone = time.time() - temps_depart_abalone
print("nb_iris: ", temps_fin_iris/100, " nb_wine: ",
      temps_fin_wine/100, " nb_abalone ", temps_fin_abalone/100)
temps_depart_knn_iris = time.time()
for i in range(100):
    knn_iris.predict(test_iris)
temps_fin_knn_iris = time.time() - temps_depart_knn_iris
temps_depart_knn_wine = time.time()
for i in range(100):
    knn_wine.predict(test_wine)
temps_fin_knn_wine = time.time() - temps_depart_knn_wine
temps_depart_knn_abalone = time.time()
for i in range(100):
    knn_abalone.predict(test_abalone)
temps_fin_knn_abalone = time.time() - temps_depart_knn_abalone
print("knn_iris: ", temps_fin_knn_iris/100, " knn_wine: ",
      temps_fin_knn_wine/100, " knn_abalone ", temps_fin_knn_abalone/100)
