## Travail pratique 3

#### Classes et modules utilisés

- `Classifier`
La classe `Classifier` est une classe de base pour les classifieurs. Elle définit une interface commune pour les méthodes d'entraînement, de prédiction et d'évaluation que les autres classifieurs doivent implémenter.

- `Knn`
La classe `Knn` implémente l'algorithme des k-plus proches voisins (k-nearest neighbors). Elle inclut des méthodes pour entraîner le modèle, faire des prédictions et évaluer les performances du modèle.

- `NaiveBayes`
La classe `NaiveBayes` implemente l'algorithme du bayes naif. Elle inclut des méthodes pour entraîner le modèle, faire des prédictions et évaluer les performances du modèle.

- `load_datasets`
Ce module contient des fonctions pour charger différents ensembles de données, tels que les datasets Iris, Wine et Abalone. Les fonctions de ce module divisent également les données en ensembles d'entraînement et de test.

- `entrainer_tester`
Ce fichier contient des fonctions pour entraîner et tester les modèles de classification. Il inclut également des fonctions pour calculer des métriques de performance telles que la précision, le rappel, le F1-score et la matrice de confusion.

#### Fonctions utilisées

- `load_iris_dataset`
La fonction `load_iris_dataset` charge le dataset Iris, le divise en ensembles d'entraînement et de test, et retourne ces ensembles.

- `load_wine_dataset`
La fonction `load_wine_dataset` charge le dataset Wine, le divise en ensembles d'entraînement et de test, et retourne ces ensembles.

- `load_abalone_dataset`
La fonction `load_abalone_dataset` charge le dataset Abalone, le divise en ensembles d'entraînement et de test, et retourne ces ensembles.

- `precision`
La fonction `precision` calcule la précision d'un modèle sur un ensemble de données donné.

- `recall`
La fonction `recall` calcule le rappel d'un modèle sur un ensemble de données donné.

- `f1_score`
La fonction `f1_score` calcule le F1-score d'un modèle sur un ensemble de données donné.

- `confusion_matrix`
La fonction `confusion_matrix` calcule la matrice de confusion d'un modèle sur un ensemble de données donné.

- `cross_validation_knn`
La fonction `cross_validation_knn` effectue une validation croisée pour sélectionner les meilleurs hyperparamètres pour le modèle KNN.


### Répartition des tâches

Pour le code nécessaire au chargement des données, l'entièreté de l'équipe a participé. Pour ce qui est des modèles, ils ont chacun été implémentés par un membre différent de l'équipe.

### Difficultés rencontrées

La lecture des jeux de données a été plus difficile à implémenter, mais en général nous n'avons rencontré aucune difficulté pour l'implémentation des modèles ou des tests dans le cas où nos résultats sont valides et cohérents. 

