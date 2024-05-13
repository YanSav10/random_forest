import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from scipy.stats import mode

np.random.seed(52)


def convert_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2

def create_bootstrap(x_array, y_array):
    mask = np.random.choice(len(x_array), size=len(x_array), replace=True)
    return x_array[mask], y_array[mask]

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=np.iinfo(np.int64).max, min_error=1e-6):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_error = min_error
        self.forest = []
        self.is_fit = False

    def fit(self, X_training, y_training):
        for _ in tqdm(range(self.n_trees), desc='Fitting trees'):
            # Create a bootstrap sample
            X_boot, y_boot = create_bootstrap(X_training, y_training)

            # Initialize a decision tree with specified parameters
            tree = DecisionTreeClassifier(
                max_features='sqrt',
                max_depth=self.max_depth,
                min_impurity_decrease=self.min_error,
            )

            # Fit the tree with the bootstrapped data
            tree.fit(X_boot, y_boot)

            # Add the trained tree to the forest
            self.forest.append(tree)

        self.is_fit = True

    def predict(self, X_testing):
        if not self.is_fit:
            raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')

        prediction = np.array([tree.predict(X_testing) for tree in self.forest])
        majority_votes, _ = mode(prediction, axis=0)
        return majority_votes.ravel()


if __name__ == '__main__':
    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )
    data.dropna(inplace=True)

    # Separate these back
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'] = X['Embarked'].apply(lambda x: convert_embarked(x))

    X_train, X_val, y_train, y_val = train_test_split(X.values, y.values, stratify=y, train_size=0.8)

    accuracies = []

    # Test the RandomForestClassifier with n_trees ranging from 1 to 600
    for n_trees in range(1, 601):
        rf = RandomForestClassifier(n_trees=n_trees)
        rf.fit(X_train, y_train)
        rf_predictions = rf.predict(X_val)
        rf_accuracy = accuracy_score(y_val, rf_predictions)
        accuracies.append(rf_accuracy)

    # Print the first 20 accuracy values rounded to three digits
    print([round(acc, 3) for acc in accuracies[:20]])
