from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import numpy as np
import pickle
#from jsonargparse import ArgumentParser
from jsonargparse import CLI
#from jsonargparse import auto_cli

def main(dataset: str = 'iris', model: str = 'decision_tree', max_depth: int = None, criterion: str = 'gini'):
    IN_DIR = f'../../data/{dataset}/processed/'
    OUT_DIR = f'../../models/{dataset}/{model}/'

    # Load    
    print( "Loading...")
    print( f"Data from {IN_DIR} and saving model to {OUT_DIR}" )
    X_train = pd.read_csv(IN_DIR + 'X_train.csv', header=None).to_numpy()
    y_train = pd.read_csv(IN_DIR + 'y_train.csv', header=None).to_numpy().ravel()
    os.makedirs(OUT_DIR, exist_ok=True)
    print( "Loading done.\n" )

    # Train
    print( "Training..." )
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)

    cm = confusion_matrix(y_train, y_train_pred)
    accuracy = accuracy_score(y_train, y_train_pred) * 100  # percent correct

    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.2f}%")
    print( "Training done.\n" )

    # Save
    print( "Saving..." )
    np.savetxt( f'{OUT_DIR}/y_train_pred.csv', y_train_pred, delimiter=',', fmt='%s') # could also do pandas
    with open(f'{OUT_DIR}/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print( "Saving done.\n" )

if __name__ == '__main__':
    print("Running training script...")

    CLI(main)