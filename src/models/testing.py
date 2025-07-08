from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import numpy as np
import pickle
from jsonargparse import CLI

def main(dataset: str = 'iris', model: str = 'decision_tree', max_depth: int = None, criterion: str = 'gini'):
    print( "Tracing arguments..." )
    for name, value in locals().items():
        print( f"{name}: {value}" )
    print( "Tracing done.\n" )

    IN_DIR = f'../../data/{dataset}/processed/'
    OUT_DIR = f'../../models/{dataset}/{model}/'

    # Load    
    print( "Loading...")
    print( f"Data from {IN_DIR} and saving to {OUT_DIR}" )
    X_test = pd.read_csv(IN_DIR + 'X_test.csv', header=None).to_numpy()
    y_test = pd.read_csv(IN_DIR + 'y_test.csv', header=None).to_numpy().ravel()
    os.makedirs(OUT_DIR, exist_ok=True)
    print( "Loading done.\n" )

    # Test
    print( "Testing..." )
    with open(f'{OUT_DIR}/decision_tree_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    # update here for new models        
    y_test_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred) * 100  # percent correct
    # @todo save these
    
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.2f}%")
    print( "Testing done.\n" )

    # Save
    print( "Saving..." )
    df_tmp = pd.DataFrame(cm)
    df_tmp.to_csv( OUT_DIR + 'test_confusion_matrix.csv', index=False, header
    np.savetxt( f'{OUT_DIR}/y_test_pred.csv', y_test_pred, delimiter=',', fmt='%s') # could also do pandas
    print( "Saving done.\n" )

if __name__ == '__main__':
    print("Running testing script...")

    CLI(main)