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
    print( f"Data from {IN_DIR} and saving to {OUT_DIR}" )

    # Load    
    print( "Loading...")
    fname = IN_DIR + 'X_test.csv'
    print( f"Loading {fname}" )
    X_test = pd.read_csv( fname, header=None).to_numpy()
    
    fname = IN_DIR + 'y_test.csv'
    print( f"Loading {fname}" )    
    y_test = pd.read_csv(IN_DIR + 'y_test.csv', header=None).to_numpy().ravel()
    os.makedirs(OUT_DIR, exist_ok=True)
    print( "Loading done.\n" )

    # Test
    print( "Testing..." )
    fname = OUT_DIR + f'{model}_model.pkl'
    with open(fname, 'rb') as f:
        clf = pickle.load(f)
    # update here for new models        
    y_test_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred) * 100.  # percent correct
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.2f}%")
    print( "Testing done.\n" )

    # Save
    print( "Saving..." )
    # confusion matrix
    df_tmp = pd.DataFrame(cm)
    fname = OUT_DIR + 'test_confusion_matrix.csv'
    if not os.path.exists(fname):
        print(f"Saving {fname}")
    else:
        print(f"Overwriting {fname}")    
    df_tmp.to_csv( fname, index=False, header=False )
    
    # testing predictions    
    fname = OUT_DIR + 'y_test_pred.csv'
    if not os.path.exists(fname):
        print(f"Saving {fname}")
    else:
        print(f"Overwriting {fname}")      
    np.savetxt( fname, y_test_pred, delimiter=',', fmt='%s') # could also do pandas
    print( "Saving done.\n" )

if __name__ == '__main__':
    print("Running testing script...")

    CLI(main)