from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import numpy as np
import pickle
from jsonargparse import CLI

def main(dataset: str = 'CODaN', model: str = 'decision_tree' ):
    print( "Tracing arguments..." )
    for name, value in locals().items():
        print( f"{name}: {value}" )
    print( "Tracing done.\n" )

    IN_DIR = f'../../data/{dataset}/processed/'
    CSV_HDR = None
    if dataset == 'CODaN':
        IN_DIR = f'../../data/{dataset}/test/'
        CSV_HDR = 0    
    OUT_DIR = f'../../models/{dataset}/{model}/'
    print( f"Data from {IN_DIR} and saving to {OUT_DIR}" )

    # Load    
    print( "Loading testing data...")
    fname = IN_DIR + 'X_test.csv'
    print( f"Loading {fname}" )
    if not os.path.exists(fname) or not os.access(fname, os.R_OK):
        raise FileNotFoundError(f'Cannot read test features file: {fname}')
    X_test = pd.read_csv( fname, header=CSV_HDR ).to_numpy()
    print("X_test first 2 rows:\n", X_test[:2])
    print("X_test last 2 rows:\n", X_test[-2:])
    
    fname = IN_DIR + 'y_test.csv'
    print( f"Loading {fname}" )    
    if not os.path.exists(fname) or not os.access(fname, os.R_OK):
        raise FileNotFoundError(f'Cannot read test labels file: {fname}')
    y_test = pd.read_csv(IN_DIR + 'y_test.csv', header=CSV_HDR ).to_numpy().ravel()
    print("y_test first 2 labels:\n", y_test[:2])
    print("y_test last 2 labels:\n", y_test[-2:])    
    
    os.makedirs(OUT_DIR, exist_ok=True)
    print( "Loading done.\n" )

    # Test
    print( "Testing..." )
    fname = OUT_DIR + f'{model}_model.pkl'
    if not os.path.exists(fname) or not os.access(fname, os.R_OK):
        raise FileNotFoundError(f'Cannot read model file: {fname}')    
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