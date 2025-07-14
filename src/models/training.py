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
    
    rng = np.random.RandomState(0) # see https://scikit-learn.org/stable/common_pitfalls.html#getting-reproducible-results-across-multiple-executions for more details

    IN_DIR = f'../../data/{dataset}/processed/'
    OUT_DIR = f'../../models/{dataset}/{model}/'
    print( f"Data from {IN_DIR} and saving to {OUT_DIR}" )

    # Load    
    print( "Loading...")
    fname = IN_DIR + 'X_train.csv'
    print( f"Loading {fname}" )
    X_train = pd.read_csv( fname, header=None).to_numpy()

    fname = IN_DIR + 'y_train.csv'    
    print( f"Loading {fname}" )
    y_train = pd.read_csv( fname, header=None).to_numpy().ravel()
    os.makedirs(OUT_DIR, exist_ok=True)
    print( "Loading done.\n" )

    # Train
    print( "Training..." )
    clf = DecisionTreeClassifier(random_state=rng) # see https://scikit-learn.org/stable/common_pitfalls.html#getting-reproducible-results-across-multiple-executions for more details
    # update here for new models

    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)

    cm = confusion_matrix(y_train, y_train_pred)
    accuracy = accuracy_score(y_train, y_train_pred) * 100.  # percent correct
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.2f}%")
    print( "Training done.\n" )

    # Save
    print( "Saving..." )
    # confusion matrix
    df_tmp = pd.DataFrame(cm)
    fname = OUT_DIR + 'train_confusion_matrix.csv'
    if not os.path.exists(fname):
        print(f"Saving {fname}")
    else:
        print(f"Overwriting {fname}")
    df_tmp.to_csv( fname, index=False, header=False  )
    
    # training predictions
    fname = OUT_DIR + 'y_train_pred.csv'
    if not os.path.exists(fname):
        print(f"Saving {fname}")
    else:
        print(f"Overwriting {fname}")    
    np.savetxt( fname, y_train_pred, delimiter=',', fmt='%s') # could also do pandas

    # model
    # update here for new models   
    fname = OUT_DIR + f'{model}_model.pkl'  
    if not os.path.exists(fname):
        print(f"Saving {fname}")
    else:
        print(f"Overwriting {fname}")              
    with open( fname, 'wb') as f:
        pickle.dump(clf, f)
     
    print( "Saving done.\n" )

if __name__ == '__main__':
    print("Running training script...")

    CLI(main)