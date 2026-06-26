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
    
    rng = np.random.RandomState(0) # see https://scikit-learn.org/stable/common_pitfalls.html#getting-reproducible-results-across-multiple-executions for more details
    
    IN_DIR = f'../../data/{dataset}/processed/'
    CSV_HDR = None
    if dataset == 'CODaN':
        IN_DIR = f'../../data/{dataset}/train/'
        CSV_HDR = 0    
    OUT_DIR = f'../../models/{dataset}/{model}/'
    print( f"Data from {IN_DIR} and saving to {OUT_DIR}" )

    # Load
    print( "Loading training data...")
    fname = IN_DIR + 'X_train.csv'
    print( f"Loading {fname}" )
    if not os.path.exists(fname) or not os.access(fname, os.R_OK):
        raise FileNotFoundError(f'Cannot read training features file: {fname}')
    X_train = pd.read_csv( fname, header=CSV_HDR ).to_numpy()
    print("X_train first 2 rows:\n", X_train[:2])
    print("X_train last 2 rows:\n", X_train[-2:])

    fname = IN_DIR + 'y_train.csv'
    print( f"Loading {fname}" )
    if not os.path.exists(fname) or not os.access(fname, os.R_OK):
        raise FileNotFoundError(f'Cannot read training labels file: {fname}')
    y_train = pd.read_csv( fname, header=CSV_HDR ).to_numpy().ravel()
    print("y_train first 2 labels:\n", y_train[:2])
    print("y_train last 2 labels:\n", y_train[-2:])

    os.makedirs(OUT_DIR, exist_ok=True)
    print( "Loading done.\n" )

    # Train
    print( "Training..." )
    if ( model == 'decision_tree' ):
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=rng) # see https://scikit-learn.org/stable/common_pitfalls.html#getting-reproducible-results-across-multiple-executions for more details
    elif ( model == 'knn' ):
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier()
    else:
        raise ValueError(f"Model {model} is not supported.")
    # update here for new models

    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)

    cm_train = confusion_matrix(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred) * 100.  # percent correct

    print("Train Confusion Matrix:\n", cm_train)
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print( "Training done.\n" )

    # Save
    print( "Saving..." )
    # train confusion matrix
    df_tmp = pd.DataFrame(cm_train)
    fname = OUT_DIR + 'train_confusion_matrix.csv'
    if not os.path.exists(fname):
        print(f"Saving {fname}")
    else:
        print(f"Overwriting {fname}")
    df_tmp.to_csv( fname, index=False, header=False, mode='w' )
    
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