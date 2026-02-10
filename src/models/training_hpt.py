from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import numpy as np
import pickle
from jsonargparse import CLI
from sklearn.model_selection import RandomizedSearchCV
import joblib
import json

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
    print( "  If you get exceptions like below, they have to do with the parameter search not releasing all resoures.")
    print( "  Should be OK if best_params.json is saved.")
    print( "  " )
    print( "  Sample exception: ")
    print( "  Exception ignored in: <function ResourceTracker.__del__ at 0x7a8feabfce00>" )
    print( "  ...")
    print( "  ChildProcessError: [Errno 10] No child processes")
    if ( model == 'decision_tree' ):
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(random_state=rng) # see https://scikit-learn.org/stable/common_pitfalls.html#getting-reproducible-results-across-multiple-executions for more details
    elif ( model == 'knn' ):
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier()
    else:
        raise ValueError(f"Model {model} is not supported.")
    # update here for new models

    #clf.fit(X_train, y_train)
    #y_train_pred = clf.predict(X_train)
    
    # hyperparameter tuning (from https://machinelearningmastery.com/beyond-gridsearchcv-advanced-hyperparameter-tuning-strategies-for-scikit-learn-models/ and )
    if ( model == 'decision_tree' ):
        param_dist = {
            'max_depth': [None, 5, 10, 20],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 5, 10],
        }
        search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_dist,
            n_iter=20,
            scoring='accuracy',
            random_state=42,
            n_jobs=-2
        )        
    search.fit(X_train, y_train)       
    # wait for completion
    joblib.externals.loky.get_reusable_executor().shutdown(wait=True) 
    print(f"Best hyperparameters: {search.best_params_}")
    best_model = search.best_estimator_
    clf = best_model
    y_train_pred = best_model.predict(X_train)
    
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

    fname = OUT_DIR + f'best_params.json'        
    if not os.path.exists(fname):
        print(f"Saving {fname}")
    else:
        print(f"Overwriting {fname}")              
    with open( fname, 'w' ) as f:
        json.dump( search.best_params_, f, indent=4 )
    #
    # Save the entire RandomizedSearchCV object
    # joblib.dump(search, 'random_search_results.pkl')
    #
    # Accessing params later is easy:
    # search = joblib.load('random_search_results.pkl')
    # print(search.best_params_)
     
    print( "Saving done.\n" )

if __name__ == '__main__':
    print("Running training script...")

    CLI(main)