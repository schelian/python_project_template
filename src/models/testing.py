from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import numpy as np
import pickle

IN_DIR = '../../data/processed/'
OUT_DIR = '../../models/decision_tree/'

X_test = pd.read_csv(IN_DIR + 'X_test.csv', header=None).to_numpy()
y_test = pd.read_csv(IN_DIR + 'y_test.csv', header=None).to_numpy().ravel()

os.makedirs(OUT_DIR, exist_ok=True)

# Test the model
with open(f'{OUT_DIR}/decision_tree_model.pkl', 'rb') as f:
    clf = pickle.load(f)
y_test_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred) * 100  # percent correct

print("Confusion Matrix:\n", cm)
print(f"Accuracy: {accuracy:.2f}%")

np.savetxt( f'{OUT_DIR}/y_test_pred.csv', y_test_pred, delimiter=',', fmt='%s') # could also do pandas