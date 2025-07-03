from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import numpy as np
import pickle

IN_DIR = '../../data/processed/'
OUT_DIR = '../../models/decision_tree/'

X_train = pd.read_csv(IN_DIR + 'X_train.csv', header=None).to_numpy()
y_train = pd.read_csv(IN_DIR + 'y_train.csv', header=None).to_numpy().ravel()

os.makedirs(OUT_DIR, exist_ok=True)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

cm = confusion_matrix(y_train, y_train_pred)
accuracy = accuracy_score(y_train, y_train_pred) * 100  # percent correct

print("Confusion Matrix:\n", cm)
print(f"Accuracy: {accuracy:.2f}%")

np.savetxt( f'{OUT_DIR}/y_train_pred.csv', y_train_pred, delimiter=',', fmt='%s') # could also do pandas

# Save the model
with open(f'{OUT_DIR}/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(clf, f)