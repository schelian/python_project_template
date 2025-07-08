import pandas as pd
import numpy as np

IN_DIR = '../../data/iris/raw/'
OUT_DIR = '../../data/iris/processed/'
df = pd.read_csv( IN_DIR + 'iris.csv' );
print( df.head(3) )     
df['Species'] = df['Species'].astype('category').cat.codes + 1  # Convert species to numeric codes starting from 1

IN_DIR = '../../data/pima/raw/'
OUT_DIR = '../../data/pima/processed/'
df = pd.read_csv( IN_DIR + 'diabetes.csv' );
print( df.head(3) )     

# update here for new datasets

X = df.to_numpy()[:, :-1]  # Features (all columns except the last)
y = df.to_numpy()[:, -1]   # Target (last column)
print( X.shape, y.shape )

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print( X_train.shape, X_test.shape, y_train.shape, y_test.shape )

df_tmp = pd.DataFrame(X_train)
df_tmp.to_csv( OUT_DIR + 'X_train.csv', index=False, header=False)

df_tmp = pd.DataFrame(X_test)
df_tmp.to_csv( OUT_DIR + 'X_test.csv', index=False, header=False)

df_tmp = pd.DataFrame(y_train)
df_tmp.to_csv( OUT_DIR + 'y_train.csv', index=False, header=False)

df_tmp = pd.DataFrame(y_test)
df_tmp.to_csv( OUT_DIR + 'y_test.csv', index=False, header=False)
