# What is this
This repository serves as a template/demonstration on how to make code easy for others to use--build, run and examine outputs.  

This demonstration is focused on machine learning and the directory structure is loosely based on this [repo](https://github.com/StigTerrebonne/AIS-LSTM).  However, feel free to adapt this to other data intensive applications (e.g., simulation, optimization, big data, etc.).  

The repo also has some tutorial material on conda, pandas, sklearn, etc.  Feel free to ignore those for other applications.  See sections with "Tutorial/reference" below for more details.

## More details (read only once or when referring back)
Documentation makes sure your code can be build, run and understood by others.  

Keep this question in mind when you write a README.md: "If I was on vacation or sick, can my colleague/co-worker build and run my code?"  

More specifically, at a minimum cover the following points in a README.md.
1. How to build the code including any installation of libraries, conda environments, etc.
1. How to run the code with different datasets or parameters
1. How to examine the results
1. EXTRA but good to have: expected output and time to run each step
1. EXTRA EXTRA: how to extend the code for new datasets, parameters, models, etc. What calls, what, etc. The format of datasets and parameters can also be described.

# One time setup
````
conda env create -f environment.yml -n python_project_template
````

# Everytime setup and cleanup
````
conda activate python_project_template
````

Do this when done:
````
conda deactivate
````

Updating the conda environment file:
Do this when you install a new package.
````
conda env export > environment.yml
````

# Running

## Setup, data preprocessing
Split data into training and testing splits.  

This reads from data/\<dataset\>/raw and writes to data/\<dataset\>/processed (X_train, y_train, etc.).
````
cd src/data
python preprocess_data.py
````

## Training
Run the training, training.py.

This reads from data/\<dataset\>/processed (X_train, y_train, etc.)and writes to models/\<dataset\>/ (y_train_pred.csv, etc.)  

This should take less than 2 minutes on laptop purchased within the last few years of 2025.

To change the dataset, use the --dataset option.  
To use a config file, use the --config option.
````
cd src/models

python training.py
# python training.py --dataset="pima"
# python training.py --config ../../config/iris_decision_tree.yaml
# how to generate a sample config file: python training.py --print_config > sample_config.yaml 

# can also do python training.py --help
````

## Testing
Run the testing, testing.py.

This reads from data/\<dataset\>/processed (X_test, y_test, etc.) and writes to models/\<dataset\>/ (y_test_pred.csv, etc.).  

This should take less than 2 minutes on laptop purchased within the last few years of 2025.

To change the dataset, use the --dataset option.  
To use a config file, use the --config option.
````
cd src/models

python testing.py
# python testing.py --dataset="pima"
# python testing.py --config ../../config/iris_decision_tree.yaml
# how to generate a sample config file: python testing.py --print_config > sample_config.yaml

# can also do python testing.py --help
````

## Examine the results
Look at models/decision_tree/y_test_pred.csv.  
It should look like this:
````
2.0
1.0
3.0
````
Be sure to include results in aggregate, not just a few datapoints.
E.g., On average, it is 75% correct.  (Other performance metrics are …)

# How to go further

## New dataset
1. Put in data under its own directory, e.g., ./data/newdata  
1. Update src/preprocess_data.py. See portions that says "update here"

## New algorithm
1. Update src/model/training.py.  See portions that says "update here"
1. Update src/model/testing.py. See portions that says "update here"

# Tutorial/reference for conda

## Creating an envt
````conda create --name myenv [ python=3.11.9 ]````

## Installing packpages: conda install <pkg>
Common ones
````
conda install -c conda-forge matplotlib

conda install -c conda-forge keras

conda install -c conda-forge opencv-python 
# if it fails, do pip install opencv-python

conda install -c conda-forge scikit-learn

conda install -c conda-forge ultralytics # YOLO
conda install pytorch::torchvision
````

If conda does not upgrade, use pip instead.  E.g.
````
conda remove jsonargparse
pip install jsonargparse
````

## Examine what's there: conda list

## Remove a package: conda remove <pkg>

# Tutorial/reference for jsonargparse

See [this link and their Github repo](https://speakerdeck.com/stecklin/jsonargparse-say-goodbye-to-configuration-hassles).  They use object-oriented programming (OOP).  It detracts the reader a little from simple uses of jsonargparse but can be helpful for greater modularity.