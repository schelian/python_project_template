# What is this
Short version of README.md.  Useful for new projects (it has less tutorial material).

# One time setup
````
conda env create -f environment.yml -n python_project_template
````
# Everytime setup and cleanup
This will activate the conda environment.
````
conda activate python_project_template
````

Do this when done; this will deactivate the conda environment.
````
conda deactivate
````

Updating the conda environment file:
Do this when you install a new package.
````
conda env export --no-builds > environment.yml
````

# Running

## Training

Run the training, training.py.

This reads from data/\<dataset\>/processed (X_train, y_train, etc.) and writes to models/\<dataset\>/\<model\> (y_train_pred.csv, etc.)  

This should take less than 2 minutes on laptop purchased within the last few years of 2025.

To change the dataset, use the --dataset option.  
To use a config file, use the --config option.
````
cd src/models

python training.py
# different dataset
# python training.py --dataset="pima"
#
# different algorithm
# python training.py --model=knn
#
# using config files
# python training.py --config ../../config/iris_decision_tree.yaml
#
# how to generate a sample config file: python training.py --print_config > sample_config.yaml 
#
# you can also do python training.py --help
````

## Testing

Run the testing, testing.py.

This reads from data/\<dataset\>/processed (X_test, y_test, etc.) and writes to models/\<dataset\>/\<model\> (y_test_pred.csv, etc.).  

This should take less than 2 minutes on laptop purchased within the last few years of 2025.

To change the dataset, use the --dataset option.  
To use a config file, use the --config option.
````
cd src/models

python testing.py
# different dataset
# python testing.py --dataset="pima"
#
# using config files
# python testing.py --config ../../config/iris_decision_tree.yaml
#
# how to generate a sample config file: python testing.py --print_config > sample_config.yaml
#
# you can also do python testing.py --help
````

## Examining results
Look at models/\<dataset\>/\<model\>/y_test_pred.csv.  
It should look like this:
````
2.0
1.0
3.0
````
Be sure to include results in aggregate, not just a few datapoints.
E.g., On average, it is 75% correct.  (Other performance metrics are …)