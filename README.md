# What is this
This serves as a template/demonstration on how to make code easy for other to use--build, run and examine outputs.  

This demonstration is focused on machine learning and the directory structure is loosely based on this [repo](https://github.com/StigTerrebonne/AIS-LSTM).  However, feel free to adapt this to other data intensive applications (e.g., simulation, optimization, big data, etc.).  

The repo has some tutorial material on conda, pandas, sklearn, etc.  Feel free to ignore those for other applications.

## More details (read only once or referring back)
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
````
conda env export > environment.yml
````

# Running

## Setup, data preprocessing
Split data into training and testing splits.  This reads from data/raw and writes to data/processed.
````
cd src/data
python preprocess_data.py
````

## Training
Run the training portion, training.py.
To check the code is working OK: use ConfigFile_checkWorking and DataSet_checkWorking.  This should take 15 minutes.  
This should take about 1 hour on a machine with a 8 GB GPU.  

To change datasets, change DataSet1 to DataSet2, etc.  
To change parameters, change ConfigFile.csv.  
* Parameters are in Config/DataSet1/ConfigFile.csv
* Input data is in Data/DataSet1/Input/TrainingDataFile.csv
* Outputs are saved to Save/DataSet1/Weights.csv (you could also save predictions, training curves, etc.)
````
cd src/models

python training.py \
config/config1.csv \
data/input/trainingData.csv \
models/Weights.csv
````

## Testing
Run the testing portion, testing.py.py.
To check the code is working OK: use ConfigFile_checkWorking and DataSet_checkWorking.  This should take 15 minutes.    
Total running time: This should take about 0.5 hours on a machine with a 8 GB GPU.  

To change datasets, change DataSet1 to DataSet2, etc.  
To change parameters, change ConfigFile.csv.  
* Parameters are in Config/DataSet1/ConfigFile.csv
* Input data is in Data/DataSet1/TestingDataFile.csv
* Weights are in Save/DataSet1/Weights.csv
* Outputs are saved to Save/DataSet1/TestingPredictions.csv
````
cd src/models

python testing.py \
Config/DataSet1/ConfigFile.csv \
Data/DataSet1/TestingDataFile.csv \
Save/DataSet1/Weights.csv \
Data/Dataset1/Final_Results/TestingPredictions.csv
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

# Tutorial/reference for conda

## Creating an envt
````conda create --name myenv [ python=3.11.9 ]````

## Installing packpages: conda install <pkg>
````
conda install -c conda-forge matplotlib
conda install -c conda-forge keras
conda install -c conda-forge opencv-python 
# if it fails, do pip install opencv-python
conda install -c conda-forge scikit-learn
conda install -c conda-forge ultralytics # YOLO
conda install pytorch::torchvision
````

## Examine what's there: conda list

## Remove a package: conda remove <pkg>