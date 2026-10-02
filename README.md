# Get data ready

In the data directory, run
1. get_codan_data.py - download CODaN data and move it into ../../data/CODaN/day and ../../data/CODaN/night
1. subset_codan_data.ipynb - select subset of images for train, val and test, and write with "day" and "night" prefixes in filenames
1. make_gt_files.ipynb - create images and labels CSV file for train, val and test
1. feature_extraction.ipynb - Converts CODaN CSV image indexes into HSV value features.

This is a sanity check.  Can HSV do as well as the ResNet?

# Train/test

##  YOLO
1. training_yolo.ipynb.  1 is day, 2 is night
1. testing_yolo.ipynb