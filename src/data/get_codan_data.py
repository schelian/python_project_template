# download CODaN data
import torch
import sys
import os

if not os.path.isdir("./CODaN"):
    print( "Cloning CODaN repository..." )
    os.system( 'git clone https://github.com/Attila94/CODaN/' )
sys.path.append("./CODaN")
from codan import CODaN

# put data in the data dir.
IN_DATA_DIR = "./CODaN/"
OUT_DATA_DIR = "../../data/CODaN"
print( f"unpacking files to {OUT_DATA_DIR}")
os.system( f'mkdir -p {OUT_DATA_DIR}' )
os.system( f'mkdir -p {OUT_DATA_DIR}/day' )
os.system( f'mkdir -p {OUT_DATA_DIR}/night' )

# only test_day and test_night have data by time of day so don't get train or val
#dataset = CODaN(IN_DATA_DIR) 
#dataset = CODaN(IN_DATA_DIR, split="val")
dataset = CODaN(IN_DATA_DIR, split="test_day")
os.system( f'mv ./CODaN/data/test_day/* {OUT_DATA_DIR}/day' )

dataset = CODaN(IN_DATA_DIR, split="test_night")
os.system( f'mv ./CODaN/data/test_night/* {OUT_DATA_DIR}/night' )

os.system( f'mkdir -p {OUT_DATA_DIR}/train' )
os.system( f'mkdir -p {OUT_DATA_DIR}/val' )
os.system( f'mkdir -p {OUT_DATA_DIR}/test' )

