#!/bin/bash

TARTANAIR_PATH=/home/campus.ncl.ac.uk/c4071391/datasets/agricultural/tartanair

python evaluation_scripts/validate_tartanair.py --datapath=$TARTANAIR_PATH --weights=droid.pth --disable_vis $@

