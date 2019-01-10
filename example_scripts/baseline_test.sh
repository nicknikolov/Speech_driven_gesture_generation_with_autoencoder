#!/usr/bin/env bash

# This script is used in "baseline_train_n_test.sh" to evaluate the baseline model
# You call use it by itself if the model is already trained
# Several aspects needs to be customized

# First - check if we have enough parameters
if [[ $# -lt 3 ]]; then
    echo 'Usage: ./baseline_test.sh FOLDER GPU_NUMBER MODEL'
    exit 0
fi

folder=$1
gpu=$2
model=$3
data_dir=/home/taras/Documents/storage/MotionJapanese/$folder
speech_features=MFCC

# Create a folder to store produced gesture sequences
mkdir -p gestures

# Remove previous results
#(I had two folders evaluation0 and evaluation1, so that I can use two GPUs simultaneously)
cd ..
rm evaluation/data/predicted/$speech_features/*

# Make predictions for all the test sequences
# (replace 1094 by 1093 for the dev sequences)
for seq in `seq 1094 2 1182`; 
        do
		echo
                echo 'Predicting sequence' $seq
                CUDA_VISIBLE_DEVICES=$gpu python predict.py $model.hdf5 $data_dir/test_inputs/X_test_audio${seq}.npy normal_prediction$seq.txt
                mv normal_prediction$seq.txt example_scripts/gestures/gesture${seq}.txt	
        done

echo 'Removing the velocities ...'
python helpers/remove_velocity.py -g example_scripts/gestures
cd example_scripts/gestures

# remove gestures with velocites
rm *.txt

# Move gestrues without velocities to the corresponding folder
mkdir -p ../../evaluation/data/predicted/$speech_features/
mv no_vel/*.txt ../../evaluation/data/predicted/$speech_features/
cd ../../evaluation

# In order for an evaluation to be correct ONLY ground truth motion 3d coords in txt format for the
# same sequences as used in the script above (1094, 1096,...) has to be in evaluation/data/original
# if evaluation/data/origibal contains all the sequences (1093,1094...) the results will be wrong
# see "evaluation" folder for the info on how to transform the true gestures from bvh to txt format

echo 'Evaluating ...'
echo "Evaluating "${model}" ..." >> ../results.txt
python calc_errors.py -g $speech_features -m ape  >> ../results.txt
python calc_errors.py -g $speech_features -m mae  >> ../results.txt
python calc_jerk.py -g $speech_features >> ../results.txt
