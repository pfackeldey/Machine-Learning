#!/bin/bash -e

# for sl6:
source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-slc6-gcc62-opt/setup.sh

# for sl7/centos7:
# source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-centos7-gcc62-opt/setup.sh

thispath=`pwd`
cd ~/master/Machine-Learning/
cd $thispath

python evaluation/keras_evaluation.py tasks/analysis/MSSM_HWW.yaml --files "$@" --tree latino
