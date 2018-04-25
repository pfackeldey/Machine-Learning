#!/bin/bash -e

# detect environment
if [[ "$( lsb_release -a )" == *6.9* ]]; then
    # Scientific Linux 6.9
    source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-slc6-gcc62-opt/setup.sh
fi
if [[ "$( lsb_release -a )" == *7.4* ]]; then
    # Scientific Linux 7.4
    source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-centos7-gcc62-opt/setup.sh
fi

thispath=`pwd`
cd ~/master/Machine-Learning/
cd $thispath

python evaluation/keras_evaluation.py tasks/analysis/MSSM_HWW.yaml --files "$@" --tree latino
