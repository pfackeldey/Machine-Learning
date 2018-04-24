#!/usr/bin/env python

import argparse
import os
import glob
import subprocess
import random

class DataSets2016(object):
    """
    Create directory structure for very specific architecture
    of HWW latino 2016 analysis. Needs to be adjusted for the
    2017 analysis.
    """
    def __init__(self, base_dir):

        self.base_dir = base_dir
        self.datarun = ["B", "C", "D", "E", "F", "G", "H"]
        self.selection = "wwSel"
        self.shifts = ["JESdo__", "JESup__", "LepElepTdo__", "LepMupTdo__", "METdo__", "METup__", "PS__", "PUdo__", "PUup__", "UEdo__", "UEup__", ""]
        self.base_path_mc = "Apr2017_summer16/lepSel__MCWeights__bSFLpTEffMulti__cleanTauMC__l2loose__hadd__l2tightOR__LepTrgFix__dorochester__formulasMC__"
        self.base_path_data = ["Apr2017_Run2016", "_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__hadd__l2tightOR__formulasDATA"]
        self.base_path_wjets = ["Apr2017_Run2016", "_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__multiFakeW__formulasFAKE__hadd"]

    def create_data_directories(self):
        return [self.base_path_data[0] + run[0] + self.base_path_data[1] + self.selection for run in self.datarun]

    def create_wjets_directories(self):
        return [self.base_path_wjets[0] + run[0] + self.base_path_wjets[1] + self.selection for run in self.datarun]

    def create_mc_directories(self):
        return [self.base_path_mc + shift + self.selection for shift in self.shifts]

    def directories(self):
        return self.create_data_directories() + self.create_wjets_directories() + self.create_mc_directories()

    def mkFullDirs(self):
        return [self.base_dir + path for path in self.directories()]


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default=["all"], choices=["all", "data", "wjets", "mc"], action="store_true")
args = parser.parse_args()

# ugly #FIXME
base_dir = "/net/data_cms/institut_3b/fackeldey/nobackup/"

datasets = DataSets2016(base_dir)

# create all paths (data, MC and all shifts)
if args.datasets == "all":
    dirs = datasets.mkFullDirs()

elif args.datasets == "data"
    dirs = [base_dir + path for path in datasets.create_data_directories()]

elif args.datasets == "wjets"
    dirs = [base_dir + path for path in datasets.create_wjets_directories()]

elif args.datasets == "mc"
    dirs = [base_dir + path for path in datasets.create_mc_directories()]

# create list of ALL files
files = []
for dir in dirs:
    for file in glob.glob(dir + "/*.root"):
        files.append(file)

# shuffle list
random.shuffle(files)

# lambda function, which resizes a list `lst` into a several sublists with size `sz`
resize_filelists = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

# resize filelist to sublist
filelists = resize_filelists(files, 50)

# create condor config and submit
for filelist in filelists:
    rfiles = " ".join(file for file in filelists[0])
    hash = hash(rfiles)
    with open("submitCondor.txt","w") as f:
        f.write("""
                Universe   = vanilla
                Executable = run_evaluation.sh
                Arguments  = {0}
                Log        = logs/{1}.log
                Output     = logs/{1}.out
                Error      = logs/{1}.error
                Queue
                """.format(rfiles, hash))
    subprocess.call(["condor_submit", "submitCondor.txt"])
