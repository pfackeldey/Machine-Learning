#!/usr/bin/env python

import argparse
import os
import glob
import subprocess
import random
from time import gmtime, strftime


class DataSets2016(object):
    """
    Create directory structure for very specific architecture
    of HWW latino 2016 analysis. Needs to be adjusted for the
    2017 analysis.
    """
    # Eleganz neu definiert...

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.datarun = ["B", "C", "D", "E", "F", "G", "H"]
        self.selection = "wwSel"
        self.shifts = ["JESdo__", "JESup__", "LepElepTdo__", "LepElepTup__", "LepMupTdo__", "LepMupTup__",
                       "METdo__", "METup__", "PS__", "PUdo__", "PUup__", "UEdo__", "UEup__", ""]
        self.base_path_mc = "Apr2017_summer16/lepSel__MCWeights__bSFLpTEffMulti__cleanTauMC__l2loose__hadd__l2tightOR__LepTrgFix__dorochester__formulasMC__"
        self.base_path_data = [
            "Apr2017_Run2016", "_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__hadd__l2tightOR__formulasDATA__"]
        self.base_path_wjets = [
            "Apr2017_Run2016", "_RemAOD/lepSel__EpTCorr__TrigMakerData__cleanTauData__l2loose__multiFakeW__formulasFAKE__hadd__"]

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
parser.add_argument(
    '--dataset', default=["all"], choices=["all", "data", "wjets", "mc"], help="Choose the filelists to be evaluated.")
args = parser.parse_args()

# ugly #FIXME
base_dir = "/net/data_cms/institut_3b/fackeldey/nobackup/"

datasets = DataSets2016(base_dir)
print args.dataset

# create all paths (data, MC and all shifts)
# UGLY FIXME
if args.dataset == "data":
    dirs = [base_dir + path for path in datasets.create_data_directories()]

elif args.dataset == "wjets":
    dirs = [base_dir + path for path in datasets.create_wjets_directories()]

elif args.dataset == "mc":
    dirs = [base_dir + path for path in datasets.create_mc_directories()]
else:
    dirs = datasets.mkFullDirs()

print dirs
# create list of ALL files
files = []
for dir in dirs:
    for _file in glob.glob(dir + "/*.root"):
        files.append(_file)

# shuffle list
random.shuffle(files)

# function, which resizes a list `lst` into a several sublists with size `sz`


def resize_filelists(lst, sz): return [lst[i:i + sz]
                                       for i in range(0, len(lst), sz)]


# resize filelist to sublist
filelists = resize_filelists(files, 50)

# create log dir for condor:
folder_logs = base_dir + \
    'logs_{}'.format(strftime("%Y_%m_%d", gmtime()))
if not os.path.exists(folder_logs):
    print "Creating directory for condor logs: ", folder_logs
    os.makedirs(folder_logs)

# create condor config and submit
for filelist, i in zip(filelists, range(len(filelists))):
    rfiles = " ".join(_file for _file in filelist)
    jobName = datasets.__class__.__name__ + "_part{0:02d}".format(i)
    with open(folder_logs + "/submitCondor_part{0:02d}.txt".format(i), "w") as f:
        f.write("""
                Universe   = vanilla
                Executable = evaluation/run_evaluation.sh
                Arguments  = {0}
                Log        = {1}/{2}.log
                Output     = {1}/{2}.out
                Error      = {1}/{2}.error
                Queue
                """.format(rfiles, folder_logs, jobName))
    subprocess.call(["condor_submit", folder_logs +
                     "/submitCondor_part{0:02d}.txt".format(i)])
