#!/usr/bin/env python

import ROOT
# disable ROOT internal argument parser
ROOT.PyConfig.IgnoreCommandLineOptions = True

import argparse
import yaml
import os
import subprocess
from array import array
import sys

import logging
logger = logging.getLogger("create_training_dataset")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def createTrainingsset(args_from_script=None):

    parser = argparse.ArgumentParser(description="Create a trainingsset.",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")
    parser.add_argument("config", help="Path to createTrainingsset config")
    args = parser.parse_args()

    # load yaml training config
    config = yaml.load(open(args.config, "r"))

    for num_fold in range(2):
        logger.info("Merge input files for fold {}.".format(num_fold))
        created_files = []
        for process in config["processes"]:
            logger.debug("Collect events of process {} for fold {}.".format(
                process, num_fold))

            # Create output file
            created_files.append(
                os.path.join(config["output_path_creation"],
                             "merge_fold{}_{}.root".format(num_fold, process)))
            file_ = ROOT.TFile(created_files[-1], "RECREATE")

            # Collect all files for this process in a chain
            chain = ROOT.TChain(config["tree_path"])
            for filename in config["processes"][process]["files"]:
                path = os.path.join(config["base_path"], filename)
                if not os.path.exists(path):
                    logger.fatal("File does not exist: {}".format(path))
                chain.AddFile(path)

            chain_numentries = chain.GetEntries()
            if not chain_numentries > 0:
                logger.fatal(
                    "Chain (before skimming) does not contain any events.")
                raise Exception
            logger.debug("Found {} events for process {}.".format(
                chain_numentries, process))

            # Skim the events with the cut string
            cut_string = "({EVENT_BRANCH}%2=={NUM_FOLD})&&({CUT_STRING})".format(
                EVENT_BRANCH=config["event_branch"],
                NUM_FOLD=num_fold,
                CUT_STRING=config["processes"][process]["cut_string"])
            logger.debug("Skim events with cut string: {}".format(cut_string))

            chain_skimmed = chain.CopyTree(cut_string)
            chain_skimmed_numentries = chain_skimmed.GetEntries()
            if not chain_skimmed_numentries > 0:
                logger.fatal(
                    "Chain (after skimming) does not contain any events.")
                raise Exception
            logger.debug("Found {} events for process {} after skimming.".
                         format(chain_skimmed_numentries, process))

            # Write training weight to new branch
            logger.debug("Add training weights with weight string: {}".format(
                config["processes"][process]["weight_string"]))
            formula = ROOT.TTreeFormula(
                "training_weight",
                config["processes"][process]["weight_string"], chain_skimmed)
            training_weight = array('f', [-999.0])
            branch_training_weight = chain_skimmed.Branch(
                config["training_weight_branch"], training_weight,
                config["training_weight_branch"] + "/F")
            for i_event in range(chain_skimmed.GetEntries()):
                chain_skimmed.GetEntry(i_event)
                training_weight[0] = formula.EvalInstance()
                branch_training_weight.Fill()

            # Rename chain to process name and write to output file
            logger.debug("Write output file for this process and fold.")
            chain_skimmed.SetName(config["processes"][process]["class"])
            chain_skimmed.Write()
            file_.Close()

        # Combine all skimmed files using `hadd`
        logger.debug("Call `hadd` to combine files of processes for fold {}.".
                     format(num_fold))
        output_file = os.path.join(config["output_path_creation"], "fold{}_{}".format(
            num_fold, config["output_filename"]))
        subprocess.call(["hadd", "-f", output_file] + created_files)
        logger.info("Created output file: {}".format(output_file))


if __name__ == "__main__" and len(sys.argv) > 1:
    createTrainingsset()
