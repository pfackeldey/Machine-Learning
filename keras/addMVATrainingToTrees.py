# -*- coding: utf-8 -*-

from treetools import TreeExtender
import ROOT
import argparse
import sys


def addMVATrainingToTrees():
    parser = argparse.ArgumentParser(description="Perform binary classification NN evaluation with kPyKeras (TMVA).",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")

    parser.add_argument("config", help="Path to evaluation config")
    args = parser.parse_args()

    # create TMVA instance
    ROOT.TMVA.Tools.Instance()
    ROOT.TMVA.PyMethodBase.PyInitialize()

    # initialize TMVA reader
    reader = ROOT.TMVA.Reader('Color:!Silent')

    # load config with training features and weight files...
    config = yaml.load(open(args.config, "r"))

    # add features to TMVA reader
    for feature in config["features"]:
        reader.AddVariable(feature)

    # book the classification method (trainings weight file)
    reader.BookMethod(["trainings_weight_file"], ["trainings_weight_file"])

    # add new branch with TreeExtender
    src = config["source_file"]
    dst = config["target_file"]
    with TreeExtender(src, dst) as te:
        te.addBranch("PyKeras_MSSM_HWW", unpackBranches=[
                     feature for feature in config["features"]])
        for entry in te:
            entry.PyKeras_MSSM_HWW[0] = reader.EvaluateMVA(
                ["trainings_weight_file"])


if __name__ == "__main__" and len(sys.argv) > 1:
    addMVATrainingToTrees()
