# -*- coding: utf-8 -*-

import os
import sys
base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)

import ROOT
import argparse
import yaml
from array import array
import keras


def multiclass_application():
    parser = argparse.ArgumentParser(description="Perform binary classification NN evaluation with kPyKeras (TMVA).",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")

    parser.add_argument("config", help="Path to evaluation config")
    args = parser.parse_args()

    # create TMVA instance
    ROOT.TMVA.Tools.Instance()
    ROOT.TMVA.PyMethodBase.PyInitialize()

    config = yaml.load(open(args.config, "r"))

    # Initialize TMVA Reader and book variables
    reader = ROOT.TMVA.Reader('Color:!Silent')
    values = {}
    for variable in config["features"]:
        values[variable] = array("f", [-999])
        reader.AddVariable(variable, values[variable])

    # Book methods for classification of different folds
    classifiers = config["trainings_weight_file"]
    for classifier in classifiers:
        reader.BookMVA(ROOT.TString(classifier), ROOT.TString(classifier))

    # Open input file and register branches with input and output variables
    files = config["source_files"]
    for f in files:
        file_ = ROOT.TFile(f, "UPDATE")
        if file_ == None:
            raise Exception("File is not existent: {}".format(f))

        tree = file_.Get('latino')
        if tree == None:
            raise Exception("Tree {} is not existent in file: {}".format(
                'latino', f))

        for feature in config["features"]:
            tree.SetBranchAddress(feature, values[featuree])

        response_branches = []
        response_single_scores = []
        prefix = 'NN_score_'
        for class_ in config["classes"]:
            response_single_scores.append(array("f", [-999]))
            response_branches.append(
                tree.Branch("{}{}".format(prefix, class_), response_single_scores[
                    -1], "{}{}/F".format(prefix, class_)))

        response_max_score = array("f", [-999])
        response_branches.append(
            tree.Branch("{}max_score".format(prefix), response_max_score,
                        "{}max_score/F".format(prefix)))

        response_max_index = array("f", [-999])
        response_branches.append(
            tree.Branch("{}max_index".format(prefix), response_max_index,
                        "{}max_index/F".format(prefix)))

        # Loop over events and add method's response to tree
        for i_event in range(tree.GetEntries()):
            # Get current event
            tree.GetEntry(i_event)

            # Get event number and calculate method's response
            event = int(getattr(tree, config["event_branch"]))
            response = reader.EvaluateMulticlass(classifiers[event % 2])

            # Find max score and index
            response_max_score[0] = -999.0
            for i, r in enumerate(response):
                response_single_scores[i][0] = r
                if r > response_max_score[0]:
                    response_max_score[0] = r
                    response_max_index[0] = i

            # Fill branches
            for branch in response_branches:
                branch.Fill()

        # Write new branches to input file
        file_.Write()
        file_.Close()


if __name__ == "__main__" and len(sys.argv) > 1:
    try:
        import tensorflow as tf
        tf.python.control_flow_ops = tf
        multiclass_application()
    except AttributeError:
        multiclass_application()
