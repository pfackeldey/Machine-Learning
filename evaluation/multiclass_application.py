# -*- coding: utf-8 -*-

import os
import sys
base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)

from utils.treetools import TreeExtender
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

    # initialize TMVA reader
    reader = ROOT.TMVA.Reader('Color:!Silent')

    # load config with training features and weight files...
    config = yaml.load(open(args.config, "r"))

    # add features to TMVA reader
    values = {}
    for feature in config["features"]:
        values[feature] = array("f", [-999])
        reader.AddVariable(feature, values[feature])

    # book the classification method (trainings weight file)
    reader.BookMVA(config["trainings_weight_file"],
                   config["trainings_weight_file"])

    # add new branch with TreeExtender
    dst = config["target_file"]
    for file_ in config["source_files"]:
        with TreeExtender(file_, dst) as te:
            te.addBranch("PyKeras_MSSM_HWW", unpackBranches=["event"])
            for entry in te:
                response = reader.EvaluateMulticlass(
                    config["trainings_weight_file"][event % 10 >= 4])

            # Find max score and index
            response_max_score[0] = -999.0
            for i, r in enumerate(response):
                response_single_scores[i][0] = r
                if r > response_max_score[0]:
                    response_max_score[0] = r
                    response_max_index[0] = i

            entry.PyKeras_MSSM_HWW[0] = response_max_score[0]


if __name__ == "__main__" and len(sys.argv) > 1:
    try:
        import tensorflow as tf
        tf.python.control_flow_ops = tf
        multiclass_application()
    except AttributeError:
        multiclass_application()
