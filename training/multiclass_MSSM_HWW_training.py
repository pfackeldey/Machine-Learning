# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import sys
import ROOT

ROOT.PyConfig.IgnoreCommandLineOptions = True

base = os.path.normpath(os.path.join(os.path.abspath(__file__), "../.."))
sys.path.append(base)

from utils.model import KerasModels
from utils.treetools import TreeExtender

import argparse
import os
import sys
import yaml


def multiclassNeuralNetwork(args_from_script=None):

    parser = argparse.ArgumentParser(description="Perform multiclassification NN training with kPyKeras (TMVA).",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")
    parser.add_argument("--epochs", default=50,
                        help="Number of training epochs. [Default: %(default)s]")
    parser.add_argument("--learning-rate", default=0.0001,
                        help="Learning rate of NN. [Default: %(default)s]")
    parser.add_argument("--batch-size", default=8000,
                        help="Batch size for training. [Default: %(default)s]")
    parser.add_argument("config", help="Path to training config")
    args = parser.parse_args()

    ROOT.TMVA.Tools.Instance()
    ROOT.TMVA.PyMethodBase.PyInitialize()

    # initialize factory:

    factory = ROOT.TMVA.Factory("TMVAMulticlassification", ROOT.TFile.Open("MSSM_HWW_multiclass_training.root", "RECREATE"),
                                "!V:!Silent:Color:!DrawProgressBar:Transformations=None:AnalysisType=multiclass")

    # load yaml training config
    config = yaml.load(open(args.config, "r"))

    dataloader = ROOT.TMVA.DataLoader("MSSM_training")

    # add training variables
    for feature in config["features"]:
        dataloader.AddVariable(feature)

    # add classes
    prepare_classes = ""
    for class_, trees in config["classes"].iteritems():
        class_chain = ROOT.TChain("em_nominal/ntuple")
        for tree in trees:
            class_chain.Add(tree)
        dataloader.AddTree(class_chain, class_,
                           config["class_weights"][class_] * config["global_weight"])
        #dataloader.SetWeightExpression(config["event_weights"], class_)
        prepare_classes += "TrainTestSplit_{}={}:".format(
            class_, config["train_test_split"])
    dataloader.PrepareTrainingAndTestTree(
        ROOT.TCut(""), prepare_classes + "SplitMode=Random:NormMode=None")

    model = KerasModels(n_features=len(config["features"]), n_classes=len(
        config["classes"]), learning_rate=args.learning_rate, plot_model=False)
    model.multiclass_MSSM_HWW_model()

    factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras, "PyKeras_MSSM_HWW",
                       "!H:!V:VarTransform=None:FileNameModel=multiclass_MSSM_HWW_model.h5:SaveBestOnly=true:TriesEarlyStopping=-1:NumEpochs={}:".format(args.epochs) + "BatchSize={}".format(args.batch_size))

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()


if __name__ == "__main__" and len(sys.argv) > 1:
    try:
        import tensorflow as tf
        tf.python.control_flow_ops = tf
        multiclassNeuralNetwork()
    except AttributeError:
        multiclassNeuralNetwork()
