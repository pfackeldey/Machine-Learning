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
    parser.add_argument("--fold", default="0", choices=["0", "1"],
                        help="Training fold. [Default: %(default)s]")
    parser.add_argument("--tree", default="latino",
                        help="Path to tree for TChain. [Default: %(default)s]")
    parser.add_argument("--epochs", default=150,
                        help="Number of training epochs. [Default: %(default)s]")
    parser.add_argument("--learning-rate", default=0.00005,
                        help="Learning rate of NN. [Default: %(default)s]")
    parser.add_argument("--batch-size", default=7500,
                        help="Batch size for training. [Default: %(default)s]")
    parser.add_argument("config", help="Path to training config")
    args = parser.parse_args()

    ROOT.TMVA.Tools.Instance()
    ROOT.TMVA.PyMethodBase.PyInitialize()

    # load yaml training config
    config = yaml.load(open(args.config, "r"))

    if args.fold == "0":
        foldcut = "event%2==0"
    elif args.fold == "1":
        foldcut = "event%2==1"
    else:
        sys.exit("Currently only fold 0 and 1 are supported")

    # initialize factory:

    factory = ROOT.TMVA.Factory("TMVAMulticlassification", ROOT.TFile.Open("MSSM_training_{}.root".format(args.fold), "RECREATE"),
                                "!V:!Silent:Color:!DrawProgressBar:Transformations=None:AnalysisType=multiclass")

    dataloader = ROOT.TMVA.DataLoader(
        "MSSM_training", "fold{}_training".format(args.fold))

    # add training variables
    for feature in config["features"]:
        dataloader.AddVariable(feature)

    # add classes
    prepare_classes = ""
    for class_, trees in config["classes"].iteritems():
        class_chain = ROOT.TChain(args.tree)
        for tree in trees:
            class_chain.Add(tree)
        dataloader.AddTree(class_chain, class_,
                           config["class_weights"][class_] * config["global_weight"], foldcut)
        #dataloader.SetWeightExpression(config["event_weights"], class_)
        prepare_classes += "TrainTestSplit_{}={}:".format(
            class_, config["train_test_split"])
    dataloader.PrepareTrainingAndTestTree(
        ROOT.TCut(""), prepare_classes + "SplitMode=Random:NormMode=None")

    model = KerasModels(n_features=len(config["features"]), n_classes=len(
        config["classes"]), learning_rate=args.learning_rate, plot_model=False, modelname="multiclass_model_fold{}.h5".format(args.fold))
    model.multiclass_MSSM_HWW_model()

    factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras, "PyKeras_MSSM_HWW",
                       "!H:!V:VarTransform=None:FileNameModel=multiclass_model_fold{}.h5".format(args.fold) + ":SaveBestOnly=true:TriesEarlyStopping=5:NumEpochs={}:".format(args.epochs) + "BatchSize={}".format(args.batch_size))

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
