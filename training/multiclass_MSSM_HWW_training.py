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

EPOCHS = 1
BATCH_SIZE = 64


def multiclassNeuralNetwork(args_from_script=None):

    parser = argparse.ArgumentParser(description="Perform multiclassification NN training with kPyKeras (TMVA).",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")

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
    for class_ in config["classes"]:
        class_tree = config["class_file"].Get(class_)
        dataloader.AddTree(class_tree, class_,
                           config["class_weights"][class_] * config["global_weight"])
        dataloader.SetWeightExpression(config["event_weights"], class_)
        prepare_classes += "TrainTestSplit_{}={}:".format(
            class_, config["train_test_split"])
    dataloader.PrepareTrainingAndTestTree(
        ROOT.TCut(""), prepare_classes + "SplitMode=Random:NormMode=None")

    model = KerasModels(n_features=len(config["features"]), n_classes=len(
        config["classes"]), learning_rate=0.00001, plot_model=False)
    model.multiclass_MSSM_HWW_model()

    factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras, "PyKeras_MSSM_HWW",
                       "!H:!V:VarTransform=None:FileNameModel=MSSM_HWW_model.h5:SaveBestOnly=true:TriesEarlyStopping=-1:NumEpochs={}:".format(EPOCHS) + "BatchSize={}".format(BATCH_SIZE))

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()


if __name__ == "__main__" and len(sys.argv) > 1:
    multiclassNeuralNetwork()
