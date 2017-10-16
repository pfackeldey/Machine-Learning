# -*- coding: utf-8 -*-
#!/usr/bin/env python


import ROOT

ROOT.PyConfig.IgnoreCommandLineOptions = True

from model import KerasModels
from treetools import TreeExtender

import argparse
import os
import sys
import yaml

EPOCHS = 5
BATCH_SIZE = 16


def classificationNeuralNetwork(args_from_script=None):

    parser = argparse.ArgumentParser(description="Perform binary classification NN training with kPyKeras (TMVA).",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")

    parser.add_argument("config", help="Path to training config")
    args = parser.parse_args()

    ROOT.TMVA.Tools.Instance()
    ROOT.TMVA.PyMethodBase.PyInitialize()

    # initialize factory:

    factory = ROOT.TMVA.Factory("TMVAclassificatoin", ROOT.TFile.Open("training_output.root", "RECREATE"),
                                "!V:!Silent:Color:!DrawProgressBar:Transformations=None:AnalysisType=Classification")

    # load yaml training config
    config = yaml.load(open(args.config, "r"))

    dataloader = ROOT.TMVA.DataLoader("example_training")

    for signal_, signal_weight in zip(config["signal_inputs"], config["signal_weights"]):
        signal = ROOT.TFile.Open(signal_)
        signal_weight = signal_weight * config["global_weight"]
        dataloader.AddSignalTree(
            signal.Get("em_nominal/ntuple"), signal_weight)

    for background_, background_weight in zip(config["background_inputs"], config["background_weights"]):
        background = ROOT.TFile.Open(background_)
        background_weight = background_weight * config["global_weight"]
        dataloader.AddBackgroundTree(
            background.Get("em_nominal/ntuple"), background_weight)

    for feature in config["features"]:
        dataloader.AddVariable(feature)

    dataloader.PrepareTrainingAndTestTree(
        ROOT.TCut(""), config["train_test_split"])

    model = KerasModels(n_features=len(config["features"]), n_classes=len(
        config["classes"]), learning_rate=0.0001)
    model.MSSM_HWW_model()

    factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras, "PyKeras_example",
                       "!H:!V:VarTransform=None:FileNameModel=example_model.h5:SaveBestOnly=true:TriesEarlyStopping=-1:NumEpochs={}:".format(EPOCHS) + "BatchSize={}".format(BATCH_SIZE))

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()


if __name__ == "__main__" and len(sys.argv) > 1:
    classificationNeuralNetwork()