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


def classificationNeuralNetwork(args_from_script=None):

    parser = argparse.ArgumentParser(description="Perform binary classification NN training with kPyKeras (TMVA).",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")

    parser.add_argument("config", help="Path to training config")
    args = parser.parse_args()

    ROOT.TMVA.Tools.Instance()
    ROOT.TMVA.PyMethodBase.PyInitialize()

    # initialize factory:

    factory = ROOT.TMVA.Factory("TMVAclassification", ROOT.TFile.Open("MSSM_HWW_training.root", "RECREATE"),
                                "!V:!Silent:Color:!DrawProgressBar:Transformations=None:AnalysisType=Classification")

    # load yaml training config
    config = yaml.load(open(args.config, "r"))

    dataloader = ROOT.TMVA.DataLoader("MSSM_training")

    signal = ROOT.TChain("em_nominal/ntuple")
    for signal_, signal_weight in zip(config["signal_inputs"], config["signal_weights"]):
        signal.Add(signal_)
        signal_weight = signal_weight * config["global_weight"]
    dataloader.AddSignalTree(signal, signal_weight)

    background = ROOT.TChain("em_nominal/ntuple")
    for background_, background_weight in zip(config["background_inputs"], config["background_weights"] * len(config["background_inputs"])):
        background.Add(background_)
        background_weight = background_weight * config["global_weight"]
    dataloader.AddBackgroundTree(background, background_weight)

    for feature in config["features"]:
        dataloader.AddVariable(feature)

    dataloader.PrepareTrainingAndTestTree(
        ROOT.TCut(config["general_cut"]), config["train_test_split"])

    model = KerasModels(n_features=len(config["features"]), n_classes=len(
        config["classes"]), learning_rate=0.00001, plot_model=False)
    model.binary_MSSM_HWW_model()

    factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras, "PyKeras_MSSM_HWW",
                       "!H:!V:VarTransform=None:FileNameModel=MSSM_HWW_model.h5:SaveBestOnly=true:TriesEarlyStopping=-1:NumEpochs={}:".format(EPOCHS) + "BatchSize={}".format(BATCH_SIZE))

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()


if __name__ == "__main__" and len(sys.argv) > 1:
    classificationNeuralNetwork()
