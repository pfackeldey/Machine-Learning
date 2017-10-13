# -*- coding: utf-8 -*-
#!/usr/bin/env python


import ROOT

ROOT.PyConfig.IgnoreCommandLineOptions = True

from model import KerasModel
from treetools import TreeExtender

import argparse
import os
import sys
import yaml

EPOCHS = 10
BATCH_SIZE = 128


def multiclassNeuralNetwork(args_from_script=None):


	parser = argparse.ArgumentParser(description="Perform Multiclass NN training with kPyKeras (TMVA).",
	                                 fromfile_prefix_chars="@", conflict_handler="resolve")

	parser.add_argument("config", help = "Path to training config")
	args = parser.parse_args()

	ROOT.TMVA.Tools.Instance()
	ROOT.TMVA.PyMethodBase.PyInitialize()

	# initialize factory:
	
	factory = ROOT.TMVA.Factory("TMVAMulticlass", ROOT.TFile.Open("training_output.root", "RECREATE"), "!V:!Silent:Color:!DrawProgressBar:Transformations=None:AnalysisType=multiclass")
	
	# load yaml training config
	config = yaml.load(open(args.config), "r")

	dataloader = ROOT.TMVA.DataLoader("example_training")
	
	for variable in config["variables"]:
		dataloader.AddVariable(variable)
	trees = {}
	for class_ in config["classes"]:
		trees[class_] = ROOT.TFile(config["training_data"]).Get(class_)
		dataloader.AddTree(trees[class_], class_, config["class_weights"][class_]*config["global_weight"])
		dataloader.SetWeightExpression(config["event_weight"], class_)
		dataloader.PrepareTrainingAndTestTree(Root.TCut(""), "TrainTestSplit_{}={}:".format(class_, config["train_test_split"])+"SplitMode=Random:NormMode=None")

	model = KerasModel(n_features = len(config["features"]), n_classes = len(config["classes"]), learning_rate = 0.1)
	model.example_model()
	model.summary()
	model.save("example_model.h5")

	factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras, "PyKeras_example", "!H:!V:VarTransform=None:FileNameModel=example_model.h5:SaveBestOnly=true:TriesEarlyStopping=-1:NumEpochs={}:".format(EPOCHS)+"BatchSize={}".format(BATCH_SIZE))

	factory.TrainAllMethods()
	factory.TestAllMethods()
	factory.EvaluateAllMethods()

if __name__ == "__main__" and len(sys.argv) > 1:
	multiclassNeuralNetwork()
