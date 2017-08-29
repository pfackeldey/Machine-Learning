#!/usr/bin/env python
# -*- coding: utf-8 -*-

#REFERENCE: https://betatim.github.io/posts/sklearn-for-TMVA-users/
####################################################################################
# This script needs scikit-learn and root_numpy modules and requieres the	   #
# following versions: Python >=2.6 or >=3.3, Numpy >=1.6.1, Scipy >=0.9 and	   #
# ROOT >=5.32.                							   #
####################################################################################


import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.cross_validation import train_test_split

import logging
log = logging.getLogger(__name__)

import argparse
import os
import sys

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

import scikitlearn_tools as tools


def scikit_classification(args_from_script=None):
	"""
	Perform scikit learn classification training.

	"""

	parser = argparse.ArgumentParser(description="Perform scikit learn classification training.",
	                                 fromfile_prefix_chars="@", conflict_handler="resolve")

	parser.add_argument("-s", "--signal", nargs="+", required=True, default=None,
	                    help="Signal file. Format same as for TChain.Add: path/to/file.root.")
	parser.add_argument("--signal-weight", default="", help="Signal weight expression. [Default: %(default)s]")
	parser.add_argument("--signal-cut", default="", help="Signal cut expression. [Default: %(default)s]")
	parser.add_argument("-b", "--background", nargs="+", required=True, default=None,
	                    help="Background file. Format same as for TChain.Add: path/to/file.root.")
	parser.add_argument("--background-weight", default="", help="Background weight expression. [Default: %(default)s]")
	parser.add_argument("--background-cut", default="", help="Background cut expression. [Default: %(default)s]")
	parser.add_argument("-f", "--folder",default=None, required=False,
	                    help="Tree in signal & background file. [Default: %(default)s]")
	parser.add_argument("-v", "--variables", nargs="+", required=True, default=None,
	                    help="Training variables.")
	parser.add_argument("--splitting", nargs="+", default="0.3 0.1 0.6",
	                    help="Set relative size of training, test and evaluation subsample (sum has to be 1). [Default: %(default)s]")
	parser.add_argument("-o", "--output-file", default="sklearnClassification/output.root",
	                    help="Output file. [Default: %(default)s]")
	parser.add_argument("--optimize-parameter", default=False, action='store_true',
	                    help="Optimize the model for a fixed hyperparameter space. [Default: %(default)s]")
	parser.add_argument("--additional-plots", default=False, action='store_true',
	                    help="Perform additional plots. [Default: %(default)s]")
	args = parser.parse_args()

	from root_numpy import root2array, rec2array, array2root
	
	#training variables
	list_of_variables = args.variables[0].split(";")

	print list_of_variables
	
	#prepare signal and background
	signal = root2array(args.signal,
			    args.folder,
			    list_of_variables,
			    selection=args.signal_cut,
			    include_weight=True if args.signal_weight else False,
			    weight_name=args.signal_weight)
	signal = rec2array(signal)

	print args.signal,args.folder,list_of_variables,args.signal_cut,args.signal_weight,args.signal_weight

	backgr = root2array(args.background,
			    args.folder,
			    list_of_variables,
			    selection=args.background_cut,
			    include_weight=True if args.background_weight else False,
			    weight_name=args.background_weight)
	backgr = rec2array(backgr)

	
		
	#sklearn needs 2D dataformat
	X = np.concatenate((signal, backgr))
	y = np.concatenate((np.ones(signal.shape[0]),
		            np.zeros(backgr.shape[0])))

	#sample splitting
	splitting = args.splitting.split()
	for i,split in zip(range(len(splitting)),splitting):
		splitting[i]=float(split)
	
	X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=splitting[2], random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=splitting[1], random_state=42)

	
	#model and training
	dt = DecisionTreeClassifier(min_samples_leaf=0.05)
		
	bdt = AdaBoostClassifier(dt,algorithm='SAMME.R',n_estimators=50,learning_rate=0.1)
	bdt.fit(X_train,y_train)

	#optimization of hyper parameter
	if args.optimize_parameter==True:

		tools.run_grid_search("sklearnClassification", bdt, X_train, y_train)

	print bdt.get_params().keys()
	#testing
	y_predicted = bdt.predict(X_test)
	y_predicted.dtype = [('score', np.float64)]
	array2root(y_predicted, args.output_file, "BDTtest", mode="recreate")

	#classification score (roc auc score):
	print "Area under ROC curve: %.4f"%(roc_auc_score(y_test, bdt.decision_function(X_test)))

	#roc-curve
	decisions = bdt.decision_function(X_test)
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, decisions)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	
	import matplotlib.pyplot as plt

	plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.3f)' % roc_auc)
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate or (1 - Specifity)')
	plt.ylabel('True Positive Rate or (Sensitivity)')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc="lower right")
	plt.savefig("ROC.png")
	plt.clf()


	#BDT output plot	    
	tools.compare_train_test(bdt, X_train, y_train, X_test, y_test)

	if args.additional_plots==True:
		tools.plot_learning_curve("sklearnClassification", bdt, X_train, y_train)
		tools.plot_correlations("sklearnClassification", list_of_variables, signal, backgr)
		tools.plot_inputs("sklearnClassification", list_of_variables, signal, None, bkg, None)
		

	#evaluation
	y_eval = bdt.predict(X_eval)
	y_eval.dtype = [('score', np.float64)]
	array2root(y_eval, args.output_file, "BDTeval")

	#finish        
	print "Training output is written to \"" + args.output_file + "\"."
