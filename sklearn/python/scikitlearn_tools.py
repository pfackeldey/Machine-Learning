# coding=utf-8
import codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.tree import export_graphviz
from sklearn import cross_validation
from sklearn.cross_validation import ShuffleSplit
from sklearn.learning_curve import learning_curve
from sklearn.metrics import roc_auc_score
from sklearn import grid_search

CV = 3
NJOBS = 6


def plot_correlations(outdir, vars, sig, bkg):
    for data, label in ((sig, "Signal"), (bkg, "Background")):
        d = pd.DataFrame(data, columns=vars)
        sns.heatmap(d.corr(), annot=True, fmt=".2f", linewidth=.5)
        plt.title(label + " Correlations")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'correlations_{}.png'.format(label.lower())))
        plt.savefig(os.path.join(outdir, 'correlations_{}.png'.format(label.lower())))
        plt.close()

def plot_inputs(outdir, vars, sig, sig_w, bkg, bkg_w):
    for n, var in enumerate(vars):
        _, bins = np.histogram(np.concatenate((sig[:, n], bkg[:, n])), bins=40)
        sns.distplot(bkg[:, n], hist_kws={'weights': bkg_w}, bins=bins, kde=False, norm_hist=True, label='background')
        sns.distplot(sig[:, n], hist_kws={'weights': sig_w}, bins=bins, kde=False, norm_hist=True, label='signal')
        plt.title(var)
        plt.legend()
        plt.savefig(os.path.join(outdir, 'input_{}.png'.format(var)))
        plt.savefig(os.path.join(outdir, 'input_{}.png'.format(var)))
        plt.close()

def visualize_tree(bdt):
	#Create tree png using graphviz.

	#Args
	#----
	#tree -- scikit-learn DecsisionTree.
	#feature_names -- list of feature names.

	with open("dt.dot", 'w') as f:
		export_graphviz(bdt, out_file=f, feature_names=list_of_variables, filled=True, rounded=True,
		special_characters=True)

	command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
	try:
		subprocess.check_call(command)
	except:
		exit("Could not run dot, ie graphviz, to "
	     "produce visualization")


def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30):
	decisions = []
	for X,y in ((X_train, y_train), (X_test, y_test)):
		d1 = clf.decision_function(X[y>0.5]).ravel()
		d2 = clf.decision_function(X[y<0.5]).ravel()
		decisions += [d1, d2]

	low = min(np.min(d) for d in decisions)
	high = max(np.max(d) for d in decisions)
	low_high = (low,high)

	plt.hist(decisions[0],
	     color='r', alpha=0.5, range=low_high, bins=bins,
	     histtype='stepfilled', normed=True,
	     label='S (train)')
	plt.hist(decisions[1],
	     color='b', alpha=0.5, range=low_high, bins=bins,
	     histtype='stepfilled', normed=True,
	     label='B (train)')

	hist, bins = np.histogram(decisions[2],
			      bins=bins, range=low_high, normed=True)
	scale = len(decisions[2]) / sum(hist)
	err = np.sqrt(hist * scale) / scale

	width = (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='S (test)')

	hist, bins = np.histogram(decisions[3],
			      bins=bins, range=low_high, normed=True)
	scale = len(decisions[2]) / sum(hist)
	err = np.sqrt(hist * scale) / scale

	plt.errorbar(center, hist, yerr=err, fmt='o', c='b', label='B (test)')

	plt.xlabel("BDT output")
	plt.ylabel("Arbitrary units")
	plt.legend(loc='best')
	plt.savefig("BDT_output.png")
	plt.clf()

def plot_learning_curve(outdir, bdt, x, y):
	logging.info("creating learning curve")
	train_sizes, train_scores, test_scores = learning_curve(bdt,
								x,
								y,
		                                                cv=ShuffleSplit(len(x),
		                                                n_iter=100,
		                                                test_size=1.0 / CV),
		                                            	n_jobs=NJOBS,
								verbosity=2,
		                                            	train_sizes=np.linspace(.1, 1., 7),
		                                            	scoring='roc_auc')
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.fill_between(train_sizes,
		     train_scores_mean - train_scores_std,
		     train_scores_mean + train_scores_std,
		     alpha=.2, color='r')
	plt.fill_between(train_sizes,
		     test_scores_mean - test_scores_std,
		     test_scores_mean + test_scores_std,
		     alpha=.2, color='g')
	plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
	plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')

	plt.xlabel("Sample size")
	plt.ylabel("Score (ROC area)")

	plt.legend()
	plt.savefig(os.path.join(outdir, 'learning-curve.png'))
	plt.savefig(os.path.join(outdir, 'learning-curve.pdf'))
	plt.close()

def run_grid_search(outdir, bdt, x, y):
	logging.info('starting hyper-parameter optimization')
	param_grid = {"n_estimators": [50,100,800,1000], 'learning_rate': [0.01,0.1,0.5]}

	clf = grid_search.GridSearchCV(bdt, param_grid, cv=CV, scoring='roc_auc', n_jobs=NJOBS, verbosity=2)
	clf.fit(x, y)

	out = '\nHyper-parameter optimization\n'
	out += '============================\n\n'
	out += 'Best estimator: {}\n'.format(clf.best_estimator_)
	out += '\nFull Scores\n'
	out += '-----------\n\n'
	for params, mean_score, scores in clf.grid_scores_:
		out += u'{:0.4f} (±{:0.4f}) for {}\n'.format(mean_score, scores.std(), params)
	with codecs.open(os.path.join(outdir, "log-hyper-parameters.txt"), "w", encoding="utf8") as fd:
		fd.write(out)

def roc_curve(false_positive_rate, true_positive_rate, score):
	plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.3f)' % score)
	plt.plot([0,1],[0,1],'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate or (1 - Specifity)')
	plt.ylabel('True Positive Rate or (Sensitivity)')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc="lower right")
	plt.savefig('ROC.png')
	plt.savefig('ROC.pdf')
	plt.close()

