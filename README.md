[![Build Status](https://travis-ci.org/CMSAachen3B/Machine-Learning.svg?branch=master)](https://travis-ci.org/CMSAachen3B/Machine-Learning)

# Machine-Learning

Modules and tools for a multivariate analysis

Installation of modules for scikit-learn:

1. scikit-learn: <http://scikit-learn.org/stable/install.html>
2. root-numpy: <http://scikit-hep.org/root_numpy/install.html>
3. python module seaborn: <https://seaborn.pydata.org/installing.html>
4. ROOT: <https://root.cern.ch/building-root> (recommended v6.08)

Installation of modules for Keras:

1. TensorFlow: <https://www.tensorflow.org/install/>
2. Keras: <https://keras.io/#installation>

For both (scikit-learn and Keras): `source /cvmfs/sft.cern.ch/lcg/views/LCG_91/x86_64-slc6-gcc62-opt/setup.sh`

More information on usage of the scripts with argument "-h" or "--help"!

## Testing scikit-learn and Keras with benchmark input files:

Download the following two benchmark files for signal and background:

1. Signal: <https://figshare.com/articles/HIGGS/1314899>
2. Background: <https://figshare.com/articles/HIGGS_background/1314900>

and move them to the repository. One has to specify the path in the example.yaml otherwise.

Call scikit-learn script:

```
python sklearn/scikitlearnclassification.py -s HIGGSsignal.root -b HIGGSbackground.root -v "lepton_pT;lepton_eta;lepton_phi;missing_energy_magnitude;missing_energy_phi;jet_1_pt;jet_1_eta;jet_1_phi;jet_1_b_tag;jet_2_pt;jet_2_eta;jet_2_phi;jet_2_b_tag;jet_3_pt;jet_3_eta;jet_3_phi;jet_3_b_tag;jet_4_pt;jet_4_eta;jet_4_phi;jet_4_b_tag;m_jj;m_jjj;m_lv;m_jlv;m_bb;m_wbb;m_wwbb"
```

Call Keras backend tensorflow script:

```
python keras/train.py example.yaml
```

More info using `--help` or `-h` option.

Write your own .yaml configuration and your own model in the KerasModels class to perform neural network classification!

## Evaluation:

for appending new branch in the root file:

```python
with TreeExtender("/source/file.root/myTree", "/target/file.root") as extender:
           extender.addBranch("myNewBranch", nLeaves=1, unpackBranches=["branchXYZ"])
           for entry in extender:
               entry.myNewBranch[0] = entry.branchXYZ * 2
```

or use the appropriate TMVA Reader producer in an Artus run! Keep in mind, that one has to add the same features/trainingvariables to the TMVA::Reader as used in the training.
