# Machine-Learning
Modules and tools for a multivariate analysis

Installation of modules for scikit-learn:
1. scikit-learn: http://scikit-learn.org/stable/install.html
2. root-numpy: http://scikit-hep.org/root_numpy/install.html
3. python module seaborn: https://seaborn.pydata.org/installing.html
4. ROOT: https://root.cern.ch/building-root (recommended v6.08)

Installation of modules for Keras:
1. TensorFlow: https://www.tensorflow.org/install/
2. Keras: https://keras.io/#installation

For both (scikit-learn and Keras): `source /cvmfs/sft.cern.ch/lcg/views/LCG_91/x86_64-slc6-gcc62-opt/setup.sh`



More information on usage of the scripts with argument "-h" or "--help"!

## Alternative to root_numpy:
Instead of using root_numpy to read in the data, check this out (not tested yet): https://github.com/artus-analysis/Artus/blob/master/Utility/python/treeTools.py

for appending new branch in the root file:
```python
with TreeExtender("/source/file.root/myTree", "/target/file.root") as extender:
           extender.addBranch("myNewBranch", nLeaves=1, unpackBranches=["branchXYZ"])
           for entry in extender:
               entry.myNewBranch[0] = entry.branchXYZ * 2
```
