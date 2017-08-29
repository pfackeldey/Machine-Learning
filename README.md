# Machine-Learning
Modules and tools for a multivariate analysis

Installation of modules (if possible via pip is recommended):
1. scikit-learn: http://scikit-learn.org/stable/install.html
2. root-numpy: http://scikit-hep.org/root_numpy/install.html
3. python module seaborn: https://seaborn.pydata.org/installing.html
4. ROOT: https://root.cern.ch/building-root (recommended v6.08)

Example console call: python scikitlearnclassification.py -s data/HIGGSsignal.root -b data/HIGGSbackground.root -v "pt;eta;phi;..."

More information with argument "-h" or "--help"!

## Alternative to root_numpy:
Instead of using root_numpy to read in the data, check this out (not tested yet): https://github.com/artus-analysis/Artus/blob/master/Utility/python/treeTools.py

for accessing the root file:
```python
with TreeExtender("/source/file.root/myTree", "/target/file.root") as extender:
           extender.addBranch("myNewBranch", nLeaves=1, unpackBranches=["branchXYZ"])
           for entry in extender:
               entry.myNewBranch[0] = entry.branchXYZ * 2
```
for writing output back to root file:
```python
with TreeMerger("/target/file.root/myMergedTree") as merger:
            merger.addTree("/source/file1.root")
            merger.addTree("/source/file2.root")
```
Thanks to Marcel Rieger for the last part!

## Workflow with ARTUS:
#todo
