[![Build Status](https://travis-ci.org/CMSAachen3B/Machine-Learning.svg?branch=master)](https://travis-ci.org/CMSAachen3B/Machine-Learning)

# Machine-Learning

Modules and tools for a multivariate analysis

Get started with:  

virtualenv (with python 3.6):
* `. ./checkout_script.sh` (only once)  
* `. ./setup.sh` (for every new terminal)

general (recommended for lxplus and lx3b, but does not support luigi... :/):
* `source /cvmfs/sft.cern.ch/lcg/views/LCG_92/x86_64-slc6-gcc62-opt/setup.sh`



The preprocessing workflow is controlled by luigi! 

Run preprocessing to get numpy arrays from ntuple:  
`python -m luigi --module run_luigi RootToNumpy --config-path config/MSSM_HWW.yaml --local-scheduler`

For a more detailed overview about the code and how to use it, take a look into
the wiki!

## License

MIT License

Copyright (c) 2017 Aachen 3B CMS Group

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
