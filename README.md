[![Build Status](https://travis-ci.org/CMSAachen3B/Machine-Learning.svg?branch=master)](https://travis-ci.org/CMSAachen3B/Machine-Learning)

# Machine-Learning

Modules and tools for a multivariate analysis

## Setup

It is necessary to change `law.cfg`, `luigi.cfg`, `hwwenv.sh`, `setup.sh` and `MSSM_HWW.yaml` to your personal paths and
environment variables!  

First setup software and environment variables for preprocessing up to the ROOT to numpy conversion:

1. `cd tasks`
2. `bash install_lx3b.sh` (only once)
3. `. setup.sh` (in every new shell)
4. `. hwwenv.sh` (in every new shell)

Now you are ready to go!

## Preprocessing

First you need to create the trainingset. This step is handled by law (based on luigi)
For this you need to initialize all law tasks:

1. `law db`

Now you can check the status of the trainingset creation:

2. `law run MergeTrainingset --CreateTrainingset-workflow local --print-status 1`

Open a second shell, setup everything and start a central scheduler via: `luigid`  
Now you can open your browser and open `localhost:8082`. Afterwards you can start the task in the previous shell:

3. `law run MergeTrainingset --CreateTrainingset-workflow local --workers 4`

The number of workers can also be changed during the running task in the central scheduler. If you omit the option
`--CreateTrainingset-workflow local` the task `CreateTrainingset` will be submitted to the HTCondor batch system.
Once the task is finished you can convert the merged trainingset to numpy arrays:

4. `law run NumpyConversion --workers 4`

Also here the number of workers can be changed with the central scheduler.

## Training

Once the preprocessing step is done, you can start the training. Here it is recommended to use a GPU (for RWTH see the wiki of this repo). More details in the wiki...

Local training can be run with:

* `python training/train.py`

For the training submission on the RZ cluster of RWTH, see [here](https://github.com/CMSAachen3B/Machine-Learning/wiki/GPU-Batch-System). An example `.submit` script can be found in the `utils` directory of this repository.

## Evaluation

For a single file:

* `python evaluation/keras_evaluation.py --file test.root --tree latino`

For all files (at RWTH Aachen via HTCondor):

* `python evaluation/process.py`

In order to evaluate only data, wjets or MC files use the option `--datasets`.
More information on script usage with `--help`  

The status of the jobs can be checked with:

1. `python evaluation/condor_status.py`
2. `condor_q [--long]`

After the jobs finished, check if a job failed with:  

* `python utils/checkJobs.py --log-dir <path/to/condor_logs>`  

If a job failed, check the corresponding `.error` file for the problem. Sometimes (very rarely) it happens that a ROOT file gets damaged. This one can check with:  

* `python utils/checkZombie.py --dir <path/to/ntuples>`  

If that happens, just copy the file again from the HWW group and resubmit the job!


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
