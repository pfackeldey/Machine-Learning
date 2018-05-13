#!/usr/bin/env python

import ROOT
import argparse

parser = argparse.ArgumentParser(description="Start TMVA Graphic User Interface.",
                                 fromfile_prefix_chars="@", conflict_handler="resolve")
parser.add_argument("-i", "--input", help="Path to TMVA output", required=True)
parser.add_argument("--multiclass", default=False, action='store_true',
                    help="Run multiclass GUI instead of binary classification GUI. [Default: %(default)s]")
args = parser.parse_args()

if args.multiclass:
    ROOT.TMVA.TMVAMultiClassGui(args.input)
else:
    ROOT.TMVA.TMVAGui(args.input)
raw_input("Press Enter to exit")
