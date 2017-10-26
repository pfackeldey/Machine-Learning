# -*- coding: utf-8 -*-

import ROOT
import argparse

parser = argparse.ArgumentParser(description="Start TMVA Graphic User Interface.",
                                 fromfile_prefix_chars="@", conflict_handler="resolve")

parser.add_argument("-i", "--input", help="Path to TMVA output", required=True)
args = parser.parse_args()

ROOT.gROOT.LoadMacro("$ROOTSYS/tmva/test/TMVAGui.C")
ROOT.TMVAGui(args.input)
raw_input("Press Enter to exit")
