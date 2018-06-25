#!/usr/bin/env python

import ROOT
import os
import glob
import argparse


def checkZombie(args_from_script=None):
    parser = argparse.ArgumentParser(description="Check if file is zombie.",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")
    parser.add_argument("--input", help="inputdir path")
    args = parser.parse_args()

    rfiles = glob.glob("{}/*.root".format(args.input))

    for filename in rfiles:
        file_ = ROOT.TFile(filename)
        if file_.IsZombie():
            print filename, "is a frickin zombie!"
            continue


if __name__ == "__main__":
    checkZombie()
