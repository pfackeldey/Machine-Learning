#!/usr/bin/env python

import os
import re
import glob
import argparse


def checkJobs():

    parser = argparse.ArgumentParser(description="Check if job was successful.",
                                     conflict_handler="resolve")
    parser.add_argument("--log-dir", help="path to condor logs")
    args = parser.parse_args()

    files = glob.glob("{}/*.error".format(args.log_dir))

    for _file in files:
        if "Traceback" in open(_file).read():
            print "\033[91mJob {} failed!\033[0m".format(re.search("part(.+?).error", _file).group(1))
        else:
            print "\033[1;32mJob {} was successful!\033[1;m".format(re.search("part(.+?).error", _file).group(1))


if __name__ == "__main__":
    checkJobs()
