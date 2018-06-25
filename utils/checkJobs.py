#!/usr/bin/env python

import os
import glob
import argparse


def checkJobs():
    parser = argparse.ArgumentParser(description="Check if job was successful.",
                                     fromfile_prefix_chars="@", conflict_handler="resolve")
    parser.add_argument("--log-dir", help="path to condor logs")
    args = parser.parse_args()

    files = glob.glob("{}/*.error".format(args.log_dir))

    for _file in files:
        if "Traceback" in open(_file).read():
            print _file, "job failed..."
            continue


if __name__ == "__main__":
    checkJobs()
