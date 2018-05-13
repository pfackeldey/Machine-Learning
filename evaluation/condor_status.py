#!/usr/bin/env python

import subprocess
import time


def getInfos():
    out = subprocess.check_output(["condor_q", "-long"])
    for jobStrings in out.split("\n\n"):
        for line in jobStrings.split("\n"):
            if line:
                break
                print[line.split(" = ")]
    return [dict([line.replace("\"", "").split(" = ") for line in jobStrings.split("\n") if " = " in line]) for jobStrings in out.split("\n\n") if jobStrings]


def getSummary():
    out = subprocess.check_output(["condor_q"])
    return out.split("\n")[-2]


def getNameFromFile(fname):
    return fname.replace(".log", "").split("/")[-1]


jobs = getInfos()
jobs = sorted(jobs, key=lambda l: l["JobStatus"] + l["Args"])
for job in jobs:
    name = getNameFromFile(job["UserLog"])
    jStatus = job["JobStatus"]
    if jStatus == "1":
        print name, "IDLE"
    elif jStatus == "2":
        print name, "\033[1;32mRUNNING\033[1;m", job["ClusterId"], " @ ", job["RemoteHost"].replace(".physik.rwth-aachen.de", "").replace("slot", "")
    elif jStatus == "7":
        susTime = (time.time() - int(job["LastSuspensionTime"])) / 60.
        print name, "\033[1;33mSUSPENDED\033[1;m since {:.2f} min".format(susTime), job["ClusterId"], "@", job["RemoteHost"].replace(".physik.rwth-aachen.de", "").replace("slot", "")
    elif jStatus == "5":
        susTime = (time.time() - int(job["LastSuspensionTime"])) / 60.
        print name, "\033[1;33mHELD\033[1;m", job["ClusterId"], "@", job["RemoteHost"].replace(".physik.rwth-aachen.de", "").replace("slot", "")
    else:
        print "job status = ", jStatus
print getSummary()
