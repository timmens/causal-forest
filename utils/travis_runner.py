#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
from glob import glob

if __name__ == "__main__":

    subdirectories = glob("*/")
    donotcheckdir = ["utils/"]
    donotcheckfiles = []

    for subdir in subdirectories:
        if subdir in donotcheckdir:
            continue
        # run specific actions for specific folders / files
