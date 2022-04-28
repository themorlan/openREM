#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_mg

"""Script to launch the mam module to import information from mammography images

    :param filename: relative or absolute path to mammography DICOM image file.
    :type filename: str.

    Tested with:
        * GE Senographe DS software versions ADS_43.10.1 and ADS_53.10.10 only.

"""

import sys
from glob import glob
from openrem.remapp.extractors.mam import mam
from openrem.remapp.tools.background import run_as_task

if len(sys.argv) < 2:
    sys.exit("Error: Supply at least one argument - the DICOM mammography image file")

for arg in sys.argv[1:]:
    for filename in glob(arg):
        run_as_task(
            mam,
            "import_mam",
            None,
            filename,
        )