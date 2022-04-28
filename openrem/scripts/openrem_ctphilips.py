#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_ctphilips

"""Script to launch the ct_philips module to import information from Philips CT 

    :param filename: relative or absolute path to Philips CT dose report DICOM image file.
    :type filename: str.

    Tested with:
        * Philips Gemini TF PET-CT v2.3.0
        * Brilliance BigBore v3.5.4.17001.
"""

import sys
from glob import glob
from openrem.remapp.extractors.ct_philips import ct_philips
from openrem.remapp.tools.background import run_in_background_with_limits, wait_task

if len(sys.argv) < 2:
    sys.exit("Error: Supply at least one argument - the Philips dose report image")

tasks = []
for arg in sys.argv[1:]:
    for filename in glob(arg):
        b = run_in_background_with_limits(
            ct_philips,
            "import_ct_philips",
            0,
            {"import_ct_philips": 1},
            filename,
        )
        tasks.append(b)

for t in tasks:
    wait_task(t)
