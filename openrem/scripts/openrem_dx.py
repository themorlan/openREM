#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_dx

"""Script to launch the mam module to import information from radiographic images

    :param filename: relative or absolute path to radiographic DICOM image file.
    :type filename: str.

    Example::

        openrem_dx.py dximage.dcm

"""

import sys
from glob import glob
from openrem.remapp.extractors.dx import dx
from openrem.remapp.tools.background import run_in_background_with_limits, wait_task

if len(sys.argv) < 2:
    sys.exit("Error: Supply at least one argument - the DICOM radiography image file")

tasks = []
for arg in sys.argv[1:]:
    for filename in glob(arg):
        b = run_in_background_with_limits(
            dx,
            "import_dx",
            0,
            {"import_dx": 1},
            filename,
        )
        tasks.append(b)

for t in tasks:
    wait_task(t)
