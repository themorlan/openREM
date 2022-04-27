#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_rdsr

"""Script to launch the rdsr to import information from DICOM Radiation SR objects 

    :param filename: relative or absolute path to Radiation Dose Structured Report.
    :type filename: str.

    Tested with:
        * CT: Siemens, Philips and GE RDSR, GE Enhanced SR.
        * Fluoro: Siemens Artis Zee RDSR
"""

import sys
from glob import glob
from openrem.remapp.extractors.rdsr import rdsr
from openrem.remapp.tools.background import run_as_task

if len(sys.argv) < 2:
    sys.exit(
        "Error: Supply at least one argument - the radiation dose structured report"
    )

for arg in sys.argv[1:]:
    for filename in glob(arg):
        run_as_task(
            rdsr,
            "import_rdsr",
            None,
            filename,
        )

sys.exit()
