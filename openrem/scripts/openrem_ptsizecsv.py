#!/usr/local/bin/python
# scripts/openrem_ptsizecsv

"""Script to launch the rdsr to import information from DICOM Radiation SR objects

    :param filename: relative or absolute path to Radiation Dose Structured Report.
    :type filename: str.

    Tested with:
        * CT: Siemens, Philips and GE RDSR, GE Enhanced SR.
        * Fluoro: Siemens Artis Zee RDSR
"""

from openrem.remapp.extractors.ptsizecsv2db import csv2db
from openrem.remapp.tools.background import run_as_task

run_as_task(
    csv2db,
    "import_size",
    None,
)
