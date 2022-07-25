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

if __name__ == "__main__":
    csv2db()
