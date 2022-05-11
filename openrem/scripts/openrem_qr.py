#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_qr

"""Script to launch the DICOM Store SCP service

"""

from openrem.remapp.netdicom.qrscu import qrscu_script

if __name__ == "__main__":
    qrscu_script()
