#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_rdsr

"""
Script to extract data from PET or NM images.

:param filename: relative or absolute path to a NM or PET image.
:type filename: str.
"""

import sys
from glob import glob
from openrem.remapp.extractors.nm_image import nm_image

if len(sys.argv) < 2:
    sys.exit(
        "Error: Supply at least one argument - the nuclear medicine image"
    )

for arg in sys.argv[1:]:
    for filename in glob(arg):
        nm_image(arg)

sys.exit()
