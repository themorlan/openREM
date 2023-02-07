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

import os
import sys
import django

if __name__ == "__main__":
    basepath = os.path.dirname(__file__)
    projectpath = os.path.abspath(os.path.join(basepath, ".."))
    if projectpath not in sys.path:
        sys.path.insert(1, projectpath)
    os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
    django.setup()
    
    from remapp.extractors.rdsr import rdsr
    import remapp.tools.default_import as default_import

    default_import.default_import(
        rdsr,
        "import_rdsr",
        "the radiation dose structured report",
        0,
        {"import_rdsr": 1},
    )
