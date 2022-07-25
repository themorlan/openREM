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

if __name__ == "__main__":
    from openrem.remapp.extractors.rdsr import rdsr
    import openrem.remapp.tools.default_import as default_import

    default_import.default_import(
        rdsr,
        "import_rdsr",
        "the radiation dose structured report",
        0,
        {"import_rdsr": 1},
    )
