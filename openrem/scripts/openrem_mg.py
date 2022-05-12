#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_mg

"""Script to launch the mam module to import information from mammography images

    :param filename: relative or absolute path to mammography DICOM image file.
    :type filename: str.

    Tested with:
        * GE Senographe DS software versions ADS_43.10.1 and ADS_53.10.10 only.

"""


if __name__ == "__main__":
    from openrem.remapp.extractors.mam import mam
    import openrem.remapp.tools.default_import as default_import

    default_import.default_import(
        mam, "import_mam", "the DICOM mammography image file", 0, {"import_mam": 1}
    )
