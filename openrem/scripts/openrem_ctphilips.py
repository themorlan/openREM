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

if __name__ == "__main__":
    from openrem.remapp.extractors.ct_philips import ct_philips
    import openrem.remapp.tools.default_import as default_import

    default_import.default_import(
        ct_philips,
        "import_ct_philips",
        "the Philips dose report image",
        0,
        {"import_ct_philips": 1},
    )
