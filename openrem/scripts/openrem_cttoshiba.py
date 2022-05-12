#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_rdsr_toshiba_ct_from_dose_images

"""Script to launch the ct_toshiba module to import information from
   Toshiba CT dose images and additional information from image tags.

    :param folder_name: absolute path to Toshiba CT study DICOM files.
    :type filename: str.

    Tested with:
        * Toshiba Aquilion CXL software version V4.40ER011
        * Toshiba Aquilion CX  software version V4.51ER014
        * Toshiba Aquilion CXL software version V4.86ER008

"""

if __name__ == "__main__":
    from openrem.remapp.extractors.ct_toshiba import ct_toshiba
    import openrem.remapp.tools.default_import as default_import

    default_import.default_import(
        ct_toshiba,
        "import_ct_toshiba",
        "the folder containing the DICOM objects",
        0,
        {"import_ct_toshiba": 1},
    )
