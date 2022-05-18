#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_rdsr

"""
Script to extract data from PET or NM images.

:param filename: relative or absolute path to a NM or PET image.
:type filename: str.
"""

if __name__ == "__main__":
    from openrem.remapp.extractors.nm_image import nm_image
    import openrem.remapp.tools.default_import as default_import

    default_import.default_import(
        nm_image,
        "import_nm",
        "the nuclear medicine image",
        0,
        {"import_rdsr": 1, "import_nm": 1},
    )
