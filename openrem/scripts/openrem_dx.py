#!/usr/local/bin/python
# This Python file uses the following encoding: utf-8
# scripts/openrem_dx

"""Script to launch the mam module to import information from radiographic images

    :param filename: relative or absolute path to radiographic DICOM image file.
    :type filename: str.

    Example::

        openrem_dx.py dximage.dcm

"""

if __name__ == "__main__":
    from openrem.remapp.extractors.dx import dx
    import openrem.remapp.tools.default_import as default_import

    default_import.default_import(
        dx, "import_dx", "the DICOM radiography image file", 0, {"import_dx": 1}
    )
