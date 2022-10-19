###################################
OpenREM Release Notes version 0.8.1
###################################

****************
Headline changes
****************

* Documentation: improved docs and added one-page complete install on Ubuntu instructions
* Install: temporary fix for dependency error
* Interface: added feature to allow users to change their own password
* Charts: fixed problem where a blank category name may not be displayed correctly
* Imports: reduced list of scanners that work with the legacy Toshiba CT extractor
* Imports: improved handling of non-conformant DX images with text in filter thickness fields
* Query-Retrieve: added non-standard option to work-around bug in Impax C-FIND SCP
* Exports: fixed bug in mammography NHSBSP exports that incorrectly reported the filter material in some circumstances
* Exports: fixed bug where sorting by AGD would cause duplicate entries for bilateral studies
* Exports: fixed another non-ASCII bug

If upgrading from 0.7.4, see also :doc:`release-0.8.0`


Specific upgrade instructions
=============================

For the original upgrade instructions, the last docs release to include them was
`0.10.0-docs <https://docs.openrem.org/en/0.10.0-docs/release-0.8.1.html>`_
