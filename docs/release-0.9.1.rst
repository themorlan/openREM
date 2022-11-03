########################
Upgrade to OpenREM 0.9.1
########################

****************
Headline changes
****************

* Imports: fixed imports for GE surgical flat panel c-arm with irregular value types and value meanings
* Interface: added feature to filter by specific number of exposure types -- CT only
* Query-retrieve: new option to get SR series when PACS returns empty series level response
* Query-retrieve: handle illegal missing instance number in image level response
* Query-retrieve: improved logging
* Exports: added export to UK PHE 2019 CT survey format
* General documentation and interface improvements, bug fixes, and changes to prepare for Python 3

Upgrade to current version
==========================

:doc:`upgrade_previous_0.10.0` and then upgrade to 1.0.

Original upgrade instructions
=============================

For the original upgrade instructions, the last docs release to include them was
`0.10.0-docs <https://docs.openrem.org/en/0.10.0-docs/release-0.9.1.html>`_
