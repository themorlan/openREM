###################################
OpenREM Release Notes version 0.8.0
###################################

****************
Headline changes
****************

* This release has extensive automated testing for large parts of the codebase (for the first time)
* Code quality is much improved, reduced duplication, better documentation, many bugs fixed
* Imports: RDSR from a wider range of systems now import properly
* Imports: Better distinction and control over defining RDSR studies as RF or DX
* Imports: Code and instructions to generate and import RDSR from older Toshiba CT scanners
* Imports: DICOM Query-Retrieve functionality has been overhauled
* Imports: Duplicate checking improved to allow cumulative and continued study RDSRs to import properly
* Imports: indicators that a study is not a patient can now be configured in the web interface
* Imports, display and export: Better handling of non-ASCII characters
* Interface: More detailed, consistent and faster rendering of the data in the web interface
* Interface: Maps of fluoroscopy radiation exposure incident on a phantom (Siemens RDSRs only)
* Interface: More and better charts, including scatter plots for mammography
* Interface: Display names dialogue has been extended to allow administration of all studies from each source
* Exports: Much faster, and more consistent
* Documentation: Extensive user documentation improvements



Specific upgrade instructions
=============================

For the original upgrade instructions, the last docs release to include them was
`0.10.0-docs <https://docs.openrem.org/en/0.10.0-docs/release-0.8.0.html>`_
