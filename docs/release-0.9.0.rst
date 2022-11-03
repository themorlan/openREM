########################
Upgrade to OpenREM 0.9.0
########################

****************
Headline changes
****************

* Interface: added feature to display workload stats in the home page modality tables
* Interface: added :doc:`i_fluoro_high_dose_alerts` feature
* Interface: dual-plane DX studies can now be handled in summary list and study detail pages
* Interface: new option to set display name when unique fields change based on device observer UID in RDSR
* Charts: added fluoroscopy charts of DAP and frequency per requested procedure, fixed bugs in links for others
* Query-retrieve: handle non-return of ModalitiesInStudy correctly
* Query-retrieve: increased query logging and summary feedback
* Query-retrieve: use time range in search (command line only)
* Imports: fix for empty NumericValues in RDSR
* Imports: fix for Toshiba RDSR with incorrect multiple values in SD field for vHP
* Imports: fix for Philips Azurion RDSR with incorrect AcquisitionDeviceType
* Imports: fix for Varian RDSRs
* Exports: made more robust for exporting malformed studies, fixed filtering bugs
* Administration: automatic e-mail alerts sent when fluoroscopy studies exceed a dose alert level
* Administration: added facility to list and delete studies where the import failed
* Administration: added interface to RabbitMQ queues and Celery tasks
* Administration: short-term fix for task performance and control on Windows
* Documentation: further refinement of the linux one-page install
* Installation: :doc:`virtual_directory`

Upgrade to current version
==========================

:doc:`upgrade_previous_0.10.0` and then upgrade to 1.0.

Original upgrade instructions
=============================

For the original upgrade instructions, the last docs release to include them was
`0.10.0-docs <https://docs.openrem.org/en/0.10.0-docs/release-0.9.0.html>`_
