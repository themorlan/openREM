###################################
OpenREM Release Notes version 0.6.0
###################################

****************
Headline changes
****************

* Charts
* Preview of DICOM Store SCP functionality
* Exports available to import into `openSkin`_
* Modalities with no data are hidden in the user interface
* Mammography import compression force behaviour changed
* Import of Toshiba planar RDSRs fixed

Changes for 0.6.2
=================

Minor update due prevent new installs from installing a non-compatible version of ``django-filter``.
The link to `openSkin`_ has also been updated in the fluoroscopy detail page.

**There is no advantage to updating to this version over 0.6.0**

Release 0.6.1 was just a documentation only change to update the link to `openSkin`_.


Specific upgrade instructions
=============================

For the original upgrade instructions, the last docs release to include them was
`0.10.0-docs <https://docs.openrem.org/en/0.10.0-docs/release-0.6.0.html>`_


***********************
Summary of new features
***********************

Charts
======

Release 0.6.0 has a range of charting options available for CT and radiographic data. These charts allow visualisation
of trends and frequencies to inform surveys and monitor performance. For more information, please see :doc:`charts`.


DICOM Store Service Class Provider
==================================

OpenREM can now act as the DICOM Store service, allowing direct sending of DICOM objects from modalities to OpenREM
without needing to use Conquest or any other DICOM Store SCP. This feature is a preview as it hasn't been extensively
tested, but it is expected to work. For more information, please see :ref:`directfrommodalities`.


Exports for openSkin
====================

Fluoroscopy studies can now be exported in a format suitable for importing into Jonathan Cole's openSkin software. The
export link is on the fluoroscopy study detail page. The software for creating the exposure incidence map can be
downloaded from https://bitbucket.org/openskin/openskin/downloads (choose the zip file), and information about the project
can be found on the `openSkin wiki`_. The software allows the user to choose between a 2D phantom that would represent
the dose to a film laying on the couch surface, or a simple 3D phantom made up of a cuboid and two semi-cylinders
(these can be seen on the `Phantom design`_ section of the wiki). For both options the output is an image of the dose
distribution in 2D, along with calculated peak skin dose information.

Automatic hiding of unused modality types
=========================================

A fresh install of OpenREM will no longer show any of the four modality types in the tables or in the navigation bar
at the top. As DICOM objects are ingested, the appropriate tables and navigation links are created.

Therefore a site that has no mammography for example will no longer have that table or navigation link in their
interface.

Mammography import compression force change
===========================================

Prior to version 0.6, the compression force extracted from the mammography image header was divided by ten before being
stored in the database. This was because the primary author only had access to GE Senograph DS units, which store the
compression force in dN, despite claiming using Newtons in the DICOM conformance statement.

The code now checks for the term *senograph ds* contained in the model name. If it matches, then the value is divided by
ten. Otherwise, the value is stored without any further change. We know that later GE units, the GE Senograph Essential
for example, and other manufacturer's units store this value in N. If you have a case that acts like the Senograph DS,
please let us know and we'll try and cater for that.

If you have existing non-GE Senograph mammography data in your database, the compression force field for those studies
is likely to be incorrect by a factor of ten (it will be too small). Studies imported after the upgrade will be correct.
If this is a problem for you, please let us know and we'll see about writing a script to correct the existing data.

Import of Toshiba Planar RDSRs fixed
====================================

Toshiba include Patient Orientation and Patient Orientation Modifier information in their cath lab RDSRs. The extractor
code was deficient for this as the RDSRs previously used didn't have this information. This has now been fixed. There
might however be an issue with Station Name not being provided - it is not yet clear if this is a configuration issue.

.. _`openSkin`: https://bitbucket.org/openskin/openskin
.. _`openSkin wiki`: https://bitbucket.org/openskin/openskin/wiki/Home
.. _`Phantom design`: https://bitbucket.org/openskin/openskin/wiki/Phantom%20design
