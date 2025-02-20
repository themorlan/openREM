DICOM import modules
====================

RDSR module
+++++++++++

Ultimately this should be the only module required as it deals with all Radiation Dose Structured Reports. This is used
for CT, fluoroscopy, mammography, digital radiography and nuclear medicine.

.. autofunction:: openrem.remapp.extractors.rdsr.rdsr


.. _mammo-module:

Mammography module
++++++++++++++++++

Mammography is interesting in that all the information required for dose
audit is contained in the image header, including patient 'size', ie thickness.
However the disadvantage over an RSDR is the requirement to process each
individual image rather than a single report for the study, which would
also capture any rejected images.

.. autofunction:: openrem.remapp.extractors.mam.mam

CR and DR module
++++++++++++++++

In practice this is only useful for DR modalities, but most of them use the
CR IOD instead of the DX one, so both are catered for. This module makes use
of the image headers much like the mammography module.

.. autofunction:: openrem.remapp.extractors.dx.dx

NM Image module
+++++++++++++++

This has the abilty to read information from the DICOM Headers of PET and 
NM images. In contrast to the other import modules this may actually complement
the data read from an RRDSR, because not all relevant data is included there.

.. autofunction:: openrem.remapp.extractors.nm_image.nm_image

CT non-standard modules
+++++++++++++++++++++++

Philips CT dose info reports
----------------------------

These have all the information that could be derived from the images also held in the DICOM header
information, making harvesting relatively easy. Used where RDSR is not available from older Philips systems.

.. autofunction:: openrem.remapp.extractors.ct_philips.ct_philips

Toshiba dose summary and images
-------------------------------

OpenREM can harvest information from older Toshiba CT systems that create dose summary images but cannot create
RDSR objects by using a combination of tools to create an RDSR that can then be imported in the normal manner.
This extractor requires that the Offis DICOM toolkit, java.exe and pixelmed.jar are available to the system.

.. autofunction:: openrem.remapp.extractors.ct_toshiba.ct_toshiba

