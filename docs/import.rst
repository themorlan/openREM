Importing data to OpenREM
*************************

From local DICOM files
======================

If you have RDSRs or RRDSRs, DX, MG, PET or NM images or Philips CT Dose Info images, you can import them directly into OpenREM:

..  toctree::
    :maxdepth: 2

    import-from-file

If you want some examples, you can find the DICOM files that we use for the automated testing in the
``openrem/remapp/tests/test_files`` folder in your OpenREM installation.

.. _directfrommodalities:

Direct from modalities
======================

For production use, you will either need the modalities to send the RDSR or images directly to your OpenREM server using
DICOM, or you will need to use query-retrieve to fetch the DICOM objects from the PACS or the modalities. In either of
these situations, you will need to run a DICOM Store service on your OpenREM server.

..  toctree::
    :maxdepth: 2

    netdicom-nodes

..  _configure_third_party_DICOM:

Third-party DICOM Stores
------------------------

The Orthanc DICOM server is recommended; Conquest or another store can be used instead:

..  toctree::
    :maxdepth: 2

    netdicom-orthanc-config
    netdicom-store

You ony need one of these - if you already have one installed it is probably easiest to stick to it.

Query-retrieve from a PACS or similar
=====================================

Before you can query-retrieve objects from a remote PACS, you need to do the following:

* Create a DICOM Store service to receive the DICOM objects - see :ref:`directfrommodalities` above.
* Configure OpenREM with the settings for the remote query-retrieve server:

  ..  toctree::
      :maxdepth: 2

      netdicom-qr-config

* Configure the settings of your DICOM store service on the PACS

* Learn how to use it:

  ..  toctree::
      :maxdepth: 2

      netdicom-qr