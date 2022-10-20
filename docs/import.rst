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

DICOM Store
-----------

The Orthanc DICOM server is recommended; another store can be used instead but documentation is not provided. Docker
installs have the Orthanc server build-in. For non-Docker installs, instructions are included in the main installation
documentation:

* Linux: :ref:`dicom_store_scp_linux`
* Windows: *to be written*


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