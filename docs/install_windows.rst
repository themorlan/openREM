**********************
Native Windows install
**********************

**Document not ready for translation**

This install is based on Windows Server 2022 using:

* Python 3.10 running in a virtualenv
* Database: PostgreSQL
* DICOM Store SCP: Orthanc running on port 104
* Webserver: Microsoft IIS
* Database files stored on D:
* OpenREM files stored on E:
* With Physics (QA) images being collected and zipped for retrieval

The instructions should work for Windows Server 2016 and 2019; and will probably work with Windows 10/11 with some
modification. Desktop editions of Windows are not recommended for a production OpenREM install.

If you are upgrading an existing installation to a new Windows server, go to the :doc:`upgrade_windows_new_server`
first.

If you are upgrading an existing Windows Server installation in-place, go to :doc:`upgrade_windows` instead.

Initial prep
============

Creating folders
^^^^^^^^^^^^^^^^

.. admonition:: Why C:, D: and E: drives?

    These folders are created on drive E: to keep the data away from the operating system drive so that it is easier
    for building/recreating the server and knowing what needs to be backed up.

    For the same reason, we will install PostgreSQL so that the database data is store on drive D: - this makes it possible
    to provide a different configuration of disk for the database drive, with different backup policies.

    However, it is also possible to store all the data on the C: drive if that works better for your installation. In
    this case, it would be advisable to create a folder C:\OpenREM\ and create all the folders specified below into that
    folder.

Create the following folders. The instructions here are for a ``CMD`` window but they can be created in Windows Explorer
instead:

.. code-block:: console

    C:\WINDOWS\system32>cd D:
    D:\>mkdir database
    D:\>cd E:
    E:\>mkdir log media pixelmed static venv orthanc\dicom orthanc\physics


Installing packages
^^^^^^^^^^^^^^^^^^^

Python
------



Orthanc
-------

PostgreSQL
----------

gettext
-------

Pixelmed and Java
-----------------

IIS
---

Installing Python packages
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create the virtualenv
---------------------

Install OpenREM
---------------

* Also install wfastcgi

OpenREM configuration and database creation
===========================================

PostgreSQL database creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


OpenREM configuration
^^^^^^^^^^^^^^^^^^^^^


Populate OpenREM database and collate static files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Webserver
=========

Configure IIS
^^^^^^^^^^^^^

Create a new website
^^^^^^^^^^^^^^^^^^^^

Configure the new website
^^^^^^^^^^^^^^^^^^^^^^^^^

Configure IIS to server the static files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test the webserver
------------------

DICOM Store SCP
===============










