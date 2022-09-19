**********************
Native Windows install
**********************

**Document not ready for translation**

This install is based on Windows Server 2022 using:

* Python 3.10 running in a virtualenv
* Database: PostgreSQL
* DICOM Store SCP: Orthanc running on port 104
* Webserver: Microsoft IIS
* Collects any Physics (QA) images and zips them

The instructions should work for Windows Server 2016 and 2019; and will probably work with Windows 10/11 with some
modification. Desktop editions of Windows are not recommended for a production OpenREM install.

If you are upgrading an existing installation to a new Windows server, go to the :doc:`upgrade_windows_new_server`
first.

If you are upgrading an existing Windows Server installation in-place, go to :doc:`upgrade_windows` instead.

Initial prep
============

Creating folders
^^^^^^^^^^^^^^^^



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










