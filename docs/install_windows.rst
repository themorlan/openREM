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

.. figure:: img/FolderLayout.png
   :figwidth: 20%
   :align: right
   :alt: Windows install folder layout
   :target: _images/FolderLayout.png

   Figure 1: Windows install folder layout

Create the following folders. The instructions here are for a ``CMD`` window but they can be created in Windows Explorer
instead:

.. code-block:: console

    C:\Users\openrem>D:
    D:\>mkdir database
    D:\>E:
    E:\>mkdir log media pixelmed static venv orthanc\dicom orthanc\physics orthanc\storage

.. admonition:: Why D: and E: drives?

    These folders are created on drive E: to keep the data away from the operating system drive so that it is easier
    for building/recreating the server and knowing what needs to be backed up.

    For the same reason, we will install PostgreSQL so that the database data is store on drive D: - this makes it possible
    to provide a different configuration of disk for the database drive, with different backup policies.

    However, it is also possible to store all the data on the C: drive if that works better for your installation. In
    this case, it would be advisable to create a folder C:\OpenREM\ and create all the folders specified below into that
    folder.

    You can also use different drive letters if that works better for your installation. In both cases paths will need
    to be modified in the instructions to suite.



Installing packages
^^^^^^^^^^^^^^^^^^^

Python
------

Download the latest version for Windows from https://www.python.org/downloads/ as long as it is in the 3.10 series.
OpenREM v1.0 has not been tested with Python 3.11 yet.

Open the downloaded file to start the installation:

* Customize installation
* Leave all the Optional Features ticked, and click ``Next``
* Tick ``Install for all users`` - this will automatically tick ``Precompile standard library``
* ``Install``
* Click to ``Disable path length limit`` - might not be necessary but might be useful!
* ``Close``

Orthanc
-------

Download the 64 bit version from https://www.orthanc-server.com/download-windows.php.

The download file might be blocked because it isn't a commonly downloaded executable. Click the ``...`` menu
and select ``Keep``. Then click ``Show more`` and ``Keep anyway``.

Open the downloaded file to start the installation:

* Click ``Next >``, accept the agreement and ``Next >`` again.
* Default install location, ``Next >``
* Select Orthanc storage directory - ``Browse...`` to ``E:\orthanc\storage``, ``OK`` and ``Next >``
* Click ``Next >`` for a Full installation
* Start Menu Folder ``Next >``
* Ready to Install ``Install``
* ``Finish``


PostgreSQL
----------

Download the latest version of PostgreSQL from https://www.enterprisedb.com/downloads/postgres-postgresql-downloads -
choose the Windows x86-64 version. OpenREM v1.0 has been tested with PostgreSQL v14.5.

Open the downloaded file to start the installation:

* Some Microsoft redistributables will install
* Click ``Next >`` to start
* Default Installation Directory ``Next >``
* All components ``Next >``
* Data Directory - browse to ``D:\database`` then ``Select folder`` and ``Next >``
* Create a password for the ``postgres`` superuser - you will need this to setup the database with pgAdmin 4 later
* Enter it twice and ``Next >``
* Default port ``Next >``
* Default Locale ``Next >``
* Pre Installation Summary ``Next >``
* Ready to Install ``Next >`` and the installation will begin
* Untick ``Launch Stack Builder at exit``
* ``Finish``

gettext
-------

Download the 64 bit static version of gettext 0.21 from https://mlocati.github.io/articles/gettext-iconv-windows.html.
Use the ``.exe`` version (software install icon, not the zip icon)

Open the downloaded file to start the installation:

* Accept the agreement ``Next >``
* Default installation directory ``Next >``
* Additional Tasks leave both boxes ticked ``Next >``
* Ready to Install ``Install``
* ``Finish``


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










