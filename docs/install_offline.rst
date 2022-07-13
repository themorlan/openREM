*******************************
Offline installation or upgrade
*******************************

In order to install or upgrade OpenREM on a server that does not have access to the internet you will need to download
all the packages and dependencies on another computer and copy them across.

If you have trouble when installing the Python packages on Windows due to incorrect architecture, you may need to either
download on a Windows system similar to the server (matching 32-bit/64-bit), or to download the files from
http://www.lfd.uci.edu/~gohlke/pythonlibs/ instead. Alternatively there are ways to tell ``pip`` to download binary
packages for specific platforms.

It is expected and highly recommended that Windows and Linux server have access to security updates even
when other internet access is blocked. A Linux server will need temporary access to the distribution's repositories to
install Python, PostgreSQL, NGINX, Orthanc, DCMTK, Java runtime environment and a few other packages on the installation
instructions. Access can then be disabled again, though it is recommended that updates be allowed through. Alternatively
it is possible to create a local repository mirror/cache, or download all the packages manually, but this is beyond the
scope of these documents.

An :doc:`install_offline_docker` might be easier on an offline Linux server, once Docker and Docker Compose are
installed.

On a computer with internet access
==================================

Download independent binaries
-----------------------------

**Python** from https://www.python.org/downloads/windows/

* Download the latest Python 3.10 Windows installer (64-bit)
* For Linux servers, Python needs to be installed through the distribution's package manager

**PostgreSQL** from https://www.enterprisedb.com/downloads/postgres-postgresql-downloads

* Download the latest PostgreSQL 14 version for Windows x86-64
* Alternatively Microsoft SQL Server can be used, but support is not available in these documents
* Some functionality might be missing if you use the built-in SQLite3 for testing or MS SQL Server

Optional - enables Toshiba and some other older CT 'dose information' images to be imported, but requires a
Java runtime environment:

**Pixelmed** from http://www.dclunie.com/pixelmed/software/webstart/pixelmed.jar


Download Python packages from PyPI
----------------------------------

In a console, navigate to a suitable place and create an empty directory to collect all the packages in, then use
``pip`` to download them all:

.. code-block:: console

    PS C:\Users\me\Desktop> mkdir openremfiles
    PS C:\Users\me\Desktop> pip3 download -d openremfiles pip
    PS C:\Users\me\Desktop> pip3 download -d openremfiles openrem

Copy everything to the Server
-----------------------------

* Copy this directory plus the binaries to the offline server.

On the server without internet access
=====================================

Follow the :doc:`install_windows`, :doc:`upgrade_windows`, :doc:`install_linux` or :doc:`upgrade_linux` instructions
until you reach the step where Python packages are installed, installing the binary packages copied across as necessary.

.. _Offline-python-packages:

Installation of Python packages
-------------------------------

In a console, navigate to the directory that your ``openremfiles`` directory is in. If the name if the directory is
different to ``openremfiles``, then change the name in the commands below as appropriate. Ensure the virtualenv has been
activated.

Ensure ``pip`` is up to date:

.. code-block:: console

    $ pip3 install --no-index --find-links=openremfiles pip -U

Install OpenREM and its dependencies:

.. code-block:: console

    $ pip3 install --no-index --find-links=openremfiles openrem

Resuming the installation
-------------------------

Now return to

* :ref:`Linux-DB` for installing on Linux servers
* :ref:`upgrade-linux-local-settings` for upgrading on Linux servers
* ``placeholder`` for installing on Windows servers
* ``placeholder`` for upgrading on Windows servers
