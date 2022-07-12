************
Installation
************

**Document not ready for translation**

Installation options
====================

There are three supported installation options for OpenREM v1.0:

* Docker
* Native install on Linux
* Native install on Windows

Docker
------

This is the quickest and easiest way of installing a fully functioning OpenREM instance, complete with database,
web server and DICOM server, on any operating system that supports Docker with Linux containers. This includes Windows
10 with Docker Desktop, but currently excludes Windows Server, though this may change with availability of WSL2 for
Windows Server 2022.

The Docker installation has mostly been tested with Ubuntu server, but has also been used successfully with Podman on
Redhat Enterprise Linux and other distributions.

Existing Windows or Linux installations of OpenREM 0.10 can be upgraded to run in a Docker installation.

It is advisable that the server OpenREM is installed on has access to the internet to get images from Docker and
security updates for the operating system. However, if this is not possible the Docker images can be obtained on a
computer that does have access to the internet and transferred to the 'offline' serer for installation.

..  toctree::
    :maxdepth: 1

    install_docker
    upgrade_docker
    install_offline_docker

Native install on Linux
-----------------------

A native installation on Linux requires Python, a webserver (eg Nginx) a database (ideally PostgreSQL) and a DICOM
server (ideally Orthanc) to be installed, with OpenREM and all the other dependencies being installed via Pip.

Existing installations of OpenREM 0.10 can be upgraded, but this release requires a different version of Python to
the older releases, and some services that were previously required are no longer needed. Full upgrade instructions are
provided, based on an Ubuntu Server installation.

..  toctree::
    :maxdepth: 1

    install_linux
    upgrade_linux
    upgrade_linux_new_server
    install_offline
    upgrade_offline

Native install on Windows
-------------------------

A native installation on Windows Server requires Python, a database (ideally PostgreSQL) and a DICOM server (ideally
Orthanc) to be installed, with OpenREM and all the other dependencies being installed via Pip. IIS is the recommended
webserver to use on Windows.

This installation process can be used with Windows 10 (and probably 11), but this is not advised for production use as
Windows 10 and 11 are not designed to be servers.

As for native Linux installs, existing installations of OpenREM 0.10 can be upgraded, but this release requires a
different version of Python to the older releases, and some services that were previously required are no longer needed.
Full upgrade instructions are provided, based on a Windows Server 2019 installation.

..  toctree::
    :maxdepth: 1

    install_windows
    upgrade_windows
    install_offline
    upgrade_offline



.. _databaselinks:

Databases
=========

..  toctree::
    :maxdepth: 2

    database

.. _webservers:

Advanced server configuration
=============================

..  toctree::
    :maxdepth: 2

    webserver_config
    virtual_directory

