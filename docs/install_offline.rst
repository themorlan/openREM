*******************************
Offline installation or upgrade
*******************************

In order to install or upgrade OpenREM on a Windows server that does not have access to the internet you will need to
download all the packages and dependencies on another computer and copy them across.

If you have trouble when installing the Python packages on Windows due to incorrect architecture, you may need to either
download on a Windows system similar to the server (matching 32-bit/64-bit), or to download the files from
http://www.lfd.uci.edu/~gohlke/pythonlibs/ instead. Alternatively there are ways to tell ``pip`` to download binary
packages for specific platforms.

It is expected and highly recommended that server operating systems have access to security updates even
when other internet access is blocked.

The instructions that follow are for a Windows server that doesn't have access to the internet. For Linux servers, it
is recommended to allow access to the distribution's repositories to install and update the software. It is technically
possible to use a local repository mirror/cache, or to download all the packages manually, but this is beyond the
scope of these instructions.

An :doc:`install_offline_docker` might be easier on an offline Linux server, once Docker and Docker Compose are
installed.

On a computer with internet access
==================================

Download independent binaries
-----------------------------

Download all the software in the :ref:`windows_install_packages` section except IIS:

* Python
* Orthanc
* PostgreSQL
* gettext
* Pixelmed
* dcmtk
* 7Zip
* Notepad++

Download Python packages from PyPI
----------------------------------

In a console, navigate to a suitable place and create an empty directory to collect all the packages in, then use
``pip`` to download them all - Python 3 (including Pip) will need to be installed on the computer with internet access
to download the packages, ideally Python 3.10:

.. code-block:: console

    C:\Users\me\Desktop> mkdir openremfiles
    C:\Users\me\Desktop> pip3 download -d openremfiles pip
    C:\Users\me\Desktop> pip3 download -d openremfiles openrem
    C:\Users\me\Desktop> pip3 download -d openremfiles wfastcgi

Copy everything to the Server
-----------------------------

* Copy this directory plus the binaries to the offline server.

On the server without internet access
=====================================

Follow the :doc:`install_windows`, :doc:`upgrade_windows_new_server`, or :doc:`upgrade_windows` instructions, installing
the binary packages that were copied across as well as IIS. Return to this page at the **Installing Python packages**
section.

.. _Offline-python-packages:

Installation of Python packages
-------------------------------

.. admonition:: Old instructions, to be replaced

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

* :ref:`windows_config_db_creation` for installing on Windows servers
* ``placeholder`` for upgrading on Windows servers
