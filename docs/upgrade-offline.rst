***************************************
Upgrade an offline OpenREM installation
***************************************

Upgrading OpenREM requires new Python packages to be available as well as the latest
version of OpenREM. These can be downloaded on any computer with Python 2.7 installed and an internet connection,
though if you have trouble when installing the packages you might need to use a similar computer to the one you are
installing on - same operating system and matching 32-bit or 64-bit.

OpenREM version 0.10 has a minimum Python version of 2.7.9. Use the instructions in the :doc:`upgrade_previous_0.10.0`
release notes to check this before downloading the new OpenREM packages. The latest Python 2.7 installation can be
obtained from https://www.python.org/downloads/windows/

* Follow the link to the 'Latest Python 2 release'
* Download either the ``Windows x86 MSI installer`` for 32-bit Windows or
* Download ``Windows x86-64 MSI installer`` for 64-bit Windows

On a computer with internet access
==================================

In a console, navigate to a suitable place and create a new directory to collect all the packages in, then use pip to
download them all:

.. code-block:: console

    mkdir openremfiles
    pip download -d openremfiles setuptools

Download specific version of Celery:

    **Linux server:**

    .. code-block:: console

        pip download -d openremfiles celery==4.2.2

    **Windows server:**

    .. code-block:: console

        pip download -d openremfiles celery==3.1.25

Download OpenREM and all other dependencies:

.. code-block:: console

    pip download -d openremfiles openrem==0.10.0b2

Copy everything to the OpenREM server
-------------------------------------

* Copy the directory to the OpenREM server

On the OpenREM server without internet access
=============================================

* Back up your database

    * For PostgreSQL on linux you can refer to :ref:`backup-psql-db`
    * For PostgreSQL on Windows you can refer to :doc:`backupRestorePostgreSQL`
    * For a non-production SQLite3 database, simply make a copy of the database file

* Stop any Celery workers

* Consider temporarily disabling your DICOM Store SCP, or redirecting the data to be processed later

* Install the new version of Python 2.7 if applicable

* If you are using a virtualenv, activate it

Upgrade setuptools:

.. code-block:: console

    pip install --no-index --find-links=openremfiles setuptools -U

Install specific version of Celery:

    **Linux server:**

    .. code-block:: console

        pip install celery==4.2.2

    **Windows server:**

    .. code-block:: console

        pip install celery==3.1.25

Install OpenREM:

.. code-block:: console

    pip install --no-index --find-links=openremfiles openrem==0.10.0b2

Now go back to :ref:`update_configuration0100`, migrate the database and finish the upgrade.
