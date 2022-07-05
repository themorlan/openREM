*******************************
Upgrading to a new Linux server
*******************************

If OpenREM has been running on an older Linux distribution, or you wish to move to Linux for host OpenREM and don't want
to use Docker, these instructions will guide you through upgrading an existing database to a new server.

This install is based on Ubuntu 22.04 using:

* Python 3.10 running in a virtualenv
* Database: PostgreSQL
* DICOM Store SCP: Orthanc running on port 104
* Webserver: NGINX with Gunicorn
* All OpenREM files in ``/var/dose/`` with group owner of ``openrem``
* Collects any Physics (QA) images and zips them

Preparation
===========

Start by following the :doc:`install_linux` instructions until the :ref:`Linux-DB` section, then return here.

.. _Upgrade Linux new server DB migration:

Database migration
==================

Get the data from the old server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get `local_settings.py` file from:

* Ubuntu 'One page install': `/var/dose/veopenrem/lib/python2.7/site-packages/openrem/openremproject/local_settings.py`
* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/openremproject/local_settings.py``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/openremproject/local_settings.py``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/openremproject/local_settings.py``
* Windows: ``C:\Python27\Lib\site-packages\openrem\openremproject\local_settings.py``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\openremproject\local_settings.py``


Export the old database on the old server - you will need the password for `openremuser` that will be in your
`local_settings.py` file:

.. code-block:: console

    $ pg_dump -U openremuser -d openremdb -F c -f pre-1-0-upgrade-dump.bak

etc etc

Complete the setup
==================

Return to the main Linux installation instructions at the :ref:`Install Linux webserver` section.
