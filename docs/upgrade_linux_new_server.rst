*******************************
Upgrading to a new Linux server
*******************************

If OpenREM has been running on an older Linux distribution, or you wish to move to Linux to host OpenREM and don't want
to use Docker, these instructions will guide you through upgrading an existing database to a new server.

This install is based on Ubuntu 22.04 using:

* Python 3.10 running in a virtualenv
* Database: PostgreSQL
* DICOM Store SCP: Orthanc running on port 104
* Webserver: NGINX with Gunicorn
* All OpenREM files in ``/var/dose/`` with group owner of ``openrem``
* Collects any Physics (QA) images and zips them

Get the local_settings.py file
==============================

Get ``local_settings.py`` file from the old server - it should be in one of these locations:

* Ubuntu 'One page install': ``/var/dose/veopenrem/lib/python2.7/site-packages/openrem/openremproject/local_settings.py``
* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/openremproject/local_settings.py``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/openremproject/local_settings.py``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/openremproject/local_settings.py``
* Windows: ``C:\Python27\Lib\site-packages\openrem\openremproject\local_settings.py``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\openremproject\local_settings.py``


Export the database
===================

Export the old database on the old server - you will need the password for ``openremuser`` that will be in your
``local_settings.py`` file:

.. code-block:: console

    $ pg_dump -U openremuser -d openremdb -F c -f pre-1-0-upgrade-dump.bak

Transfer the files
==================

Copy these two files to your new server.

Continue on the new server
==========================

Now follow the :doc:`install_linux` instructions looking out for the additional steps for upgrading to a new Linux
server.

