*********************************
Upgrading to a new Windows server
*********************************

If OpenREM has been running on an older Windows server version, or you wish to move to Windows Server to host OpenREM,
these instructions will guide you through upgrading an existing database to a new server.

This install is based on Windows Server 2022 - for details see the main :doc:`install_windows` docs.

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

Export the old database on the old server - get details from the ``local_settings.py`` file:

* Check the database username and change in the command below as necessary (``openremuser``)
* Check the database name and change in the command below as necessary (``openremdb``)
* You will need the password for ``openremuser``
* You will need to edit the command for the path to ``pg_dump.exe`` - the ``14`` is likely to be a lower number

.. code-block:: console

    C:\Users\openrem>"c:\Program Files\PostgreSQL\14\bin\pg_dump.exe" -U openremuser -d openremdb -F c -f windump.bak

Transfer the files
==================

Copy these two files to your new server.

Continue on the new server
==========================

Now follow the :doc:`install_windows` instructions looking out for the additional steps for upgrading to a new
server.
