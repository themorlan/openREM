#######################
Database administration
#######################

**Document not ready for translation**

***************
Database backup
***************

* Open a shell (command prompt) in the Docker folder

.. code-block:: console

    $ docker-compose exec db pg_dump -U openrem_user -d openrem_prod -F c -f /db_backup/openremdump.bak

* To automate a regular backup (**recommended**) adapt the following command in a bash script:

.. code-block:: bash

    #!/bin/bash
    TODAY=$(date "+%Y-%m-%d")
    docker-compose -f /path/to/docker-compose.yml exec db pg_dump -U openrem_user -d openrem_prod -F c -f "/db_backup/openremdump-"$TODAY".bak"

* or powershell script:

.. code-block:: powershell

    $dateString = "{0:yyyy-MM-dd}" -f (get-date)
    docker-compose -f C:\Path\To\docker-compose.yml exec db pg_dump -U openrem_user -d openrem_prod -F c -f /db_backup/openremdump-$dateString.bak

You will need to ensure the backups are either regularly deleted/moved, or overwritten so that the backups don't fill
the disk.

****************
Database restore
****************

To restore a database backup to a new Docker container, install using the :doc:`installation` instructions and bring
the containers up, but don't run the database commands. These instructions can also be used to create a duplicate
server on a different system for testing or other purposes.

* Requires exactly the same version of OpenREM to be installed as the database was exported from
* Copy the database backup to the ``db_backup/`` folder of the new install (the name is assumed to be
  ``openremdump.bak``, change as necessary)
* Open a shell (command prompt) in the new install folder (where ``docker-compose.yml`` is)

.. code-block:: console

    $ docker-compose exec db pg_restore --no-privileges --no-owner -U openrem_user -d openrem_prod /db_backup/openremdump.bak

You may get an error about the public schema, this is normal.

* Get the database ready and set up Django:

.. code-block:: console

    $ docker-compose exec openrem python manage.py migrate --fake-initial

.. code-block:: console

    $ docker-compose exec openrem python manage.py makemigrations remapp

.. code-block:: console

    $ docker-compose exec openrem python manage.py migrate --fake

.. code-block:: console

    $ docker-compose exec openrem python manage.py collectstatic --noinput --clear

.. code-block:: console

    $ docker-compose exec openrem python django-admin compilemessages

The OpenREM server should now be ready to use again.

********
Advanced
********

These methods should not be required in normal use; only do this if you know what you are doing!

psql
====

Start the PostgreSQL console:

.. code-block:: console

    $ docker-compose exec db psql -U openrem_user openrem_prod

.. sourcecode:: psql

    -- List users
    \du

    -- List databases
    \l

    -- Exit the console
    \q

pgAdmin or other PostgreSQL connections
=======================================

To access the database directly by pgAdmin or other software, the ports must be exposed.

* Edit ``docker-compose.yml`` to add the ports:

.. code-block:: yaml

    db:
      ports:
        - 5432:5432

* If you have a database already running on the host machine, this port will prevent the container
  starting. In this case, change the first number in the pair to an alternative port.
* The service will be accessible on the host machine after the containers are taken down and up again:

.. code-block:: console

    $ docker-compose down
    $ docker-compose up -d

********************************
Linux-only non-Docker PostgreSQL
********************************

.. _backup-psql-db:

Database backup
===============

Ad hoc:

.. code-block:: console

    $ sudo -u postgres pg_dump -U openremuser -d openremdb -F c -f openremdump.bak

Bash script example:

.. sourcecode:: bash

    #! /bin/bash
    rm -rf /path/to/db/backups/*
    PGPASSWORD="mysecretpassword" /usr/bin/pg_dump -U openremuser -d openremdb -F c -f /path/to/db/backups/openremdump.bak

.. _restore-psql-linux:

Database restore
================

* Requires exactly the same version of OpenREM to be installed as the database was exported from
* Requires the same username to have been created in PostgreSQL

    * ``sudo -u postgres createuser -P openremuser`` if required
    * Check ``local_settings.py`` for username previously used!

* ``openrem/remapp/migrations/`` should be empty except ``__init__.py``

.. sourcecode:: console

    sudo -u postgres createdb -T template0 new_openremdb_name
    sudo -u postgres pg_restore -d new_openremdb_name /db_backup/openremdump.bak

* Update the ``local_settings.py`` file with the new database details, as per :ref:`updatelinuxconfig`
* Set up the new database with Django/OpenREM:

.. sourcecode:: console

    python manage.py migrate --fake-initial
    python manage.py makemigrations remapp
    python manage.py migrate remapp --fake

* If this restore was to a new system prior to upgrade, you can now proceed with the upgrade instructions.

.. _database-windows:

*****************************
Legacy - databases on Windows
*****************************

Windows is only a supported platform for OpenREM v1 and later when using Docker. Therefore there
are no instructions for maintenance of databases on that platform, except in the upgrade guide in
the release notes: :ref:`release1-0upgrade`.

For upgrades between older versions, please refer to the docs that accompany those versions.