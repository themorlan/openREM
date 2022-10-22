#######################
Database administration
#######################

**Document not ready for translation**

********************
Docker installations
********************

Database backup
===============

* Open a shell (command prompt) in the Docker folder

.. code-block:: console

    $ docker-compose exec db pg_dump -U openremuser -d openrem_prod -F c -f /db_backup/openremdump.bak

* To automate a regular backup (**recommended**) adapt the following command in a bash script:

.. code-block:: bash

    #!/bin/bash
    TODAY=$(date "+%Y-%m-%d")
    docker-compose -f /path/to/docker-compose.yml exec db pg_dump -U openremuser -d openrem_prod -F c -f "/db_backup/openremdump-"$TODAY".bak"

* or powershell script:

.. code-block:: powershell

    $dateString = "{0:yyyy-MM-dd}" -f (get-date)
    docker-compose -f C:\Path\To\docker-compose.yml exec db pg_dump -U openremuser -d openrem_prod -F c -f /db_backup/openremdump-$dateString.bak

You will need to ensure the backups are either regularly deleted/moved, or overwritten so that the backups don't fill
the disk.

Database restore
================

To restore a database backup to a new Docker container, install using the :doc:`installation` instructions and bring
the containers up, but don't run the database commands. These instructions can also be used to create a duplicate
server on a different system for testing or other purposes.

* Requires exactly the same version of OpenREM to be installed as the database was exported from
* Copy the database backup to the ``db_backup/`` folder of the new install (the name is assumed to be
  ``openremdump.bak``, change as necessary)
* Open a shell (command prompt) in the new install folder (where ``docker-compose.yml`` is)

.. code-block:: console

    $ docker-compose exec db pg_restore --no-privileges --no-owner -U openremuser -d openrem_prod /db_backup/openremdump.bak

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

Advanced
========

These methods should not be required in normal use; only do this if you know what you are doing!

psql
^^^^

Start the PostgreSQL console:

.. code-block:: console

    $ docker-compose exec db psql -U openremuser openrem_prod

.. sourcecode:: psql

    -- List users
    \du

    -- List databases
    \l

    -- Exit the console
    \q

pgAdmin or other PostgreSQL connections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

*******************
Linux installations
*******************

.. _backup-psql-db:

Database backup
===============

* Check the database username and change in the command below as necessary (``openremuser``)
* Check the database name and change in the command below as necessary (``openremdb``)
* You will need the password for ``openremuser``

* Ad hoc:

.. code-block:: console

    $ sudo -u postgres pg_dump -U openremuser -d openremdb -F c -f openremdump.bak

* To automate a regular backup (**recommended**) adapt the following command in a bash script:

.. sourcecode:: bash

    #! /bin/bash
    rm -rf /path/to/db/backups/*
    PGPASSWORD="mysecretpassword" /usr/bin/pg_dump -U openremuser -d openremdb -F c -f /path/to/db/backups/openremdump.bak

.. _restore-psql-linux:

Database restore
================

* Requires the same version of OpenREM to be installed as the database was exported from,
  unless you are :doc:`upgrade_linux` or :doc:`upgrade_linux_new_server`.
* Username can be changed on restore by specifying the new user in the restore command. The user must
  exist in PostgreSQL though - ``sudo -u postgres createuser -P openremuser`` if required
* ``openrem/remapp/migrations/`` should be empty except ``__init__.py``

.. sourcecode:: console

    $ sudo -u postgres createdb -T template0 new_openremdb_name
    $ sudo -u postgres pg_restore --no-privileges --no-owner -U openremuser -d new_openremdb_name path-to/openremdump.bak

* Update the ``local_settings.py`` file with the new database details, as per :ref:`updatelinuxconfig`
* Set up the new database with Django/OpenREM after activating the virtualenv and moving to the
  ``site-packages/openrem`` folder:

.. sourcecode:: console

    $ python manage.py migrate --fake-initial
    $ python manage.py migrate remapp --fake
    $ python manage.py makemigrations remapp
    $ python manage.py migrate

.. _database-windows:

*********************
Windows installations
*********************

Database backup
===============

* Check the database username and change in the command below as necessary (``openremuser``)
* Check the database name and change in the command below as necessary (``openremdb``)
* You will need the password for ``openremuser``
* You will need to edit the command for the path to ``pg_dump.exe`` - the ``14`` is likely to be a lower number

* Ad hoc:

.. code-block:: console

    C:\Users\openrem>"c:\Program Files\PostgreSQL\14\bin\pg_dump.exe" -U openremuser -d openremdb -F c -f windump.bak

* To automate a regular backup (**recommended**) adapt the following command in a bat script:

.. warning::

    Content to be added!

Database restore
================

* Requires the same version of OpenREM to be installed as the database was exported from,
  unless you are :doc:`upgrade_windows` or :doc:`upgrade_windows_new_server`.
* Username can be changed on restore by specifying the new user in the restore command. The user must
  exist in PostgreSQL though - create the user in pgAdmin if required
* ``openrem\remapp\migrations\`` should be empty except ``__init__.py``

.. code-block::

    C:\Users\openrem>"c:\Program Files\PostgreSQL\14\bin\pg_restore.exe" --no-privileges --no-owner -U openremuser -d openremdb -W windump.bak

* Update the ``local_settings.py`` file with the new database details, as per :ref:`updatewindowsconfig`
* Set up the new database with Django/OpenREM after activating the virtualenv and moving to the
  ``site-packages\openrem`` folder:

.. code-block:: console

    (venv) E:\venv\Lib\site-packages\openrem>python manage.py migrate --fake-initial
    (venv) E:\venv\Lib\site-packages\openrem>python manage.py migrate remapp --fake
    (venv) E:\venv\Lib\site-packages\openrem>python manage.py makemigrations remapp
    (venv) E:\venv\Lib\site-packages\openrem>python manage.py migrate

