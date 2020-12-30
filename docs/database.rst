########
Database
########

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

****************
Database restore
****************

To restore a database backup to a new Docker container, install using the :doc:`installation` instructions and bring
the containers up, but don't run the database commands. These instructions can also be used to create a duplicate
server on a different system for testing or other purposes.

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

    $ docker-compose exec openrem python manage.py migrate remapp --fake

.. code-block:: console

    $ docker-compose exec openrem python manage.py makemigrations remapp

.. code-block:: console

    $ docker-compose exec openrem python manage.py migrate

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

