#################
Upgrade to Docker
#################

These instructions assume:

* You are upgrading from 0.10.0.
* You are using a PostgreSQL database in the existing installation.
* That existing Linux installs followed the instructions in the previous releases, with the *openrem-function*
  format that changed in the 0.9.1 release (:ref:`service_name_change`).

If not you will need to adapt the instructions as necessary.

* **Upgrades from 0.9.1 or earlier should review** :doc:`upgrade_previous_0.10.0`. -- needs changing

..  toctree::
    :maxdepth: 1

    upgrade_previous_0.10.0

.. _post_upgrade0100:

******************************************
Upgrade process from a PostgresQL database
******************************************

Stop the existing services
==========================

* Linux:

    .. code-block:: console

        $ sudo systemctl stop orthanc
        $ sudo systemctl stop nginx
        $ sudo systemctl stop openrem-gunicorn
        $ sudo systemctl stop openrem-flower
        $ sudo systemctl stop openrem-celery
        $ sudo systemctl stop rabbitmq-server
        $ sudo systemctl disable orthanc
        $ sudo systemctl disable nginx
        $ sudo systemctl disable openrem-gunicorn
        $ sudo systemctl disable openrem-flower
        $ sudo systemctl disable openrem-celery
        $ sudo systemctl disable rabbitmq-server

* Windows: stop the following services

    * Orthanc or Conquest
    * IIS OpenREM site or other webserver
    * Flower
    * Celery
    * RabbitMQ

Establish existing database details
===================================

Review the current ``local_settings.py`` for the database settings and location of the ``MEDIA_ROOT`` folder. The file
is in:

* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/openremproject/local_settings.py``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/openremproject/local_settings.py``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/openremproject/local_settings.py``
* Windows: ``C:\Python27\Lib\site-packages\openrem\openremproject\local_settings.py``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\openremproject\local_settings.py``

.. _release1-0upgrade:

Export the database
===================

* Open a command line window
* Windows: go to Postgres bin folder, for example:

    .. code-block:: console

        $ cd "C:\Program Files\PostgreSQL\9.6\bin"

* Dump the database:

    * Use the username (``-U openremuser``) and database name (``-d openremdb``) from ``local_settings.py``
    * Use the password from ``local_settings.py`` when prompted
    * For linux, the command is ``pg_dump`` (no ``.exe``)
    * Set the path to somewhere suitable to dump the exported database file

    .. code-block:: console

        $ pg_dump.exe -U openremuser -d openremdb -F c -f path/to/export/openremdump.bak

Set up the new installation
===========================

.. _update_configuration0100:

* Install Docker
* Download and extract https://bitbucket.org/openrem/docker/get/develop.zip and open a shell (command window) in the
  new folder
* Customise variables in ``.env.prod``, the ``orthanc_1`` section in ``docker-compose.yml``
  and in ``orthanc_1.json`` as necessary.  A full description of the options are found in:

..  toctree::
    :maxdepth: 1

    env_variables
    docker_orthanc

Start the containers with:

.. code-block:: console

    $ docker-compose up -d

Copy the database backup to the postgres docker container and import it. If you have changed the database variables,
ensure that:

* the database user (``-U openremuser``) matches ``POSTGRES_USER`` in ``.env.prod``
* the database name (``-d openrem_prod``) matches ``POSTGRES_DB`` in ``.env.prod``

They don't have to match the old database settings. The filename in both commands (``openremdump.bak``) should match
your backup filename.

.. code-block:: console

    $ docker cp /path/to/openremdump.bak openrem-db:/db_backup/

.. code-block:: console

    $ docker-compose exec db pg_restore --no-privileges --no-owner -U openremuser -d openrem_prod /db_backup/openremdump.bak

It is normal to get an error about the public schema, for example:

.. code-block:: none

    pg_restore: while PROCESSING TOC:
    pg_restore: from TOC entry 3; 2615 2200 SCHEMA public postgres
    pg_restore: error: could not execute query: ERROR:  schema "public" already exists
    Command was: CREATE SCHEMA public;

    pg_restore: warning: errors ignored on restore: 1

Rename the 0.10 upgrade migration file, migrate the database (the steps and fakes are required as it is not a new
database), and create the static files:

.. code-block:: console

    $ docker-compose exec openrem mv remapp/migrations/0001_initial.py.1-0-upgrade remapp/migrations/0001_initial.py

.. code-block:: console

    $ docker-compose exec openrem python manage.py migrate --fake-initial

.. code-block:: console

    $ docker-compose exec openrem python manage.py migrate remapp --fake

.. code-block:: console

    $ docker-compose exec openrem python manage.py makemigrations remapp

.. code-block:: console

    $ docker-compose exec openrem python manage.py migrate

.. code-block:: console

    $ docker-compose exec openrem python manage.py loaddata openskin_safelist.json

.. code-block:: console

    $ docker-compose exec openrem python manage.py collectstatic --noinput --clear

Generate translation binary files

.. code-block:: console

    $ docker-compose exec openrem python manage.py compilemessages

The new OpenREM installation should now be ready to be used.
