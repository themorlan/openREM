########################
Upgrade to OpenREM 1.0.0
########################

****************
Headline changes
****************

* Python 3
* Django 2.2
* Docker

* Performing physician added to standard fluoroscopy exports (:issue:`840`)

*******************
Upgrade preparation
*******************

* These instructions assume you are upgrading from 0.10.0.
* **Upgrades from 0.9.1 or earlier should review** :doc:`upgrade_previous_0.10.0`. -- needs changing

..  toctree::
    :maxdepth: 1

    upgrade_previous_0.10.0

.. _post_upgrade0100:

******************************************
Upgrade process from a PostgresQL database
******************************************

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

    $ docker-compose exec db pg_restore --no-privileges --no-owner -U openrem_user -d openrem_prod /db_backup/openremdump.bak

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

    $ docker-compose exec openrem python manage.py collectstatic --noinput --clear

Generate translation binary files

.. code-block:: console

    $ docker-compose exec openrem python manage.py compilemessages

The new OpenREM installation should now be ready to be used.

***************************************************
Upgrading an OpenREM server with no internet access
***************************************************

Follow the instructions found at :doc:`upgrade-offline`, before returning here to update the configuration, migrate the
database and complete the upgrade.

**********************************************************
Upgrading an OpenREM server that uses a different database
**********************************************************



*******************************************
Upgrading without using Docker - linux only
*******************************************

Upgrading without using Docker is not recommended, and not supported on Windows. Instructions are only provided for
Linux and assume a configuration similar to the 'One page complete Ubuntu install' provided with release 0.8.1 and
later.

Preparation
===========

Back up the database:

.. code-block:: console

    $ pg_dump -U openremuser -d openremdb -F c -f pre-1-0-upgrade-dump.bak

Stop any Celery workers, Flower and Gunicorn, disable DICOM Store SCP:

.. code-block:: console

    $ sudo systemctl stop openrem-celery
    $ sudo systemctl stop openrem-flower
    $ sudo systemctl stop openrem-gunicorn
    $ sudo systmectl stop orthanc

Install Python 3.8 and create a new virtualenv:

.. code-block:: console

    $ sudo apt install python3.8 python3.8-dev python3.8-distutils python3.8-venv

.. code-block:: console

    $ cd /var/dose
    $ python3.8 -m venv veopenrem3
    $ . veopenrem3/bin/activate

Install the new version of OpenREM
==================================

.. code-block:: console

    $ pip install --upgrade pip

.. code-block:: console

    $ pip install openrem==1.0.0b1

Update the local_settings.py file
=================================

* Remove the first line ``LOCAL_SETTINGS = True``
* Change second line to ``from .settings import *``
* Compare file to local_settings.py.example to see if there are other sections that should be updated

Migrate the database
====================

In a shell/command window, move into the ``openrem`` folder:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.8/site-packages/openrem/

Prepare the migrations folder:

* Delete everything except ``__init__.py`` and ``0001_initial.py.1-0-upgrade`` in ``remapp/migrations``
* Rename ``0001_initial.py.1-0-upgrade`` to ``0001_initial.py``

.. code-block:: console

    $ rm remapp/migrations/0*.py
    $ rm remapp/migrations/0*.pyc  # may result in 'cannot remove' if there are none
    $ mv remapp/migrations/0001_initial.py{.1-0-upgrade,}

Migrate the database:

.. code-block:: console

    $ python manage.py migrate --fake-initial

.. code-block:: console

    $ python manage.py migrate remapp --fake

.. code-block:: console

    $ python manage.py makemigrations remapp

.. code-block:: console

    $ python manage.py migrate


Update static files
===================

.. code-block:: console

    $ python manage.py collectstatic --clear

..  admonition:: Virtual directory users

    If you are running your website in a virtual directory, you also have to update the reverse.js file.
    To get the file in the correct path, take care that you insert just after the declaration of
    ``STATIC_ROOT`` the following line in your ``local_settings.py`` (see also the sample ``local_settings.py.example``):

    .. code-block:: none

        JS_REVERSE_OUTPUT_PATH = os.path.join(STATIC_ROOT, 'js', 'django_reverse')

    To update the reverse.js file execute the following command:

    .. code-block:: console

        $ python manage.py collectstatic_js_reverse

    See  :doc:`virtual_directory` for more details.

Generate translation binary files

.. code-block:: console

    $ python django-admin compilemessages

Update all the services configurations
======================================

Edit the Gunicorn systemd file ``WorkingDirectory`` and ``ExecStart``:

.. code-block:: console

    $ sudo nano /etc/systemd/system/openrem-gunicorn.service

.. code-block:: none

    WorkingDirectory=/var/dose/veopenrem3/lib/python3.8/site-packages/openrem

    ExecStart=/var/dose/veopenrem3/bin/gunicorn \
        --bind unix:/tmp/openrem-server.socket \
        openremproject.wsgi:application --timeout 300 --workers 4

Edit the Celery configuration file ``CELERY_BIN``:

.. code-block:: console

    $ nano /var/dose/celery/celery.conf

.. code-block:: none

    CELERY_BIN="/var/dose/veopenrem3/bin/celery"

Edit the Celery systemd file ``WorkingDirectory``:

.. code-block:: console

    $ sudo nano /etc/systemd/system/openrem-celery.service

.. code-block:: none

    WorkingDirectory=/var/dose/veopenrem3/lib/python3.8/site-packages/openrem

Edit the Flower systemd file ``WorkingDirectory``:

.. code-block:: console

    $ sudo nano /etc/systemd/system/openrem-flower.service

.. code-block:: none

    WorkingDirectory=/var/dose/veopenrem3/lib/python3.8/site-packages/openrem

Reload systemd and restart the services
=======================================

.. code-block:: console

    $ sudo systemctl daemon-reload
    $ sudo systemctl restart openrem-gunicorn.service
    $ sudo systemctl restart nginx.service
    $ sudo systemctl start openrem-celery.service
    $ sudo systemctl start openrem-flower.service
    $ sudo systemctl start orthanc.service


