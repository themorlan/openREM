########################
Upgrade to OpenREM 1.0.0
########################

****************
Headline changes
****************

* Python 3!
* Django 2.2!
* Docker!

*******************
Upgrade preparation
*******************

* These instructions assume you are upgrading from 0.10.0.
* **Upgrades from 0.9.1 or earlier should review** :doc:`upgrade_previous_0.10.0`. -- needs changing


******************************************
Upgrade process from a PostgresQL database
******************************************

Establish existing database details
===================================

Review the current ``local_settings.py`` for the database settings. The file is in:

* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/openremproject/local_settings.py``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/openremproject/local_settings.py``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/openremproject/local_settings.py``
* Windows: ``C:\Python27\Lib\site-packages\openrem\openremproject\local_settings.py``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\openremproject\local_settings.py``


Export the database
===================

* Open a command line window
* Windows: go to Postgres bin folder, for example:

    .. code-block:: none

        cd "C:\Program Files\PostgreSQL\9.6\bin"

* Dump the database:

    * Use the username and database name from ``local_settings.py``
    * Use the password from ``local_settings.py`` when prompted
    * For linux, the command is ``pg_dump`` (no ``.exe``)
    * Set the path to somewhere suitable to dump the exported database file

    .. code-block:: none

        pg_dump.exe -U openremuser -d openremdb -F c -f path/to/export/openremdump.bak

Set up the new installation
===========================

* Install Docker
* Download and extract https://bitbucket.org/openrem/docker/get/develop.zip and open a shell (command window) in the
  new folder
* Customise variables in ``.env.prod`` and in the ``orthanc_1`` section in ``docker-compose.yml`` as necessary.
  Make sure the database user matches the details in the current ``local_settings.py``.
  A full description of the options are found in:

..  toctree::
    :maxdepth: 1

    env_variables
    docker_orthanc

Start the containers with:

.. code-block:: none

    docker-compose up -d

Copy the database backup to the postgres docker container and import it:

.. code-block:: none

    docker cp /path/to/openremdump.bak  openrem-db:/db_backup
    docker-compose exec db pg_restore -U openremuser -d openrem_prod /db_backup/openremdump.bak

Rename the 0.10 upgrade migration file, migrate the database (the steps and fakes are required as it is not a new
database), and create the static files:

.. code-block:: none

    docker-compose exec openrem mv remapp/migrations/0001_initial.py{.1-0-upgrade,}
    docker-compose exec openrem python manage.py migrate --fake-initial
    docker-compose exec openrem python manage.py migrate remapp --fake
    docker-compose exec openrem python manage.py makemigrations remapp
    docker-compose exec openrem python manage.py migrate
    docker-compose exec openrem python manage.py collectstatic --noinput --clear

The new OpenREM installation should now be ready to be used.

***************************************************
Upgrading an OpenREM server with no internet access
***************************************************

Follow the instructions found at :doc:`upgrade-offline`, before returning here to update the configuration, migrate the
database and complete the upgrade.

**********************************************************
Upgrading an OpenREM server that uses a different database
**********************************************************



***************************************************************
Old style, deprecated, to be pruned down for Ubuntu alternative
***************************************************************


Upgrade
=======

* Back up your database

    * For PostgreSQL on linux you can refer to :ref:`backup-psql-db`
    * For PostgreSQL on Windows you can refer to :doc:`backupRestorePostgreSQL`
    * For a non-production SQLite3 database, simply make a copy of the database file

* Stop any Celery workers

* Consider temporarily disabling your DICOM Store SCP, or redirecting the data to be processed later

* Create a new virtualenv with Python 3:

.. code-block:: console

    python3 -m venv virtualenv3
    . virtualenv3/bin/activate
    # add location and Windows alternatives - go with strong recommendation for virtualenv this time...


*Ubuntu one page instructions*::

    sudo systemctl stop openrem-celery
    sudo systemctl stop orthanc
    . /var/dose/veopenrem/bin/activate

* Install the new version of OpenREM:

    .. code-block:: console

        pip install openrem==1.0.0b1

* Install ``gunicorn`` if required.

.. _update_configuration0100:

Update the local_settings.py file
=================================

* Remove the first line ``LOCAL_SETTINGS = True``
* Change second line to ``from .settings import *``
* Compare file to local_settings.py.example to see if there are other sections that should be updated

Migrate the database
====================

In a shell/command window, move into the ``openrem`` folder:

* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/``
* Windows: ``C:\Python27\Lib\site-packages\openrem\``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\``

Prepare the migrations folder:

* Delete everything except ``__init__.py`` in ``remapp/migrations``
* Rename ``0001_initial.py.1-0-upgrade`` to ``0001_initial.py``

.. code-block:: console

    python manage.py migrate --fake-initial
    python manage.py migrate remapp --fake
    python manage.py makemigrations remapp
    python manage.py migrate


Update static files
===================

In the same shell/command window as you used above run the following command to clear the static files
belonging to your previous OpenREM version and replace them with those belonging to the version you have
just installed (assuming you are using a production web server...):

.. code-block:: console

    python manage.py collectstatic --clear

..  admonition:: Virtual directory users

    If you are running your website in a virtual directory, you also have to update the reverse.js file.
    To get the file in the correct path, take care that you insert just after the declaration of
    ``STATIC_ROOT`` the following line in your ``local_settings.py`` (see also the sample ``local_settings.py.example``):

    .. code-block:: console

        JS_REVERSE_OUTPUT_PATH = os.path.join(STATIC_ROOT, 'js', 'django_reverse')

    To update the reverse.js file execute the following command:

    .. code-block:: console

        python manage.py collectstatic_js_reverse

    See  :doc:`virtual_directory` for more details.


Update all the services configurations
======================================

* Change paths to python, celery and flower binaries to Python 3 versions

Restart all the services
========================

Follow the guide at :doc:`startservices`.

    *Ubuntu one page instructions*::

        sudo systemctl start openrem-celery
        sudo systemctl start orthanc
        sudo systemctl restart openrem-gunicorn

.. _post_upgrade0100:


.. _CP1676: https://www.dicomstandard.org/cps/