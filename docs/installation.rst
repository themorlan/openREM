************
Installation
************

Windows or Linux: Docker install
=================================

Preparation
-----------
* Install Docker
* Download https://bitbucket.org/openrem/docker/get/develop.zip

.. _dockerinstall:

Install
-------
* Extract the ZIP file and open a shell (command prompt) in the new folder
* Customise variables in the following three files:

    * ``.env.prod``
    * the ``orthanc_1`` section of ``docker-compose.yml``
    * ``orthanc_1.json``

A full description of the options are found in:

..  toctree::
    :maxdepth: 1

    env_variables
    docker_orthanc

Start the containers with:

.. code-block:: console

    $ docker-compose up -d

Get the database and translations ready:

.. code-block:: console

    $ docker-compose exec openrem python manage.py makemigrations remapp --noinput
    $ docker-compose exec openrem python manage.py migrate --noinput
    $ docker-compose exec openrem python manage.py collectstatic --noinput --clear
    $ docker-compose exec openrem python manage.py compilemessages
    $ docker-compose exec openrem python manage.py createsuperuser

Open a web browser and go to http://localhost/

Non-Docker alternative - Linux only
===================================

We recommend all installations to use the Docker method described above. However, it is possible to install without
Docker, but only on Linux. The instructions are a prescriptive install on Ubuntu:

..  toctree::
    :maxdepth: 2

    quick_start_linux


Offline Docker installations
============================

..  toctree::
    :maxdepth: 1

    install-offline


Upgrading an existing installation
==================================

..  toctree::
    :maxdepth: 2

    release-1.0.0
    upgrade-offline

.. _databaselinks:

Databases
=========

..  toctree::
    :maxdepth: 2

    database


Remaining pages to be reviewed to see if any information needs to be retained, then deleted:

..  toctree::
    :maxdepth: 2

    postgresql
    postgresql_windows
    backupRestorePostgreSQL
    backupMySQLWindows

.. _webservers:

Web servers
===========

..  toctree::
    :maxdepth: 1

    webserver_config
    virtual_directory


