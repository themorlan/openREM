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
* Extract the zip file and open a shell (command window) in the new folder
* Customise any variables in ``.env.prod`` and in the ``orthanc_1`` section
  in ``docker-compose.yml`` as necessary. A full description of the options are found in:

..  toctree::
    :maxdepth: 1

    env_variables
    docker_orthanc

Start the containers with:

.. code-block:: none

    $ docker-compose up -d

Get the database ready:

.. code-block:: console

    $ docker-compose exec openrem python manage.py makemigrations remapp --noinput
    $ docker-compose exec openrem python manage.py migrate --noinput
    $ docker-compose exec openrem python manage.py collectstatic --noinput --clear
    $ docker-compose exec openrem python manage.py createsuperuser

Open a web browser and go to http://localhost/

Non-docker alternative - Linux only
===================================

We recommend all installations to use the Docker method described above. However, it is possible to install without
Docker, but only on Linux. The instructions are a prescriptive install on Ubuntu:

..  toctree::
    :maxdepth: 1

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

To be removed here, but we need some of the content in other pages, so leaving for now

..  toctree::
    :maxdepth: 2

    postgresql
    postgresql_windows
    backupRestorePostgreSQL
    backupMySQLWindows

.. _webservers:

Web servers
===========

Don't need this section, but need to work out what to do about virtual_directory installs

..  toctree::
    :maxdepth: 1

    virtual_directory


