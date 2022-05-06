************
Installation
************

**Document not ready for translation**

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
* Customise variables in the following two files:

    * ``.env.prod``
    * the ``orthanc_1`` section of ``docker-compose.yml``

* If you are using SELinux, you will also need to edit the nginx and orthanc bind mounts in ``docker-compose.yml``

A full description of the options are found in:

..  toctree::
    :maxdepth: 1

    env_variables
    docker_orthanc
    docker_selinux

Start the containers with:

.. code-block:: console

    $ docker-compose up -d

Get the database and translations ready:

.. code-block:: console

    $ docker-compose exec openrem python manage.py makemigrations remapp --noinput
    $ docker-compose exec openrem python manage.py migrate --noinput
    $ docker-compose exec openrem python manage.py loaddata openskin_safelist.json
    $ docker-compose exec openrem python manage.py collectstatic --noinput --clear
    $ docker-compose exec openrem python manage.py compilemessages
    $ docker-compose exec openrem python manage.py createsuperuser

Open a web browser and go to http://localhost/

Non-Docker alternative
======================

We recommend all installations to use the Docker method described above. However, it is possible to install without
Docker. The instructions are a prescriptive install on Ubuntu:

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

.. _webservers:

Advanced server configuration
=============================

..  toctree::
    :maxdepth: 2

    webserver_config
    virtual_directory

