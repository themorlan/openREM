Docker install
==============

Preparation
-----------
* Install Docker and Docker Compose (may be installed automatically with Docker)
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

If you want to run the OpenREM in a virtual directory (like http://server/dms/) there is further configuration to be
done - go to :doc:`virtual_directory`.

Troubleshooting
^^^^^^^^^^^^^^^

If there is a timeout on the first run and some of the containers fail to start, take them down again, make sure
none are running, and try again.

To see how to do this, and for other Docker commands, look at :doc:`troubleshooting`.
