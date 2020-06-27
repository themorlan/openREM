***************************
Offline Docker installation
***************************

OpenREM can be run on a server that is not connected to the internet if required, though access to
https://hub.docker.com would make installation and upgrades much easier.

The server will need to have Docker installed.

Collect installation files
==========================

On a computer with internet access:

* Install Docker - this is required to download the images
* Download https://bitbucket.org/openrem/docker/get/develop.zip
* Download and save the Docker images as tar files:

.. code-block:: none

    docker pull openrem/openrem:develop
    docker pull postgres:12.0-alpine
    docker pull openrem/nginx
    docker pull rabbitmq:3-management-alpine
    docker pull openrem/orthanc

    docker save -o openrem.tar openrem/openrem:develop
    docker save -o openrem-postgres.tar postgres:12.0-alpine
    docker save -o openrem-nginx.tar openrem/nginx
    docker save -o openrem-rabbitmq.tar rabbitmq:3-management-alpine
    docker save -o openrem-orthanc.tar openrem/orthanc

If both the computer with internet access and the target server are Linux or MacOS the images can be made smaller using
gzip, for example:

.. code-block:: none

    docker save openrem/openrem:develop | gzip > openrem.tar.gz

Copy all the tar files and the zip file to the server where OpenREM is to be installed.

Load the docker images
======================

On the server where OpenREM is to be installed, in the folder containing the Docker images:

.. code-block:: none

    docker load -i openrem.tar
    docker load -i openrem-postgres.tar
    docker load -i openrem-nginx.tar
    docker load -i openrem-rabbitmq.tar
    docker load -i openrem-orthanc.tar

If you have compressed the images with gzip the command is the same but with the ``.gz`` suffix, for example:

.. code-block:: none

    docker load -i openrem.tar.gz

Check that the images have been loaded:

.. code-block:: none

    docker images

Continue to :ref:`dockerinstall`
