###############################
Celery and Flower configuration
###############################

Celery and RabbitMQ are used by OpenREM to run tasks in the background, like exports and DICOM queries. RabbitMQ
handles the message queue and Celery provides the 'workers' that perform the tasks.

.. _celery_concurrency:

Celery concurrency
==================

Multiple tasks can be processed at the same time. By default, Celery sets the concurrency (number of worker processes)
to match the number of CPUs available.

The easiest way to increase the number of workers is to scale up the ``worker`` Docker container.

* Open a shell (command prompt) in the Docker folder

.. code-block:: console

    $ docker-compose up -d --scale worker=2

To reduce the number of workers again:

.. code-block:: console

    $ docker-compose up -d --scale worker=1


Linux-only non-Docker
^^^^^^^^^^^^^^^^^^^^^

Change the number of workers by specifying the concurrency in the Celery conf file ``CELERYD_OPTS``.

.. code-block:: console

    $ nano /var/dose/celery/celery.conf

.. code-block:: bash

    CELERYD_OPTS="-O=fair --queues=default --concurrency=4"

