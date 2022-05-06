########################################
Celery and Flower configuration - Legacy
########################################

**Document not ready for translation**

Celery and RabbitMQ are used by OpenREM to run tasks in the background, like exports and DICOM queries. RabbitMQ
handles the message queue and Celery provides the 'workers' that perform the tasks. Since release 1.0 OpenREM
uses standard python multiprocessing - hence none of those tools is required anymore.

..  toctree::
    :maxdepth: 2

    celery-windows
    celery-linux

.. _celery_concurrency:

Linux-only non-Docker
^^^^^^^^^^^^^^^^^^^^^

Change the number of workers by specifying the concurrency in the Celery conf file ``CELERYD_OPTS``.

.. code-block:: console

    $ nano /var/dose/celery/celery.conf

.. code-block:: bash

    CELERYD_OPTS="-O=fair --queues=default --concurrency=4"

