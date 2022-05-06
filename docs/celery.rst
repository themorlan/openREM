########################################
Celery and Flower configuration - Legacy
########################################

**Document not ready for translation**

Celery and RabbitMQ were used by OpenREM to run tasks in the background, like exports and DICOM queries. RabbitMQ
handles the message queue and Celery provides the 'workers' that perform the tasks. Since release 1.0 OpenREM
uses standard python multiprocessing - hence none of those tools is required anymore.

..  toctree::
    :maxdepth: 2

    celery-windows
    celery-linux

.. _celery_concurrency:

Celery concurrency
^^^^^^^^^^^^^^^^^^

Set the number of workers (concurrency, ``-c``) according to how many processor cores you have available. The more you
have, the more processes (imports, exports, query-retrieve operations etc) can take place simultaneously. However, each
extra worker uses extra memory and if you have too many they will be competing for CPU resources too.

.. admonition:: Problems with Celery 4 on Windows

    Full support for Celery on Windows was dropped with version 4 due to lack of Windows based developers. Therefore
    for Windows the instructions fix Celery at version ``3.1.25`` to retain full functionality.

To stop the celery queues in Linux:

.. sourcecode:: console

    celery multi stop default --pidfile=/path/to/media/celery/%N.pid

For Windows, just press ``Ctrl+c``

You will need to do this twice if there are running tasks you wish to kill.

For production use, see `Daemonising Celery`_ below.


.. _one_page_linux_celery:

Celery and Flower
^^^^^^^^^^^^^^^^^

First, create a Celery configuration file:

.. code-block:: console

    $ nano /var/dose/celery/celery.conf

.. code-block:: bash

    # Name of nodes to start
    CELERYD_NODES="default"

    # Absolute or relative path to the 'celery' command:
    CELERY_BIN="/var/dose/veopenrem3/bin/celery"

    # App instance to use
    CELERY_APP="openremproject"

    # How to call manage.py
    CELERYD_MULTI="multi"

    # Extra command-line arguments to the worker
    CELERYD_OPTS="-O=fair --queues=default"

    # - %n will be replaced with the first part of the nodename.
    # - %I will be replaced with the current child process index
    #   and is important when using the prefork pool to avoid race conditions.
    CELERYD_PID_FILE="/var/dose/celery/%n.pid"
    CELERYD_LOG_FILE="/var/dose/log/%n%I.log"
    CELERYD_LOG_LEVEL="INFO"

    # Flower configuration options
    FLOWER_PORT=5555
    FLOWER_LOG_PREFIX="/var/dose/log/flower.log"
    FLOWER_LOG_LEVEL="INFO"

Now create the systemd service files:

.. code-block:: console

    $ sudo nano /etc/systemd/system/openrem-celery.service

.. code-block:: bash

    [Unit]
    Description=Celery Service
    After=network.target

    [Service]
    Type=forking
    Restart=on-failure
    User=www-data
    Group=www-data
    EnvironmentFile=/var/dose/celery/celery.conf
    WorkingDirectory=/var/dose/veopenrem3/lib/python3.8/site-packages/openrem
    ExecStart=/bin/sh -c '${CELERY_BIN} multi start ${CELERYD_NODES} \
      -A ${CELERY_APP} --pidfile=${CELERYD_PID_FILE} \
      --logfile=${CELERYD_LOG_FILE} --loglevel=${CELERYD_LOG_LEVEL} ${CELERYD_OPTS}'
    ExecStop=/bin/sh -c '${CELERY_BIN} multi stopwait ${CELERYD_NODES} \
      --pidfile=${CELERYD_PID_FILE}'
    ExecReload=/bin/sh -c '${CELERY_BIN} multi restart ${CELERYD_NODES} \
      -A ${CELERY_APP} --pidfile=${CELERYD_PID_FILE} \
      --logfile=${CELERYD_LOG_FILE} --loglevel=${CELERYD_LOG_LEVEL} ${CELERYD_OPTS}'

    [Install]
    WantedBy=multi-user.target

.. code-block:: console

    $ sudo nano /etc/systemd/system/openrem-flower.service

.. code-block:: bash

    [Unit]
    Description=Flower Celery Service
    After=network.target

    [Service]
    User=www-data
    Group=www-data
    EnvironmentFile=/var/dose/celery/celery.conf
    WorkingDirectory=/var/dose/veopenrem3/lib/python3.8/site-packages/openrem
    ExecStart=/bin/sh -c '${CELERY_BIN} flower -A ${CELERY_APP} --port=${FLOWER_PORT} \
      --address=127.0.0.1 --log-file-prefix=${FLOWER_LOG_PREFIX} --loglevel=${FLOWER_LOG_LEVEL}'
    Restart=on-failure
    Type=simple

    [Install]
    WantedBy=multi-user.target

Now register, set to start on boot, and start the services:

.. code-block:: console

    $ sudo systemctl daemon-reload
    $ sudo systemctl enable openrem-celery.service
    $ sudo systemctl start openrem-celery.service
    $ sudo systemctl enable openrem-flower.service
    $ sudo systemctl start openrem-flower.service


Enable RadbbitMQ queue management interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ sudo rabbitmq-plugins enable rabbitmq_management

**Optional -- RabbitMQ Administrator**

This is not required unless you wish to interact with the RabbitMQ management interface directly. Most
functions can be carried out in the OpenREM interface instead. If you do wish to create a user for this
purpose, see the general instructions to :ref:`enableRabbitMQ`.

Log locations
^^^^^^^^^^^^^

* OpenREM: ``/var/dose/log/``
* Celery: ``/var/dose/log/default.log``
* Celery systemd: ``sudo journalctl -u openrem-celery``
* NGINX: ``/var/log/nginx/``
* Orthanc: ``/var/log/orthanc/Orthanc.log``
* Gunicorn systemd: ``sudo journalctl -u openrem-gunicorn``


.. _`WinSCP`: https://winscp.net
