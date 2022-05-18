########################################
Celery and Flower - Legacy
########################################

**Document not ready for translation**

Celery and RabbitMQ were used by OpenREM to run tasks in the background, like exports and DICOM queries.
Since release 1.0 OpenREM uses standard python multiprocessing - hence none of those tools is required anymore.
RabbitMQ handles the message queue and Celery provides the 'workers'  that perform the tasks. 

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


Log locations
^^^^^^^^^^^^^

* OpenREM: ``/var/dose/log/``
* Celery: ``/var/dose/log/default.log``
* Celery systemd: ``sudo journalctl -u openrem-celery``
* NGINX: ``/var/log/nginx/``
* Orthanc: ``/var/log/orthanc/Orthanc.log``
* Gunicorn systemd: ``sudo journalctl -u openrem-gunicorn``

.. _celery-task-queue:

Celery task queue
=================

Celery will have been automatically installed with OpenREM, and along with
RabbitMQ allows for asynchronous task processing for imports, exports and DICOM networking tasks.

..  Note::

    Celery needs to be able to write to the place where the Celery logs and pid file are to be stored, so make sure:

    * the folder exists (the suggestion below is to create a folder in the ``MEDIA_ROOT`` location)
    * the user that starts Celery can write to that folder

You can put the folder wherever you like, for example you might like to create a ``/var/log/openrem/`` folder on a linux
system.

If you are using the built-in Test web server then Celery and the webserver will be running as your user. If you are
running a production webserver, such as Apache or nginx on linux, then the user that runs those daemons will need to
be able to write to the ``MEDIA_ROOT`` and the Celery log files folder. In this case, you need to change the ownership
of the folders and change to the right user before running Celery. On Ubuntu:

.. sourcecode:: console

    mkdir /path/to/media/celery  # change as appropriate
    sudo chown www-data /path/to/media  # change as appropriate
    sudo su -p www-data

Now start celery...

Move into the openrem folder:

* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/`` (remember to activate the virtualenv)
* Windows: ``C:\Python27\Lib\site-packages\openrem\``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\`` (remember to activate the virtualenv)

Linux - ``\`` is the line continuation character:

.. sourcecode:: console

    celery multi start default -Ofair -A openremproject -c 4 -Q default \
    --pidfile=/path/to/media/celery/%N.pid --logfile=/path/to/media/celery/%N.log

Windows - ``celery multi`` doesn't work on Windows, and ``^`` is the continuation character:

.. sourcecode:: console

    celery worker -n default -Ofair -A openremproject -c 4 -Q default ^
    --pidfile=C:\path\to\media\celery\default.pid --logfile=C:\path\to\media\celery\default.log

.. _start_flower:

Celery task management: Flower
==============================

Flower will have been automatically installed with OpenREM and enables monitoring and management of Celery tasks.

You should start Flower with the same user that you started Celery with, and put the log file in the same place too.

Move into the openrem folder:

* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/`` (remember to activate the virtualenv)
* Windows: ``C:\Python27\Lib\site-packages\openrem\``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\`` (remember to activate the virtualenv)

If you need to change the default port from 5555 then you need to make the same change in
``openremproject\local_settings.py`` to add/modify the line ``FLOWER_PORT = 5555``

If you wish to be able to use the Flower management interface independently of OpenREM, then omit the ``--address``
part of the command. Flower will then be available from any PC on the network at http://yourdoseservernameorIP:5555/

Linux - ``\`` is the line continuation character:

.. sourcecode:: console

    celery flower -A openremproject --port=5555 --address=127.0.0.1  --loglevel=INFO \
    ---log-file-prefix=/path/to/media/celery/flower.log

Windows - ``^`` is the line continuation character:

.. sourcecode:: console

    celery flower -A openremproject --port=5555 --address=127.0.0.1  --loglevel=INFO ^
    ---log-file-prefix=C:\path\to\media\celery\flower.log

.. _celery-beat:

Celery periodic tasks: beat
===========================

.. note::

    Celery beat is only required if you are using the :ref:`nativestore`. Please read the warnings there before deciding
    if you need to run Celery beat. At the current time, using a third party DICOM store service is recommended for
    most users. See the :ref:`configure_third_party_DICOM` documentation for more details

Celery beat is a scheduler. If it is running, then every 60 seconds a task is run to check if any of the DICOM
Store SCP nodes are set to ``keep_alive``, and if they are, it tries to verify they are running with a DICOM echo.
If this is not successful, then the Store SCP is started.

To run celery beat, open a new shell and move into the openrem folder:

* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/`` (remember to activate the virtualenv)
* Windows: ``C:\Python27\Lib\site-packages\openrem\``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\`` (remember to activate the virtualenv)

Linux::

    celery -A openremproject beat -s /path/to/media/celery/celerybeat-schedule \
    -f /path/to/media/celery/celerybeat.log \
    --pidfile=/path/to/media/celery/celerybeat.pid

Windows::

    celery -A openremproject beat -s C:\path\to\media\celery\celerybeat-schedule ^
    -f C:\path\to\media\celery\celerybeat.log ^
    --pidfile=C:\path\to\media\celery\celerybeat.pid

As with starting the Celery workers, the folder that the pid, log and for beat, schedule files are to be written
**must already exist** and the user starting Celery beat must be able write to that folder.

To stop Celery beat, just press ``Ctrl+c``

.. _user-settings: