***************
Troubleshooting
***************

General Docker troubleshooting
==============================

All commands should be run from the folder where ``docker-compose.yml`` is.

To list the active containers:

.. code-block:: console

    $ docker-compose ps

To list active containers running anywhere on the system:

.. code-block:: console

    $ docker ps

To start the containers and detach (so you get the command prompt back instead of seeing all the logging):

.. code-block:: console

    $ docker-compose up -d

To stop the containers:

.. code-block:: console

    $ docker-compose down

To see logs of all the containers in follow mode (``-f``) and with timestamps (``-t``):

.. code-block:: console

    $ docker-compose logs -ft

To see logs of just one container in follow mode - use the service name from the ``docker-compose.yml`` file, choose
from ``openrem``, ``worker`` (Celery), ``flower`` (Flower), ``db`` (PostgreSQL), ``nginx`` (web server),
``broker`` (RabbitMQ), ``orthanc_1`` (DICOM server):

.. code-block:: console

    $ docker-compose logs -f orthanc_1

Other Docker errors
===================

.. toctree::
    :maxdepth: 2

    docker_up

Older stuff
===========

..  toctree::
    :maxdepth: 1

    trouble500
    troubledbtlaterality

If you have a modality where every study has one event (usually CT), review

.. toctree::
    :maxdepth: 1

    import_multirdsr

If planar X-ray studies are appearing in fluoroscopy or vice-versa, review

* :doc:`i_displaynames`

For DICOM networking:

* :ref:`qrtroubleshooting` for query retrieve
* :ref:`storetroubleshooting` for DICOM store

For RabbitMQ/task management:

* :doc:`rabbitmq_management`

Log files
=========

Log file location, naming and verbosity were configured in the ``local_settings.py`` configuration - see the
:ref:`local_settings_logfile` configuration docs for details.

If the defaults have not been modified, then there will be three log files in your ``MEDIAROOT`` folder which you
configured at installation. See the install config section on :ref:`mediarootsettings` for details.

The ``openrem.log`` has general logging information, the other two are specific to the DICOM store and DICOM
query-retrieve functions if you are making use of them.

You can increase the verbosity of the log files by changing the log 'level' to ``DEBUG``, or you can decrease the
verbosity to ``WARNING``, ``ERROR``, or ``CRITICAL``. The default is ``INFO``.

Starting again!
===============

If for any reason you want to start again with the database, then this is how you might do it:

SLQite3 database
----------------

* Delete or rename your existing database file (location will be described in your ``local_settings.py`` file)
* :ref:`database_creation`

Any database
------------

These instructions will also allow you to keep any user settings if you use an SQLite3 database.

In a shell/command window, move into the openrem folder:

* Ubuntu linux: ``cd /usr/local/lib/python2.7/dist-packages/openrem/``
* Other linux: ``cd /usr/lib/python2.7/site-packages/openrem/``
* Linux virtualenv: ``cd virtualenvfolder/lib/python2.7/site-packages/openrem/``
* Windows: ``cd C:\Python27\Lib\site-packages\openrem\``
* Windows virtualenv: ``cd virtualenvfolder\Lib\site-packages\openrem\``

Run the django python shell:

.. code-block:: console

    $ python manage.py shell

.. code-block:: python

    >>> from remapp.models import GeneralStudyModuleAttr
    >>> a = GeneralStudyModuleAttr.objects.all()
    >>> a.count()  # Just to see that we are doing something!
    53423

And if you are sure you want to delete all the studies...

.. code-block:: python

    >>> a.delete()
    >>> a.count()
    0

    >>> exit()
