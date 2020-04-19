Docker env configuration
========================

Edit the ``.env.prod`` file to customise your installation. There should be no space between the variable name, the
``=`` and the value. Everything after the ``=`` until the end of the line is transferred as the value. These settings
take effect when docker-compose is started or restarted.

Variables that should always be changed
---------------------------------------

.. code-block:: none

    SECRET_KEY=

    Set a new secret key. Create your own, or generate one by using a tool like
    http://www.miniwebtool.com/django-secret-key-generator/ for this.

.. code-block:: none

    DJANGO_ALLOWED_HOSTS=

    Should be a single string of hosts with a space between each. For example:
    ``DJANGO_ALLOWED_HOSTS=localhost 127.0.0.1 [::1] myservername``


Variables to help with debugging problems
-----------------------------------------

.. code-block:: none

    DEBUG=

Set to 1 to enable Django debugging mode.

.. code-block:: none

    LOG_LEVEL=
    LOG_LEVEL_QRSCU=
    LOG_LEVEL_TOSHIBA=

Set the log level. Options are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, and ``CRITICAL``, with
progressively less logging.

Variables to be changed for your environment
--------------------------------------------

E-mail server settings
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

    EMAIL_HOST=
    EMAIL_PORT=
    EMAIL_HOST_USER=
    EMAIL_HOST_PASSWORD=
    EMAIL_USE_TLS=
    EMAIL_USE_SSL=
    EMAIL_DOSE_ALERT_SENDER=
    EMAIL_OPENREM_URL=

The host name and port of the e-mail server that you wish to use must be entered in the ``EMAIL_HOST`` and
``EMAIL_PORT`` fields. ``EMAIL_HOST`` might be your institution's Outlook/Exchange server.

If the e-mail server is set to only allow authenticated users to send messages then a suitable user and password
must be entered in the ``EMAIL_HOST_USER`` and ``EMAIL_HOST_PASSWORD`` fields. If this approach is used then it
may be useful to request that an e-mail account be created specifically for sending these OpenREM alert messages.

It may be possible to configure the e-mail server to allow sending of messages that originate from the OpenREM
server without authentication, in which case the user and password settings should not be required.

The ``EMAIL_USE_TLS`` and ``EMAIL_USE_TLS`` options should be configured to match the encryption requirements of
the e-mail server.

The ``EMAIL_DOSE_ALERT_SENDER`` should contain the e-mail address that you want to use as the sender address.

The ``EMAIL_OPENREM_URL`` must contain the URL of your OpenREM installation in order for hyperlinks in the e-mail
alert messages to function correctly.

Regionalisation
^^^^^^^^^^^^^^^

Local time zone for this installation. Choices can be found here:
http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
although not all choices may be available on all operating systems:

    .. code-block:: none

        TIME_ZONE=Europe/London

Language code for this installation. All choices can be found here:
http://www.i18nguy.com/unicode/language-identifiers.html

    .. code-block:: none

        LANGUAGE_CODE=en-us

If you set this to False, Django will make some optimizations so as not to load the internationalization machinery:

    .. code-block:: none

        USE_I18N=True

If you set this to False, Django will not format dates, numbers and calendars according to the current locale:

    .. code-block:: none

        USE_L10N=True

If you set this to False (default), Django will not use timezone-aware datetimes:

    .. code-block:: none

        USE_TZ=False

XLSX date and time settings for exports:

    .. code-block:: none

        XLSX_DATE=dd/mm/yyyy
        XLSX_TIME=hh:mm:ss

Variables that should only be changed if you know what you are doing
--------------------------------------------------------------------

.. code-block:: none

    ## Database settings
    SQL_HOST=db
    SQL_ENGINE=django.db.backends.postgresql
    SQL_PORT=5432
    DATABASE=postgres
    POSTGRES_USER=openrem_user
    POSTGRES_PASSWORD=openrem_pass
    POSTGRES_DB=openrem_prod

    ## Paths
    MEDIA_ROOT=/home/app/openrem/mediafiles
    STATIC_ROOT=/home/app/openrem/staticfiles
    LOG_ROOT=/logs

    ## RabbitMQ/Celery/Flower
    BROKER_MGMT_URL=http://broker:15672/
    FLOWER_URL=http://flower

Variables that shouldn't be changed
-----------------------------------

Changing this will mean some OpenREM functions will fail

.. code-block:: none

    DOCKER_INSTALL=1
