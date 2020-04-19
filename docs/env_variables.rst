Docker env configuration
========================

Edit the ``.env.prod`` file to customise your installation. There should be no space between the variable name, the
``=`` and the value. Everything after the ``=`` until the end of the line is transferred as the value. These settings
take effect when docker-compose is started or restarted.

Variables that should always be changed
---------------------------------------

* ``SECRET_KEY=``

    Set a new secret key. Create your own, or generate one by using a tool like
    http://www.miniwebtool.com/django-secret-key-generator/ for this.

* ``DJANGO_ALLOWED_HOSTS=``

    Should be a single string of hosts with a space between each. For example:
    ``DJANGO_ALLOWED_HOSTS=localhost 127.0.0.1 [::1] myservername``


Variables to help with debugging problems
-----------------------------------------

.. sourcecode::

    DEBUG=

Set to 1 to enable Django debugging mode.

.. sourcecode::

    LOG_LEVEL=``
    LOG_LEVEL_QRSCU=``
    LOG_LEVEL_TOSHIBA=``

Set the log level. Options are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, and ``CRITICAL``, with
progressively less logging.

Variables to be changed for your environment
--------------------------------------------

E-mail server settings
^^^^^^^^^^^^^^^^^^^^^^

.. sourcecode::

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

