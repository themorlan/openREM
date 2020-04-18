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

* ``DEBUG=``

    Set to 1 to enable Django debugging mode.

* ``LOG_LEVEL=``
* ``LOG_LEVEL_QRSCU=``
* ``LOG_LEVEL_TOSHIBA=``

    Set the log level. Options are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``,
    and ``CRITICAL``, with progressively less logging.
