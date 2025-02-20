********************************
Upgrading a native Linux install
********************************

These instructions assume a configuration similar to the 'One page complete Ubuntu install' provided with release
0.8.1 and later. If you are running an older distribution, consider upgrading the operating system or migrating
the service to a new host. The test system for these upgrade instructions was upgraded from 18.04 to 20.04 and then
22.04 before the OpenREM upgrade was started. If you are using a different distribution or have set up your system
differently, it might be better to start afresh following or adapting the :doc:`upgrade_linux_new_server` docs instead.

If upgrading to a new host, follow the :doc:`upgrade_linux_new_server` docs.

This release will run on Python 3.8 or 3.9, but Python 3.10 is recommended. If a different release of Python is being
used, substitute 3.10 for that version where necessary below.

If you are upgrading OpenREM on a Linux server with limited internet access, go to the :doc:`install_offline` docs.

* **Upgrades from 0.9.1 or earlier should review** :doc:`upgrade_previous_0.10.0` first. Upgrading to 1.0 is only
  possible from 0.10.0.

Preparation
===========

Back up the database - you will need the password for ``openremuser`` that will be in your
``local_settings.py`` file. You'll need this file again later so open it in a different window:

.. code-block:: console

    $ less /var/dose/veopenrem/lib/python2.7/site-packages/openrem/openremproject/local_settings.py

Backup the database, in the main window:

.. code-block:: console

    $ pg_dump -U openremuser -d openremdb -F c -f pre-1-0-upgrade-dump.bak

Stop any Celery workers, Flower, RabbitMQ, Gunicorn, NGINX, and Orthanc (OpenREM service names will be
reversed if they weren't changed with the 0.9.1 upgrade):

.. code-block:: console

    $ sudo systemctl stop openrem-celery
    $ sudo systemctl stop openrem-flower
    $ sudo systemctl stop openrem-gunicorn
    $ sudo systemctl stop rabbitmq-server
    $ sudo systemctl stop nginx
    $ sudo systemctl stop orthanc

Update apt and install any updates:

.. code-block:: console

    $ sudo -- sh -c 'apt update && apt upgrade'

Install Python 3.10 and other packages:

.. code-block:: console

    $ sudo apt install acl python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip \
    postgresql nginx orthanc dcmtk default-jre zip gettext

Reset the permissions for the ``/var/dose`` folder:

.. code-block:: console

    $ sudo chmod -R 775 /var/dose
    $ sudo chown -R $USER:openrem /var/dose
    $ sudo chmod -R g+s /var/dose/*

Now find the ``uid`` of your user and the ``gid`` of the ``openrem`` group:

.. code-block:: console

    $ id
    $ getent group openrem

Take note of the ``uid`` number and the ``gid`` in the third field of the group information and use it in the next
command, replacing ``1001`` (user ``uid``) and ``1002`` (``openrem`` group ``gid``) as appropriate:

.. code-block:: console

    $ sudo setfacl -PRdm u:1001:rwx,g:1002:rwx,o::r /var/dose/

.. admonition:: What are we doing with the permissions?

    These settings enable the web server user ``www-data``, the DICOM server user ``orthanc`` and the OpenREM server
    users (you and your colleagues) to all read, write and execute the OpenREM files. The ``setfacl`` command
    relies on Access Control Lists being available on your system - they are usually enabled on ext4 and can be
    enabled on others. See :ref:`add_linux_user` for adding colleagues access to the Linux folders.

Create a new Python virtual environment:

.. code-block:: console

    $ python3.10 -m venv /var/dose/veopenrem3

Activate the virtualenv:

.. code-block:: console

    $ . /var/dose/veopenrem3/bin/activate

Install the new version of OpenREM
==================================

Ensure the new virtualenv is active — prompt will look like

.. code-block:: console

    (veopenrem3)username@hostname:~$

Upgrade Pip and install OpenREM

.. code-block:: console

    $ pip install --upgrade pip

.. code-block:: console

    $ pip install openrem==1.0.0b1

.. _upgrade-linux-local-settings:

Configure the local_settings.py file
====================================

Navigate to the Python openrem folder and copy the example ``local_settings.py`` and ``wsgi.py`` files to remove the
``.linux`` and ``.example`` suffixes:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.10/site-packages/openrem/
    $ cp openremproject/local_settings.py{.linux,}
    $ cp openremproject/wsgi.py{.example,}

Review the old ``local_settings.py`` file that was opened earlier - see the first part of the Preparation section. Edit
the new ``local_settings.py`` as needed - make sure you update the database ``NAME``, ``USER`` and ``PASSWORD``, the
``ALLOWED_HOSTS`` list and the ``EMAIL`` configuration and check all the other settings. Change the ``SECRET_KEY`` from
the default:

.. code-block:: console

    $ nano openremproject/local_settings.py

.. code-block:: python
    :emphasize-lines: 4-6, 16-17, 25-28, 51,56,59,70-77

    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': 'openremdb',
            'USER': 'openremuser',
            'PASSWORD': 'mysecretpassword',     # This is the password you set earlier
            'HOST': '',
            'PORT': '',
        }
    }

    MEDIA_ROOT = '/var/dose/media/'

    STATIC_ROOT = '/var/dose/static/'

    # Change secret key
    SECRET_KEY = 'hmj#)-$smzqk*=wuz9^a46rex30^$_j$rghp+1#y&amp;i+pys5b@$'

    # DEBUG mode: leave the hash in place for now, but remove it and the space (so DEBUG
    # is at the start of the line) as soon as something doesn't work. Put it back
    # when you get it working again.
    # DEBUG = True

    ALLOWED_HOSTS = [
        # Add the names and IP address of your host, for example:
        'openrem-server',
        'openrem-server.ad.abc.nhs.uk',
        '10.123.213.22',
    ]

    LOG_ROOT = '/var/dose/log'
    LOG_FILENAME = os.path.join(LOG_ROOT, 'openrem.log')
    QR_FILENAME = os.path.join(LOG_ROOT, 'openrem_qr.log')
    EXTRACTOR_FILENAME = os.path.join(LOG_ROOT, 'openrem_extractor.log')

    # Removed comment hashes to enable log file rotation:
    LOGGING['handlers']['file']['class'] = 'logging.handlers.RotatingFileHandler'
    LOGGING['handlers']['file']['maxBytes'] = 10 * 1024 * 1024  # 10*1024*1024 = 10 MB
    LOGGING['handlers']['file']['backupCount'] = 5  # number of log files to keep before deleting the oldest one
    LOGGING['handlers']['qr_file']['class'] = 'logging.handlers.RotatingFileHandler'
    LOGGING['handlers']['qr_file']['maxBytes'] = 10 * 1024 * 1024  # 10*1024*1024 = 10 MB
    LOGGING['handlers']['qr_file']['backupCount'] = 5  # number of log files to keep before deleting the oldest one
    LOGGING['handlers']['extractor_file']['class'] = 'logging.handlers.RotatingFileHandler'
    LOGGING['handlers']['extractor_file']['maxBytes'] = 10 * 1024 * 1024  # 10*1024*1024 = 10 MB
    LOGGING['handlers']['extractor_file']['backupCount'] = 5  # number of log files to keep before deleting the oldest one

    # Regionalisation settings
    #   Date format for exporting data to Excel xlsx files.
    #   Default in OpenREM is dd/mm/yyyy. Override it by uncommenting and customising below; a full list of codes is
    #   available at https://msdn.microsoft.com/en-us/library/ee634398.aspx.
    # XLSX_DATE = 'mm/dd/yyyy'
    #   Local time zone for this installation. Choices can be found here:
    #   http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
    #   although not all choices may be available on all operating systems.
    #   In a Windows environment this must be set to your system time zone.
    TIME_ZONE = 'Europe/London'
    #   Language code for this installation. All choices can be found here:
    #   http://www.i18nguy.com/unicode/language-identifiers.html
    LANGUAGE_CODE = 'en-us'

    DCMTK_PATH = '/usr/bin'
    DCMCONV = os.path.join(DCMTK_PATH, 'dcmconv')
    DCMMKDIR = os.path.join(DCMTK_PATH, 'dcmmkdir')
    JAVA_EXE = '/usr/bin/java'
    JAVA_OPTIONS = '-Xms256m -Xmx512m -Xss1m -cp'
    PIXELMED_JAR = '/var/dose/pixelmed/pixelmed.jar'
    PIXELMED_JAR_OPTIONS = '-Djava.awt.headless=true com.pixelmed.doseocr.OCR -'

    # E-mail server settings - see https://docs.djangoproject.com/en/2.2/topics/email/
    EMAIL_HOST = 'localhost'
    EMAIL_PORT = 25
    EMAIL_HOST_USER = ''
    EMAIL_HOST_PASSWORD = ''
    EMAIL_USE_TLS = 0         # Use 0 for False, 1 for True
    EMAIL_USE_SSL = 0         # Use 0 for False, 1 for True
    EMAIL_DOSE_ALERT_SENDER = 'your.alert@email.address'
    EMAIL_OPENREM_URL = 'http://your.openrem.server'

Migrate the database
====================

In a shell/command window, move into the ``openrem`` folder:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.10/site-packages/openrem/

Prepare the migrations folder:

* Rename ``0001_initial.py.1-0-upgrade`` to ``0001_initial.py``

.. code-block:: console

    $ mv remapp/migrations/0001_initial.py{.1-0-upgrade,}

Migrate the database:

.. code-block:: console

    $ python manage.py migrate --fake-initial

.. code-block:: console

    $ python manage.py migrate remapp --fake

.. code-block:: console

    $ python manage.py makemigrations remapp

.. admonition:: Rename questions

    There will be some questions about fields being renamed - answer ``N`` to all of them.

.. code-block:: console

    $ python manage.py migrate

.. code-block:: console

    $ python manage.py loaddata openskin_safelist.json


Update static files and translations
====================================

.. code-block:: console

    $ python manage.py collectstatic --clear

.. admonition:: Warning about deleting all files

    You will get a warning about all files in the static files location being deleted. As long as the folder is correct,
    type ``yes`` to continue.

..  admonition:: Virtual directory users

    If you are running your website in a virtual directory, you also have to update the reverse.js file.
    To get the file in the correct path, take care that you insert just after the declaration of
    ``STATIC_ROOT`` the following line in your ``local_settings.py`` (see also the sample ``local_settings.py.example``):

    .. code-block:: none

        JS_REVERSE_OUTPUT_PATH = os.path.join(STATIC_ROOT, 'js', 'django_reverse')

    To update the reverse.js file execute the following command:

    .. code-block:: console

        $ python manage.py collectstatic_js_reverse

    See  :doc:`virtual_directory` for more details.

Generate translation binary files

.. code-block:: console

    $ python manage.py compilemessages

Update all the services configurations
======================================

Edit the Gunicorn systemd file ``WorkingDirectory`` and ``ExecStart``:

.. code-block:: console

    $ sudo nano /etc/systemd/system/openrem-gunicorn.service

.. code-block:: none
    :emphasize-lines: 1,3

    WorkingDirectory=/var/dose/veopenrem3/lib/python3.10/site-packages/openrem

    ExecStart=/var/dose/veopenrem3/bin/gunicorn \
        --bind unix:/tmp/openrem-server.socket \
        openremproject.wsgi:application --timeout 300 --workers 4

Celery, Flower and RabbitMQ are no longer required for this release, so their Systemd control files
can be disabled, and RabbitMQ can be removed (assuming it is not in use for any other services on this
server):

.. code-block:: console

    $ sudo systemctl disable openrem-celery.service
    $ sudo systemctl disable openrem-flower.service

.. code-block:: console

    $ sudo apt remove rabbitmq-server
    $ sudo apt purge rabbitmq-server

Reload systemd and restart the services
=======================================

.. code-block:: console

    $ sudo systemctl daemon-reload

Start and check Gunicorn:

.. code-block:: console

    $ sudo systemctl start openrem-gunicorn.service
    $ sudo systemctl status openrem-gunicorn.service

Start and check NGINX:

.. code-block:: console

    $ sudo systemctl start nginx.service
    $ sudo systemctl status nginx.service

Start and check Orthanc:

.. code-block:: console

    $ sudo systemctl start orthanc.service
    $ sudo systemctl status orthanc.service

.. admonition:: Registered Users error

    If Orthanc fails to start, check the Orthanc log file:

    .. code-block:: console

        $ sudo less /var/log/orthanc/Orthanc.log

    If there is an error: ``Bad file format: The configuration section "RegisteredUsers" is defined in
    2 different configuration files`` this might be due to changes in the installed version of Orthanc.

    Edit the main Orthanc configuration file to remove the setting, as it is now in a ``credentials.json``
    configuration file.

    .. code-block:: console

        $ sudo nano /etc/orthanc/orthanc.json

    Remove the ``RegisteredUsers`` setting and try again:

    .. code-block:: console

        $ sudo systemctl start orthanc.service
        $ sudo systemctl status orthanc.service

    If there is still an issue, check the log again. If the problem this time is due to the ``TCP port of the DICOM
    server``, you might need to give it permission again:

    .. code-block:: console

        $ sudo setcap CAP_NET_BIND_SERVICE=+eip /usr/sbin/Orthanc

    And restart Orthanc once more.

Test the webserver
==================

You should now be able to browse to the web interface of your upgraded OpenREM system and have a look around.

Update the DICOM Store settings
===============================

Log in to the web interface, and navigate to ``Config``, ``DICOM networking``.

The remote nodes should be correct from the old system, but the DICOM Store SCP settings will need
updating. Modify the store, and add the hostname ``localhost``.

After you have clicked ``Submit``, the status page should show the server is alive. If it isn't, go and check the
status of Orthanc again (we may have checked it too quickly before).
