Native Linux install
====================

**Document not ready for translation**

This install is based on Ubuntu 22.04 using:

* Python 3.10 running in a virtualenv
* Database: PostgreSQL
* DICOM Store SCP: Orthanc running on port 104
* Webserver: NGINX with Gunicorn
* All OpenREM files in ``/var/dose/`` with group owner of ``openrem``
* Collects any Physics (QA) images and zips them

The instructions should work for Ubuntu 20.04 too, references to jammy will be focal instead.

There are various commands and paths that reference the Python version 3.10 in these instructions. If you are using
Python 3.8 or Python 3.9 then these will need to be modified accordingly.

If you are upgrading an existing installation to a new Linux server, go to the :doc:`upgrade_linux_new_server` docs
first.

If you are installing OpenREM on a Linux server with limited internet access, go to the :doc:`install_offline` docs.

If you are installing on a different Linux OS you can adapt these instructions or consider using a
:doc:`install_docker` instead.

Initial prep
^^^^^^^^^^^^


Install apt packages
--------------------
**Apt sources**

We will need the ``universe`` repository enabled. Check first:

.. code-block:: console

    $ less /etc/apt/sources.list

Look for::

    deb http://archive.ubuntu.com/ubuntu/ jammy universe
    deb http://archive.ubuntu.com/ubuntu/ jammy-updates universe

If these two lines are not there or are commented out (line starts with a ``#``), add them in or remove the ``#``
(``sudo nano /etc/apt/sources.list``).

.. code-block:: console

    $ sudo -- sh -c 'apt update && apt upgrade'

.. code-block:: console

    $ sudo apt install acl python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip \
    postgresql nginx orthanc dcmtk default-jre zip gettext

Folders and permissions
-----------------------

**Groups**

Now create new group ``openrem`` and add your user to it (``$USER`` will automatically substitute for the user you are
running as):

.. code-block:: console

    $ sudo groupadd openrem
    $ sudo adduser $USER openrem

Add orthanc and www-data users to openrem group:

.. code-block:: console

    $ sudo -- sh -c 'adduser orthanc openrem && adduser www-data openrem'

.. note::

    At a later stage, to add a second administrator just add them to the ``openrem`` group in the same way.

**Folders**

Create the folders we need, and set the permissions. The 'sticky' group setting and the access control list
setting (``setfacl``) below will enable both ``orthanc`` user and ``www-data`` user as well as you and your colleagues
to write to the logs and access the 'Physics' images etc:

.. code-block:: console

    $ sudo -- sh -c 'mkdir /var/dose && chmod 775 /var/dose'

.. code-block:: console

    $ sudo chown $USER:openrem /var/dose

.. code-block:: console

    $ cd /var/dose

.. code-block:: console

    $ mkdir {log,media,pixelmed,static,veopenrem3}

.. code-block:: console

    $ mkdir -p orthanc/dicom && mkdir -p orthanc/physics

.. code-block:: console

    $ sudo chown -R $USER:openrem /var/dose/*

.. code-block:: console

    $ sudo chmod -R g+s /var/dose/*

Find the ``uid`` of your user and the ``gid`` of the ``openrem`` group:

.. code-block:: console

    $ id
    $ getent group openrem

Take note of the ``uid`` number and the ``gid`` in the third field of the group information and use it in the next
command, replacing ``1001`` (user ``uid``) and ``1002`` (``openrem`` group ``gid``) as appropriate:

.. code-block:: console

    $ sudo setfacl -PRdm u:1001:rwx,g:1002:rwx,o::r /var/dose/


Pixelmed download
-----------------

.. code-block:: console

    $ cd /var/dose/pixelmed
    $ wget http://www.dclunie.com/pixelmed/software/webstart/pixelmed.jar

Create the virtualenv
---------------------

Create a virtualenv (Python local environment) in the folder we created:

.. code-block:: console

    $ python3.10 -m venv /var/dose/veopenrem3

.. _activatevirtualenv:

Activate the virtualenv
-----------------------

Activate the virtualenv (note the ``.`` -- you can also use the word ``source``):

.. code-block:: console

    $ . /var/dose/veopenrem3/bin/activate

Install Python packages
-----------------------

.. note::

    If you are installing this server offline, return to the Offline installation docs for
    :ref:`Offline-python-packages`

.. code-block:: console

    $ pip install --upgrade pip

.. code-block:: console

    $ pip install openrem

.. _Linux-DB:

Database and OpenREM config
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup PostgreSQL database
-------------------------

Create a postgres user, and create the database. You will be asked to enter a new password (twice). This will be needed
when configuring the ``local_settings.py`` file later:

.. code-block:: console

    $ sudo -u postgres createuser -P openremuser

.. code-block:: console

    $ sudo -u postgres createdb -T template1 -O openremuser -E 'UTF8' openremdb

.. admonition:: For upgrades use a different template

    If this is an upgrade to a new Linux server and not a new install, use ``template0`` instead:

    .. code-block:: console

            $ sudo -u postgres createdb -T template0 -O openremuser -E 'UTF8' openremdb

Update the PostgreSQL client authentication configuration. Add the following line anywhere near the bottom of the file,
for example in the gap before ``# DO NOT DISABLE`` or anywhere in the table that follows. The number of spaces between
each word is not important (one or more). If you are not using PostgreSQL 14 then substitute the version number in the
file path.

.. code-block:: console

    $ sudo nano /etc/postgresql/14/main/pg_hba.conf

.. code-block:: none

    local   all     openremuser                 md5

Reload postgres:

.. code-block:: console

    $ sudo systemctl reload postgresql

.. _updatelinuxconfig:

Configure OpenREM
-----------------

Navigate to the Python openrem folder and copy the example ``local_settings.py`` and ``wsgi.py`` files to remove the
``.linux`` and ``.example`` suffixes:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.10/site-packages/openrem/
    $ cp openremproject/local_settings.py{.linux,}
    $ cp openremproject/wsgi.py{.example,}

Edit ``local_settings.py`` as needed - make sure you change the ``PASSWORD``, the ``SECRET_KEY`` (to anything, just
change it), the ``ALLOWED_HOSTS`` list, regionalisation settings and the ``EMAIL`` configuration. You can modify the
email settings later if necessary. Some settings are not shown here but are documented
in the settings file or elsewhere in the docs.

.. admonition:: Upgrading to a new server

    If you are upgrading to a new Linux server, review the ``local_settings.py`` file from the old server to copy over
    the ``NAME``, ``USER`` and ``PASSWORD``, ``ALLOWED_HOSTS`` list and the ``EMAIL`` configuration, and check all the
    other settings. Change the ``SECRET_KEY`` from the default, but it doesn't have to match the one on the old server.

.. code-block:: console

    $ nano openremproject/local_settings.py

.. code-block:: python
    :emphasize-lines: 4-6, 17-18,26-29,52,57,60,71-78

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
    JS_REVERSE_OUTPUT_PATH = os.path.join(STATIC_ROOT, 'js', 'django_reverse')

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

Now create the database. Make sure you are still in the openrem python folder and
the virtualenv is active — prompt will look like

.. code-block:: console

    (veopenrem3)username@hostname:/var/dose/veopenrem3/lib/python3.10/site-packages/openrem/$

Otherwise see :ref:`activatevirtualenv` and navigate back to that folder.

.. note::

    If you are upgrading to a new Linux server, use these additional commands before continuing with those below:

    .. code-block:: console

        $ mv remapp/migrations/0001_initial.py{.1-0-upgrade,}

    Import the database - update the path to the database backup file you copied from the old server:

    .. code-block:: console

        $ pg_restore -U openremuser -d openremdb /path/to/pre-1-0-upgrade-dump.bak


    Migrate the database:

    .. code-block:: console

        $ python manage.py migrate --fake-initial

    .. code-block:: console

        $ python manage.py migrate remapp --fake


.. code-block:: console

    $ python manage.py makemigrations remapp
    $ python manage.py migrate
    $ python manage.py loaddata openskin_safelist.json
    $ python manage.py collectstatic --no-input --clear
    $ python manage.py compilemessages
    $ python manage.py createsuperuser

.. _Install Linux webserver:

Webserver
^^^^^^^^^

Configure NGINX and Gunicorn
----------------------------

Copy in the OpenREM site config file

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.10/site-packages/openrem/
    $ sudo cp sample-config/openrem-server /etc/nginx/sites-available/openrem-server

.. note::

    Content of NGINX config file:

    .. code-block:: nginx

        server {
            listen 80;
            server_name openrem-server;

            location /static {
                alias /var/dose/static;
            }

            location / {
                proxy_pass http://unix:/tmp/openrem-server.socket;
                proxy_set_header Host $host;
                proxy_read_timeout 300s;
            }
        }

Remove the default config and make ours active:

.. code-block:: console

    $ sudo rm /etc/nginx/sites-enabled/default

.. code-block:: console

    $ sudo ln -s /etc/nginx/sites-available/openrem-server /etc/nginx/sites-enabled/openrem-server

Copy the Gunicorn systemd service file into place:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.10/site-packages/openrem/
    $ sudo cp sample-config/openrem-gunicorn.service /etc/systemd/system/openrem-gunicorn.service

.. note::

    Content of systemd file:

    .. code-block:: bash

        [Unit]
        Description=Gunicorn server for OpenREM

        [Service]
        Restart=on-failure
        User=www-data
        WorkingDirectory=/var/dose/veopenrem3/lib/python3.10/site-packages/openrem

        ExecStart=/var/dose/veopenrem3/bin/gunicorn \
            --bind unix:/tmp/openrem-server.socket \
            openremproject.wsgi:application --timeout 300

        [Install]
        WantedBy=multi-user.target

Load the new systemd configurations:

.. code-block:: console

    $ sudo systemctl daemon-reload

Set the new Gunicorn service to start on boot:

.. code-block:: console

    $ sudo systemctl enable openrem-gunicorn.service

Start the Gunicorn service, and restart the NGINX service:

.. code-block:: console

    $ sudo -- sh -c 'systemctl start openrem-gunicorn.service && systemctl restart nginx.service'

Test the webserver
------------------

You should now be able to browse to the OpenREM server from another PC.

You can check that NGINX and Gunicorn are running with the following two commands:

.. code-block:: console

    $ sudo systemctl status openrem-gunicorn.service

.. code-block:: console

    $ sudo systemctl status nginx.service

.. _dicom_store_scp_linux:

DICOM Store SCP
^^^^^^^^^^^^^^^

Copy the Lua file to the Orthanc folder. This will control how we process the incoming DICOM objects.

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.10/site-packages/openrem/
    $ cp sample-config/openrem_orthanc_config.lua.linux /var/dose/orthanc/openrem_orthanc_config.lua

Edit the Orthanc Lua configuration options:

.. code-block:: console

    $ nano /var/dose/orthanc/openrem_orthanc_config.lua

Set ``use_physics_filtering`` to true if you want Orthanc to keep physics test studies, and have it put them in the
``/var/dose/orthanc/physics/`` folder. Set it to ``false`` to disable this feature. Add names or IDs to
``physics_to_keep`` as a comma separated list.

.. code-block:: lua
    :emphasize-lines: 3,7

    -- Set this to true if you want Orthanc to keep physics test studies, and have it
    -- put them in the physics_to_keep_folder. Set it to false to disable this feature
    local use_physics_filtering = true

    -- A list to check against patient name and ID to see if the images should be kept.
    -- Orthanc will put anything that matches this in the physics_to_keep_folder.
    local physics_to_keep = {'physics'}

Lists of things to ignore. Orthanc will ignore anything matching the content of these comma separated lists; they will
not be imported into OpenREM.

.. code-block:: lua
    :emphasize-lines: 3-7

    -- Lists of things to ignore. Orthanc will ignore anything matching the content of
    -- these lists: they will not be imported into OpenREM.
    local manufacturers_to_ignore = {'Faxitron X-Ray LLC', 'Gendex-KaVo'}
    local model_names_to_ignore = {'CR 85', 'CR 75', 'CR 35', 'CR 25', 'ADC_5146', 'CR975'}
    local station_names_to_ignore = {'CR85 Main', 'CR75 Main'}
    local software_versions_to_ignore = {'VixWin Platinum v3.3'}
    local device_serial_numbers_to_ignore = {'SCB1312016'}

Enable or disable additional functionality to extract dose information from older Toshiba and GE scanners, and specify
which CT scanners should use this method. Each system should be listed as ``{'Manufacturer', 'Model name'}``, with
systems in a comma separated list within curly brackets, as per the example below:

.. code-block:: lua
    :emphasize-lines: 3,7-10

    -- Set this to true if you want to use the OpenREM Toshiba CT extractor. Set it to
    -- false to disable this feature.
    local use_toshiba_ct_extractor = true

    -- A list of CT make and model pairs that are known to have worked with the Toshiba CT extractor.
    -- You can add to this list, but you will need to verify that the dose data created matches what you expect.
    local toshiba_extractor_systems = {
            {'Toshiba', 'Aquilion'},
            {'GE Medical Systems', 'Discovery STE'},
    }

Edit the Orthanc configuration:

.. code-block:: console

    $ sudo nano /etc/orthanc/orthanc.json

Add the Lua script to the Orthanc config:

.. code-block:: json-object
    :emphasize-lines: 4

    // List of paths to the custom Lua scripts that are to be loaded
    // into this instance of Orthanc
    "LuaScripts" : [
    "/var/dose/orthanc/openrem_orthanc_config.lua"
    ],

Set the AE Title and port:

.. code-block:: json-object
    :emphasize-lines: 2,5

    // The DICOM Application Entity Title
    "DicomAet" : "OPENREM",

    // The DICOM port
    "DicomPort" : 104,

.. note::

    Optionally, you may also like to enable the HTTP server interface for Orthanc (although if the Lua script is removing
    all the objects as soon as they are processed, you won't see much!):

    .. code-block:: json-object

        // Whether remote hosts can connect to the HTTP server
        "RemoteAccessAllowed" : true,

        // Whether or not the password protection is enabled
        "AuthenticationEnabled" : false,

    To see the Orthanc web interface, go to http://openremserver:8042/ -- of course change the server name to that of your
    server!

Allow Orthanc to use DICOM port
-------------------------------

By default, Orthanc uses port 4242. If you wish to use a lower port, specifically the DICOM port of 104, you will need
to give the Orthanc binary special permission to do so:

.. code-block:: console

    $ sudo setcap CAP_NET_BIND_SERVICE=+eip /usr/sbin/Orthanc


Finish off
----------

Restart Orthanc:

.. code-block:: console

    $ sudo systemctl restart orthanc.service

.. _add_linux_user:

New users, and quick access to physics folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _`WinSCP`: https://winscp.net

This is for new Linux users; for new OpenREM users, refer to :ref:`user-settings`

If you left ``local use_physics_filtering = true`` in the Orthanc configuration, you might like to give your colleagues
a quick method of accessing
the physics folder from their home folder. Then if they use a program like `WinSCP`_ it is easy to find and copy the QA
images to another (Windows) computer on the network. WinSCP can also be run directly from a USB stick if you are unable
to install software :-)

Add the new user (replace ``newusername`` as appropriate):

.. code-block:: console

    $ sudo adduser newusername

Then add the new user to the `openrem` group (again, replace the user name):

.. code-block:: console

    $ sudo adduser newusername openrem

Now add a 'sym-link' to the new users home directory (again, replace the user name):

.. code-block:: console

    $ sudo ln -sT /var/dose/orthanc/physics /home/newusername/physicsimages

The new user should now be able to get to the physics folder by clicking on the ``physicsimages`` link when they log in,
and should be able to browse, copy and delete the zip files and folders.

Asciinema demo of this install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Link to `asciinema <https://asciinema.org/a/8CqCcLMlUG5DlWj7NhrQV8b8L>`_ demo of this install
