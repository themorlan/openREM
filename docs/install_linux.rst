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

The instructions should work for Ubuntu 20.04 too, references to jammy will be focal instead and Python 3.9.

Initial prep
^^^^^^^^^^^^


Install apt packages and direct downloads
-----------------------------------------
**Apt sources**

We will need the ``universe`` repository enabled. Check first:

.. code-block:: console

    $ less /etc/apt/sources.list

Look for::

    deb http://archive.ubuntu.com/ubuntu/ jammy universe
    deb http://archive.ubuntu.com/ubuntu/ jammy-updates universe

If these two lines are not there, add them in (``sudo nano /etc/apt/sources.list``).

.. code-block:: console

    $ sudo apt update && sudo apt upgrade

.. code-block:: console

    $ sudo apt install acl python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip postgresql nginx orthanc dcmtk default-jre zip

Folders and permissions
-----------------------

**Groups**

Now create new group ``openrem`` and add your user to it (``$USER`` will automatically substitute for the user you are
running as) :

.. code-block:: console

    $ sudo groupadd openrem

.. code-block:: console

    $ sudo adduser $USER openrem

.. note::

    At a later stage, to add a second administrator just add them to the ``openrem`` group in the same way.

**Folders**

Create the folders we need, and set the permissions. In due course, the ``orthanc`` user and the ``www-data`` user will
be added to the ``openrem`` group, and the 'sticky' group setting below will enable both users to write to the logs etc:

.. code-block:: console

    $ sudo mkdir /var/dose

.. code-block:: console

    $ sudo chown $USER:openrem /var/dose

.. code-block:: console

    $ sudo chmod 775 /var/dose

.. code-block:: console

    $ cd /var/dose

.. code-block:: console

    $ mkdir log

.. code-block:: console

    $ mkdir media

.. code-block:: console

    $ mkdir -p orthanc/dicom

.. code-block:: console

    $ mkdir -p orthanc/physics

.. code-block:: console

    $ mkdir pixelmed

.. code-block:: console

    $ mkdir static

.. code-block:: console

    $ mkdir veopenrem3

.. code-block:: console

    $ sudo chown -R $USER:openrem /var/dose/*

.. code-block:: console

    $ sudo chmod -R g+s /var/dose/*

.. code-block:: console

    $ sudo setfacl -R -dm u::rwx,g::rwx,o::r /var/dose/


Pixelmed download
-----------------

.. code-block:: console

    $ sudo apt update && sudo apt upgrade
    $ sudo apt install acl python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip postgresql  \
      nginx orthanc dcmtk default-jre zip

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

.. code-block:: console

    $ pip install --upgrade pip

.. code-block:: console

    $ pip install openrem

Add orthanc and www-data users to openrem group
-----------------------------------------------

.. code-block:: console

    $ sudo adduser orthanc openrem

.. code-block:: console

    $ sudo adduser www-data openrem

.. _Linux-DB:

Database and OpenREM config
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup PostgreSQL database
-------------------------

Create a postgres user, and create the database. You will be asked to enter a new password (twice). This will be needed
when configuring OpenREM:

.. code-block:: console

    $ sudo -u postgres createuser -P openremuser

.. code-block:: console

    $ sudo -u postgres createdb -T template1 -O openremuser -E 'UTF8' openremdb

If you are migrating from another server, you could at this point create a template0 database to restore into. See
:ref:`restore-psql-linux` for details.

Update the PostgreSQL client authentication configuration. Add the following line anywhere near the bottom of the file,
for example in the gap before ``# DO NOT DISABLE`` or anywhere in the table that follows. The number of spaces between
each word is not important (one or more).

.. code-block:: console

    $ sudo nano /etc/postgresql/10/main/pg_hba.conf

.. code-block:: none

    local   all     openremuser                 md5

Reload postgres:

.. code-block:: console

    $ sudo systemctl reload postgresql

.. _updatelinuxconfig:

Configure OpenREM
-----------------

First navigate to the Python openrem folder and copy the example local_settings and wsgi files to remove the
``.example`` suffixes:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.8/site-packages/openrem/
    $ cp openremproject/local_settings.py{.example,}
    $ cp openremproject/wsgi.py{.example,}

Edit the new local_settings file

.. code-block:: console

    $ nano openremproject/local_settings.py

.. code-block:: python

    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql_psycopg2',
            'NAME': 'openremdb',
            'USER': 'openremuser',
            'PASSWORD': 'mysecretpassword',     # This needs changing, hopefully!
            'HOST': '',
            'PORT': '',
        }
    }

    MEDIA_ROOT = '/var/dose/media/'

    STATIC_ROOT = '/var/dose/static/'

    # Change secret key

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

    LOG_ROOT = "/var/dose/log"
    LOG_FILENAME = os.path.join(LOG_ROOT, "openrem.log")
    QR_FILENAME = os.path.join(LOG_ROOT, "openrem_qr.log")
    STORE_FILENAME = os.path.join(LOG_ROOT, "openrem_store.log")
    EXTRACTOR_FILENAME = os.path.join(LOG_ROOT, "openrem_extractor.log")

    # Removed comment hashes to enable log file rotation:
    LOGGING['handlers']['file']['class'] = 'logging.handlers.RotatingFileHandler'
    LOGGING['handlers']['file']['maxBytes'] = 10 * 1024 * 1024  # 10*1024*1024 = 10 MB
    LOGGING['handlers']['file']['backupCount'] = 5  # number of log files to keep before deleting the oldest one
    LOGGING['handlers']['qr_file']['class'] = 'logging.handlers.RotatingFileHandler'
    LOGGING['handlers']['qr_file']['maxBytes'] = 10 * 1024 * 1024  # 10*1024*1024 = 10 MB
    LOGGING['handlers']['qr_file']['backupCount'] = 5  # number of log files to keep before deleting the oldest one
    LOGGING['handlers']['store_file']['class'] = 'logging.handlers.RotatingFileHandler'
    LOGGING['handlers']['store_file']['maxBytes'] = 10 * 1024 * 1024  # 10*1024*1024 = 10 MB
    LOGGING['handlers']['store_file']['backupCount'] = 5  # number of log files to keep before deleting the oldest one
    LOGGING['handlers']['extractor_file']['class'] = 'logging.handlers.RotatingFileHandler'
    LOGGING['handlers']['extractor_file']['maxBytes'] = 10 * 1024 * 1024  # 10*1024*1024 = 10 MB
    LOGGING['handlers']['extractor_file']['backupCount'] = 5  # number of log files to keep before deleting the oldest one

    DCMTK_PATH = '/usr/bin'
    DCMCONV = os.path.join(DCMTK_PATH, 'dcmconv')
    DCMMKDIR = os.path.join(DCMTK_PATH, 'dcmmkdir')
    JAVA_EXE = '/usr/bin/java'
    JAVA_OPTIONS = '-Xms256m -Xmx512m -Xss1m -cp'
    PIXELMED_JAR = '/var/dose/pixelmed/pixelmed.jar'
    PIXELMED_JAR_OPTIONS = '-Djava.awt.headless=true com.pixelmed.doseocr.OCR -'

Now create the database. Make sure you are still in the openrem python folder and
the virtualenv is active — prompt will look like

.. code-block:: console

    (veopenrem3)username@hostname:/var/dose/veopenrem3/lib/python3.8/site-packages/openrem/$

Otherwise see :ref:`activatevirtualenv` and navigate back to that folder:

.. code-block:: console

    $ python manage.py makemigrations remapp

.. code-block:: console

    $ python manage.py migrate

.. code-block:: console

    $ python manage.py loaddata openskin_safelist.json

.. code-block:: console

    $ python manage.py createsuperuser

.. code-block:: console

    $ mv remapp/migrations/0002_0_7_fresh_install_add_median.py{.inactive,}
    $ python manage.py migrate

Generate translation binary files

.. code-block:: console

    $ python django-admin compilemessages

Webserver
^^^^^^^^^

Configure NGINX and Gunicorn
----------------------------

Create the OpenREM site config file

.. code-block:: console

    $ sudo nano /etc/nginx/sites-available/openrem-server

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

Add the static files to the static folder for NGINX to serve. Again, you need to ensure the virtualenv is active in your
console and you are in the ``site-packages/openrem/`` folder:

.. code-block:: console

    $ python manage.py collectstatic

Create the Gunicorn systemd service file:

.. code-block:: console

    $ sudo nano /etc/systemd/system/openrem-gunicorn.service

.. code-block:: bash

    [Unit]
    Description=Gunicorn server for OpenREM

    [Service]
    Restart=on-failure
    User=www-data
    WorkingDirectory=/var/dose/veopenrem3/lib/python3.8/site-packages/openrem

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

    $ sudo systemctl start openrem-gunicorn.service
    $ sudo systemctl restart nginx.service

Test the webserver
------------------

You should now be able to browse to the OpenREM server from another PC.

You can check that NGINX and Gunicorn are running with the following two commands:

.. code-block:: console

    $ sudo systemctl status openrem-gunicorn.service

.. code-block:: console

    $ sudo systemctl status nginx.service


DICOM Store SCP
^^^^^^^^^^^^^^^

Open the following link in a new tab and copy the content (select all then Ctrl-c): |openrem_orthanc_conf_link|

Create the lua file to control how we process the incoming DICOM objects and paste the content in (Shift-Ctrl-v if
working directly in the Ubuntu terminal, something else if you are using PuTTY etc):

.. code-block:: console

    $ nano /var/dose/orthanc/openrem_orthanc_config.lua

Then edit the top section as follows -- keeping Physics test images has been configured, set to false to change this.
There are other settings too that you might like to change in the second section (not displayed here):

.. code-block:: lua

    -------------------------------------------------------------------------------------
    -- OpenREM python environment and other settings

    -- Set this to the path and name of the python executable used by OpenREM
    local python_executable = '/var/dose/veopenrem3/bin/python'

    -- Set this to the path of the python scripts folder used by OpenREM
    local python_scripts_path = '/var/dose/veopenrem3/bin/'

    -- Set this to the path where you want Orthanc to temporarily store DICOM files
    local temp_path = '/var/dose/orthanc/dicom/'

    -- Set this to 'mkdir' on Windows, or 'mkdir -p' on Linux
    local mkdir_cmd = 'mkdir -p'

    -- Set this to '\\'' on Windows, or '/' on Linux
    local dir_sep = '/'

    -- Set this to true if you want Orthanc to keep physics test studies, and have it
    -- put them in the physics_to_keep_folder. Set it to false to disable this feature
    local use_physics_filtering = true

    -- Set this to the path where you want to keep physics-related DICOM images
    local physics_to_keep_folder = '/var/dose/orthanc/physics/'

    -- Set this to the path and name of your zip utility, and include any switches that
    -- are needed to create an archive (used with physics-related images)
    local zip_executable = '/usr/bin/zip -r'

    -- Set this to the path and name of your remove folder command, including switches
    -- for it to be quiet (used with physics-related images)
    local rmdir_cmd = 'rm -r'
    -------------------------------------------------------------------------------------

Add the Lua script to the Orthanc config:

.. code-block:: console

    $ sudo nano /etc/orthanc/orthanc.json

.. code-block:: json-object

    // List of paths to the custom Lua scripts that are to be loaded
    // into this instance of Orthanc
    "LuaScripts" : [
    "/var/dose/orthanc/openrem_orthanc_config.lua"
    ],

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

Then edit the Orthanc configuration again:

.. code-block:: console

    $ sudo nano /etc/orthanc/orthanc.json

.. code-block:: json-object

    // The DICOM Application Entity Title
    "DicomAet" : "OPENREM",

    // The DICOM port
    "DicomPort" : 104,

Finish off
----------

Restart Orthanc:

.. code-block:: console

    $ sudo systemctl restart orthanc.service

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
