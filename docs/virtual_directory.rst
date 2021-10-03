**************************************************
Running the OpenREM website in a virtual directory
**************************************************

If you want to run the OpenREM in a virtual directory (like http://server/dms/) you need to configure this in your
web server configuration as well as in the OpenREM configuration.

The following steps are necessary:

- Configure virtual directory settings in the Docker ``.env.prod`` file
- Update Nginx webserver configuration
- Update the ``reverse.js`` file
- Restart the containers

Docker setup
============

Stop the containers if they are running before changing the configuration, using a shell (command prompt) in the Docker
OpenREM installation folder

.. code-block:: console

    $ docker-compose down

Configure virtual directory settings in .env.prod
-------------------------------------------------

Django needs to know the virtual directory name and which URLs the static and media files are served from.

Edit ``.env.prod``, uncomment the following lines (remove the ``#``) and set them as appropriate. For example, to serve
the website from a subfolder/virtual directory named ``dms``:

.. code-block:: none

    ## For installations in a virtual directory
    VIRTUAL_DIRECTORY=dms/
    MEDIA_URL=/dms/media/
    STATIC_URL=/dms/static/

Modify webserver configuration
------------------------------

Edit ``nginx-conf/conf.d/openrem.conf`` to update the locations — again using the example virtual directory ``dms``:

.. code-block:: nginx

    server {
        listen 80;
        location /dms/ {
            proxy_pass http://openremproject;
            # ...
        }
        location /dms/static/ {
            alias /home/app/openrem/staticfiles/;
        }
        location /dms/media/ {
            alias /home/app/openrem/mediafiles/;
        }
    }

Start the containers
--------------------

.. code-block:: console

    $ docker-compose up -d

Update reverse.js
-----------------

The static reverse.js file should be updated in order to change the URLs in the static javascript files.

Open a shell (command prompt) and navigate to the Docker OpenREM installation folder

.. code-block:: console

    $ docker-compose exec openrem python manage.py collectstatic_js_reverse


Test!
-----

You should now be able to reach the OpenREM interface using the virtual directory address.


Non-Docker Linux install
========================

.. code-block:: console

    $ sudo systemctl stop openrem-gunicorn.service
    $ sudo systemctl stop nginx.service

Update local_settings.py
------------------------

Update ``local_settings.py`` with the same variables as in the ``.env.prod`` file. If the values aren't in your copy
of the file just add them in:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.8/site-packages/openrem/
    $ nano openremproject/local_settings.py

.. code-block:: python

    VIRTUAL_DIRECTORY = "dms/"
    STATIC_URL = "/dms/static/"
    MEDIA_URL = "/dms/media/"

Modify webserver configuration
------------------------------

.. code-block:: console

    $ sudo nano /etc/nginx/sites-available/openrem-server

.. code-block:: nginx

    server {
        # ...
        location /dms/static {
            alias /var/dose/static;
        }
        location /dms {
            proxy_pass http://unix:/tmp/openrem-server.socket;
            # ...
        }
    }

Update reverse.js
-----------------

.. code-block:: console

    $ . /var/dose/veopenrem3/bin/activate
    $ cd /var/dose/veopenrem3/lib/python3.8/site-packages/openrem/
    $ python manage.py collectstatic_js_reverse

Restart the services
--------------------

.. code-block:: console

    $ sudo systemctl start openrem-gunicorn.service
    $ sudo systemctl start nginx.service
