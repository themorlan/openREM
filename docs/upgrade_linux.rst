********************************
Upgrading a native Linux install
********************************

These instructions assume a configuration similar to the 'One page complete Ubuntu install' provided with release
0.8.1 and later. If you are running an older distribution, consider upgrading the operating system or migrating
the service to a new host. This release will run on Python 3.8 or 3.9, but Python 3.10 is recommended.

If upgrading to a new host, follow the :doc:`upgrade_linux_new_server` docs instead.

If a different release of Python is being used, substitute 3.10 for that version where necessary below.

If you are upgrading OpenREM on a Linux server with limited internet access, go to the :doc:`install_offline` docs.

Preparation
===========

Back up the database:

.. code-block:: console

    $ pg_dump -U openremuser -d openremdb -F c -f pre-1-0-upgrade-dump.bak

Stop any Celery workers, Flower, RabbitMQ, Gunicorn, NGINX, and Orthanc (OpenREM service names will be
reversed if they weren't changed with the 0.9.1 upgrade):

.. code-block:: console

    $ sudo systemctl stop openrem-celery
    $ sudo systemctl stop openrem-flower
    $ sudo systemctl stop openrem-gunicorn
    $ sudo systemctl stop rabbitmq-server
    $ sudo systmectl stop nginx
    $ sudo systmectl stop orthanc

Update apt and install any updates:

.. code-block:: console

    $ sudo -- sh -c 'apt update && apt upgrade'

Install Python 3.10 and create a new virtualenv:

.. code-block:: console

    $ sudo apt install acl python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip \
    postgresql nginx orthanc dcmtk default-jre zip gettext

.. code-block:: console

    $ python3.10 -m venv /var/dose/veopenrem3
    $ . /var/dose/veopenrem3/bin/activate

Install the new version of OpenREM
==================================

.. note::

    If you are upgrading this server offline, return to the Offline installation docs for
    :ref:`Offline-python-packages`

.. code-block:: console

    $ pip install --upgrade pip

.. code-block:: console

    $ pip install openrem==1.0.0b1

.. _upgrade-linux-local-settings:

Update the local_settings.py file
=================================

Copy the old ``local_settings.py`` file to the new venv:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.10/site-packages/openrem/
    $ cp /var/dose/veopenrem/lib/python2.7/site-packages/openrem/openremproject/local_settings.py openremproject/local_settings.py

* Remove the first line ``LOCAL_SETTINGS = True``
* Change second line to ``from .settings import *``
* Compare file to ``local_settings.py.linux`` to see if there are other sections that should be updated

Migrate the database
====================

In a shell/command window, move into the ``openrem`` folder:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.10/site-packages/openrem/

Prepare the migrations folder:

* Delete everything except ``__init__.py`` and ``0001_initial.py.1-0-upgrade`` in ``remapp/migrations``
* Rename ``0001_initial.py.1-0-upgrade`` to ``0001_initial.py``

.. code-block:: console

    $ rm -r remapp/migrations/0*.py
    $ rm -r remapp/migrations/0*.pyc  # may result in 'cannot remove' if there are none
    $ mv remapp/migrations/0001_initial.py{.1-0-upgrade,}

Migrate the database:

.. code-block:: console

    $ python manage.py migrate --fake-initial

.. code-block:: console

    $ python manage.py migrate remapp --fake

.. code-block:: console

    $ python manage.py makemigrations remapp

.. code-block:: console

    $ python manage.py migrate

.. code-block:: console

    $ python manage.py loaddata openskin_safelist.json


Update static files
===================

.. code-block:: console

    $ python manage.py collectstatic --clear

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

    WorkingDirectory=/var/dose/veopenrem3/lib/python3.8/site-packages/openrem

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
    $ sudo systemctl restart openrem-gunicorn.service
    $ sudo systemctl restart nginx.service
    $ sudo systemctl start orthanc.service
