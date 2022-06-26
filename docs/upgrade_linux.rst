********************************
Upgrading a native Linux install
********************************

These instructions assume a configuration similar to the 'One page complete Ubuntu install' provided with release
0.8.1 and later.

Preparation
===========

Back up the database:

.. code-block:: console

    $ pg_dump -U openremuser -d openremdb -F c -f pre-1-0-upgrade-dump.bak

Stop any Celery workers, Flower and Gunicorn, disable DICOM Store SCP:

.. code-block:: console

    $ sudo systemctl stop openrem-celery
    $ sudo systemctl stop openrem-flower
    $ sudo systemctl stop openrem-gunicorn
    $ sudo systmectl stop orthanc

Install Python 3.8 and create a new virtualenv:

.. code-block:: console

    $ sudo apt install python3.8 python3.8-dev python3.8-distutils python3.8-venv

.. code-block:: console

    $ cd /var/dose
    $ python3.8 -m venv veopenrem3
    $ . veopenrem3/bin/activate

Install the new version of OpenREM
==================================

.. code-block:: console

    $ pip install --upgrade pip

.. code-block:: console

    $ pip install openrem==1.0.0b1

Update the local_settings.py file
=================================

* Remove the first line ``LOCAL_SETTINGS = True``
* Change second line to ``from .settings import *``
* Compare file to local_settings.py.example to see if there are other sections that should be updated

Migrate the database
====================

In a shell/command window, move into the ``openrem`` folder:

.. code-block:: console

    $ cd /var/dose/veopenrem3/lib/python3.8/site-packages/openrem/

Prepare the migrations folder:

* Delete everything except ``__init__.py`` and ``0001_initial.py.1-0-upgrade`` in ``remapp/migrations``
* Rename ``0001_initial.py.1-0-upgrade`` to ``0001_initial.py``

.. code-block:: console

    $ rm remapp/migrations/0*.py
    $ rm remapp/migrations/0*.pyc  # may result in 'cannot remove' if there are none
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

    $ python django-admin compilemessages

Update all the services configurations
======================================

Edit the Gunicorn systemd file ``WorkingDirectory`` and ``ExecStart``:

.. code-block:: console

    $ sudo nano /etc/systemd/system/openrem-gunicorn.service

.. code-block:: none

    WorkingDirectory=/var/dose/veopenrem3/lib/python3.8/site-packages/openrem

    ExecStart=/var/dose/veopenrem3/bin/gunicorn \
        --bind unix:/tmp/openrem-server.socket \
        openremproject.wsgi:application --timeout 300 --workers 4


Reload systemd and restart the services
=======================================

.. code-block:: console

    $ sudo systemctl daemon-reload
    $ sudo systemctl restart openrem-gunicorn.service
    $ sudo systemctl restart nginx.service
    $ sudo systemctl start orthanc.service
