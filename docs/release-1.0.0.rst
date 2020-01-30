########################
Upgrade to OpenREM 1.0.0
########################

****************
Headline changes
****************

* Python 3!
* Django 2.2!

*******************
Upgrade preparation
*******************

* These instructions assume you are upgrading from 0.10.0.
* **Upgrades from 0.9.1 or earlier should review** :doc:`upgrade_previous_0.10.0`. -- needs changing

***************************************************
Upgrading an OpenREM server with no internet access
***************************************************

Follow the instructions found at :doc:`upgrade-offline`, before returning here to update the configuration, migrate the
database and complete the upgrade.

***************
Upgrade process
***************

Upgrade
=======

* Back up your database

    * For PostgreSQL on linux you can refer to :ref:`backup-psql-db`
    * For PostgreSQL on Windows you can refer to :doc:`backupRestorePostgreSQL`
    * For a non-production SQLite3 database, simply make a copy of the database file

* Stop any Celery workers

* Consider temporarily disabling your DICOM Store SCP, or redirecting the data to be processed later

* If you are using a virtualenv, activate it

    *Ubuntu one page instructions*::

        sudo systemctl stop openrem-celery
        sudo systemctl stop orthanc
        . /var/dose/veopenrem/bin/activate

* Install the new version of OpenREM:

    .. code-block:: console

        pip install openrem==1.0.0b1

.. _update_configuration0100:

Migrate the database
====================

In a shell/command window, move into the ``openrem`` folder:

* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/``
* Windows: ``C:\Python27\Lib\site-packages\openrem\``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\``

* Delete everything except ``__init__.py`` in ``remapp/migrations``
* Rename ``0001_initial.py.1-0-upgrade`` to ``0001_initial.py``

.. code-block:: console

    python manage.py migrate --fake-initial
    python manage.py migrate remapp --fake
    python manage.py makemigrations remapp
    python manage.py migrate


Update static files
===================

In the same shell/command window as you used above run the following command to clear the static files
belonging to your previous OpenREM version and replace them with those belonging to the version you have
just installed (assuming you are using a production web server...):

.. code-block:: console

    python manage.py collectstatic --clear

..  admonition:: Virtual directory users

    If you are running your website in a virtual directory, you also have to update the reverse.js file.
    To get the file in the correct path, take care that you insert just after the declaration of
    ``STATIC_ROOT`` the following line in your ``local_settings.py`` (see also the sample ``local_settings.py.example``):

    .. code-block:: console

        JS_REVERSE_OUTPUT_PATH = os.path.join(STATIC_ROOT, 'js', 'django_reverse')

    To update the reverse.js file execute the following command:

    .. code-block:: console

        python manage.py collectstatic_js_reverse

    See  :doc:`virtual_directory` for more details.


Restart all the services
========================

Follow the guide at :doc:`startservices`.

    *Ubuntu one page instructions*::

        sudo systemctl start openrem-celery
        sudo systemctl start orthanc
        sudo systemctl restart openrem-gunicorn

.. _post_upgrade0100:


.. _CP1676: https://www.dicomstandard.org/cps/