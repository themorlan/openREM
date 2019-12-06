#########################
Upgrade to OpenREM 0.10.0
#########################

****************
Headline changes
****************

* Database: new summary fields introduced to improve the responsiveness of the interface - requires additional migration
  step
* Imports: enabled import of GE Elite Mini View C-arm, Opera Swing R/F and Philips BigBore CT RDSRs that have issues
* Imports: updated event level laterality to import from new location after DICOM standard change proposal CP1676_
* Interface: highlight row when dose alert exceeded
* Exports: added fluoroscopy and radiography exports tailored for UK PHE dose survey
* General: Lots of fixes to imports, interface, charts etc

*******************
Upgrade preparation
*******************

* These instructions assume you are upgrading from 0.9.1.
* **Upgrades from 0.9.0 or earlier should review** :doc:`upgrade_previous_0.10.0`.

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

        pip install openrem==0.10.0

.. _update_configuration0100:

Migrate the database
====================

In a shell/command window, move into the ``openrem`` folder:

* Ubuntu linux: ``/usr/local/lib/python2.7/dist-packages/openrem/``
* Other linux: ``/usr/lib/python2.7/site-packages/openrem/``
* Linux virtualenv: ``vitualenvfolder/lib/python2.7/site-packages/openrem/``
* Windows: ``C:\Python27\Lib\site-packages\openrem\``
* Windows virtualenv: ``virtualenvfolder\Lib\site-packages\openrem\``

.. code-block:: console

    python manage.py makemigrations remapp
    python manage.py migrate remapp


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

****************************************
Post-upgrade migration of summary fields
****************************************

Populate new summary fields
===========================

..  figure:: img/0_10_Migration_Login.png
    :figwidth: 100%
    :align: center
    :alt: 0.10 upgrade panel before log in

With RabbitMQ, Celery and the web server running, log in as an administrator to start the migration process. If you have
a large number of studies in your database this can take some time. A large database (several hundred studies) on slow
disks might take a day or two, on faster disks or with a smaller database it could take from a few minutes to an hour
or so. You will be able to monitor the progress on the home page as seen in the figure at the bottom of this page.

..  figure:: img/0_10_Migration_Loggedin.png
    :figwidth: 100%
    :align: center
    :alt: 0.10 upgrade panel after log in as administrator

One task per modality type (CT, fluoroscopy, mammography and radiography) is generated to create a task per study in
each modality to populate the new fields for that study. If the number of workers is the same or less than the number
of modality types in your database then the study level tasks will all be created before any of them are executed as
all the workers will be busy. Therefore there might be a delay before the progress indicators on the OpenREM front
page start to update. You can review the number of tasks being created on the ``Config -> Tasks`` page.

Before the migration is complete, some of the information on the modality pages of OpenREM will be missing, such as the
dose information for example, but otherwise everything that doesn't rely on Celery workers will work as normal. Studies
sent directly to be imported will carry on during the migration, but query-retrieve tasks will get stuck behind the
migration tasks.

..  figure:: img/0_10_Migration_Processing.png
    :figwidth: 100%
    :align: center
    :alt: 0.10 upgrade panel, population of fields in progress

When the process is complete the 'Summary data fields migration' panel will disappear and will not be seen again.

Post migration activity
=======================

Any scheduled query-retrieve tasks may not have executed properly during the migration. If they
haven't, it is worth replicating the missing tasks using the web interface 'Query remote server'.

.. _CP1676: https://www.dicomstandard.org/cps/