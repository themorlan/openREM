**********************************
Upgrading a native Windows install
**********************************

Release 1.0 of OpenREM uses a newer version of Python and no longer uses RabbitMQ, Erlang and Celery. Instructions
are only provided for Orthanc DICOM server, and no longer for Conquest. The built-in DICOM Store node has been removed.

Consider upgrading to a new Windows server instead of upgrading in place. Instructions for
:doc:`upgrade_windows_new_server` are provided including exporting and importing the existing PostgreSQL database.

* something about a clean install, and/or not having old services that are no longer required
* something about being a standardised approach which will make upgrade docs and examples easier to follow

Then best effort upgrade docs... a lot of this can be copied from the :doc:`install_windows` instructions, or depending
on what it ends up looking like, we might point there with a few admonitions to point out differences?

* Export database for backup
* Stop all the services
* Install Python 3.10
* Update PostgreSQL, Orthanc, DCMTK, Pixelmed
* Add/update as necessary gettext, 7Zip, Notepad++
* Install IIS if Apache/NGINX previously in use

* Create virtualenv, activate
* Install new OpenREM, wfastcgi

* Configure OpenREM - use new local_settings.py.windows, adjust database name etc
* Will database be available in new version of PostgreSQL? Or does it need to be imported?

* Rename 0001_initial.py file
* Do the fake-initial etc stuff
* Do the rest of the manage.py stuff

* Configure/reconfigure IIS

* Configure/reconfigure Orthanc

