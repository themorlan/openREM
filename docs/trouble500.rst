Server 500 errors
=================

**Turn on debug mode**

This will render a debug report in the browser - usually revealing the problem.

Docker installs
---------------

Edit the ``.env.prod`` file. Find the following line and change it from ``0`` to ``1``:

.. code-block:: none

    DEBUG=1

Restart the containers using a command line in the folder containing your installation. This might be enough:

.. code-block:: console

    docker-compose up -d

If the webserver fails, then restart all the containers:

.. code-block:: console

    docker-compose down
    docker-compose up -d

Non-Docker installs
-------------------

Locate and edit your local_settings file

.. code-block:: console

    nano /var/dose/veopenrem3/lib/python3.8/site-packages/openrem/local_settings.py

Find the following line and make it active:

.. code-block:: python

    DEBUG = True

Restart the web service:

.. code-block:: console

    sudo systemctl reload openrem-gunicorn.service

Returning to normal mode
------------------------

You should always disable debug mode when you have fixed the error. If you leave debug mode
in place, the system is likely to run out of memory as database queries are cached in this mode.

Docker:

* Edit ``.env.prod`` to set ``DEBUG=0``
* Restart ``docker-compose``

Non-docker:

* Edit ``local_settings.py`` again to comment out the ``DEBUG`` line (add a ``#`` to the start) or set it to ``False``
* Reload the web service
