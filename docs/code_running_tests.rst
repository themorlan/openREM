**********************
Running the test suite
**********************

**TODO: Update for Python 3, OpenREM 1.0**

Code formatting and tests
=========================

Steps before pushing to Bitbucket. Commands assume you are in the root directory of the git repository,
at the same level as README.rst and requirements.txt etc, and that you have activated a virtualenv with
the project requirements installed (``pip install -e .``) plus Black (``pip install black``)

Run black against the code:

.. code-block:: console

    $ black --exclude stuff/ .

Check the changes made, edit where necessary. Black is an opinionated Python formatter and in general
OpenREM code should be subjected to it. The flake8 tests are tuned to agree with Black.

Run the Django tests:

.. code-block:: console

    $ python openrem/manage.py test remapp --parallel



**old stuff to be updated**

Preparation
===========

Install the dependencies and OpenREM
------------------------------------

OpenREM is a Django application, and therefore we use Django's test-execution framework to test OpenREM.

The first thing to do is to create a local copy of the git repository, then install all of OpenREM's dependencies in a
virtualenv.

You will need ``python``, ``pip``, ``git`` and ``virtualenv`` installed - see the links on the : doc : `install-prep` docs
for the latter, but you might try ``pip install virtualenv``.

.. sourcecode:: console

    mkdir openremrepo
    git clone https://bitbucket.org/openrem/openrem.git openremrepo

Now create the virtualenv:

.. sourcecode:: console

    mkdir veOpenREM
    virtualenv veOpenREM
    . veOpenREM/bin/activate  # Linux
    veOpenREM\Scripts\activate  # Windows

At this stage there should be a ``(veOpenREM)`` prefix to our prompt telling us the virtualenv is activated.

Now install the dependencies:

.. sourcecode:: console

    pip install -e openremrepo/
    pip install https://bitbucket.org/edmcdonagh/pynetdicom/get/default.tar.gz#egg=pynetdicom-0.8.2b2

In the future it might be necessary to install numpy too for testing.

Configure OpenREM
-----------------

Rename and configure ``openremproject/local_settings.py.example`` and ``openremproject/wsgi.py.example`` as per the
: doc :`install` docs.

Create a database following the same : doc :`install` instructions.

Run the tests!
==============

Making sure the virtualenv is activated, move to ``openremrepo/openrem`` and run:

.. sourcecode:: console

    python manage.py test remapp

All the tests that exit in ``openrem/remapp/tests/`` will now be run.


Related tools
=============

Enabling django-debug-toolbar
-----------------------------

See :doc:`enabling_debug_toolbar`

Creating test versions of production systems
============================================

If you wish to create a duplicate install to test upgrades etc, refer to :ref:`restore-psql-linux` and the preceding
text regarding making backups.