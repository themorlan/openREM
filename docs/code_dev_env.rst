##################################
Creating a development environment
##################################

**Document not ready for translation**

Install Python 3.6+, preferably Python 3.8: check your Linux distribution docs to see how to install a particular
version; for Windows go to https://www.python.org/downloads/. Check "Add Python 3.8 to PATH" during installation.

Install git: ``sudo apt install git`` or equivalent on Linux; for Windows go to https://git-scm.com/download/win

*Recommended* - install an integrated development environment such as `PyCharm <https://www.jetbrains.com/pycharm/>`_
or `Visual Studio Code <https://code.visualstudio.com/>`_ (many others are available).

*Recommended* - install `PostgreSQL database <https://www.enterprisedb.com/downloads/postgres-postgresql-downloads>`_

Check out git repo
==================

Either clone the main OpenREM repository, or fork it first and then clone that (adapt the command accordingly):

.. code-block:: console

    $ git clone https://bitbucket.org/openrem/openrem.git

The OpenREM source code will now be in a folder called openrem. If you wish to specify the folder, you could do this
by adding the folder name to the clone command.

Create Python virtual environment
=================================

Linux - install the Python package ``venv`` using ``pip`` (Windows users, ``venv`` should have been
installed with Python automatically):

.. code-block:: console

    $ sudo apt install python3-venv

Then create the Python virtual environment in a folder called ``openrem-venv`` (change as required):

Linux:

.. code-block:: console

    $ python3.8 -m venv openrem-venv
    $ . openrem-venv/bin/activate

Windows PowerShell (for ``cmd.exe`` substitute ``Activate.ps1`` with ``activate.bat``)

.. code-block:: powershell

    PS C:\Path\To\Coding Folder> C:\Python38\python -m venv openrem-venv
    PS C:\Path\To\Coding Folder> .\openrem-venv\Scripts\Activate.ps1

For users of VS Code, it can be useful to create the virtual environment in a folder called ``.venv``
within your project folder (where you checked out the git repo), then VS Code will find it automatically.
If you are using PyCharm you can click on the Python interpreter at the bottom right and click 'Add
Interpreter'.

Install the Python libraries
============================

Assumes:

* git repository is in a sub-folder called ``openrem`` - change as necessary
* venv is activated

.. code-block:: console

    $ pip install -e openrem/

Setup OpenREM
=============

You'll need a basic configuration of OpenREM to run any code locally - copy the
``openremproject/local_settings.py.example``  to ``openremproject/local_settings.py`` and set a path for a SQLite
database etc.

To use PosgreSQL instead of SQLite3, set up a user in pgAdmin 4 on Windows, and an empty database with the
same user as owner, or use the :ref:`Linux-DB` instructions on Linux.

Run test webserver
==================

To see the changes you have made with the web interface, you can use the built-in Django webserver:

.. sourcecode:: console

    python manage.py runserver --insecure

In a web browser on the same computer, go to http://localhost:8000/ - you should now see the message about
creating users.

Get coding
==========

Create a branch in the git repository, and start making your changes, adding your features etc!

When you are done, push it back to Bitbucket and send in a pull request! Ideally, try and use the ``refs #123``
syntax in commit messages to reference the issue on Bitbucket you are working on.
