##################################
Creating a development environment
##################################

Install Python 3.6+, preferably Python 3.8 - check your Linux distribution docs to see how to install a particular
version; for Windows go to https://www.python.org/downloads/

Install git - ``sudo apt install git`` or equivalent on Linux; for Windows go to https://git-scm.com/download/win

*Recommended* - install an integrated development environment such as `PyCharm <https://www.jetbrains.com/pycharm/>`_
or `Visual Studio Code <https://code.visualstudio.com/>`_ (many others are available).

Check out git repo
==================

Either clone the main OpenREM repository, or fork it first and then clone that (adapt the command accordingly):

.. code-block:: console

    $ git clone https://bitbucket.org/openrem/openrem.git

The OpenREM source code will now be in a folder called openrem. If you wish to specify the folder, you could do this
by adding the folder name to the clone command.

Create Python virtual environment
=================================

Linux:

.. code-block:: console

    $ python3.8 -m venv openrem-venv
    $ . openrem-venv/bin/activate

Windows PowerShell (for ``cmd.exe`` substitute ``Activate.ps1`` with ``activate.bat``)

.. code-block:: powershell

    PS C:\Path\To\Coding Folder> C:\Python38\python -m venv openrem-venv
    PS C:\Path\To\Coding Folder> .\openrem-venv\Scripts\Activate.ps1

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

Get coding
==========

Create a branch in the git repository, and start making your changes, adding your features etc!

When you are done, push it back to Bitbucket and send in a pull request! Ideally, try and use the ``refs #123``
syntax in commit messages to reference the issue on Bitbucket you are working on.
