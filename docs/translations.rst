***************************
Translating OpenREM strings
***************************

OpenREM's primary language is British English (en_GB). Users and developers with knowledge
of other languages can create translations of the interface strings, export file strings
and the documentation. These will then be exposed when the web browser language is set to
match the new translation language (OpenREM interface) or when the language is selected
for the documentation.

Web based translations
======================

We have applied to https://hosted.weblate.org/ for gratis hosting of this project. If our
application is successful, all the translatable strings will be available on that site
for translations to be provided, which can then be pulled into the main project.

For strings in the OpenREM interface that are not yet marked as translatable, see below.

All the strings in the documentation are available for translation.

Offline translations
====================

If you have write access to the OpenREM Bitbucket repository, create a new branch from
develop. You can do this on your PC or online. If you don't, go to
https://bitbucket.org/openrem/openrem/src/develop/ and
click on the ``+`` in the far left bar and fork the repository and clone it to your PC
and create a new branch from develop as before.

Documentation translations
--------------------------

First you need to install some packages. You will need Python 3.6+, preferably Python 3.8.

Open a shell/command line in the folder above the repository folder.

Linux:

.. code-block:: console

    $ python3.8 -m venv translate-venv
    $ . translate-venv/bin/activate

Windows PowerShell (for ``cmd.exe`` substitute ``Activate.ps1`` for ``activate.bat``)

.. code-block:: powershell

    PS C:\Path\To\Coding Folder> C:\Python38\python -m venv translate-venv
    PS C:\Path\To\Coding Folder> .\translate-venv\Scripts\Activate.ps1

Move into the repository docs folder:

.. code-block:: console

    $ cd openrem/docs

Install the packages from pip:

.. code-block:: console

    $ pip install -r rtdrequirements.txt
    $ pip install sphinx
    $ pip install sphinx-intl

Generate translatable file templates (``.pot`` files):

.. code-block:: console

    $ sphinx-build -b gettext . _build/gettext

This will leave the generated files in the folder ``_build/gettext``.

Generate the translation files - for German and Portuguese/Brazil for example:

.. code-block:: console

    $ sphinx-intl update -p _build/gettext -l de -l pt_BR

This will create a ``locale`` folder with a translation file (``.po``) per ``.rst`` file in the documentation per
language you requested, like this::

    locale
    ├── de
    │   └── LC_MESSAGES
    │       └── index.po
    └── pt_BR
      └── LC_MESSAGES
          └── index.po

These ``.po`` files can now be edited with a text editor or a Po editor such as https://poedit.net/, taking
care to retain any reST notation.

The new or updated files can now be committed and pushed back to Bitbucket and a pull request created to merge
them into develop.

To build the documentation in the translated language locally, use the following command (using German as
the example):

.. code-block:: console

    $ sphinx-build -b html -D language=de . _build/html/de

The German documentation will now be in the ``_build/html/de`` folder. Any strings that were not translated
will still be in British English, so you don't need to do everything at once.

OpenREM interface translations - existing translatable strings
--------------------------------------------------------------

Using the virtual environment created above, move to the openrem folder within the repository clone,
at the same level as ``manage.py``, eg:

.. code-block:: console

    $ cd ../openrem

Create or update message files, again using German for the example:

.. code-block:: console

    $ django-admin makemessages -l de

All the strings that have been marked for translation in either the python code or the templates will now
have been extracted and added/updated a file called ``django.po`` that will be in
``openrem/locale/de/LC_MESSAGES/``

*Windows users* - ``makemessages`` requires ``gettext`` to be installed. To create or update the ``.po`` files
on Windows, download `a precompiled binary installer <https://mlocati.github.io/articles/gettext-iconv-windows.html>`_

Alternatively, if none of the original strings have been updated or made translatable, you can copy the
``openrem/locale/en/LC_MESSAGES/django.po`` into an appropriately named folder and work on that - it is just
an empty translation file.

As with the documentation ``.po`` files, these can be updated with a text editor or using dedicated software. You can
see examples of translated strings in the existing German version. Some strings have translator comments with
them, some will have options for plurals, some will have variables in them.

For Python code strings, the variables will be in brace format and easy to recognise:

.. code-block:: po

    #. Translators: CT xlsx export progress
    #: remapp/exports/ct_export.py:160
    #, python-brace-format
    msgid ""
    "Writing study {row} of {numrows} to All data sheet and individual protocol "
    "sheets"
    msgstr ""
    "Schreiben von Studien-{row} von {numrows} in All data blatt und einzelnen "
    "Protokollblätter"

This example also demonstrates that for multi-row strings, the first line is an empty pair of double quotes,
and the text occurs on the following lines. The original string that will be matched is the ``msgid`` and the
new translation is ``msgstr``.

For template strings, the ``{{ }}`` braces become ``%( )s`` — it is important to keep the ``s`` at the end.
For example:

.. code-block:: po

    #: remapp/templates/remapp/home-list-modalities.html:11
    #: remapp/templates/remapp/home-list-modalities.html:13
    #: remapp/templates/remapp/home.html:206 remapp/templates/remapp/home.html:208
    #, python-format
    msgid "Number in last %(day_delta)s days"
    msgstr "Nummer in den letzten %(day_delta)s-Tagen"

This example shows the original string and translation in the same line as ``msgid`` and ``msgstr``. It also
shows that this one string is found four times in two templates, but the same string will be replaced in the
same way in all four occurances.

When the translations have been completed, they need to be compiled into a binary ``.mo`` file. For testing
locally, this is done with the following command, again in the virtual environment in the ``openrem``
folder where ``manage.py`` is:

.. code-block:: console

    $ django-admin compilemessages

If you now run your webserver (``runserver`` or using a real webserver), and set your browser language to the
language you have created the translations for, the translations should appear.

The new locale folders/files should now be committed to the repository and pushed as a new branch to Bitbucket
with a Pull Request made to incorporate the changes into the core code.

Making strings translatable
---------------------------

For now, please refer to https://docs.djangoproject.com/en/2.2/topics/i18n/translation/ for instructions.