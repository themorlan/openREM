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

https://hosted.weblate.org/ have granted us gratis hosting of this project. In time, all the translatable strings will
be available on that site for translations to be provided, which can then be pulled into the main project.

For strings in the OpenREM interface that are not yet marked as translatable, see below.

These documents will be added to Weblate for translations when they are reasonably stable. The first documents are
available for translation now.

Adding to existing translations
-------------------------------

Create an account on https://hosted.weblate.org. You can suggest translation strings anonymously, but we would very much
prefer it if you created an account so we can attribute the work and build a relationship!

The OpenREM project is at https://hosted.weblate.org/projects/openrem/

Each page in the Read The Docs documentation (https://docs.openrem.org) is a separate 'component' in Weblate, and they
have been named 'RTD document name'. The web interface strings are all in one 'component'.

Choose a component, and on the next page you can select one of the existing translations which you can review, edit and
propose new translation strings.

Creating new language translations
----------------------------------

At the component level, you will see an option to create a new translation. You will need to do this for each component
individually I think.

Code syntax in strings
----------------------

Be careful not to edit code syntax within strings. For example, Python code might be::

    Writing study {row} of {numrows} to All data sheet and individual protocol sheets

This is translated into Norwegian Bokmål as::

    Skriver studie av {row} av {numrows} til alle datablad og individuelle protokollblader

Notice that the ``{}`` and their contents is unchanged - but may be moved around within the sentence to produce the
correct grammar for the language being used.

Similarly with Django HTML template strings::

    Number in last %(day_delta)s days

becomes::

    Antall de siste %(day_delta)s dagene

It is essential that the ``%()s`` as well as the string inside the brackets stay intact.

For the RTD translations, there will be Sphinx codes that should be left untranslated, for example:

.. code-block::

    :ref:`genindex`

Bringing translations back into Read The Docs and OpenREM
---------------------------------------------------------

When new strings have been translated, the translation files can be downloaded in a ZIP file. A new branch should be
created from ``develop`` - either in the openrem repository or in a fork, and the new files added/existing files
replaced. This can be done by anyone, doesn't need to be the person who has done the translations.

The Norwegian Read The Docs Sphinx translation files need to go in the ``no`` folder, not in ``nb_NO`` else Read The
Docs doesn't pick them up.

Generating po and pot files for translations
============================================

Weblate has the ``develop`` branch available to it currently, so changes to strings need to be in ``develop`` before
they can be translated.


Documentation strings
---------------------

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

This will leave the translation template ``pot`` files in the folder ``_build/gettext``.

To generate or update the translation files - for German and Portuguese/Brazil for example (this step probably isn't
necessary, it can be done in Weblate):

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
them into develop. Or the files can be committed without any further translations to be merged into develop to be
translated on Weblate.

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

    $ django-admin makemessages -l de --keep-pot

All the strings that have been marked for translation in either the python code or the templates will now
have been extracted and added to or updated in a template file called ``django.pot`` and files called ``django.po``
that will be in ``openrem/locale/xx/LC_MESSAGES/`` where ``xx`` is the language code, such as ``de``.

*Windows users* - ``makemessages`` requires ``gettext`` to be installed. To create or update the ``.pot`` and ``.po``
files on Windows, download
`a precompiled binary installer <https://mlocati.github.io/articles/gettext-iconv-windows.html>`_

As with the documentation ``.po`` files, these can be updated with a text editor or using dedicated software. You can
see examples of translated strings in the existing German version. Some strings have translator comments with
them, some will have options for plurals, some will have variables in them. Or just create a pull request on Bitbucket
and they will be available on Weblate once merged.

For local use, when the translations have been completed, they need to be compiled into a binary ``.mo`` file. This is
done with the following command, again in the virtual environment in the ``openrem`` folder where ``manage.py`` is:

.. code-block:: console

    $ django-admin compilemessages

If you now run your webserver (``runserver`` or using a real webserver), and set your browser language to the
language you have created the translations for, the translations should appear.

The new locale folders/files should now be committed to the repository and pushed as a new branch to Bitbucket
with a Pull Request made to incorporate the changes into the core code.

Making strings translatable
---------------------------

For now, please refer to https://docs.djangoproject.com/en/2.2/topics/i18n/translation/ for instructions.