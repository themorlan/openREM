***************************
Translating OpenREM strings
***************************

OpenREM's primary language is British English (en_GB). Users and developers with knowledge
of other languages can create translations of the interface strings, export file strings
and the documentation. These will then be exposed when the web browser language is set to
match the new translation language (OpenREM interface) or when the language is selected
for the documentation.

A web-based service for managing translations has kindly been provided to OpenREM by Weblate. Their hosting is free
to OpenREM, and they `welcome donations <https://weblate.org/en-gb/donate/>`_.

Translators
===========

* Create an account at https://hosted.weblate.org
* The OpenREM project is at https://hosted.weblate.org/projects/openrem/
* Each page in the Read The Docs documentation (https://docs.openrem.org) is a separate 'component' in Weblate, and they
  have been named 'RTD document name'. The web interface strings are all in one 'component'.
* Choose a component, and on the next page you can select one of the existing translations which you can review, edit
  and propose new translation strings.
* Once approved, they will be merged in by developers

Creating new language translations
----------------------------------

At the component level, you will see an option to create a new translation. This might need to be done for each
component individually.

Code syntax in strings
----------------------

Be careful not to edit code syntax within strings. For example, Python code might be:

.. code-block:: none

    Writing study {row} of {numrows} to All data sheet and individual protocol sheets

This is translated into Norwegian Bokmål as:

.. code-block:: none

    Skriver studie av {row} av {numrows} til alle datablad og individuelle protokollblader

Notice that the ``{}`` and their contents is unchanged - but may be moved around within the sentence to produce the
correct grammar for the language being used.

Similarly with Django HTML template strings:

.. code-block:: none

    Number in last %(day_delta)s days

becomes:

.. code-block:: none

    Antall de siste %(day_delta)s dagene

It is essential that the ``%()s`` as well as the string inside the brackets stay intact.

For the RTD translations, there will be Sphinx codes that should be left untranslated, for example:

.. code-block:: none

    :ref:`genindex`


Developers
==========

Install pre-requisites
----------------------

**gettext**

Linux: ``sudo apt install gettext`` or equivalent for your distribution. For Windows: download
`a precompiled binary installer <https://mlocati.github.io/articles/gettext-iconv-windows.html>`_

**sphinx-intl**

Activate development environment - see :doc:`code_dev_env` for details - and add the sphinx packages:

.. code-block:: console

    $ pip install sphinx
    $ pip install sphinx-intl

Update .pot and .po files
-------------------------

Activate the development environment and move to the root of the OpenREM repository - with the ``docs`` folder and
``openrem`` folder etc:

.. code-block:: console

    $ sphinx-build -b gettext docs/ docs/_build/gettext
    $ sphinx-intl update -p docs/_build/gettext
    $ django-admin makemessages --keep-pot

Adding new interface strings for translation
--------------------------------------------

Please refer to https://docs.djangoproject.com/en/2.2/topics/i18n/translation/ for instructions.

In brief, the following will help get you started, but does not cover lazy translations, plurals and many other things!

All the Sphinx/Read The Docs strings are translatable - if a page does not appear in Weblate that is because it has
not been configured as a component there yet.

Python code
-----------

First, import ``gettext`` from Django:

.. code-block:: python

    from django.utils.translation import gettext as _

Then wrap strings to be translated with ``_()`` so

.. code-block:: python

    query.stage = "Checking to see if any response studies are already in the OpenREM database"

becomes

.. code-block:: python

    query.stage = _(
        "Checking to see if any response studies are already in the OpenREM database"
    )

The same is done for strings that contain variables. Unfortunately ``gettext`` cannot work with f-strings so we are
stuck with ``.format()`` instead. It is easier to understand how to translate the text though if we use named variables
rather than position based ones, like this:

.. code-block:: python

    query.stage = _("Filter at {level} level on {filter_name} that {filter_type} {filter_list}".format(
        level=level, filter_name=filter_name, filter_type=filter_type, filter_list=filter_list
    ))

Remember we cannot assume the grammar of the translated string so try and pass the whole sentence or paragraph to be
translated.

Template code
-------------

Add the following at the top of the template file, just after any ``extends`` code:

.. code-block:: html

    {% load i18n %}

This can be done with *inline* translations and *block* translations. For inline,

.. code-block:: html

    <th style="width:25%">System name</th>

becomes

.. code-block:: html

    <th style="width:25%">{% trans "System name" %}</th>

If there are variables, a block translation is required, for example:

.. code-block:: html

    {% if home_config.display_workload_stats %}
        <th style="width:12.5%">{% blocktrans with home_config.day_delta_a as day_delta trimmed %}
            Number in last {{ day_delta }} days{% endblocktrans %}</th>
        <th style="width:12.5%">{% blocktrans with home_config.day_delta_b as day_delta trimmed %}
            Number in last {{ day_delta }} days{% endblocktrans %}</th>
    {% endif %}

Comments can be added to aid translators, for example:

.. code-block:: html

    {# Translators: Number of studies in DB listed above home-page table. No final full-stop in English due to a.m./p.m. #}
    {% now "DATETIME_FORMAT" as current_time %}
    {% blocktrans with total_studies=homedata.total trimmed%}
        There are {{ total_studies }} studies in this database. Page last refreshed on {{ current_time }}
    {% endblocktrans %}


Making use of updated strings on local system
---------------------------------------------

Specify the language to build for Sphinx docs, eg for German:

.. code-block:: console

    $ sphinx-build -b html -D language=de . _build/html/de

For Django strings:

.. code-block:: console

    $ django-admin compilemessages


Incorporating translations into main repo
=========================================

In the git repository:

.. code-block::

    $ git remote add weblate https://hosted.weblate.org/git/openrem/web-interface/

* Checkout the ``weblate\develop`` branch as a new local branch
* Push the branch to Bitbucket
* Create a pull request to develop


