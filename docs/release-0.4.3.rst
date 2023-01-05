OpenREM Release Notes version 0.4.3
***********************************

Headline changes
================


* Export of study information is now handled by a task queue - no more export time-outs.
* Patient size information in csv files can now be uploaded and imported via a web interface.
* Proprietary projection image object created by Hologic tomography units can now be interrogated for details of the tomosynthesis exam.
* Settings.py now ships with its proper name, this will overwrite important local settings if upgrade is from 0.3.9 or earlier.
* Time since last study is no longer wrong just because of daylight saving time!
* Django release set to 1.6; OpenREM isn't ready for Django 1.7 yet
* The inner ``openrem`` Django project folder is now called ``openremproject`` to avoid import conflicts with Celery on Windows
* DEBUG mode now defaults to False

Specific upgrade instructions
=============================

For the original upgrade instructions, the last docs release to include them was
`0.10.0-docs <https://docs.openrem.org/en/0.10.0-docs/release-0.4.3.html>`_
