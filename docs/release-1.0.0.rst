########################
Upgrade to OpenREM 1.0.0
########################

**Document not ready for translation**

****************
Headline changes
****************

* Python 3
* Django 2.2
* Docker or direct install on Windows and Linux
* Celery, Flower and RabbitMQ removed from requirements

* Performing physician added to standard fluoroscopy exports (:issue:`840`)
* Station name checked at series level only, option to check at study level only instead (:issue:`772`)

*******************
Upgrade preparation
*******************

* These instructions assume you are upgrading from 0.10.0.
* **Upgrades from 0.9.1 or earlier should review** :doc:`upgrade_previous_0.10.0`. -- needs changing

..  toctree::
    :maxdepth: 1

    upgrade_previous_0.10.0

.. _post_upgrade0100:





