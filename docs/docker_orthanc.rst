Orthanc configuration - docker-compose.yml
==========================================

The ``docker-compose.yml`` file defines all the containers that are needed for OpenREM, including the Orthanc
container(s) that provide the DICOM Store functionality to enable scanners to send directly to OpenREM, and for
query-retrieve to function.

Unlike the ``.env.prod`` file, this file is formatted as YAML. Strings need to quoted, a ``:`` and a space separate
the variable name and the value, and spaces are used at the start of the line to create a hierarchy.

Orthanc configuration
---------------------

Edit the ``docker-compose.yml`` file to make the changes. They will take effect next time ``docker-compose`` is started.

Find the ``orthanc_1`` definition near the end of the file.

Port
^^^^

The default port for DICOM store is set to ``104``.

To use a different port, change both the ``ports`` section and the ``environment`` section as required. In the ports
section the first number is the port exposed outside of Docker, the second number is used internally. Set them to
be the same and set the environment port accordingly to enable the same port to be used within the OpenREM interface:
For example, to use port 8104:

.. code-block:: yaml

    ports:
      - 8104:8104
    environment:
      ORTHANC__DICOM_PORT: 8104

DICOM Application Entity Title
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Application Entity Title of the Store Server. Should be up to 16 characters, no spaces. This server isn't fussy
by default, so if remote nodes connect using a different AETitle that is ok.

.. code-block:: yaml

    environment:
      ORTHANC__DICOM_AET: "OPENREM"

Objects to be ignored
^^^^^^^^^^^^^^^^^^^^^

Lists of things to ignore. Orthanc will ignore anything matching the content of these comma separated lists: they will
not be imported into OpenREM. Some examples have been added below - note the formatting syntax:

.. code-block:: yaml

    environment:
      MANUFACTURERS_TO_IGNORE: "{'Faxitron X-Ray LLC', 'Gendex-KaVo'}"
      MODEL_NAMES_TO_IGNORE: "{'CR 85', 'CR 75'}"
      STATION_NAMES_TO_IGNORE: "{'CR85 Main', 'CR75 Main'}"
      SOFTWARE_VERSIONS_TO_IGNORE: "{'VixWin Platinum v3.3'}"
      DEVICE_SERIAL_NUMBERS_TO_IGNORE: "{'SCB1312016'}"

Extractor for older Toshiba CT dose summary files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enable or disable additional functionality to extract dose information from older Toshiba and GE scanners, and specify
which CT scanners should use this method. Each system should be listed as ``{'Manufacturer', 'Model name'}``, with
systems in a comma separated list within curly brackets, as per the example below:

.. code-block:: yaml

    environment:
      USE_TOSHIBA_CT_EXTRACTOR: "true"
      TOSHIBA_EXTRACTOR_SYSTEMS: |
        {{'Toshiba', 'Aquilion'}, {'GE Medical Systems', 'Discovery STE'},}

Physics Filtering
^^^^^^^^^^^^^^^^^

Set this to true if you want Orthanc to keep physics test studies, and have it
put them in the ``physics_to_keep_folder``. Set it to ``"false"`` to disable this feature

.. code-block:: yaml

    environment:
      USE_PHYSICS_FILTERING: "true"

A list to check against patient name and ID to see if the images should be kept.
Orthanc will put anything that matches this in the ``physics_to_keep_folder``.

.. code-block:: yaml

    environment:
      PHYSICS_TO_KEEP: "{'physics',}"

Orthanc web interface
^^^^^^^^^^^^^^^^^^^^^

There will normally not be any studies in the Orthanc database once they have been processed, but if you want to
enable the Orthanc web viewer, change ``ORTHANC__REMOTE_ACCESS_ALLOWED`` to ``"true"`` and uncomment the port
declaration, changing the first number if required:

.. code-block:: yaml

    ports:
      - 8042:8042
    environment:
      ORTHANC__REMOTE_ACCESS_ALLOWED: "true"
      ORTHANC__AUTHENTICATION_ENABLED: "true"
      ORTHANC__REGISTERED_USERS: |
        {"orthancuser": "demo"}

Lua script path
^^^^^^^^^^^^^^^

The path within the Orthanc container for the OpenREM Lua script is specified here - this should not be changed.

Multiple Orthanc Store nodes
----------------------------

If you need more than one DICOM Store server, to listen on a different port for example, copy the whole ``orthanc_1``
section and paste it just after and before the ``volumes`` section. Rename to ``orthanc_2`` and make the port and
any other changes as necessary.

Next time ``docker-compose`` is started the additional Orthanc container will be started. ``docker-compose.yml`` is
also used to stop the containers, so if you are removing the additional Orthanc container stop the containers first.

Additional Orthanc configuration options
----------------------------------------

More configuration options can be found on the `osimis/orthanc Docker Images page
<https://osimis.atlassian.net/wiki/spaces/OKB/pages/26738689/How+to+use+osimis+orthanc+Docker+images#Howtouseosimis/orthancDockerimages?-DICOM>`_

