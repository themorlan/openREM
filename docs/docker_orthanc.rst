DICOM store configuration (Orthanc)
===================================

Orthanc provides the DICOM Store functionality to enable scanners to send directly to OpenREM, and for
query-retrieve to function. Configuration is in the orthanc section of ``docker-compose.yml``

OpenREM Lua script configuration
--------------------------------

This file is formatted as YAML:

* Strings need to quoted or placed on a new line after a ``|``
* A ``:`` and a space separate the variable name and the value, and spaces are used at the start of the line to create
  a hierarchy. See the examples below.

Edit the ``docker-compose.yml`` file to make the changes. They will take effect next time ``docker-compose up -d``
is run.

Find the ``orthanc_1`` definition near the end of the file.


Objects to be ignored
^^^^^^^^^^^^^^^^^^^^^

Lists of things to ignore. Orthanc will ignore anything matching the content of these comma separated lists: they will
not be imported into OpenREM. Some examples have been added below - note the formatting syntax.
``STATION_NAMES_TO_IGNORE`` has the value on a new line with a ``|`` instead of being quoted, to show this syntax
option:

.. code-block:: yaml

    environment:
      MANUFACTURERS_TO_IGNORE: "{'Faxitron X-Ray LLC', 'Gendex-KaVo'}"
      MODEL_NAMES_TO_IGNORE: "{'CR 85', 'CR 75'}"
      STATION_NAMES_TO_IGNORE: |
        {'CR85 Main', 'CR75 Main'}
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
put them in the ``imports/physics/`` folder. Set it to ``"false"`` to disable this feature

.. code-block:: yaml

    environment:
      USE_PHYSICS_FILTERING: "true"

A list to check against patient name and ID to see if the images should be kept.

.. code-block:: yaml

    environment:
      PHYSICS_TO_KEEP: "{'physics',}"

Orthanc Configuration
---------------------

This section is formatted as JSON. It can contain any configuration options that appear in the standard Orthanc
``orthanc.json`` file, but the ones that are needed for OpenREM are included
as standard and described below.

* Strings need to quoted with double quotes ``"``.

DICOM Application Entity Title
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Application Entity Title of the Store Server. Should be up to 16 characters, no spaces. This server isn't fussy
by default, so if remote nodes connect using a different AETitle that is ok.

.. code-block:: yaml

    ORTHANC_JSON: |
      {
        // DICOM Store configuration
        "DicomAet" : "OPENREM",
      }

DICOM Port
^^^^^^^^^^

The default port for DICOM store is set to ``104``.

To use a different port, change the first number of the pair in ports. The first number is the port exposed outside of
Docker, the second number is used internally by the Orthanc container.

For example, to use port 8104:

.. code-block:: yaml

    ports:
    # DICOM store port (first number)
      - 8104:4242

Orthanc web interface
^^^^^^^^^^^^^^^^^^^^^

There will normally not be any studies in the Orthanc database once they have been processed, but if you want to
enable the Orthanc web viewer, enable the port in and set ``RemoteAccessAllowed`` to ``true`` in the ``ORTHANC_JSON``
section. The first number in the port configuration can be changed if required:

.. code-block:: yaml

    ports:
    # Othanc web interface
      - 8042:8042

.. code-block:: yaml

    ORTHANC_JSON: |
      {
        "Name" : "OpenREM Orthanc",
        "RemoteAccessAllowed" : true,
        "AuthenticationEnabled" : true,
        "RegisteredUsers" : {
          "orthancuser": "demo"
        },
      }

Lua script path
^^^^^^^^^^^^^^^

The path within the Orthanc container for the OpenREM Lua script is specified here - this should not be changed
(see below for advanced options).

.. code-block:: yaml

    ORTHANC_JSON: |
      {
        // OpenREM Lua Script
        "LuaScripts" : [
          "/etc/share/orthanc/scripts/openrem_orthanc_config_docker.lua"
        ]
      }

Advanced options
----------------

Multiple stores
^^^^^^^^^^^^^^^

If you need more than one DICOM Store server, to listen on a different port for example, copy the whole ``orthanc_1``
section in ``docker-compose.yml`` and paste it after the ``orthanc_1`` block. Rename to ``orthanc_2``.

Next time ``docker-compose`` is started the additional Orthanc container will be started. ``docker-compose.yml`` is
also used to stop the containers, so if you are removing the additional Orthanc container stop the containers first.

Advanced Orthanc configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any of the Orthanc configuration settings can be set in the ``ORTHANC_JSON`` section. The default configuration
can be seen `on the Orthanc Server webpages
<https://hg.orthanc-server.com/orthanc/file/Orthanc-1.8.2/OrthancServer/Resources/Configuration.json>`_ including
documentation as to how they are used.

A custom version of the ``openrem_orthanc_config_docker.lua`` script can be used if required. Copy the existing one
and place the new one, with a new name, in the ``orthanc/`` folder, and set the ``LuaScripts`` value in
``ORTHANC_JSON`` to match.

**Pay special attention to the first sections**, up to the ``ToAscii`` function,
these sections have been changed for the Docker implementation.
