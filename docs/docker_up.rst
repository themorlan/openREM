Errors at docker-compose up
===========================

**Document not ready for translation**

Cannot start service nginx
--------------------------

Error message when running ``docker-compose up -d`` (example from Windows)::

    ERROR: for nginx Cannot start service nginx: driver failed programming external connectivity on endpoint
    openrem-nginx (...): Error starting userland proxy: listen tcp 0.0.0.0:80: bind: An attempt was made to access a
    socket in a way forbidden by its access permissions.
    ERROR: Encountered errors while bringing up the project.

This error indicates port 80 is not available for Docker/OpenREM. Try:

* Shutting down other web servers, such as IIS
* Using an alternative port temporarily for OpenREM:

    * Edit the ``docker-compose.yml`` file
    * Find the section that includes

    .. code-block:: yaml

          nginx:
            ports:
              - 80:80

    * Change the external facing port to a high number, for example:

    .. code-block:: yaml

          nginx:
            ports:
              - 8080:80

Now stop and start the containers again:

.. code-block:: console

    $ docker-compose down
    $ docker-compose up -d

If there are no errors, check that the containers are up and which ports are in use:

.. code-block:: console

    $ docker-compose ps

Connection was reset, Orthanc restarting
----------------------------------------
* After installation, browsing to the webservice reports "The connection was reset".
* ``docker-compose ps`` reports:

.. code-block::

    openrem-orthanc-1                      /docker-entrypoint.sh /tmp ...   Restarting

* Orthanc Docker logs include:

.. code-block::

    openrem-orthanc-1 | E1208 12:51:29.599961 OrthancException.cpp:57] The specified path does not point to a regular file: The path does not point to a regular file: /etc/share/orthanc/scripts/openrem_orthanc_config_docker.lua
    openrem-orthanc-1 | E1208 12:51:29.600051 ServerIndex.cpp:706] INTERNAL ERROR: ServerIndex::Stop() should be invoked manually to avoid mess in the destruction order!

This might indicate that the bind mounts have not worked. This might be due to SELinux, particularly if you are using
Red Hat or Fedora or related distributions.

See :doc:`docker_selinux`
