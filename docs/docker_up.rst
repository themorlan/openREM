docker-compose up errors
========================

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


