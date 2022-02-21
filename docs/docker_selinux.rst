Docker SELinux configuration
============================

**Document not ready for translation**

SELinux will prevent bind mounts in Docker with the standard configuration, which will be seen because Orthanc fails
to start. SELinux is commonly enabled on Red Hat, Fedora and associated distributions.

The  ``docker-compose.yml`` file needs to be edited to fix this.

Change nginx configuration
--------------------------

Find the following section:

.. code-block:: yaml

    nginx:
      container_name: openrem-nginx
      restart: unless-stopped
      image: nginx:1.17.8-alpine
      volumes:
        - media_volume:/home/app/openrem/mediafiles
        - static_volume:/home/app/openrem/staticfiles
    # For SELinux (RedHat, Fedora etc), add :z to the end of next two lines
        - ./nginx-conf/conf.d:/etc/nginx/conf.d
        - ./nginx-conf/certs:/etc/ssl/private

Follow the instruction to edit the ``nginx-conf`` lines, like this:

.. code-block:: yaml

    # For SELinux (RedHat, Fedora etc), add :z to the end of next two lines
        - ./nginx-conf/conf.d:/etc/nginx/conf.d:z
        - ./nginx-conf/certs:/etc/ssl/private:z

Change the Orthanc configuration
--------------------------------

Find the following section:

.. code-block:: yaml

    orthanc_1:
      container_name: openrem-orthanc-1
      restart: unless-stopped
      image: openrem/orthanc
      volumes:
        - imports_volume:/imports
    # For SELinux (RedHat, Fedora etc), add :z to the end of next line
        - ./orthanc:/etc/share/orthanc/scripts/

Follow the instruction to edit the ``orthanc_1`` line, like this:

.. code-block:: yaml

    # For SELinux (RedHat, Fedora etc), add :z to the end of next line
        - ./orthanc:/etc/share/orthanc/scripts/:z
