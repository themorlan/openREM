***********************
Webserver configuration
***********************

Webserver timeout
=================

Some long running actions can cause webserver errors if they take longer than the timeout setting in the webserver,
particularly generating fluoroscopy :doc:`skindosemap`. The default setting is 300 seconds, or five minutes. To modify
this, change the following two settings:

Edit ``docker-compose.yml`` in the Docker OpenREM installation folder and change the timeout setting on the following
line:

.. code-block:: yaml

    services:
      openrem:
        container_name: openrem
        command: gunicorn openremproject.wsgi:application --bind 0.0.0.0:8000 --timeout 300

Edit ``nginx-conf/conf.d/openrem.conf`` and set the same timeout:

.. code-block:: nginx

    server {
        listen 80;
        location / {
            proxy_pass http://openremproject;
            # ...
            proxy_read_timeout 300s;
        }

Reload the containers:

.. code-block:: console

    $ docker-compose down
    $ docker-compose up -d

Non-Docker Linux install
------------------------

Change the same settings as for the Docker install above:

.. code-block:: console

    $ sudo nano /etc/nginx/sites-available/openrem-server

and

.. code-block:: console

    $ sudo nano /etc/systemd/system/openrem-gunicorn.service

.. code-block:: none

    ExecStart=/var/dose/veopenrem3/bin/gunicorn \
        --bind unix:/tmp/openrem-server.socket \
        openremproject.wsgi:application --timeout 300

Adding an SSL certificate
=========================

It is advisable to add an SSL certificate to the web server even though it might only be accessible within an
institution. There are several reasons for this, but one main one is that over time web browsers will give more and more
warnings about entering passwords into non-HTTPS websites.

It is likely that within your institution there will be a corporate trusted root certificate and a mechanism of getting
certificates you generate for your servers signed by that root certificate. How to generate a certificate signing
request (CSR) and private key are beyond the scope of these documents, but this blog post was helpful when we were
learning how to do this at our institution:
https://www.endpoint.com/blog/2014/10/30/openssl-csr-with-alternative-names-one

Once you have a signed certificate, place it and the key in ``nginx-conf/certs``, where it will be available in the
Nginx container at ``/etc/ssl/private``.

There are two conf files in ``nginx-conf/conf.d`` - the default one is ``openrem.conf``. There is an alternative one
named ``openrem-secure.conf.example``. Edit the second file as required, then rename them both so the secure version
is the only one to have a ``.conf`` ending.

Ensure the the following lines are updated for the name of your server and the names of your signed certificate and key:

.. code-block:: nginx

    server {
        listen 443 ssl;
        server_name add_server_name_here;
        ssl_certificate /etc/ssl/private/openrem.cer;
        ssl_certificate_key /etc/ssl/private/openrem.key;

        # ...
    }
