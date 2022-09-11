###########################
DICOM Network Configuration
###########################

****************************************
Configuring DICOM store nodes in OpenREM
****************************************

You need to configure details of the DICOM store node to enable the query-retrieve functionality.

To configure a DICOM Store SCP, on the ``Config`` menu select ``DICOM networking``, then click
``Add new Store`` and fill in the details (see figure 1):

.. figure:: img/netdicomstorescp.png
   :figwidth: 50%
   :align: right
   :alt: DICOM Store SCP configuration
   :target: _images/netdicomstorescp.png

   Figure 1: DICOM Store SCP configuration

* Name of local store node: This is the *friendly name*, such as ``OpenREM store``
* Application Entity Title of the node: This is the DICOM name for the store, and must be letters or numbers only, no
  spaces, and a maximum of 16 characters
* Port for store node: Port 104 is the reserved DICOM port, but it is common to use *high* ports such as 8104, partly
  because ports up to 1024 usually need more privileges than for the high ports. However, if there is a firewall
  between the remote nodes (modalities, PACS) and the OpenREM server, then you need to make sure that the firewall is
  configured to allow the port you choose here

Third-party DICOM store node for scripted import to OpenREM
===========================================================

*******************************
Status of DICOM Store SCP nodes
*******************************

DICOM Store SCP advanced configuration

.. figure:: img/storenodealive.png
   :figwidth: 50%
   :align: right
   :alt: DICOM Store SCP status "Alive"

.. figure:: img/storenodefail.png
   :figwidth: 50%
   :align: right
   :alt: DICOM Store SCP status "Association fail"

   Figure 3: DICOM Store SCP status - Alive and Association failed

DICOM Store SCP nodes that have been configured are listed in the left column of the DICOM network configuration page.
For each server, the basic details are displayed, including the Database ID which is required for command line/scripted
use of the query-retrieve function.

In the title row of the Store SCP config panel, the status will be reported either as 'Server is alive' or 'Error:
Association fail - server not running?' - see figure 3


****************************************************************
Query retrieve of third-party system, such as a PACS or modality
****************************************************************

To Query-Retrieve a remote host, you will need to configure both a local Store SCP and the remote host.

To configure a remote query retrieve SCP, on the ``Config`` menu select ``DICOM networking``, then click
``Add new QR Node`` and fill in the details:

* Name of QR node: This is the *friendly name*, such as ``PACS QR``
* AE Title of the remote node: This is the DICOM name of the remote node, 16 or fewer letters and numbers, no spaces
* AE Title this server: This is the DICOM name that the query (DICOM C-Find) will come from. This may be important if
  the remote node filters access based on *calling aet*. Normal rules of 16 or fewer letters and numbers, no spaces
* Remote port: Enter the port the remote node is using (eg 104)
* Remote IP address: The IP address of the remote node, for example ``192.168.1.100``
* Remote hostname: Alternatively, if your network has a DNS server that can resolve the hostnames, you can enter the
  hostname instead. If the hostname is entered, it will be used in preference to the IP address, so only enter it if
  you know it will be resolved.

Now go to the :doc:`netdicom-qr` documentation to learn how to use it.


.. _storetroubleshooting:

**********************************
Troubleshooting: openrem_store.log
**********************************

If the default logging settings haven't been changed then there will be a log files to refer to. The default
location is within your ``MEDIAROOT`` folder:

This file contains information about each echo and association that is made against the store node, and any objects that
are sent to it.

The following is an example of the log for a Philips *dose info* image being received:


.. sourcecode:: console

    [21/Feb/2016 21:13:43] INFO [remapp.netdicom.storescp:310] Starting AE... AET:MYSTOREAE01, port:8104
    [21/Feb/2016 21:13:43] INFO [remapp.netdicom.storescp:314] Started AE... AET:MYSTOREAE01, port:8104
    [21/Feb/2016 21:13:43] INFO [remapp.netdicom.storescp:46] Store SCP: association requested
    [21/Feb/2016 21:13:44] INFO [remapp.netdicom.storescp:54] Store SCP: Echo received
    [21/Feb/2016 21:13:46] INFO [remapp.netdicom.storescp:46] Store SCP: association requested
    [21/Feb/2016 21:13:46] INFO [remapp.netdicom.storescp:54] Store SCP: Echo received
    [21/Feb/2016 21:13:49] INFO [remapp.netdicom.storescp:46] Store SCP: association requested
    [21/Feb/2016 21:13:49] INFO [remapp.netdicom.storescp:54] Store SCP: Echo received
    [21/Feb/2016 21:13:50] INFO [remapp.netdicom.storescp:46] Store SCP: association requested
    [21/Feb/2016 21:13:50] INFO [remapp.netdicom.storescp:54] Store SCP: Echo received
    [21/Feb/2016 21:13:51] INFO [remapp.netdicom.storescp:46] Store SCP: association requested
    [21/Feb/2016 21:13:51] INFO [remapp.netdicom.storescp:54] Store SCP: Echo received
    [21/Feb/2016 21:14:39] INFO [remapp.netdicom.storescp:46] Store SCP: association requested
    [21/Feb/2016 21:14:39] INFO [remapp.netdicom.storescp:78] Received C-Store. Stn name NM-54316, Modality CT,
    SOPClassUID Secondary Capture Image Storage, Study UID 1.2.840.113564.9.1.2843752344.47.2.5000947881 and Instance
    UID 1.2.840.113704.7.1.1.4188.1234134540.349
    [21/Feb/2016 21:14:39] INFO [remapp.netdicom.storescp:232] File
    /var/openrem/media/dicom_in/1.2.840.113704.7.1.1.4188.1453134540.349.dcm written
    [21/Feb/2016 21:14:39] INFO [remapp.netdicom.storescp:263] Processing as Philips Dose Info series
    ...etc











.. _`Issue #337`: https://bitbucket.org/openrem/openrem/issues/337/storescp-is-killed-if-daemonized-when