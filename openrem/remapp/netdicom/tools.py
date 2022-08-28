# This Python file uses the following encoding: utf-8

# OpenREM root UID: 1.2.826.0.1.3680043.9.5224.
# Provided by Medical Connections https://www.medicalconnections.co.uk/FreeUID

# OpenREM root UID: 1.3.6.1.4.1.45593.
# Provided by IANA as a private enterprise number

# ImplementationUID 1.2.826.0.1.3680043.9.5224.1.0.6.0.1
# = 1.2.826.0.1.3680043.9.5224.1.versionnumber.betanumber
# IANA version
# = 1.3.6.1.4.1.45593.1.0.7.0.1

# UID root for objects
# = 1.2.826.0.1.3680043.9.5224.2.machine-root.machineID.numberperimage
# where numberperimage  might consist of yyyymmddhhmmssss.number

# pydicom has a UID generator of the form:
# root + mac + pid + second + microsecond, eg
# 1.2.826.0.1.3680043.9.5224.2.+8796759879378+15483+44+908342
# 1.2.826.0.1.3680043.9.5224.2.87967598793781548344908342
# which is 54 characters but process ID could be longer.

# 1.3.6.1.4.1.45593.1.2.879675987937815483yyyymmddssssssss
# would be 55 characters - process ID could be longer.
# Includes an extra 1. after the root UID to enable future use for
# anything else.

# UIDs used for test code will be pydicom.uid.generate_uid with prefix of
# 1.3.6.1.4.1.45593.999.


import logging

from django.conf import settings
from django.utils.translation import gettext as _
from pynetdicom import AE, VerificationPresentationContexts

from remapp.models import DicomRemoteQR, DicomStoreSCP

logger = logging.getLogger("remapp.netdicom.qrscu")


def echoscu(scp_pk=None, store_scp=False, qr_scp=False):
    """
    Function to check if built-in Store SCP or remote Query-Retrieve SCP returns a DICOM echo
    :param scp_pk: Primary key if either Store or QR SCP in database
    :param store_scp: True if checking Store SCP
    :param qr_scp: True if checking QR SCP
    :return: 'AssocFail', Success or ?
    """

    if store_scp and scp_pk:
        scp = DicomStoreSCP.objects.get(pk=scp_pk)
        if not scp.peer:
            if settings.DOCKER_INSTALL:
                msg = _("Store Docker container name is missing")
            else:
                msg = _("Store hostname is missing (normally localhost)")
            logger.error(f"{scp.name} (Database ID {scp_pk}): {msg}")
            return f"{msg} - modify to add"
        remote_host = scp.peer
        our_aet = "OPENREMECHO"
    elif qr_scp and scp_pk:
        scp = DicomRemoteQR.objects.get(pk=scp_pk)
        if scp.hostname:
            remote_host = scp.hostname
        else:
            remote_host = scp.ip
        our_aet = scp.callingaet
        if not our_aet:
            our_aet = "OPENREMECHO"
    else:
        logger.warning("echoscu called without SCP information")
        return 0

    remote_port = scp.port
    remote_aet = scp.aetitle

    ae = AE()
    ae.requested_contexts = VerificationPresentationContexts
    ae.ae_title = our_aet

    assoc = ae.associate(remote_host, remote_port, ae_title=remote_aet)

    if assoc.is_established:
        status = assoc.send_c_echo()

        if status:
            if status.Status == 0x0000:
                logger.info(
                    f"Returning Success response from echo to {remote_host} {remote_port} {remote_aet}"
                )
                assoc.release()
                return "Success"
            else:
                logger.info(
                    "Returning EchoFail response from echo to {0} {1} {2}. Type is {3}.".format(
                        remote_host, remote_port, remote_aet, status.Status
                    )
                )
                assoc.release()
                return "Association created, but EchoFail"
        else:
            print("Connection timed out, was aborted or received invalid response")
            logger.info(
                "Returning EchoFail response from echo to {0} {1} {2}. No status.".format(
                    remote_host, remote_port, remote_aet
                )
            )
            assoc.release()
            return "EchoFail"
    else:
        if assoc.is_rejected:
            msg = "{0}: {1}".format(
                assoc.acceptor.primitive.result_str, assoc.acceptor.primitive.reason_str
            )
            logger.info(
                "Association rejected from {0} {1} {2}. {3}".format(
                    remote_host, remote_port, remote_aet, msg
                )
            )
            return msg
        if assoc.is_aborted:
            msg = "Association aborted or never connected"
            logger.info(
                "{3} to {0} {1} {2}".format(remote_host, remote_port, remote_aet, msg)
            )
            return msg
        msg = "Association Failed"
        logger.info(
            "{3} with {0} {1} {2}".format(remote_host, remote_port, remote_aet, msg)
        )
        return msg
