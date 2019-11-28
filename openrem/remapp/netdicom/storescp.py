import django
import logging
import os
import sys

logger = logging.getLogger('remapp.netdicom.storescp')

# setup django/OpenREM
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'
django.setup()

from pydicom.dataset import Dataset

from pynetdicom import (
    AE, evt,
    StoragePresentationContexts,
)
from pynetdicom.sop_class import VerificationSOPClass


# Implement a handler evt.EVT_C_STORE
def handle_store(event):
    """Handle a C-STORE request event."""
    # Decode the C-STORE request's *Data Set* parameter to a pydicom Dataset
    ds = event.dataset

    # Add the File Meta Information
    ds.file_meta = event.file_meta

    # Save the dataset using the SOP Instance UID as the filename
    ds.save_as(ds.SOPInstanceUID, write_like_original=False)

    # Return a 'Success' status
    return 0x0000


def start_store(store_pk=None):
    import socket
    import time
    from ..models import DicomStoreSCP
    from django.core.exceptions import ObjectDoesNotExist

    try:
        conf = DicomStoreSCP.objects.get(pk__exact=store_pk)
        our_aet = conf.aetitle
        our_port = conf.port
        conf.run = True
        conf.save()
    except ObjectDoesNotExist:
        logger.error(u"Attempt to start DICOM Store SCP with an invalid database pk")
        sys.exit(u"Attempt to start DICOM Store SCP with an invalid database pk")

    handlers = [(evt.EVT_C_STORE, handle_store)]

    # Initialise the Application Entity
    ae = AE()

    # Add the supported presentation contexts
    ae.supported_contexts = StoragePresentationContexts
    ae.add_supported_context(VerificationSOPClass)
    ae.implementation_class_uid = '1.3.6.1.4.1.45593.1.1'
    ae.implementation_version_name = 'OpenREM_1_0_0'

    # Start listening for incoming association requests
    try:
        msg = f'Starting Store SCP AET {our_aet}, port {our_port}'
        logger.info(msg)
        conf.status = msg
        conf.save()
        scp = ae.start_server(('', our_port), ae_title=our_aet, evt_handlers=handlers, block=False)
        msg = f'Started Store SCP AET {our_aet}, port {our_port}'
        logger.info(msg)
        conf.status = msg
        conf.save()

        while 1:
            time.sleep(1)
            stay_alive = DicomStoreSCP.objects.get(pk__exact=store_pk)
            if not stay_alive.run:
                scp.shutdown()
                logger.info(f'Stopped Store SCP AET {our_aet}, port {our_port}')
                break

    except PermissionError:
        msg = f"Starting Store SCP AE AET:{our_aet}, port:{our_port} failed: permission denied."
        conf.status = msg
        conf.save()
        logger.error = msg
    except OSError as e:
        msg = f"Starting Store SCP AE AET:{our_aet}, port:{our_port} failed:"
        if e.errno == 98:
            conf.status = msg + " port already in use."
            conf.save()
            logger.error = msg + f' Err {e.errno} {e.strerror}'
        else:
            conf.status = msg + f'Err {e.errno} {e.strerror}'
            conf.save()
            logger.error = msg + f'Err {e.errno} {e.strerror}'


def _interrupt(store_pk=None):
    from ..models import DicomStoreSCP
    stay_alive = DicomStoreSCP.objects.get(pk__exact=store_pk)
    stay_alive.run = False
    stay_alive.status = "Store interrupted from the shell"
    stay_alive.save()
