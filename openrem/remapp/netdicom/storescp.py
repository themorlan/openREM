#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2012-2019  The Royal Marsden NHS Foundation Trust
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    Additional permission under section 7 of GPLv3:
#    You shall not make any use of the name of The Royal Marsden NHS
#    Foundation trust in connection with this Program in any press or
#    other public announcement without the prior written consent of
#    The Royal Marsden NHS Foundation Trust.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
..  module:: storescp
    :synopsis: Module to provide a DICOM STORE SCP function for OpenREM

..  moduleauthor:: Ed McDonagh

"""
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

from pynetdicom import (
    AE, evt,
    StoragePresentationContexts,
)
from pynetdicom.sop_class import VerificationSOPClass
from ..version import __implementation_uid__ as OPENREM_UID
from ..version import __openrem_root_uid__ as ROOT_UID
from ..version import __version__ as OPENREM_VERSION


# Implement a handler evt.EVT_C_STORE
def handle_store(event):
    """Handle a C-STORE request event."""
    from openremproject.settings import MEDIA_ROOT
    from ..extractors.dx import dx
    from ..extractors.mam import mam
    from ..extractors.rdsr import rdsr
    from ..extractors.ct_philips import ct_philips
    from ..models import DicomDeleteSettings

    del_settings = DicomDeleteSettings.objects.get()
    # Decode the C-STORE request's *Data Set* parameter to a pydicom Dataset
    ds = event.dataset

    # Add the File Meta Information
    ds.file_meta = event.file_meta
    ds.file_meta.ImplementationClassUID = OPENREM_UID
    ds.file_meta.ImplementationVersionName = f'OpenREM_{OPENREM_VERSION}'

    # Save the dataset using the SOP Instance UID as the filename
    path = os.path.join(
        MEDIA_ROOT, "dicom_in"
    )
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, "{0}.dcm".format(ds.SOPInstanceUID))
    ds.save_as(filename=filename, write_like_original=False)

    if (ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.67'  # X-Ray Radiation Dose SR
        or ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.22'  # Enhanced SR, as used by GE
        ):
        logger.info("Processing as RDSR")
        rdsr.delay(filename)
    elif (ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1'  # CR Image Storage
          or ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.1'  # Digital X-Ray Image Storage for Presentation
          or ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.1.1'  # Digital X-Ray Image Storage for Processing
          ):
        logger.info("Processing as DX")
        dx.delay(filename)
    elif (ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.2'  # Digital Mammography X-Ray Image Storage for Presentation
          or ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.2.1'  # Digital Mammography X-Ray Image Storage for Processing
          or (ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage, for processing
              and ds.Modality == 'MG'  # Selenia proprietary DBT projection objects
              and 'ORIGINAL' in ds.ImageType
              )
          ):
        logger.info("Processing as MG")
        mam.delay(filename)
    elif ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.7':
        try:
            manufacturer = ds.Manufacturer
            series_description = ds.SeriesDescription
        except AttributeError:
            if del_settings.del_no_match:
                os.remove(filename)
                logger.info("Secondary capture object with either no manufacturer or series description. Deleted.")
            return 0x0000
        if manufacturer == 'Philips' and series_description == 'Dose Info':
            logger.info("Processing as Philips Dose Info series")
            ct_philips.delay(filename)
        elif del_settings.del_no_match:
            os.remove(filename)
            logger.info("Can't find anything to do with this file - it has been deleted")
    elif del_settings.del_no_match:
        os.remove(filename)
        logger.info("Can't find anything to do with this file - it has been deleted")

    # Return a 'Success' status
    return 0x0000


def start_store(store_pk=None):
    """Function to start DICOM Store SCP"""
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
        logger.error("Attempt to start DICOM Store SCP with an invalid database pk")
        sys.exit("Attempt to start DICOM Store SCP with an invalid database pk")

    handlers = [(evt.EVT_C_STORE, handle_store)]

    # Initialise the Application Entity
    ae = AE()

    # Add the supported presentation contexts
    ae.supported_contexts = StoragePresentationContexts
    ae.add_supported_context(VerificationSOPClass)
    ae.implementation_class_uid = OPENREM_UID
    ae.implementation_version_name = f'OpenREM_{OPENREM_VERSION}'

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
