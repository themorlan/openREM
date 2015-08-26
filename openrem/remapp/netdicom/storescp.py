#!/usr/bin/python
"""
Storage SCP example.

This demonstrates a simple application entity that support a number of
Storage service classes. For this example to work, you need an SCU
sending to this host on specified port.

For help on usage,
python storescp.py -h
"""
import os
import sys
import errno

# setup django/OpenREM
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1,projectpath)
os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'

try:
    import netdicom
    from distutils.version import StrictVersion
    if StrictVersion(netdicom.__version__.__version__) <= StrictVersion('0.8.1'):
        sys.exit('Pynedicom > 0.8.1 needs to be installed, see http://docs.openrem.org/en/latest/install.html')
except ImportError:
    sys.exit('Pynedicom > 0.8.1 needs to be installed, see http://docs.openrem.org/en/latest/install.html')
from netdicom import AE, StorageSOPClass, VerificationSOPClass
from dicom.UID import ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian
from dicom.dataset import Dataset, FileDataset
import tempfile
from django.views.decorators.csrf import csrf_exempt

# netdicom.debug()

# callbacks
def OnAssociateRequest(association):
    print "association requested"


def OnAssociateResponse(association):
    print "Association response received"


def OnReceiveEcho(self):
    print "Echo received"


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


@csrf_exempt
def OnReceiveStore(SOPClass, DS):
    import datetime
    from remapp.extractors.dx import dx
    from remapp.extractors.mam import mam
    from remapp.extractors.rdsr import rdsr
    from remapp.extractors.ct_philips import ct_philips
    from openremproject.settings import MEDIA_ROOT
    from openremproject.settings import RM_DCM_NOMATCH

    print "Received C-STORE"
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = DS.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = DS.SOPInstanceUID
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.9.5224.1.0.6.0.1"  # Using Medical Connections allocated UID
    datestamp = datetime.datetime.now()
    path = os.path.join(
#        MEDIA_ROOT, "dicom_in", datestamp.strftime("%Y"), datestamp.strftime("%m"), datestamp.strftime("%d")
        MEDIA_ROOT, "dicom_in"
    )
    mkdir_p(path)
    filename = os.path.join(path, "{0}.dcm".format(DS.SOPInstanceUID))
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble="\0" * 128)
    ds.update(DS)
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.save_as(filename)
    print "File %s written" % filename
    print DS.SOPClassUID
    if (DS.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.67'     # X-Ray Radiation Dose SR
        or DS.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.22'  # Enhanced SR, as used by GE
    ):
        print "RDSR"
        rdsr.delay(filename)
    elif ( DS.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1'      # CR Image Storage
        or DS.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.1'    # Digital X-Ray Image Storage for Presentation
        or DS.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.1.1'  # Digital X-Ray Image Storage for Processing
    ):
        print "DX"
        dx.delay(filename)
    elif ( DS.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.2'    # Digital Mammography X-Ray Image Storage for Presentation
        or DS.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.2.1'  # Digital Mammography X-Ray Image Storage for Processing
        or (DS.SOPClassUID == '1.2.840.10008.5.1.4.1.1.7'     # Secondary Capture Image Storage, for processing
            and DS.Modality == 'MG'                           # Selenia proprietary DBT projection objects
            and 'ORIGINAL' in DS.ImageType
        )
    ):
        print "Mammo"
        mam.delay(filename)
    elif (DS.SOPClassUID == '1.2.840.10008.5.1.4.1.1.7'
          and DS.Manufacturer == 'Philips'
          and DS.SeriesDescription == 'Dose Info'
    ):
        print "Philips CT Dose Info image"
        ct_philips.delay(filename)
    elif RM_DCM_NOMATCH:
        os.remove(filename)
        print "DICOM not RDSR, DX, Mammo or Philips CT dose info; file deleted. Modality type was {0}".format(DS.Modality)

    # must return appropriate status
    return SOPClass.Success

from celery import shared_task


@shared_task(queue='stores')
def web_store(store_pk=None):
    import time
    from remapp.models import DicomStoreSCP
    from django.core.exceptions import ObjectDoesNotExist

    try:
        conf = DicomStoreSCP.objects.get(pk__exact=store_pk)
        aet = conf.aetitle
        port = conf.port
        conf.task_id = web_store.request.id
        conf.save()
    except ObjectDoesNotExist:
        sys.exit("Attempt to start DICOM Store SCP with an invalid database pk")

    # setup AE
    MyAE = AE(
        aet, port, [],
        [StorageSOPClass, VerificationSOPClass],
        [ExplicitVRLittleEndian, ImplicitVRLittleEndian]
    )
    MyAE.OnAssociateRequest = OnAssociateRequest
    MyAE.OnAssociateResponse = OnAssociateResponse
    MyAE.OnReceiveStore = OnReceiveStore
    MyAE.OnReceiveEcho = OnReceiveEcho

    # start AE
    conf.status = "Starting AE... AET:{0}, port:{1}".format(aet, port)
    conf.save()
    MyAE.start()
    conf.status = "Started AE... AET:{0}, port:{1}".format(aet, port)
    conf.save()
    print "Started AE... AET:{0}, port:{1}".format(aet, port)

    while 1:
        time.sleep(1)
        stay_alive = DicomStoreSCP.objects.get(pk__exact=store_pk)
        if not stay_alive.run:
            MyAE.Quit()
            print "AE Stopped... AET:{0}, port:{1}".format(aet, port)
            break


if __name__ == "__main__":
    import sys
    sys.exit(store())
