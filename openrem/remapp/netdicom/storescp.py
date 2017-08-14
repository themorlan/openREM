# This Python file uses the following encoding: utf-8
#!/usr/bin/python
"""
Storage SCP
"""
import errno
import logging
import os
import socket
import sys

import django
from django.views.decorators.csrf import csrf_exempt
from pydicom.dataset import Dataset, FileDataset
from pydicom.filewriter import write_file
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, \
    ExplicitVRBigEndian, DeflatedExplicitVRLittleEndian
from pynetdicom3 import AE, StorageSOPClassList, VerificationSOPClass

# setup django/OpenREM
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'
django.setup()

logger = logging.getLogger(name='remapp.netdicom.storescp')

_OPENREM_STORAGE_CLASSES = {'ComputedRadiographyImageStorage': '1.2.840.10008.5.1.4.1.1.1',
                            'DigitalXRayImagePresentationStorage': '1.2.840.10008.5.1.4.1.1.1.1',
                            'DigitalXRayImageProcessingStorage': '1.2.840.10008.5.1.4.1.1.1.1.1.1',
                            'DigitalMammographyXRayImagePresentationStorage': '1.2.840.10008.5.1.4.1.1.1.2',
                            'DigitalMammographyXRayImageProcessingStorage': '1.2.840.10008.5.1.4.1.1.1.2.1',
                            'DigitalIntraOralXRayImagePresentationStorage': '1.2.840.10008.5.1.4.1.1.1.3',
                            'DigitalIntraOralXRayImageProcessingStorage': '1.2.840.10008.5.1.1.4.1.1.3.1',
                            'CTImageStorage': '1.2.840.10008.5.1.4.1.1.2',
                            'EnhancedCTImageStorage': '1.2.840.10008.5.1.4.1.1.2.1',
                            'LegacyConvertedEnhancedCTImageStorage': '1.2.840.10008.5.1.4.1.1.2.2',
                            'SecondaryCaptureImageStorage': '1.2.840.10008.5.1.4.1.1.7',
                            'MultiframeSingleBitSecondaryCaptureImageStorage': '1.2.840.10008.5.1.4.1.1.7.1',
                            'MultiframeGrayscaleByteSecondaryCaptureImageStorage': '1.2.840.10008.5.1.4.1.1.7.2',
                            'MultiframeGrayscaleWordSecondaryCaptureImageStorage': '1.2.840.10008.5.1.4.1.1.7.3',
                            'MultiframeTrueColorSecondaryCaptureImageStorage': '1.2.840.10008.5.1.4.1.1.7.4',
                            'TwelveLeadECGWaveformStorage': '1.2.840.10008.5.1.4.1.1.9.1.1',
                            'GeneralECGWaveformStorage': '1.2.840.10008.5.1.4.1.1.9.1.2',
                            'AmbulatoryECGWaveformStorage': '1.2.840.10008.5.1.4.1.1.9.1.3',
                            'HemodynamicWaveformStorage': '1.2.840.10008.5.1.4.1.1.9.2.1',
                            'CardiacElectrophysiologyWaveformStorage': '1.2.840.10008.5.1.4.1.1.9.3.1',
                            'ArterialPulseWaveformStorage': '1.2.840.10008.5.1.4.1.1.9.5.1',
                            'RespiratoryWaveformStorage': '1.2.840.10008.5.1.4.1.1.9.6.1',
                            'XRayAngiographicImageStorage': '1.2.840.10008.5.1.4.1.1.12.1',
                            'EnhancedXAImageStorage': '1.2.840.10008.5.1.4.1.1.12.1.1',
                            'XRayRadiofluoroscopicImageStorage': '1.2.840.10008.5.1.4.1.1.12.2',
                            'EnhancedXRFImageStorage': '1.2.840.10008.5.1.4.1.1.12.2.1',
                            'XRay3DAngiographicImageStorage': '1.2.840.10008.5.1.4.1.1.13.1.1',
                            'XRay3DCraniofacialImageStorage': '1.2.840.10008.5.1.4.1.1.13.1.2',
                            'BreastTomosynthesisImageStorage': '1.2.840.10008.5.1.4.1.1.13.1.3',
                            'BreastProjectionXRayImagePresentationStorage': '1.2.840.10008.5.1.4.1.1.13.1.4',
                            'BreastProjectionXRayImageProcessingStorage': '1.2.840.10008.5.1.4.1.1.13.1.5',
                            'NuclearMedicineImageStorage': '1.2.840.10008.5.1.4.1.1.20',
                            'RawDataStorage': '1.2.840.10008.5.1.4.1.1.66',
                            'BasicTextSRStorage': '1.2.840.10008.5.1.4.1.1.88.11',
                            'EnhancedSRStorage': '1.2.840.10008.5.1.4.1.1.88.22',
                            'ComprehensiveSRStorage': '1.2.840.10008.5.1.4.1.1.88.33',
                            'Comprehenseice3DSRStorage': '1.2.840.10008.5.1.4.1.1.88.34',
                            'ExtensibleSRStorage': '1.2.840.10008.5.1.4.1.1.88.35',
                            'ProcedureSRStorage': '1.2.840.10008.5.1.4.1.1.88.40',
                            'MammographyCADSRStorage': '1.2.840.10008.5.1.4.1.1.88.50',
                            'KeyObjectSelectionStorage': '1.2.840.10008.5.1.4.1.1.88.59',
                            'ChestCADSRStorage': '1.2.840.10008.5.1.4.1.1.88.65',
                            'XRayRadiationDoseSRStorage': '1.2.840.10008.5.1.4.1.1.88.67',
                            'RadiopharmaceuticalRadiationDoseSRStorage': '1.2.840.10008.5.1.4.1.1.88.68',
                            'ColonCADSRStorage': '1.2.840.10008.5.1.4.1.1.88.69',
                            'ImplantationPlanSRDocumentStorage': '1.2.840.10008.5.1.4.1.1.88.70',
                            'EncapsulatedPDFStorage': '1.2.840.10008.5.1.4.1.1.104.1',
                            'EncapsulatedCDAStorage': '1.2.840.10008.5.1.4.1.1.104.2',
                            'PositronEmissionTomographyImageStorage': '1.2.840.10008.5.1.4.1.1.128',
                            'EnhancedPETImageStorage': '1.2.840.10008.5.1.4.1.1.130',
                            'LegacyConvertedEnhancedPETImageStorage': '1.2.840.10008.5.1.4.1.1.128.1',
                            'BasicStructuredDisplayStorage': '1.2.840.10008.5.1.4.1.1.131',
                            'RTImageStorage': '1.2.840.10008.5.1.4.1.1.481.1',
                            'RTDoseStorage': '1.2.840.10008.5.1.4.1.1.481.2',
                            'RTStructureSetStorage': '1.2.840.10008.5.1.4.1.1.481.3',
                            'RTBeamsTreatmentRecordStorage': '1.2.840.10008.5.1.4.1.1.481.4',
                            'RTPlanStorage': '1.2.840.10008.5.1.4.1.1.481.5',
                            'RTBrachyTreatmentRecordStorage': '1.2.840.10008.5.1.4.1.1.481.6',
                            'RTTreatmentSummaryRecordStorage': '1.2.840.10008.5.1.4.1.1.481.7',
                            'RTIonPlanStorage': '1.2.840.10008.5.1.4.1.1.481.8',
                            'RTIonBeamsTreatmentRecordStorage': '1.2.840.10008.5.1.4.1.1.481.9',
                            'RTBeamsDeliveryInstructionStorage': '1.2.840.10008.5.1.4.34.7',
                            }


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


@csrf_exempt
def on_c_store(dataset):
    from remapp.extractors.dx import dx
    from remapp.extractors.mam import mam
    from remapp.extractors.rdsr import rdsr
    from remapp.extractors.ct_philips import ct_philips
    from remapp.models import DicomDeleteSettings
    from remapp.version import __netdicom_implementation_version__
    from remapp.version import __version__ as openrem_version
    from openremproject.settings import MEDIA_ROOT

    try:
        logger.info(u"Received C-Store. Stn name %s, Modality %s, SOPClassUID %s, Study UID %s and Instance UID %s",
                    dataset.StationName, dataset.Modality, dataset.SOPClassUID, dataset.StudyInstanceUID, dataset.SOPInstanceUID)
    except AttributeError:
        try:
            logger.info(
                u"Received C-Store - station name missing. Modality %s, SOPClassUID %s, Study UID %s and "
                u"Instance UID %s",
                dataset.Modality, dataset.SOPClassUID, dataset.StudyInstanceUID, dataset.SOPInstanceUID)
        except AttributeError:
            logger.warning(u"Received C-Store - error in logging details")

    if 'TransferSyntaxUID' in dataset:
        del dataset.TransferSyntaxUID  # Don't know why this has become necessary - possibly isn't with pynetdicom3

    del_settings = DicomDeleteSettings.objects.get()
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    file_meta.ImplementationClassUID = "1.3.6.1.4.1.45593.1.{0}".format(__netdicom_implementation_version__)
    file_meta.ImplementationVersionName = "OpenREM_{0}".format(openrem_version)
    path = os.path.join(
        MEDIA_ROOT, "dicom_in"
    )
    mkdir_p(path)
    filename = os.path.join(path, u"{0}.dcm".format(dataset.SOPInstanceUID))
    ds_new = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds_new.update(dataset)
    ds_new.is_little_endian = True
    ds_new.is_implicit_VR = True

    while True:
        try:
            station_name = dataset.StationName
        except:
            station_name = u"missing"
        try:
            ds_new.save_as(filename)
            break
        except ValueError as e:
            # Black magic pydicom method suggested by Darcy Mason:
            # https://groups.google.com/forum/?hl=en-GB#!topic/pydicom/x_WsC2gCLck
            # Not sure if still necessary with pydicom 1.0, and not sure if it would work anyway?
            if "Invalid tag (0018, 7052)" in e.message or "Invalid tag (0018, 7054)" in e.message:
                logger.debug(u"Found illegal use of multiple values of filter thickness using comma. "
                             u"Changing before saving.")
                thickmin = dict.__getitem__(ds_new, 0x187052)
                thickvalmin = thickmin.__getattribute__('value')
                if ',' in thickvalmin:
                    thickvalmin = thickvalmin.replace(',', '\\')
                    thicknewmin = thickmin._replace(value = thickvalmin)
                    dict.__setitem__(ds_new, 0x187052, thicknewmin)
                thickmax = dict.__getitem__(ds_new, 0x187054)
                thickvalmax = thickmax.__getattribute__('value')
                if ',' in thickvalmax:
                    thickvalmax = thickvalmax.replace(',', '\\')
                    thicknewmax = thickmax._replace(value = thickvalmax)
                    dict.__setitem__(ds_new, 0x187054, thicknewmax)
            elif "Invalid tag (01f1, 1027)" in e.message:
                logger.warning(u"Invalid value in tag (01f1,1027), 'exposure time per rotation'. Tag value deleted. "
                               u"Stn name {0}, modality {1}, SOPClass UID {2}, Study UID {3}, Instance UID {4}".format(
                                station_name, dataset.Modality, dataset.SOPClassUID, dataset.StudyInstanceUID, dataset.SOPInstanceUID))
                priv_exp_time = dict.__getitem__(ds_new, 0x1f11027)
                blank_val = priv_exp_time._replace(value='')
                dict.__setitem__(ds_new, 0x1f11027, blank_val)
            elif "Invalid tag (01f1, 1033)" in e.message:
                logger.warning(u"Invalid value in unknown private tag (01f1,1033). Tag value deleted. "
                               u"Stn name {0}, modality {1}, SOPClass UID {2}, Study UID {3}, Instance UID {4}".format(
                                station_name, dataset.Modality, dataset.SOPClassUID, dataset.StudyInstanceUID, dataset.SOPInstanceUID))
                priv_tag = dict.__getitem__(ds_new, 0x1f11033)
                blank_val = priv_tag._replace(value='')
                dict.__setitem__(ds_new, 0x1f11033, blank_val)
            else:
                logger.error(
                    u"ValueError on DCM save {0}. Stn name {1}, modality {2}, SOPClass UID {3}, Study UID {4}, "
                    u"Instance UID {5}".format(
                        e.message, station_name, dataset.Modality, dataset.SOPClassUID, dataset.StudyInstanceUID, dataset.SOPInstanceUID))
                return 0xA700  # Failed - out of resources. Should find a better error
        except IOError as e:
            logger.error(
                    u"IOError on DCM save {0} - does the user running storescp have write rights in the {1} "
                    u"folder?".format(e.message, path))
            return 0xA700  # Failed - out of resources
        except:
            logger.error(
                u"Unexpected error on DCM save: {0}. Stn name {1}, modality {2}, SOPClass UID {3}, Study UID {4}, "
                u"Instance UID {5}".format(
                    sys.exc_info()[0], dataset.StationName, dataset.Modality, dataset.SOPClassUID, dataset.StudyInstanceUID,
                    dataset.SOPInstanceUID))
            return 0xA700  # Failed - out of resources. Should find a better error

    return 0x0000  # Success

    logger.info(u"File %s written", filename)
    if (dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.67'  # X-Ray Radiation Dose SR
        or dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.88.22'  # Enhanced SR, as used by GE
        ):
        logger.info(u"Processing as RDSR")
        rdsr.delay(filename)
    elif (dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1'  # CR Image Storage
          or dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.1'  # Digital X-Ray Image Storage for Presentation
          or dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.1.1'  # Digital X-Ray Image Storage for Processing
          ):
        logger.info(u"Processing as DX")
        dx.delay(filename)
    elif (dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.2'  # Digital Mammography X-Ray Image Storage for Presentation
          or dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.2.1'  # Digital Mammography X-Ray Image Storage for Processing
          or (dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage, for processing
              and dataset.Modality == 'MG'  # Selenia proprietary DBT projection objects
              and 'ORIGINAL' in dataset.ImageType
              )
          ):
        logger.info(u"Processing as MG")
        mam.delay(filename)
    elif dataset.SOPClassUID == '1.2.840.10008.5.1.4.1.1.7':
        try:
            manufacturer = dataset.Manufacturer
            series_description = dataset.SeriesDescription
        except AttributeError:
            if del_settings.del_no_match:
                os.remove(filename)
                logger.info(u"Secondary capture object with either no manufacturer or series description. Deleted.")
            return SOPClass.Success
        if manufacturer == 'Philips' and series_description == 'Dose Info':
            logger.info(u"Processing as Philips Dose Info series")
            ct_philips.delay(filename)
        elif del_settings.del_no_match:
            os.remove(filename)
            logger.info(u"Can't find anything to do with this file - it has been deleted")
    elif del_settings.del_no_match:
        os.remove(filename)
        logger.info(u"Can't find anything to do with this file - it has been deleted")



def web_store(store_pk=None):
    import socket
    import time
    from remapp.models import DicomStoreSCP
    from remapp.netdicom.tools import _create_ae
    from django.core.exceptions import ObjectDoesNotExist

    try:
        conf = DicomStoreSCP.objects.get(pk__exact=store_pk)
        aet = conf.aetitle
        port = conf.port
        conf.run = True
        conf.save()
    except ObjectDoesNotExist:
        logger.error(u"Attempt to start DICOM Store SCP with an invalid database pk")
        sys.exit(u"Attempt to start DICOM Store SCP with an invalid database pk")

    scp_classes = [x for x in _OPENREM_STORAGE_CLASSES]
    # scp_classes = [x for x in StorageSOPClassList]
    scp_classes.append(VerificationSOPClass)
    transfer_syntax = [ExplicitVRLittleEndian, ImplicitVRLittleEndian, ExplicitVRBigEndian]

    # setup AE
    try:
        my_ae = _create_ae(aet,
                           port=port,
                           sop_scu=[],
                           sop_scp=scp_classes,
                           transfer_syntax=transfer_syntax)
        my_ae.on_c_store = on_c_store
        # my_ae.OnReceiveEcho = OnReceiveEcho

        my_ae.maximum_pdu_size = 16384

        # Set timeouts
        my_ae.network_timeout = None
        my_ae.acse_timeout = 60
        my_ae.dimse_timeout = None

        # start AE
        logger.info(u"Starting  Store SCP AE... AET:{0}, port:{1}".format(aet, port))
        conf.status = u"Starting Store SCP AE... AET:{0}, port:{1}".format(aet, port)
        conf.save()
        my_ae.start()
        conf.status = u"Started Store SCP AE... AET:{0}, port:{1}".format(aet, port)
        conf.save()
        logger.info(u"Started Store SCP AE... AET:%s, port:%s", aet, port)

        while 1:
            time.sleep(1)
            stay_alive = DicomStoreSCP.objects.get(pk__exact=store_pk)
            if not stay_alive.run:
                my_ae.quit()
                logger.info(u"Stopped Store SCP AE... AET:%s, port:%s", aet, port)
                break
    except socket.error as socket_err:
        if socket_err.errno == errno.EADDRINUSE:
            conf.status = u"Starting Store SCP AE AET:{0}, port:{1} failed; address already in use!".format(aet, port)
            logger.warning(u"Starting Store SCP AE AET:{0}, port:{1} failed: {2}".format(aet, port, socket_err))
            conf.save()
        elif socket_err.errno == errno.EACCES:
            conf.status = u"Starting Store SCP AE AET:{0}, port:{1} failed; " \
                          u"permission denied (try above 1024?)!".format(aet, port)
            logger.warning(u"Starting Store SCP AE AET:{0}, port:{1} failed: {2}".format(aet, port, socket_err))
            conf.save()
        else:
            conf.status = u"Starting Store SCP AE AET:{0}, port:{1} failed; see logfile".format(aet, port)
            logger.error(u"Starting Store SCP AE AET:{0}, port:{1} failed: {2}".format(aet, port, socket_err))
            conf.save()


def _interrupt(store_pk=None):
    from remapp.models import DicomStoreSCP
    stay_alive = DicomStoreSCP.objects.get(pk__exact=store_pk)
    stay_alive.run = False
    stay_alive.status = u"Store interrupted from the shell"
    stay_alive.save()
