# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2012,2013  The Royal Marsden NHS Foundation Trust
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
..  module:: ctphilips.
    :synopsis: Module to extract radiation dose structured report related data from Philips CT dose report images

..  moduleauthor:: Ed McDonagh

"""
from datetime import datetime, timedelta
import logging
import os
import sys

from decimal import Decimal
import django
from django.db.models import Max, Min, ObjectDoesNotExist
import pydicom

from openrem.remapp.tools.background import (
    record_task_error_exit,
    record_task_related_query,
    record_task_info,
)

from ..tools.dcmdatetime import get_date_time, get_date, get_time
from ..tools.get_values import (
    get_value_kw,
    get_value_num,
    get_or_create_cid,
    get_seq_code_meaning,
    get_seq_code_value,
    list_to_string,
)
from ..tools.hash_id import hash_id
from remapp.tools.send_high_dose_alert_emails import send_ct_high_dose_alert_email

# setup django/OpenREM
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from .extract_common import (  # pylint: disable=wrong-import-order, wrong-import-position
    ct_event_type_count,
    patient_module_attributes,
    add_standard_names,
)
from remapp.models import (  # pylint: disable=wrong-import-order, wrong-import-position
    CtAccumulatedDoseData,
    CtIrradiationEventData,
    CtRadiationDose,
    CtXRaySourceParameters,
    DicomDeleteSettings,
    GeneralEquipmentModuleAttr,
    GeneralStudyModuleAttr,
    PatientIDSettings,
    PatientStudyModuleAttr,
    ScanningLength,
    UniqueEquipmentNames,
)

logger = logging.getLogger(__name__)


def _scanninglength(dataset, event):  # TID 10014
    scanlen = ScanningLength.objects.create(ct_irradiation_event_data=event)
    scanlen.scanning_length = get_value_kw("ScanLength", dataset)
    scanlen.save()


def _ctxraysourceparameters(dataset, event):
    param = CtXRaySourceParameters.objects.create(ct_irradiation_event_data=event)
    param.identification_of_the_xray_source = "A"
    param.kvp = get_value_kw("KVP", dataset)
    mA = get_value_kw("XRayTubeCurrentInuA", dataset)
    if mA:
        param.xray_tube_current = mA / 1000.0
    # exposure time per rotation mandatory for non-localizer exposures, but we don't have it.
    param.save()


def _ctirradiationeventdata(dataset, ct):  # TID 10013
    event = CtIrradiationEventData.objects.create(ct_radiation_dose=ct)
    event.acquisition_protocol = get_value_kw("SeriesDescription", dataset)
    # target region is mandatory, but I don't have it
    acqtype = get_value_kw("AcquisitionType", dataset)
    if acqtype == "CONSTANT_ANGLE":
        event.ct_acquisition_type = get_or_create_cid(
            "113805", "Constant Angle Acquisition"
        )
    elif acqtype == "SPIRAL":
        event.ct_acquisition_type = get_or_create_cid("P5-08001", "Spiral Acquisition")
    elif acqtype == "SEQUENCED":  # guessed
        event.ct_acquisition_type = get_or_create_cid("113804", "Sequenced Acquisition")
    elif acqtype == "STATIONARY":  # guessed
        event.ct_acquisition_type = get_or_create_cid(
            "113806", "Stationary Acquisition"
        )
    elif acqtype == "FREE":  # guessed, for completeness
        event.ct_acquisition_type = get_or_create_cid("113807", "Free Acquisition")
    # procedure context is optional and not reported (contrast or not)
    # irradiation event uid would be available in image headers, but assuming just working from dose report image:
    event.irradiation_event_uid = pydicom.uid.generate_uid()
    exptime = get_value_kw("ExposureTime", dataset)
    if exptime:
        event.exposure_time = exptime / 1000.0
    _scanninglength(dataset, event)
    event.nominal_single_collimation_width = get_value_kw(
        "SingleCollimationWidth", dataset
    )
    event.nominal_total_collimation_width = get_value_kw(
        "TotalCollimationWidth", dataset
    )
    event.pitch_factor = get_value_kw(
        "SpiralPitchFactor", dataset
    )  # not sure what would be there for an axial scan: SequencedPitchFactor?
    event.number_of_xray_sources = 1
    ctdiwphantom = get_value_num(0x01E11026, dataset)  # Philips private tag
    if ctdiwphantom == "16 CM":
        event.ctdiw_phantom_type = get_or_create_cid(
            "113690", "IEC Head Dosimetry Phantom"
        )
    if ctdiwphantom == "32 CM":
        event.ctdiw_phantom_type = get_or_create_cid(
            "113691", "IEC Body Dosimetry Phantom"
        )
    event.save()
    _ctxraysourceparameters(dataset, event)
    event.mean_ctdivol = get_value_kw("CTDIvol", dataset)
    event.dlp = Decimal(get_value_num(0x00E11021, dataset))  # Philips private tag
    event.date_time_started = get_date_time("AcquisitionDateTime", dataset)
    #    event.series_description = get_value_kw('SeriesDescription',dataset)
    event.save()


def _ctaccumulateddosedata(dataset, ct):  # TID 10012
    ctacc = CtAccumulatedDoseData.objects.create(ct_radiation_dose=ct)
    ctacc.total_number_of_irradiation_events = get_value_kw(
        "TotalNumberOfExposures", dataset
    )
    try:
        ctacc.ct_dose_length_product_total = Decimal(
            get_value_num(0x00E11021, dataset)
        )  # Philips private tag
    except TypeError:
        pass
    ctacc.comment = get_value_kw("CommentsOnRadiationDose", dataset)
    ctacc.save()


def _ctradiationdose(dataset, g):
    proj = CtRadiationDose.objects.create(general_study_module_attributes=g)
    proj.procedure_reported = get_or_create_cid("P5-08000", "Computed Tomography X-Ray")
    proj.has_intent = get_or_create_cid("R-408C3", "Diagnostic Intent")
    proj.scope_of_accumulation = get_or_create_cid("113014", "Study")
    comment_dose = get_value_kw("CommentsOnRadiationDose", dataset)
    comment_protocol_file = get_value_num(0x00E11061, dataset)
    comment_study_description = get_value_kw("StudyDescription", dataset)
    if not comment_dose:
        comment_dose = ""
    if not comment_protocol_file:
        comment_protocol_file = ""
    if not comment_study_description:
        comment_study_description = ""
    proj.comment = (
        f"StudyDescription: {comment_study_description}. Comments on radiation dose: {comment_dose}. "
        f"ProtocolFilename: {comment_protocol_file}"
    )
    proj.source_of_dose_information = get_or_create_cid(
        "113866", "Copied From Image Attributes"
    )
    proj.save()
    _ctaccumulateddosedata(dataset, proj)
    for series in dataset.ExposureDoseSequence:
        if "AcquisitionType" in series:
            _ctirradiationeventdata(series, proj)
    events = proj.ctirradiationeventdata_set.all()
    if not events:
        logger.warning(
            f"There were no events in ct_philips import, or they couldn't be read. "
            f"{get_value_kw('StationName', dataset)}"
            f"{get_value_kw('Manufacturer', dataset)}"
            f"{get_value_kw('ManufacturerModelName', dataset)}"
            f"{g.study_date.date()}"
            f"{g.study_time.time()}"
            f"{g.accession_number}"
        )
    else:
        # Come back and set start and end of irradiation after creating the x-ray events
        proj.start_of_xray_irradiation = events.aggregate(Min("date_time_started"))[
            "date_time_started__min"
        ]
        try:
            latestlength = int(
                events.latest("date_time_started").exposure_time * 1000
            )  # in microseconds
            lastevent = events.aggregate(Max("date_time_started"))[
                "date_time_started__max"
            ]
            if lastevent and latestlength:
                last = lastevent + timedelta(microseconds=latestlength)
                proj.end_of_xray_irradiation = last
        except TypeError:
            pass
        proj.save()


def _generalequipmentmoduleattributes(dataset, study):
    equip = GeneralEquipmentModuleAttr.objects.create(
        general_study_module_attributes=study
    )
    equip.manufacturer = get_value_kw("Manufacturer", dataset)
    equip.institution_name = get_value_kw("InstitutionName", dataset)
    equip.institution_address = get_value_kw("InstitutionAddress", dataset)
    equip.station_name = get_value_kw("StationName", dataset)
    equip.institutional_department_name = get_value_kw(
        "InstitutionalDepartmentName", dataset
    )
    equip.manufacturer_model_name = get_value_kw("ManufacturerModelName", dataset)
    equip.device_serial_number = get_value_kw("DeviceSerialNumber", dataset)
    equip.software_versions = get_value_kw("SoftwareVersions", dataset)
    equip.gantry_id = get_value_kw("GantryID", dataset)
    equip.spatial_resolution = get_value_kw(
        "SpatialResolution", dataset
    )  # might fall over if field present but blank - check!
    equip.date_of_last_calibration = get_date("DateOfLastCalibration", dataset)
    equip.time_of_last_calibration = get_time("TimeOfLastCalibration", dataset)

    equip_display_name, created = UniqueEquipmentNames.objects.get_or_create(
        manufacturer=equip.manufacturer,
        manufacturer_hash=hash_id(equip.manufacturer),
        institution_name=equip.institution_name,
        institution_name_hash=hash_id(equip.institution_name),
        station_name=equip.station_name,
        station_name_hash=hash_id(equip.station_name),
        institutional_department_name=equip.institutional_department_name,
        institutional_department_name_hash=hash_id(equip.institutional_department_name),
        manufacturer_model_name=equip.manufacturer_model_name,
        manufacturer_model_name_hash=hash_id(equip.manufacturer_model_name),
        device_serial_number=equip.device_serial_number,
        device_serial_number_hash=hash_id(equip.device_serial_number),
        software_versions=equip.software_versions,
        software_versions_hash=hash_id(equip.software_versions),
        gantry_id=equip.gantry_id,
        gantry_id_hash=hash_id(equip.gantry_id),
        hash_generated=True,
        device_observer_uid=None,
        device_observer_uid_hash=None,
    )
    if created:
        if equip.institution_name and equip.station_name:
            equip_display_name.display_name = (
                equip.institution_name + " " + equip.station_name
            )
        elif equip.institution_name:
            equip_display_name.display_name = equip.institution_name
        elif equip.station_name:
            equip_display_name.display_name = equip.station_name
        else:
            equip_display_name.display_name = "Blank"
        equip_display_name.save()

    equip.unique_equipment_name = UniqueEquipmentNames(pk=equip_display_name.pk)

    equip.save()


def _patientstudymoduleattributes(dataset, g):  # C.7.2.2
    patientatt = PatientStudyModuleAttr.objects.create(
        general_study_module_attributes=g
    )
    patientatt.patient_age = get_value_kw("PatientAge", dataset)
    patientatt.patient_weight = get_value_kw("PatientWeight", dataset)
    patientatt.save()


def _generalstudymoduleattributes(dataset, g):
    g.study_instance_uid = get_value_kw("StudyInstanceUID", dataset)
    g.study_date = get_date("StudyDate", dataset)
    g.study_time = get_time("StudyTime", dataset)
    g.study_workload_chart_time = datetime.combine(
        datetime.date(datetime(1900, 1, 1)), datetime.time(g.study_time)
    )
    g.referring_physician_name = list_to_string(
        get_value_kw("RequestingPhysician", dataset)
    )
    g.study_id = get_value_kw("StudyID", dataset)
    accession_number = get_value_kw("AccessionNumber", dataset)
    patient_id_settings = PatientIDSettings.objects.get()
    if accession_number and patient_id_settings.accession_hashed:
        accession_number = hash_id(accession_number)
        g.accession_hashed = True
    g.accession_number = accession_number
    g.modality_type = "CT"
    g.study_description = get_value_kw("ProtocolName", dataset)
    g.operator_name = list_to_string(get_value_kw("OperatorsName", dataset))
    if "RequestAttributesSequence" in dataset:
        g.procedure_code_value = get_seq_code_value(
            "ScheduledProtocolCodeSequence", dataset.RequestAttributesSequence[0]
        )
        g.procedure_code_meaning = get_seq_code_meaning(
            "ScheduledProtocolCodeSequence", dataset.RequestAttributesSequence[0]
        )
    g.requested_procedure_code_meaning = get_value_kw(
        "RequestedProcedureDescription", dataset
    )
    g.save()
    _ctradiationdose(dataset, g)
    try:
        g.number_of_events = (
            g.ctradiationdose_set.get().ctirradiationeventdata_set.count()
        )
        g.save()
    except ObjectDoesNotExist:
        logger.warning(
            "Study UID {0} of modality {1}. Unable to get event count!".format(
                g.study_instance_uid, get_value_kw("ManufacturerModelName", dataset)
            )
        )
    ct_event_type_count(g)
    try:
        g.total_dlp = (
            g.ctradiationdose_set.get()
            .ctaccumulateddosedata_set.get()
            .ct_dose_length_product_total
        )
        g.save()
    except ObjectDoesNotExist:
        logger.warning(
            "Study UID {0} of modality {1}. Unable to set summary total_dlp".format(
                g.study_instance_uid, get_value_kw("ManufacturerModelName", dataset)
            )
        )


def _philips_ct2db(dataset):
    if "StudyInstanceUID" in dataset:
        study_instance_uid = dataset.StudyInstanceUID
        record_task_info(f"UID: {study_instance_uid.replace('.', '. ')}")
        record_task_related_query(study_instance_uid)
        existing = GeneralStudyModuleAttr.objects.filter(
            study_instance_uid__exact=study_instance_uid
        )
        if existing:
            return

    g = GeneralStudyModuleAttr.objects.create()
    _generalstudymoduleattributes(dataset, g)
    _generalequipmentmoduleattributes(dataset, g)
    _patientstudymoduleattributes(dataset, g)
    patient_module_attributes(dataset, g)

    # Add standard names
    add_standard_names(g)

    # Am Ende der Funktion nach erfolgreicher Speicherung
    send_ct_high_dose_alert_email(study_pk=g.pk)


def ct_philips(philips_file):
    """Extract radiation dose structured report related data from Philips CT dose report images

    :param filename: relative or absolute path to Philips CT dose report DICOM image file.
    :type filename: str.

    Tested with:
        * Philips Gemini TF PET-CT v2.3.0
        * Brilliance BigBore v3.5.4.17001.
    """

    try:
        del_settings = DicomDeleteSettings.objects.get()
        del_ct_phil = del_settings.del_ct_phil
    except ObjectDoesNotExist:
        del_ct_phil = False

    try:
        dataset = pydicom.dcmread(philips_file)
    except FileNotFoundError:
        logger.warning(
            f"ct_philips.py not attempting to extract from {philips_file}, the file does not exist"
        )
        record_task_error_exit(
            f"Not attempting to extract from {philips_file}, the file does not exist"
        )
        return 1

    dataset.decode()
    if (
        dataset.SOPClassUID != "1.2.840.10008.5.1.4.1.1.7"
        or dataset.Manufacturer != "Philips"
        or dataset.SeriesDescription != "Dose Info"
    ):
        error = "{0} is not a Philips CT dose report image".format(philips_file)
        logger.error(error)
        record_task_error_exit(error)
        return 1

    _philips_ct2db(dataset)

    if del_ct_phil:
        os.remove(philips_file)

    return 0
