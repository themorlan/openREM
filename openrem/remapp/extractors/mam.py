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
..  module:: mam.
    :synopsis: Module to extract radiation dose related data from mammography image objects.

..  moduleauthor:: Ed McDonagh

"""
import os
import sys
import django
import logging
from pydicom.charset import default_encoding

from openrem.remapp.tools.background import (
    record_task_error_exit,
    record_task_related_query,
    record_task_info,
)

logger = logging.getLogger("remapp.extractors.mam")

# setup django/OpenREM
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from .extract_common import (  # pylint: disable=wrong-import-order, wrong-import-position
    patient_module_attributes,
    add_standard_names,
)


def _xrayfilters(dataset, source):
    from remapp.models import XrayFilters
    from remapp.tools.get_values import get_value_kw, get_or_create_cid

    filters = XrayFilters.objects.create(irradiation_event_xray_source_data=source)
    xray_filter_material = get_value_kw("FilterMaterial", dataset)
    try:
        xray_filter_material = xray_filter_material.decode(default_encoding)
    except AttributeError:
        pass
    if xray_filter_material:
        if xray_filter_material.strip().lower() == "molybdenum":
            filters.xray_filter_material = get_or_create_cid(
                "C-150F9", "Molybdenum or Molybdenum compound"
            )
        if xray_filter_material.strip().lower() == "rhodium":
            filters.xray_filter_material = get_or_create_cid(
                "C-167F9", "Rhodium or Rhodium compound"
            )
        if xray_filter_material.strip().lower() == "silver":
            filters.xray_filter_material = get_or_create_cid(
                "C-137F9", "Silver or Silver compound"
            )
        if (
            xray_filter_material.strip().lower() == "aluminum"
            or xray_filter_material.strip().lower() == "aluminium"
        ):
            filters.xray_filter_material = get_or_create_cid(
                "C-120F9", "Aluminum or Aluminum compound"
            )

        filters.save()


def _kvp(dataset, source):
    from remapp.models import Kvp
    from remapp.tools.get_values import get_value_kw

    kv = Kvp.objects.create(irradiation_event_xray_source_data=source)
    kv.kvp = get_value_kw("KVP", dataset)
    kv.save()


def _exposure(dataset, source):
    from remapp.models import Exposure

    exp = Exposure.objects.create(irradiation_event_xray_source_data=source)
    from remapp.tools.get_values import get_value_kw

    try:
        exp.exposure = get_value_kw("ExposureInuAs", dataset).decode()  # uA.s
    except AttributeError:
        exp.exposure = get_value_kw("ExposureInuAs", dataset)
    exp.save()


def _xraygrid(gridcode, source):
    from remapp.models import XrayGrid
    from remapp.tools.get_values import get_or_create_cid

    grid = XrayGrid.objects.create(irradiation_event_xray_source_data=source)
    if gridcode == "111646":
        grid.xray_grid = get_or_create_cid("111646", "No grid")
    if gridcode == "111642":
        grid.xray_grid = get_or_create_cid("111642", "Focused grid")
    if gridcode == "111643":
        grid.xray_grid = get_or_create_cid("111643", "Reciprocating grid")
    grid.save()


def _xraytubecurrent(current_value, source):
    from remapp.models import XrayTubeCurrent

    tubecurrent = XrayTubeCurrent.objects.create(
        irradiation_event_xray_source_data=source
    )
    tubecurrent.xray_tube_current = current_value
    tubecurrent.save()


def _irradiationeventxraysourcedata(dataset, event):
    # TODO: review model to convert to cid where appropriate, and add additional fields, such as height and width
    from remapp.models import IrradEventXRaySourceData
    from remapp.tools.get_values import get_value_kw, get_or_create_cid

    source = IrradEventXRaySourceData.objects.create(irradiation_event_xray_data=event)
    # AGD/MGD is dGy in Mammo headers, and was dGy in Radiation Dose SR - CP1194 changes this to mGy!
    agd_dgy = get_value_kw("OrganDose", dataset)  # AGD in dGy
    if agd_dgy:
        source.average_glandular_dose = float(agd_dgy) * 100.0  # AGD in mGy
    source.average_xray_tube_current = get_value_kw("XRayTubeCurrent", dataset)
    _xraytubecurrent(source.average_xray_tube_current, source)
    source.exposure_time = get_value_kw("ExposureTime", dataset)
    source.focal_spot_size = get_value_kw("FocalSpots", dataset)
    anode_target_material = get_value_kw("AnodeTargetMaterial", dataset)
    try:
        anode_target_material = anode_target_material.decode(default_encoding)
    except AttributeError:
        pass
    if anode_target_material:
        if anode_target_material.strip().lower() == "molybdenum":
            source.anode_target_material = get_or_create_cid(
                "C-150F9", "Molybdenum or Molybdenum compound"
            )
        if anode_target_material.strip().lower() == "rhodium":
            source.anode_target_material = get_or_create_cid(
                "C-167F9", "Rhodium or Rhodium compound"
            )
        if anode_target_material.strip().lower() == "tungsten":
            source.anode_target_material = get_or_create_cid(
                "C-164F9", "Tungsten or Tungsten compound"
            )
    collimated_field_area = get_value_kw("FieldOfViewDimensions", dataset)
    if collimated_field_area:
        source.collimated_field_area = (
            float(collimated_field_area[0]) * float(collimated_field_area[1]) / 1000000
        )
    source.exposure_control_mode = get_value_kw("ExposureControlMode", dataset)
    source.save()
    _xrayfilters(dataset, source)
    _kvp(dataset, source)
    _exposure(dataset, source)
    xray_grid = get_value_kw("Grid", dataset)
    if xray_grid:
        if xray_grid == "NONE":
            _xraygrid("111646", source)
        elif xray_grid == ["RECIPROCATING", "FOCUSED"]:
            _xraygrid("111642", source)
            _xraygrid("111643", source)


def _doserelateddistancemeasurements(dataset, mech):
    from remapp.models import DoseRelatedDistanceMeasurements
    from remapp.tools.get_values import get_value_kw, get_value_num

    dist = DoseRelatedDistanceMeasurements.objects.create(
        irradiation_event_xray_mechanical_data=mech
    )
    dist.distance_source_to_detector = get_value_kw("DistanceSourceToDetector", dataset)
    dist.distance_source_to_entrance_surface = get_value_kw(
        "DistanceSourceToEntrance", dataset
    )
    dist.radiological_thickness = get_value_num(0x00451049, dataset)
    dist.save()


def _irradiationeventxraymechanicaldata(dataset, event):
    from remapp.models import IrradEventXRayMechanicalData
    from remapp.tools.get_values import get_value_kw

    mech = IrradEventXRayMechanicalData.objects.create(
        irradiation_event_xray_data=event
    )
    try:
        mech.compression_thickness = get_value_kw("BodyPartThickness", dataset).decode()
    except AttributeError:
        mech.compression_thickness = get_value_kw("BodyPartThickness", dataset)
    compression_force = get_value_kw("CompressionForce", dataset)
    if compression_force:
        mech.compression_force = float(compression_force)
    mech.magnification_factor = get_value_kw(
        "EstimatedRadiographicMagnificationFactor", dataset
    )
    mech.column_angulation = get_value_kw("PositionerPrimaryAngle", dataset)
    mech.save()
    _doserelateddistancemeasurements(dataset, mech)


def _accumulatedmammo_update(event):  # TID 10005
    from remapp.tools.get_values import get_or_create_cid
    from remapp.models import AccumMammographyXRayDose

    accum = event.projection_xray_radiation_dose.accumxraydose_set.get()
    accummams = accum.accummammographyxraydose_set.all()
    event_added = False
    for accummam in accummams:
        if not accummam.laterality:
            if event.laterality.code_meaning == "Right":
                accummam.laterality = get_or_create_cid("T-04020", "Right breast")
            elif event.laterality.code_meaning == "Left":
                accummam.laterality = get_or_create_cid("T-04030", "Left breast")
            accummam.accumulated_average_glandular_dose += (
                event.irradeventxraysourcedata_set.get().average_glandular_dose
            )
            accummam.save()
            event_added = True
        elif event.laterality.code_meaning in accummam.laterality.code_meaning:
            accummam.accumulated_average_glandular_dose += (
                event.irradeventxraysourcedata_set.get().average_glandular_dose
            )
            accummam.save()
            event_added = True
    if not event_added:
        accummam = AccumMammographyXRayDose.objects.create(accumulated_xray_dose=accum)
        if event.laterality.code_meaning == "Right":
            accummam.laterality = get_or_create_cid("T-04020", "Right breast")
        elif event.laterality.code_meaning == "Left":
            accummam.laterality = get_or_create_cid("T-04030", "Left breast")
        accummam.accumulated_average_glandular_dose = (
            event.irradeventxraysourcedata_set.get().average_glandular_dose
        )
        accummam.save()
    accummam.save()


def _irradiationeventxraydata(dataset, proj):  # TID 10003
    # TODO: review model to convert to cid where appropriate, and add additional fields
    from remapp.models import IrradEventXRayData
    from remapp.tools.get_values import (
        get_value_kw,
        get_or_create_cid,
        get_seq_code_value,
        get_seq_code_meaning,
    )
    from remapp.tools.dcmdatetime import make_date_time

    event = IrradEventXRayData.objects.create(projection_xray_radiation_dose=proj)
    event.acquisition_plane = get_or_create_cid("113622", "Single Plane")
    event.irradiation_event_uid = get_value_kw("SOPInstanceUID", dataset)
    event_time = get_value_kw("AcquisitionTime", dataset)
    event_date = get_value_kw("AcquisitionDate", dataset)
    event.date_time_started = make_date_time(f"{event_date}{event_time}")
    event.irradiation_event_type = get_or_create_cid("113611", "Stationary Acquisition")
    event.acquisition_protocol = get_value_kw("ProtocolName", dataset)
    event.anatomical_structure = get_or_create_cid(
        get_seq_code_value("AnatomicRegionSequence", dataset),
        get_seq_code_meaning("AnatomicRegionSequence", dataset),
    )
    laterality = get_value_kw("ImageLaterality", dataset)
    if not laterality:
        laterality = get_value_kw("Laterality", dataset)
    if laterality:
        if laterality.strip() == "R":
            event.laterality = get_or_create_cid("G-A100", "Right")
        if laterality.strip() == "L":
            event.laterality = get_or_create_cid("G-A101", "Left")
    event.image_view = get_or_create_cid(
        get_seq_code_value("ViewCodeSequence", dataset),
        get_seq_code_meaning("ViewCodeSequence", dataset),
    )
    # image view modifier?
    if event.anatomical_structure:
        event.target_region = event.anatomical_structure
    try:
        event.entrance_exposure_at_rp = get_value_kw(
            "EntranceDoseInmGy", dataset
        ).decode()
    except AttributeError:
        event.entrance_exposure_at_rp = get_value_kw("EntranceDoseInmGy", dataset)
    # reference point definition?
    pc_fibroglandular = get_value_kw("CommentsOnRadiationDose", dataset)
    if pc_fibroglandular:
        if "%" in pc_fibroglandular:
            event.percent_fibroglandular_tissue = pc_fibroglandular.replace(
                "%", ""
            ).strip()
    event.comment = get_value_kw("ExposureControlModeDescription", dataset)
    event.save()

    #    irradiationeventxraydetectordata(dataset,event)
    _irradiationeventxraysourcedata(dataset, event)
    _irradiationeventxraymechanicaldata(dataset, event)
    if (
        event.laterality
        and event.irradeventxraysourcedata_set.get().average_glandular_dose
    ):
        _accumulatedmammo_update(event)


def _accumulatedxraydose(proj):
    from remapp.models import AccumXRayDose, AccumMammographyXRayDose
    from remapp.tools.get_values import get_or_create_cid

    accum = AccumXRayDose.objects.create(projection_xray_radiation_dose=proj)
    accum.acquisition_plane = get_or_create_cid("113622", "Single Plane")
    accum.save()
    accummam = AccumMammographyXRayDose.objects.create(accumulated_xray_dose=accum)
    accummam.accumulated_average_glandular_dose = 0.0
    accummam.save()


def _projectionxrayradiationdose(dataset, g):
    from remapp.models import ProjectionXRayRadiationDose
    from remapp.tools.get_values import get_or_create_cid

    proj = ProjectionXRayRadiationDose.objects.create(general_study_module_attributes=g)
    proj.procedure_reported = get_or_create_cid("P5-40010", "Mammography")
    proj.has_intent = get_or_create_cid("R-408C3", "Diagnostic Intent")
    proj.scope_of_accumulation = get_or_create_cid("113014", "Study")
    proj.source_of_dose_information = get_or_create_cid(
        "113866", "Copied From Image Attributes"
    )
    proj.xray_detector_data_available = get_or_create_cid("R-00339", "No")
    proj.xray_source_data_available = get_or_create_cid("R-0038D", "Yes")
    proj.xray_mechanical_data_available = get_or_create_cid("R-0038D", "Yes")
    proj.save()
    _accumulatedxraydose(proj)
    _irradiationeventxraydata(dataset, proj)


def _generalequipmentmoduleattributes(dataset, study):
    from remapp.models import GeneralEquipmentModuleAttr, UniqueEquipmentNames
    from remapp.tools.dcmdatetime import get_date, get_time
    from remapp.tools.get_values import get_value_kw
    from remapp.tools.hash_id import hash_id

    equip = GeneralEquipmentModuleAttr.objects.create(
        general_study_module_attributes=study
    )
    equip.manufacturer = get_value_kw("Manufacturer", dataset)
    equip.institution_name = get_value_kw("InstitutionName", dataset)
    try:
        "In extractor, {0} of type {1}".format(
            equip.institution_name, type(equip.institution_name)
        )
    except UnicodeDecodeError:
        logger.error(
            "Non ASCII string not properly decoded: {0}".format(equip.institution_name)
        )
    equip.institution_address = get_value_kw("InstitutionAddress", dataset)
    equip.station_name = get_value_kw("StationName", dataset)
    equip.institutional_department_name = get_value_kw(
        "InstitutionalDepartmentName", dataset
    )
    equip.manufacturer_model_name = get_value_kw("ManufacturerModelName", dataset)
    equip.device_serial_number = get_value_kw("DeviceSerialNumber", dataset)
    equip.software_versions = get_value_kw("SoftwareVersions", dataset)
    equip.gantry_id = get_value_kw("GantryID", dataset)
    equip.spatial_resolution = get_value_kw("SpatialResolution", dataset)
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
    from remapp.models import PatientStudyModuleAttr
    from remapp.tools.get_values import get_value_kw

    patientatt = PatientStudyModuleAttr.objects.create(
        general_study_module_attributes=g
    )
    patientatt.patient_age = get_value_kw("PatientAge", dataset)
    patientatt.save()


def _generalstudymoduleattributes(dataset, g):
    from datetime import datetime
    from remapp.extractors.extract_common import populate_mammo_agd_summary
    from remapp.models import PatientIDSettings
    from remapp.tools.get_values import (
        get_value_kw,
        get_seq_code_meaning,
        get_seq_code_value,
        list_to_string,
    )
    from remapp.tools.dcmdatetime import get_date, get_time
    from remapp.tools.hash_id import hash_id

    g.study_instance_uid = get_value_kw("StudyInstanceUID", dataset)
    logger.debug("Populating mammo study %s", g.study_instance_uid)
    g.study_date = get_date("StudyDate", dataset)
    g.study_time = get_time("StudyTime", dataset)
    g.study_workload_chart_time = datetime.combine(
        datetime.date(datetime(1900, 1, 1)), datetime.time(g.study_time)
    )
    g.referring_physician_name = list_to_string(
        get_value_kw("ReferringPhysicianName", dataset)
    )
    g.study_id = get_value_kw("StudyID", dataset)
    accession_number = get_value_kw("AccessionNumber", dataset)
    patient_id_settings = PatientIDSettings.objects.get()
    if accession_number and patient_id_settings.accession_hashed:
        accession_number = hash_id(accession_number)
        g.accession_hashed = True
    g.accession_number = accession_number
    g.study_description = get_value_kw("StudyDescription", dataset)
    g.modality_type = get_value_kw("Modality", dataset)
    g.physician_of_record = list_to_string(get_value_kw("PhysiciansOfRecord", dataset))
    g.name_of_physician_reading_study = list_to_string(
        get_value_kw("NameOfPhysiciansReadingStudy", dataset)
    )
    g.performing_physician_name = list_to_string(
        get_value_kw("PerformingPhysicianName", dataset)
    )
    g.operator_name = list_to_string(get_value_kw("OperatorsName", dataset))
    g.procedure_code_meaning = get_value_kw(
        "ProtocolName", dataset
    )  # Being used to summarise protocol for study
    g.requested_procedure_code_value = get_seq_code_value(
        "RequestedProcedureCodeSequence", dataset
    )
    g.requested_procedure_code_meaning = get_seq_code_meaning(
        "RequestedProcedureCodeSequence", dataset
    )
    g.save()

    _generalequipmentmoduleattributes(dataset, g)
    _projectionxrayradiationdose(dataset, g)
    _patientstudymoduleattributes(dataset, g)
    patient_module_attributes(dataset, g)
    populate_mammo_agd_summary(g)
    g.number_of_events = (
        g.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
    )
    g.save()

    # Add standard names
    add_standard_names(g)


def _test_if_mammo(dataset):
    """Test if dicom object passed is a mammo file by looking at SOP Class UID"""
    if (
        dataset.SOPClassUID == "1.2.840.10008.5.1.4.1.1.1.2.1"
        or dataset.SOPClassUID == "1.2.840.10008.5.1.4.1.1.1.2"
    ):
        return 1
    elif (
        dataset.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7"
        and dataset.Modality == "MG"
        and "ORIGINAL" in dataset.ImageType
    ):
        return 1
    return 0


def _mammo2db(dataset):
    from time import sleep
    from random import random
    from remapp.extractors.extract_common import (
        get_study_check_dup,
        populate_mammo_agd_summary,
    )
    from remapp.models import GeneralStudyModuleAttr
    from remapp.tools import check_uid
    from remapp.tools.get_values import get_value_kw

    study_uid = get_value_kw("StudyInstanceUID", dataset)
    if not study_uid:
        error = "In mammo import no UID present. Stop import."
        logger.error(error)
        record_task_error_exit(error)
        return
    study_in_db = check_uid.check_uid(study_uid)
    record_task_info(f"UID: {study_uid.replace('.', '. ')}")
    record_task_related_query(study_uid)
    logger.debug("In mam.py. Study_UID %s, study_in_db %s", study_uid, study_in_db)

    if study_in_db:
        sleep(
            2.0
        )  # Give initial event a chance to get to save on _projectionxrayradiationdose
        this_study = get_study_check_dup(dataset, modality="MG")
        if this_study:
            _irradiationeventxraydata(
                dataset, this_study.projectionxrayradiationdose_set.get()
            )
            populate_mammo_agd_summary(this_study)
            this_study.number_of_events = (
                this_study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
            )
            this_study.save()

            # Update any matching standard names
            add_standard_names(this_study)

    if not study_in_db:
        # study doesn't exist, start from scratch
        g = GeneralStudyModuleAttr.objects.create()
        g.study_instance_uid = get_value_kw("StudyInstanceUID", dataset)
        g.save()
        event_uid = get_value_kw("SOPInstanceUID", dataset)
        check_uid.record_sop_instance_uid(g, event_uid)
        logger.debug("Created new mammo study %s, event %s", study_uid, event_uid)
        # check again
        study_in_db = check_uid.check_uid(study_uid)
        if study_in_db == 1:
            _generalstudymoduleattributes(dataset, g)
        elif not study_in_db:
            record_task_error_exit(
                "Something went wrong, GeneralStudyModuleAttr wasn't created"
            )
            return
        elif study_in_db > 1:
            sleep(random())  # nosec - not being used for cryptography
            # Check if other instance(s) has deleted the study yet
            study_in_db = check_uid.check_uid(study_uid)
            if study_in_db == 1:
                _generalstudymoduleattributes(dataset, g)
            elif study_in_db > 1:
                g.delete()
                study_in_db = check_uid.check_uid(study_uid)
                if not study_in_db:
                    # both must have been deleted simultaneously!
                    sleep(random())  # nosec - not being used for cryptography
                    # Check if other instance has created the study again yet
                    study_in_db = check_uid.check_uid(study_uid)
                    if study_in_db == 1:
                        sleep(
                            2.0
                        )  # Give initial event a chance to get to save on _projectionxrayradiationdose
                        this_study = get_study_check_dup(dataset, modality="MG")
                        if this_study:
                            _irradiationeventxraydata(
                                dataset,
                                this_study.projectionxrayradiationdose_set.get(),
                            )
                    while not study_in_db:
                        g = GeneralStudyModuleAttr.objects.create()
                        g.study_instance_uid = get_value_kw("StudyInstanceUID", dataset)
                        g.save()
                        check_uid.record_sop_instance_uid(g, event_uid)
                        # check again
                        study_in_db = check_uid.check_uid(study_uid)
                        if study_in_db == 1:
                            _generalstudymoduleattributes(dataset, g)
                        elif study_in_db > 1:
                            g.delete()
                            sleep(random())  # nosec - not being used for cryptography
                            study_in_db = check_uid.check_uid(study_uid)
                            if study_in_db == 1:
                                sleep(
                                    2.0
                                )  # Give initial event a chance to get to save on _projectionxrayradiationdose
                                this_study = get_study_check_dup(dataset, modality="MG")
                                if this_study:
                                    _irradiationeventxraydata(
                                        dataset,
                                        this_study.projectionxrayradiationdose_set.get(),
                                    )
                elif study_in_db == 1:
                    sleep(
                        2.0
                    )  # Give initial event a chance to get to save on _projectionxrayradiationdose
                    this_study = get_study_check_dup(dataset, modality="MG")
                    if this_study:
                        _irradiationeventxraydata(
                            dataset, this_study.projectionxrayradiationdose_set.get()
                        )


def mam(mg_file):
    """Extract radiation dose structured report related data from mammography images

    :param mg_file: relative or absolute path to mammography DICOM image file.
    :type mg_file: str.

    """

    import pydicom
    from django.core.exceptions import ObjectDoesNotExist
    from remapp.models import DicomDeleteSettings

    try:
        del_settings = DicomDeleteSettings.objects.get()
        del_mg_im = del_settings.del_mg_im
        del_no_match = del_settings.del_no_match
    except ObjectDoesNotExist:
        del_mg_im = False
        del_no_match = True

    try:
        dataset = pydicom.dcmread(mg_file)
    except FileNotFoundError:
        logger.warning(
            f"mam.py not attempting to extract from {mg_file}, the file does not exist"
        )
        record_task_error_exit(
            f"Not attempting to extract from {mg_file}, the file does not exist"
        )
        return 1

    dataset.decode()
    ismammo = _test_if_mammo(dataset)
    if not ismammo:
        if del_no_match:
            logger.debug("%s is not a mammo file, deleting", mg_file)
            os.remove(mg_file)
        record_task_error_exit(f"{mg_file} is not a mammo file")
        return 1

    _mammo2db(dataset)

    if del_mg_im:
        logger.debug("Mammo %s processing complete, deleting file", mg_file)
        os.remove(mg_file)
    else:
        logger.debug("Mammo %s processing complete, file remains", mg_file)

    return 0
