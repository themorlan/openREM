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
#
#
#    This file (dx.py) is intended to extract radiation dose related data from
#    DX images. It is based on mam.py.
#    David Platten, 28/3/2014
#
"""
..  module:: dx.
    :synopsis: Module to extract radiation dose related data from DX image objects.

..  moduleauthor:: David Platten, Ed McDonagh

"""
from datetime import datetime
from decimal import Decimal, DecimalException
import logging
import os
from random import random
import sys
from time import sleep

import django
from django.core.exceptions import ObjectDoesNotExist
import pydicom
from pydicom import config
from pydicom.valuerep import MultiValue

from openrem.remapp.tools.background import (
    record_task_error_exit,
    record_task_related_query,
    record_task_info,
)

from ..tools import check_uid
from ..tools.dcmdatetime import get_date, get_time, make_date_time
from ..tools.get_values import (
    get_value_kw,
    get_value_num,
    get_or_create_cid,
    get_seq_code_value,
    get_seq_code_meaning,
    list_to_string,
)
from ..tools.hash_id import hash_id

# setup django/OpenREM
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from .extract_common import (  # pylint: disable=wrong-import-order, wrong-import-position
    get_study_check_dup,
    populate_dx_rf_summary,
    patient_module_attributes,
    add_standard_names,
)
from remapp.models import (  # pylint: disable=wrong-import-order, wrong-import-position
    AccumXRayDose,
    AccumIntegratedProjRadiogDose,
    DicomDeleteSettings,
    DoseRelatedDistanceMeasurements,
    Exposure,
    GeneralEquipmentModuleAttr,
    GeneralStudyModuleAttr,
    IrradEventXRayData,
    IrradEventXRayDetectorData,
    IrradEventXRayMechanicalData,
    IrradEventXRaySourceData,
    Kvp,
    PatientIDSettings,
    PatientStudyModuleAttr,
    ProjectionXRayRadiationDose,
    UniqueEquipmentNames,
    XrayFilters,
    XrayGrid,
)

logger = logging.getLogger(
    "remapp.extractors.dx"
)  # Explicitly named so that it is still handled when using __main__


def _xrayfilters(filttype, material, thickmax, thickmin, source):

    filters = XrayFilters.objects.create(irradiation_event_xray_source_data=source)
    if filttype:
        filter_types = {
            "STRIP": {"code": "113650", "meaning": "Strip filter"},
            "WEDGE": {"code": "113651", "meaning": "Wedge filter"},
            "BUTTERFLY": {"code:": "113652", "meaning": "Butterfly filter"},
            "NONE": {"code": "111609", "meaning": "No filter"},
            "FLAT": {"code": "113653", "meaning": "Flat filter"},
        }
        if filttype in filter_types:
            filters.xray_filter_type = get_or_create_cid(
                filter_types[filttype]["code"], filter_types[filttype]["meaning"]
            )
    if material:
        logger.debug(
            f"In _xrayfilters, attempting to match material {material.strip().lower()}"
        )
        if material.strip().lower() == "molybdenum":
            filters.xray_filter_material = get_or_create_cid(
                "C-150F9", "Molybdenum or Molybdenum compound"
            )
        if material.strip().lower() == "rhodium":
            filters.xray_filter_material = get_or_create_cid(
                "C-167F9", "Rhodium or Rhodium compound"
            )
        if material.strip().lower() == "silver":
            filters.xray_filter_material = get_or_create_cid(
                "C-137F9", "Silver or Silver compound"
            )
        if material.strip().lower() in [
            "aluminum",
            "aluminium",
        ]:  # Illegal spelling of Aluminium found in Philips DiDi
            filters.xray_filter_material = get_or_create_cid(
                "C-120F9", "Aluminum or Aluminum compound"
            )
        if material.strip().lower() == "copper":
            filters.xray_filter_material = get_or_create_cid(
                "C-127F9", "Copper or Copper compound"
            )
        if material.strip().lower() == "niobium":
            filters.xray_filter_material = get_or_create_cid(
                "C-1190E", "Niobium or Niobium compound"
            )
        if material.strip().lower() == "europium":
            filters.xray_filter_material = get_or_create_cid(
                "C-1190F", "Europium or Europium compound"
            )
        if material.strip().lower() == "lead":
            filters.xray_filter_material = get_or_create_cid(
                "C-132F9", "Lead or Lead compound"
            )
        if material.strip().lower() == "tantalum":
            filters.xray_filter_material = get_or_create_cid(
                "C-156F9", "Tantalum or Tantalum compound"
            )
    if thickmax is not None and thickmin is not None:
        if thickmax < thickmin:
            tempmin = thickmax
            thickmax = thickmin
            thickmin = tempmin
    if thickmax is not None:
        filters.xray_filter_thickness_maximum = thickmax
    if thickmin is not None:
        filters.xray_filter_thickness_minimum = thickmin
    filters.save()


def _xrayfiltersnone(source):
    filters = XrayFilters.objects.create(irradiation_event_xray_source_data=source)
    filters.xray_filter_type = get_or_create_cid("111609", "No filter")
    filters.save()


def _xray_filters_multiple(
    xray_filter_material,
    xray_filter_thickness_maximum,
    xray_filter_thickness_minimum,
    source,
):
    for i, material in enumerate(xray_filter_material):
        try:
            thickmax = None
            thickmin = None
            if isinstance(xray_filter_thickness_maximum, list):
                thickmax = xray_filter_thickness_maximum[i]
            if isinstance(xray_filter_thickness_minimum, list):
                thickmin = xray_filter_thickness_minimum[i]
            _xrayfilters("FLAT", material, thickmax, thickmin, source)
        except IndexError:
            pass


def _xray_filters_prep(dataset, source):
    xray_filter_type = get_value_kw("FilterType", dataset)
    xray_filter_material = get_value_kw("FilterMaterial", dataset)

    # Explicit no filter, register as such
    if xray_filter_type == "NONE":
        _xrayfiltersnone(source)
        return
    # Implicit no filter, just ignore
    if xray_filter_type is None:
        return

    # Get multiple filters into pydicom MultiValue or lists
    if (
        xray_filter_material
        and "," in xray_filter_material
        and not isinstance(xray_filter_material, MultiValue)
    ):
        xray_filter_material = xray_filter_material.split(",")

    xray_filter_thickness_minimum = get_value_kw("FilterThicknessMinimum", dataset)
    xray_filter_thickness_maximum = get_value_kw("FilterThicknessMaximum", dataset)
    if xray_filter_thickness_minimum and not isinstance(
        xray_filter_thickness_minimum, (MultiValue, list)
    ):
        try:
            float(xray_filter_thickness_minimum)
        except ValueError:
            if "," in xray_filter_thickness_minimum:
                xray_filter_thickness_minimum = xray_filter_thickness_minimum.split(",")
    if xray_filter_thickness_maximum and not isinstance(
        xray_filter_thickness_maximum, (MultiValue, list)
    ):
        try:
            float(xray_filter_thickness_maximum)
        except ValueError:
            if "," in xray_filter_thickness_maximum:
                xray_filter_thickness_maximum = xray_filter_thickness_maximum.split(",")

    if xray_filter_material and isinstance(xray_filter_material, (MultiValue, list)):
        _xray_filters_multiple(
            xray_filter_material,
            xray_filter_thickness_maximum,
            xray_filter_thickness_minimum,
            source,
        )
    else:
        # deal with known Siemens filter records
        siemens_filters = ("CU_0.1_MM", "CU_0.2_MM", "CU_0.3_MM")
        if xray_filter_type in siemens_filters:
            if xray_filter_type == "CU_0.1_MM":
                thickmax = 0.1
                thickmin = 0.1
            elif xray_filter_type == "CU_0.2_MM":
                thickmax = 0.2
                thickmin = 0.2
            elif xray_filter_type == "CU_0.3_MM":
                thickmax = 0.3
                thickmin = 0.3
            _xrayfilters("FLAT", "COPPER", thickmax, thickmin, source)
        else:
            _xrayfilters(
                xray_filter_type,
                xray_filter_material,
                xray_filter_thickness_maximum,
                xray_filter_thickness_minimum,
                source,
            )


def _kvp(dataset, source):
    kv = Kvp.objects.create(irradiation_event_xray_source_data=source)
    kv.kvp = get_value_kw("KVP", dataset)
    kv.save()


def _exposure(dataset, source):
    exp = Exposure.objects.create(irradiation_event_xray_source_data=source)

    exp.exposure = get_value_kw("ExposureInuAs", dataset)  # uAs
    if not exp.exposure:
        exposure = get_value_kw("Exposure", dataset)
        if exposure:
            exp.exposure = exposure * 1000
    exp.save()


def _xraygrid(gridcode, source):
    grid = XrayGrid.objects.create(irradiation_event_xray_source_data=source)
    if gridcode == "111646":
        grid.xray_grid = get_or_create_cid("111646", "No grid")
    elif gridcode == "111641":
        grid.xray_grid = get_or_create_cid("111641", "Fixed grid")
    elif gridcode == "111642":
        grid.xray_grid = get_or_create_cid("111642", "Focused grid")
    elif gridcode == "111643":
        grid.xray_grid = get_or_create_cid("111643", "Reciprocating grid")
    elif gridcode == "111644":
        grid.xray_grid = get_or_create_cid("111644", "Parallel grid")
    elif gridcode == "111645":
        grid.xray_grid = get_or_create_cid("111645", "Crossed grid")
    grid.save()


def _irradiationeventxraydetectordata(dataset, event):
    detector = IrradEventXRayDetectorData.objects.create(
        irradiation_event_xray_data=event
    )
    detector.exposure_index = get_value_kw("ExposureIndex", dataset)
    detector.relative_xray_exposure = get_value_kw("RelativeXRayExposure", dataset)
    manufacturer = detector.irradiation_event_xray_data.projection_xray_radiation_dose.general_study_module_attributes.generalequipmentmoduleattr_set.all()[
        0
    ].manufacturer.lower()
    if "fuji" in manufacturer:
        detector.relative_exposure_unit = "S ()"
    elif "carestream" in manufacturer:
        detector.relative_exposure_unit = "EI (Mbels)"
    elif "kodak" in manufacturer:
        detector.relative_exposure_unit = "EI (Mbels)"
    elif "agfa" in manufacturer:
        detector.relative_exposure_unit = "lgM (Bels)"
    elif "konica" in manufacturer:
        detector.relative_exposure_unit = "S ()"
    elif "canon" in manufacturer:
        detector.relative_exposure_unit = "REX ()"
    elif "swissray" in manufacturer:
        detector.relative_exposure_unit = "DI ()"
    elif "philips" in manufacturer:
        detector.relative_exposure_unit = "EI ()"
    elif "siemens" in manufacturer:
        detector.relative_exposure_unit = "EXI (μGy)"
    detector.sensitivity = get_value_kw("Sensitivity", dataset)
    detector.target_exposure_index = get_value_kw("TargetExposureIndex", dataset)
    detector.deviation_index = get_value_kw("DeviationIndex", dataset)
    detector.save()


def _irradiationeventxraysourcedata(dataset, event):
    # TODO: review model to convert to cid where appropriate, and add additional fields such as field height and width
    source = IrradEventXRaySourceData.objects.create(irradiation_event_xray_data=event)
    source.average_xray_tube_current = get_value_kw("XRayTubeCurrent", dataset)
    if not source.average_xray_tube_current:
        source.average_xray_tube_current = get_value_kw(
            "AverageXRayTubeCurrent", dataset
        )
    source.exposure_time = get_value_kw("ExposureTime", dataset)
    source.focal_spot_size = get_value_kw("FocalSpots", dataset)
    collimated_field_area = get_value_kw("FieldOfViewDimensions", dataset)
    if collimated_field_area:
        source.collimated_field_area = (
            float(collimated_field_area[0]) * float(collimated_field_area[1]) / 1000000
        )
    exp_ctrl_mode = get_value_kw("ExposureControlMode", dataset)
    if exp_ctrl_mode:
        source.exposure_control_mode = exp_ctrl_mode
    xray_grid = get_value_kw("Grid", dataset)
    if xray_grid:
        if xray_grid == "NONE":
            _xraygrid("111646", source)
        else:
            for gtype in xray_grid:
                if (
                    "FI" in gtype
                ):  # Fixed; abbreviated due to fitting two keywords in 16 characters
                    _xraygrid("111641", source)
                elif "FO" in gtype:  # Focused
                    _xraygrid("111642", source)
                elif "RE" in gtype:  # Reciprocating
                    _xraygrid("111643", source)
                elif "PA" in gtype:  # Parallel
                    _xraygrid("111644", source)
                elif "CR" in gtype:  # Crossed
                    _xraygrid("111645", source)
    source.grid_absorbing_material = get_value_kw("GridAbsorbingMaterial", dataset)
    source.grid_spacing_material = get_value_kw("GridSpacingMaterial", dataset)
    source.grid_thickness = get_value_kw("GridThickness", dataset)
    source.grid_pitch = get_value_kw("GridPitch", dataset)
    source.grid_aspect_ratio = get_value_kw("GridAspectRatio", dataset)
    source.grid_period = get_value_kw("GridPeriod", dataset)
    source.grid_focal_distance = get_value_kw("GridFocalDistance", dataset)
    source.save()
    _xray_filters_prep(dataset, source)
    _kvp(dataset, source)
    _exposure(dataset, source)


def _doserelateddistancemeasurements(dataset, mech):
    dist = DoseRelatedDistanceMeasurements.objects.create(
        irradiation_event_xray_mechanical_data=mech
    )
    manufacturer = dist.irradiation_event_xray_mechanical_data.irradiation_event_xray_data.projection_xray_radiation_dose.general_study_module_attributes.generalequipmentmoduleattr_set.all()[
        0
    ].manufacturer
    model_name = dist.irradiation_event_xray_mechanical_data.irradiation_event_xray_data.projection_xray_radiation_dose.general_study_module_attributes.generalequipmentmoduleattr_set.all()[
        0
    ].manufacturer_model_name
    dist.distance_source_to_detector = get_value_kw("DistanceSourceToDetector", dataset)
    if (
        dist.distance_source_to_detector
        and manufacturer
        and model_name
        and "kodak" in manufacturer.lower()
        and "dr 7500" in model_name.lower()
    ):
        dist.distance_source_to_detector *= 100  # convert dm to mm
    dist.distance_source_to_entrance_surface = get_value_kw(
        "DistanceSourceToPatient", dataset
    )
    dist.distance_source_to_isocenter = get_value_kw(
        "DistanceSourceToIsocenter", dataset
    )
    # DistanceSourceToReferencePoint isn't a DICOM tag. Same as DistanceSourceToPatient?
    #    dist.distance_source_to_reference_point = get_value_kw('DistanceSourceToReferencePoint',dataset)
    # Table longitudinal and lateral positions not DICOM elements.
    #    dist.table_longitudinal_position = get_value_kw('TableLongitudinalPosition',dataset)
    #    dist.table_lateral_position = get_value_kw('TableLateralPosition',dataset)
    dist.table_height_position = get_value_kw("TableHeight", dataset)
    # DistanceSourceToTablePlane not a DICOM tag.
    #    dist.distance_source_to_table_plane = get_value_kw('DistanceSourceToTablePlane',dataset)
    dist.radiological_thickness = get_value_num(0x00451049, dataset)
    dist.save()


def _irradiationeventxraymechanicaldata(dataset, event):
    mech = IrradEventXRayMechanicalData.objects.create(
        irradiation_event_xray_data=event
    )
    mech.magnification_factor = get_value_kw(
        "EstimatedRadiographicMagnificationFactor", dataset
    )
    mech.primary_angle = get_value_kw("PositionerPrimaryAngle", dataset)
    mech.secondary_angle = get_value_kw("PositionerSecondaryAngle", dataset)
    mech.column_angulation = get_value_kw("ColumnAngulation", dataset)
    mech.table_head_tilt_angle = get_value_kw("TableHeadTiltAngle", dataset)
    mech.table_horizontal_rotation_angle = get_value_kw(
        "TableHorizontalRotationAngle", dataset
    )
    mech.table_cradle_tilt_angle = get_value_kw("TableCradleTiltAngle", dataset)
    mech.save()
    _doserelateddistancemeasurements(dataset, mech)


def _irradiationeventxraydata(dataset, proj):  # TID 10003
    # TODO: review model to convert to cid where appropriate, and add additional fields

    event = IrradEventXRayData.objects.create(projection_xray_radiation_dose=proj)
    event.acquisition_plane = get_or_create_cid("113622", "Single Plane")
    event.irradiation_event_uid = get_value_kw("SOPInstanceUID", dataset)
    event_time = get_value_kw("AcquisitionTime", dataset)
    if not event_time:
        event_time = get_value_kw("ContentTime", dataset)
    if not event_time:
        event_time = get_value_kw("StudyTime", dataset)
    event_date = get_value_kw("AcquisitionDate", dataset)
    if not event_date:
        event_date = get_value_kw("ContentDate", dataset)
    if not event_date:
        event_date = get_value_kw("StudyDate", dataset)
    event.date_time_started = make_date_time("{0}{1}".format(event_date, event_time))
    event.irradiation_event_type = get_or_create_cid("113611", "Stationary Acquisition")
    event.acquisition_protocol = get_value_kw("ProtocolName", dataset)
    if not event.acquisition_protocol:
        manufacturer = get_value_kw("Manufacturer", dataset)
        software_versions = get_value_kw("SoftwareVersions", dataset)
        if manufacturer == "TOSHIBA_MEC" and software_versions == "TM_TFD_1.0":
            event.acquisition_protocol = get_value_kw("ImageComments", dataset)
    if not event.acquisition_protocol:
        event.acquisition_protocol = get_value_kw("SeriesDescription", dataset)
    if not event.acquisition_protocol:
        event.acquisition_protocol = get_seq_code_meaning(
            "PerformedProtocolCodeSequence", dataset
        )
    series_description = get_value_kw("SeriesDescription", dataset)
    if series_description:
        event.comment = series_description
    event.anatomical_structure = get_or_create_cid(
        get_seq_code_value("AnatomicRegionSequence", dataset),
        get_seq_code_meaning("AnatomicRegionSequence", dataset),
    )
    laterality = get_value_kw("ImageLaterality", dataset)
    if laterality:
        if laterality.strip() == "R":
            event.laterality = get_or_create_cid("G-A100", "Right")
        if laterality.strip() == "L":
            event.laterality = get_or_create_cid("G-A101", "Left")

    event.image_view = get_or_create_cid(
        get_seq_code_value("ViewCodeSequence", dataset),
        get_seq_code_meaning("ViewCodeSequence", dataset),
    )
    if not event.image_view:
        projection = get_value_kw("ViewPosition", dataset)
        if projection == "AP":
            event.image_view = get_or_create_cid("R-10206", "antero-posterior")
        elif projection == "PA":
            event.image_view = get_or_create_cid("R-10214", "postero-anterior")
        elif projection == "LL":
            event.image_view = get_or_create_cid("R-10236", "left lateral")
        elif projection == "RL":
            event.image_view = get_or_create_cid("R-10232", "right lateral")
            # http://dicomlookup.com/lookup.asp?sw=Tnumber&q=(0018,5101) lists four other views: RLD (Right Lateral Decubitus),
            # LLD (Left Lateral Decubitus), RLO (Right Lateral Oblique) and LLO (Left Lateral Oblique). There isn't an exact
            # match for these views in the CID 4010 DX View (http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_4010.html)

    # image view modifier?
    if event.anatomical_structure:
        event.target_region = event.anatomical_structure
    event.entrance_exposure_at_rp = get_value_kw("EntranceDoseInmGy", dataset)
    # reference point definition?
    pc_fibroglandular = get_value_kw("CommentsOnRadiationDose", dataset)
    if pc_fibroglandular:
        if "%" in pc_fibroglandular:
            event.percent_fibroglandular_tissue = pc_fibroglandular.replace(
                "%", ""
            ).strip()
    exposure_control = get_value_kw("ExposureControlModeDescription", dataset)

    if event.comment and exposure_control:
        event.comment = event.comment + ", " + exposure_control

    dap = get_value_kw("ImageAndFluoroscopyAreaDoseProduct", dataset)
    if dap:
        event.dose_area_product = (
            dap / 100000
        )  # Value of DICOM tag (0018,115e) in dGy.cm2, converted to Gy.m2
    event.save()

    _irradiationeventxraydetectordata(dataset, event)
    _irradiationeventxraysourcedata(dataset, event)
    _irradiationeventxraymechanicaldata(dataset, event)
    _accumulatedxraydose_update(event)


def _accumulatedxraydose(proj):
    accum = AccumXRayDose.objects.create(projection_xray_radiation_dose=proj)
    accum.acquisition_plane = get_or_create_cid("113622", "Single Plane")
    accum.save()
    accumint = AccumIntegratedProjRadiogDose.objects.create(accumulated_xray_dose=accum)
    accumint.dose_area_product_total = 0.0
    accumint.total_number_of_radiographic_frames = 0
    accumint.save()


def _accumulatedxraydose_update(event):
    accumint = (
        event.projection_xray_radiation_dose.accumxraydose_set.get().accumintegratedprojradiogdose_set.get()
    )
    if accumint.total_number_of_radiographic_frames is not None:
        accumint.total_number_of_radiographic_frames = (
            accumint.total_number_of_radiographic_frames + 1
        )
    else:
        accumint.total_number_of_radiographic_frames = 1

    if event.dose_area_product:
        accumint.dose_area_product_total += Decimal(event.dose_area_product)
    accumint.save()


def _projectionxrayradiationdose(dataset, g):
    proj = ProjectionXRayRadiationDose.objects.create(general_study_module_attributes=g)
    proj.procedure_reported = get_or_create_cid("113704", "Projection X-Ray")
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
    patientatt = PatientStudyModuleAttr.objects.create(
        general_study_module_attributes=g
    )
    patientatt.patient_age = get_value_kw("PatientAge", dataset)
    patientatt.patient_weight = get_value_kw("PatientWeight", dataset)
    patientatt.patient_size = get_value_kw("PatientSize", dataset)
    try:
        Decimal(patientatt.patient_size)
    except DecimalException:
        patientatt.patient_size = None
    except TypeError:
        pass
    patientatt.save()


def _generalstudymoduleattributes(dataset, g):
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
    if not g.study_description:
        g.study_description = get_value_kw("SeriesDescription", dataset)
    if not g.study_description:
        g.study_description = get_seq_code_meaning("ProcedureCodeSequence", dataset)
    g.modality_type = get_value_kw("Modality", dataset)
    g.physician_of_record = list_to_string(get_value_kw("PhysiciansOfRecord", dataset))
    g.name_of_physician_reading_study = list_to_string(
        get_value_kw("NameOfPhysiciansReadingStudy", dataset)
    )
    g.performing_physician_name = list_to_string(
        get_value_kw("PerformingPhysicianName", dataset)
    )
    g.operator_name = list_to_string(get_value_kw("OperatorsName", dataset))
    # Being used to summarise protocol for study:
    g.procedure_code_meaning = get_seq_code_meaning("ProcedureCodeSequence", dataset)
    if not g.procedure_code_meaning:
        g.procedure_code_meaning = get_value_kw("ProtocolName", dataset)
    if not g.procedure_code_meaning:
        g.procedure_code_meaning = get_value_kw("StudyDescription", dataset)
    if not g.procedure_code_meaning:
        g.procedure_code_meaning = get_value_kw("SeriesDescription", dataset)
    g.requested_procedure_code_value = get_seq_code_value(
        "RequestedProcedureCodeSequence", dataset
    )
    g.requested_procedure_code_meaning = get_seq_code_meaning(
        "RequestedProcedureCodeSequence", dataset
    )
    if not g.requested_procedure_code_value:
        g.requested_procedure_code_value = get_seq_code_value(
            "RequestAttributesSequence", dataset
        )
    if not g.requested_procedure_code_value:
        g.requested_procedure_code_value = get_seq_code_value(
            "ProcedureCodeSequence", dataset
        )
    if not g.requested_procedure_code_value:
        g.requested_procedure_code_value = get_seq_code_value(
            "PerformedProtocolCodeSequence", dataset
        )
    if not g.requested_procedure_code_meaning:
        g.requested_procedure_code_meaning = get_seq_code_meaning(
            "RequestAttributesSequence", dataset
        )
    if not g.requested_procedure_code_meaning:
        g.requested_procedure_code_meaning = get_seq_code_meaning(
            "ProcedureCodeSequence", dataset
        )
    if not g.requested_procedure_code_meaning:
        g.requested_procedure_code_meaning = get_value_num(0x00321060, dataset)
    if not g.requested_procedure_code_meaning:
        g.requested_procedure_code_meaning = get_seq_code_meaning(
            "PerformedProtocolCodeSequence", dataset
        )
    if not g.requested_procedure_code_meaning:
        manufacturer = get_value_kw("Manufacturer", dataset)
        model = get_value_kw("ManufacturerModelName", dataset)
        if (
            manufacturer
            and model
            and "canon" in manufacturer.lower()
            and "cxdi" in model.lower()
        ):
            g.requested_procedure_code_meaning = get_value_num(0x00081030, dataset)
        if (
            manufacturer
            and model
            and "carestream health" in manufacturer.lower()
            and "drx-revolution" in model.lower()
        ):
            g.requested_procedure_code_meaning = get_value_num(0x00081030, dataset)
    g.save()

    _generalequipmentmoduleattributes(dataset, g)
    _projectionxrayradiationdose(dataset, g)
    _patientstudymoduleattributes(dataset, g)
    patient_module_attributes(dataset, g)
    populate_dx_rf_summary(g)
    g.number_of_events = (
        g.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
    )
    g.save()

    # Add standard names
    add_standard_names(g)


# The routine will accept three types of image:
# CR image storage                               (SOP UID = '1.2.840.10008.5.1.4.1.1.1')
# Digital x-ray image storage - for presentation (SOP UID = '1.2.840.10008.5.1.4.1.1.1.1')
# Digital x-ray image storage - for processing   (SOP UID = '1.2.840.10008.5.1.4.1.1.1.1.1')
# These SOP UIDs were taken from http://www.dicomlibrary.com/dicom/sop/
def _test_if_dx(dataset):
    """Test if dicom object passed is a DX or CR radiographic file by looking at SOP Class UID"""
    if (
        dataset.SOPClassUID != "1.2.840.10008.5.1.4.1.1.1"
        and dataset.SOPClassUID != "1.2.840.10008.5.1.4.1.1.1.1"
        and dataset.SOPClassUID != "1.2.840.10008.5.1.4.1.1.1.1.1"
    ):
        return 0
    return 1


def _dx2db(dataset):
    study_uid = get_value_kw("StudyInstanceUID", dataset)
    if not study_uid:
        error = "In dx import: No UID returned"
        logger.error(error)
        record_task_error_exit(error)
        return
    record_task_info(f"UID: {study_uid.replace('.', '. ')}")
    record_task_related_query(study_uid)
    study_in_db = check_uid.check_uid(study_uid)

    if study_in_db:
        sleep(
            2.0
        )  # Give initial event a chance to get to save on _projectionxrayradiationdose
        this_study = get_study_check_dup(dataset, modality="DX")
        if this_study:
            _irradiationeventxraydata(
                dataset, this_study.projectionxrayradiationdose_set.get()
            )
            populate_dx_rf_summary(this_study)
            this_study.number_of_events = (
                this_study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
            )
            this_study.save()

            # Update any matching standard names
            add_standard_names(this_study)

        else:
            error = f"Study {study_uid.replace('.', '. ')} already in DB"
            logger.error(error)
            record_task_error_exit(error)
            return

    if not study_in_db:
        # study doesn't exist, start from scratch
        g = GeneralStudyModuleAttr.objects.create()
        g.study_instance_uid = get_value_kw("StudyInstanceUID", dataset)
        g.save()
        logger.debug(
            "Started importing DX with Study Instance UID of {0}".format(
                g.study_instance_uid
            )
        )
        event_uid = get_value_kw("SOPInstanceUID", dataset)
        check_uid.record_sop_instance_uid(g, event_uid)
        # check study again
        study_in_db = check_uid.check_uid(study_uid)
        if study_in_db == 1:
            _generalstudymoduleattributes(dataset, g)
        elif not study_in_db:
            error = "Something went wrong, GeneralStudyModuleAttr wasn't created"
            record_task_error_exit(error)
            logger.error(error)
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
                        this_study = get_study_check_dup(dataset, modality="DX")
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
                                this_study = get_study_check_dup(dataset, modality="DX")
                                if this_study:
                                    _irradiationeventxraydata(
                                        dataset,
                                        this_study.projectionxrayradiationdose_set.get(),
                                    )
                elif study_in_db == 1:
                    sleep(
                        2.0
                    )  # Give initial event a chance to get to save on _projectionxrayradiationdose
                    this_study = get_study_check_dup(dataset, modality="DX")
                    if this_study:
                        _irradiationeventxraydata(
                            dataset,
                            this_study.projectionxrayradiationdose_set.get(),
                        )


def _fix_kodak_filters(dataset):
    """
    Replace floats with commas in with multivalue floats: as found in older Carestream/Kodak units such as the DR7500
    :param dataset: DICOM dataset
    :return: Repaired DICOM dataset
    """

    try:  # Black magic pydicom method suggested by Darcy Mason: https://groups.google.com/forum/?hl=en-GB#!topic/pydicom/x_WsC2gCLck
        xray_filter_thickness_minimum = get_value_kw("FilterThicknessMinimum", dataset)
    except (
        ValueError
    ):  # Assumes ValueError will be a comma separated pair of numbers, as per Kodak.
        thick = dict.__getitem__(
            dataset, 0x187052
        )  # pydicom black magic as suggested by
        thickval = thick.__getattribute__("value")
        if "," in thickval:
            thickval = thickval.replace(",", "\\")
            thick2 = thick._replace(value=thickval)
            dict.__setitem__(dataset, 0x187052, thick2)

    try:
        xray_filter_thickness_maximum = get_value_kw("FilterThicknessMaximum", dataset)
    except (
        ValueError
    ):  # Assumes ValueError will be a comma separated pair of numbers, as per Kodak.
        thick = dict.__getitem__(
            dataset, 0x187054
        )  # pydicom black magic as suggested by
        thickval = thick.__getattribute__("value")
        if "," in thickval:
            thickval = thickval.replace(",", "\\")
            thick2 = thick._replace(value=thickval)
            dict.__setitem__(dataset, 0x187054, thick2)


def _remove_spaces_decimal(value):
    """Remove padding and convert to decimal"""
    while value and value.endswith((" ", "\x00")):
        value = value[:-1]
    return Decimal(value)


def _fix_exposure_values(dataset):
    """
    Replace decimal values in ExposureTime, XRayTubeCurrent and Exposure with integer values; move original values to
    appropriate fields
    :param dataset: DICOM dataset
    :return: Repaired DICOM dataset
    """
    try:
        dataset.ExposureTime
    except TypeError:
        exposure_time = dataset.get_item("ExposureTime").value.decode()
        exposure_time = _remove_spaces_decimal(exposure_time)
        del dataset.ExposureTime
        dataset.ExposureTime = round(exposure_time, 0)
        if "ExposureTimeInuS" not in dataset:
            dataset.ExposureTimeInuS = round(exposure_time * 1000, 0)
        logger.warning(
            f"ExposureTime (VR=IS) contained illegal value {exposure_time}. Now set "
            f"to {dataset.ExposureTime}. ExposureTimeInuS now set to "
            f"{dataset.ExposureTimeInuS}."
        )
    try:
        dataset.XRayTubeCurrent
    except TypeError:
        xray_tube_current = dataset.get_item("XRayTubeCurrent").value.decode()
        xray_tube_current = _remove_spaces_decimal(xray_tube_current)
        del dataset.XRayTubeCurrent
        dataset.XRayTubeCurrent = round(xray_tube_current, 0)
        if "XRayTubeCurrentInuA" not in dataset:
            dataset.XRayTubeCurrentInuA = round(xray_tube_current * 1000, 0)
        logger.warning(
            f"ExposureTime (VR=IS) contained illegal value {xray_tube_current}. Now set "
            f"to {dataset.XRayTubeCurrent}. ExposureTimeInuS now set to "
            f"{dataset.XRayTubeCurrentInuA}."
        )
    try:
        dataset.Exposure
    except TypeError:
        exposure = dataset.get_item("Exposure").value.decode()
        exposure = _remove_spaces_decimal(exposure)
        del dataset.Exposure
        dataset.Exposure = round(exposure, 0)
        if "ExposureInuAs" not in dataset:
            dataset.ExposureInuAs = round(exposure * 1000, 0)
        logger.warning(
            f"Exposure (VR=IS) contained illegal value {exposure}. Now set "
            f"to {dataset.Exposure}. ExposureTimeInuS now set to "
            f"{dataset.ExposureInuAs}."
        )


def dx(dig_file):
    """Extract radiation dose structured report related data from DX radiographic images

    :param filename: relative or absolute path to DICOM DX radiographic image file.
    :type filename: str.

    """

    try:
        del_settings = DicomDeleteSettings.objects.get()
        del_dx_im = del_settings.del_dx_im
    except ObjectDoesNotExist:
        del_dx_im = False

    # Set convert_wrong_length_to_UN = True to prevent the wrong length causing an error.
    config.convert_wrong_length_to_UN = True

    logger.debug("About to read DX")

    try:
        dataset = pydicom.dcmread(dig_file)
    except FileNotFoundError:
        logger.warning(
            f"dx.py not attempting to extract from {dig_file}, the file does not exist"
        )
        record_task_error_exit(
            f"Not attempting to extract from {dig_file}, the file does not exist"
        )
        return 1

    try:
        dataset.decode()
    except ValueError as err:
        if "could not convert string to float" in str(err):
            _fix_kodak_filters(dataset)
            dataset.decode()
    except TypeError as err:
        if "Could not convert value to integer without loss" in str(err):
            _fix_exposure_values(dataset)
            dataset.decode()
    isdx = _test_if_dx(dataset)
    if not isdx:
        error = "{0} is not a DICOM DX radiographic image".format(dig_file)
        logger.error(error)
        record_task_error_exit(error)
        return 1

    logger.debug("About to launch _dx2db")
    _dx2db(dataset)

    if del_dx_im:
        os.remove(dig_file)

    return 0
