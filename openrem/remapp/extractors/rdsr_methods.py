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

..  module:: rdsr_methods
    :synopsis: methods used to read the standard irradiation part of rdsr
        objects. This is used by rdsr.py.

..  moduleauthor:: Ed McDonagh

"""

from decimal import Decimal
import logging

from defusedxml.ElementTree import fromstring, ParseError
from django.db.models import Avg, ObjectDoesNotExist, Max
from django.core.exceptions import ValidationError

from ..tools.dcmdatetime import make_date_time
from ..tools.get_values import (
    get_or_create_cid,
    test_numeric_value,
)
from .extract_common import person_participant, observercontext
from remapp.models import (  # pylint: disable=wrong-import-order, wrong-import-position
    AccumCassetteBsdProjRadiogDose,
    AccumIntegratedProjRadiogDose,
    AccumMammographyXRayDose,
    AccumProjXRayDose,
    AccumXRayDose,
    Calibration,
    CtAccumulatedDoseData,
    CtDoseCheckDetails,
    CtIrradiationEventData,
    CtRadiationDose,
    CtXRaySourceParameters,
    DeviceParticipant,
    DoseRelatedDistanceMeasurements,
    Exposure,
    GeneralEquipmentModuleAttr,
    ImageViewModifier,
    IrradEventXRayData,
    IrradEventXRayDetectorData,
    IrradEventXRayMechanicalData,
    IrradEventXRaySourceData,
    Kvp,
    ObserverContext,
    ProjectionXRayRadiationDose,
    PulseWidth,
    ScanningLength,
    XrayFilters,
    XrayGrid,
    XrayTubeCurrent,
)

logger = logging.getLogger("remapp.extractors.rdsr")


def _deviceparticipant(dataset, eventdatatype, foreignkey):

    if eventdatatype == "detector":
        device = DeviceParticipant.objects.create(
            irradiation_event_xray_detector_data=foreignkey
        )
    elif eventdatatype == "source":
        device = DeviceParticipant.objects.create(
            irradiation_event_xray_source_data=foreignkey
        )
    elif eventdatatype == "accumulated":
        device = DeviceParticipant.objects.create(accumulated_xray_dose=foreignkey)
    elif eventdatatype == "ct_accumulated":
        device = DeviceParticipant.objects.create(ct_accumulated_dose_data=foreignkey)
    elif eventdatatype == "ct_event":
        device = DeviceParticipant.objects.create(ct_irradiation_event_data=foreignkey)
    else:
        logger.warning(
            f"RDSR import, in _deviceparticipant, but no suitable eventdatatype (is {eventdatatype})"
        )
        return
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Device Role in Procedure":
            device.device_role_in_procedure = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
            for cont2 in cont.ContentSequence:
                if cont2.ConceptNameCodeSequence[0].CodeMeaning == "Device Name":
                    device.device_name = cont2.TextValue
                elif (
                    cont2.ConceptNameCodeSequence[0].CodeMeaning
                    == "Device Manufacturer"
                ):
                    device.device_manufacturer = cont2.TextValue
                elif (
                    cont2.ConceptNameCodeSequence[0].CodeMeaning == "Device Model Name"
                ):
                    device.device_model_name = cont2.TextValue
                elif (
                    cont2.ConceptNameCodeSequence[0].CodeMeaning
                    == "Device Serial Number"
                ):
                    device.device_serial_number = cont2.TextValue
                elif (
                    cont2.ConceptNameCodeSequence[0].CodeMeaning
                    == "Device Observer UID"
                ):
                    device.device_observer_uid = cont2.UID
    device.save()


def _pulsewidth(pulse_width_value, source):
    """Takes pulse width values and populates PulseWidth table

    :param pulse_width_value: Decimal or list of decimals
    :param source: database object in IrradEventXRaySourceData table
    :return: None
    """

    try:
        pulse = PulseWidth.objects.create(irradiation_event_xray_source_data=source)
        pulse.pulse_width = pulse_width_value
        pulse.save()
    except (ValueError, TypeError, ValidationError):
        if not hasattr(pulse_width_value, "strip") and (
            hasattr(pulse_width_value, "__getitem__")
            or hasattr(pulse_width_value, "__iter__")
        ):
            for per_pulse_pulse_width in pulse_width_value:
                pulse = PulseWidth.objects.create(
                    irradiation_event_xray_source_data=source
                )
                pulse.pulse_width = per_pulse_pulse_width
                pulse.save()


def _kvptable(kvp_value, source):
    """Takes kVp values and populates kvp table

    :param kvp_value: Decimal or list of decimals
    :param source: database object in IrradEventXRaySourceData table
    :return: None
    """

    try:
        kvpdata = Kvp.objects.create(irradiation_event_xray_source_data=source)
        kvpdata.kvp = kvp_value
        kvpdata.save()
    except (ValueError, TypeError, ValidationError):
        if not hasattr(kvp_value, "strip") and (
            hasattr(kvp_value, "__getitem__") or hasattr(kvp_value, "__iter__")
        ):
            for per_pulse_kvp in kvp_value:
                kvp = Kvp.objects.create(irradiation_event_xray_source_data=source)
                kvp.kvp = per_pulse_kvp
                kvp.save()


def _xraytubecurrent(current_value, source):
    """Takes X-ray tube current values and populates XrayTubeCurrent table

    :param current_value: Decimal or list of decimals
    :param source: database object in IrradEventXRaySourceData table
    :return: None
    """

    try:
        tubecurrent = XrayTubeCurrent.objects.create(
            irradiation_event_xray_source_data=source
        )
        tubecurrent.xray_tube_current = current_value
        tubecurrent.save()
    except (ValueError, TypeError, ValidationError):
        if not hasattr(current_value, "strip") and (
            hasattr(current_value, "__getitem__") or hasattr(current_value, "__iter__")
        ):
            for per_pulse_current in current_value:
                tubecurrent = XrayTubeCurrent.objects.create(
                    irradiation_event_xray_source_data=source
                )
                tubecurrent.xray_tube_current = per_pulse_current
                tubecurrent.save()


def _exposure(exposure_value, source):
    """Takes exposure (uA.s) values and populates Exposure table

    :param exposure_value: Decimal or list of decimals
    :param source: database object in IrradEventXRaySourceData table
    :return: None
    """

    try:
        exposure = Exposure.objects.create(irradiation_event_xray_source_data=source)
        exposure.exposure = exposure_value
        exposure.save()
    except (ValueError, TypeError, ValidationError):
        if not hasattr(exposure_value, "strip") and (
            hasattr(exposure_value, "__getitem__")
            or hasattr(exposure_value, "__iter__")
        ):
            for per_pulse_exposure in exposure_value:
                exposure = Exposure.objects.create(
                    irradiation_event_xray_source_data=source
                )
                exposure.exposure = per_pulse_exposure
                exposure.save()


def _xrayfilters(content_sequence, source):

    filters = XrayFilters.objects.create(irradiation_event_xray_source_data=source)
    for cont2 in content_sequence:
        if cont2.ConceptNameCodeSequence[0].CodeMeaning == "X-Ray Filter Type":
            filters.xray_filter_type = get_or_create_cid(
                cont2.ConceptCodeSequence[0].CodeValue,
                cont2.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont2.ConceptNameCodeSequence[0].CodeMeaning == "X-Ray Filter Material":
            filters.xray_filter_material = get_or_create_cid(
                cont2.ConceptCodeSequence[0].CodeValue,
                cont2.ConceptCodeSequence[0].CodeMeaning,
            )
        elif (
            cont2.ConceptNameCodeSequence[0].CodeMeaning
            == "X-Ray Filter Thickness Minimum"
        ):
            filters.xray_filter_thickness_minimum = test_numeric_value(
                cont2.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont2.ConceptNameCodeSequence[0].CodeMeaning
            == "X-Ray Filter Thickness Maximum"
        ):
            filters.xray_filter_thickness_maximum = test_numeric_value(
                cont2.MeasuredValueSequence[0].NumericValue
            )
    filters.save()


def _doserelateddistancemeasurements(dataset, mech):  # CID 10008

    distance = DoseRelatedDistanceMeasurements.objects.create(
        irradiation_event_xray_mechanical_data=mech
    )
    codes = {
        "Distance Source to Isocenter": "distance_source_to_isocenter",
        "Distance Source to Reference Point": "distance_source_to_reference_point",
        "Distance Source to Detector": "distance_source_to_detector",
        "Table Longitudinal Position": "table_longitudinal_position",
        "Table Lateral Position": "table_lateral_position",
        "Table Height Position": "table_height_position",
        "Distance Source to Table Plane": "distance_source_to_table_plane",
    }
    # For Philips Allura XPer systems you get the privately defined 'Table Height Position' with CodingSchemeDesignator
    # '99PHI-IXR-XPER' instead of the DICOM defined 'Table Height Position'.
    # It seems they are defined the same
    for cont in dataset.ContentSequence:
        try:
            setattr(
                distance,
                codes[cont.ConceptNameCodeSequence[0].CodeMeaning],
                cont.MeasuredValueSequence[0].NumericValue,
            )
        except KeyError:
            pass
    distance.save()


def _irradiationeventxraymechanicaldata(dataset, event):  # TID 10003c

    mech = IrradEventXRayMechanicalData.objects.create(
        irradiation_event_xray_data=event
    )
    for cont in dataset.ContentSequence:
        if (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "CR/DR Mechanical Configuration"
        ):
            mech.crdr_mechanical_configuration = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Positioner Primary Angle":
            try:
                mech.positioner_primary_angle = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning == "Positioner Secondary Angle"
        ):
            try:
                mech.positioner_secondary_angle = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Positioner Primary End Angle"
        ):
            try:
                mech.positioner_primary_end_angle = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Positioner Secondary End Angle"
        ):
            try:
                mech.positioner_secondary_end_angle = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Column Angulation":
            try:
                mech.column_angulation = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Table Head Tilt Angle":
            try:
                mech.table_head_tilt_angle = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Table Horizontal Rotation Angle"
        ):
            try:
                mech.table_horizontal_rotation_angle = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Table Cradle Tilt Angle":
            try:
                mech.table_cradle_tilt_angle = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Compression Thickness":
            try:
                mech.compression_thickness = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif cont.ConceptNameCodeSequence[0].CodeValue == "111647":  # Compression Force
            try:
                mech.compression_force = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "111648"
        ):  # Compression Pressure
            try:
                mech.compression_pressure = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "111649"
        ):  # Compression Contact Area
            try:
                mech.compression_contact_area = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
    _doserelateddistancemeasurements(dataset, mech)
    mech.save()


def _check_dap_units(dap_sequence):
    """Check for non-conformant DAP units of dGycm2 before storing value

    :param dap_sequence: MeasuredValueSequence[0] from ConceptNameCodeSequence of any of the Dose Area Products
    :return: dose_area_product in Gy.m2
    """
    dap = test_numeric_value(dap_sequence.NumericValue)
    try:
        if dap and dap_sequence.MeasurementUnitsCodeSequence[0].CodeValue == "dGy.cm2":
            return dap * 0.00001
        else:
            return dap
    except AttributeError:
        return dap


def _check_rp_dose_units(rp_dose_sequence):
    """Check for non-conformant dose at reference point units of mGy before storing value

    :param rp_dose_sequence: MeasuredValueSequence[0] from ConceptNameCodeSequence of any dose at RP
    :return: dose at reference point in Gy
    """
    rp_dose = test_numeric_value(rp_dose_sequence.NumericValue)
    try:
        if (
            rp_dose
            and rp_dose_sequence.MeasurementUnitsCodeSequence[0].CodeValue == "mGy"
        ):
            return rp_dose * 0.001
        else:
            return rp_dose
    except AttributeError:
        return rp_dose


def _irradiationeventxraysourcedata(dataset, event):  # TID 10003b
    # Name in DICOM standard for TID 10003B is Irradiation Event X-Ray Source Data
    # See http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_TID_10003B.html
    # TODO: review model to convert to cid where appropriate, and add additional fields

    # Variables below are used if privately defined parameters are available
    private_collimated_field_height = None
    private_collimated_field_width = None
    private_collimated_field_area = None

    source = IrradEventXRaySourceData.objects.create(irradiation_event_xray_data=event)
    for cont in dataset.ContentSequence:
        try:
            if cont.ConceptNameCodeSequence[0].CodeValue == "113738":  # = 'Dose (RP)'
                source.dose_rp = _check_rp_dose_units(cont.MeasuredValueSequence[0])
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Reference Point Definition"
            ):
                try:
                    source.reference_point_definition_code = get_or_create_cid(
                        cont.ConceptCodeSequence[0].CodeValue,
                        cont.ConceptCodeSequence[0].CodeMeaning,
                    )
                except AttributeError:
                    source.reference_point_definition = cont.TextValue
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning == "Average Glandular Dose"
            ):
                source.average_glandular_dose = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Fluoro Mode":
                source.fluoro_mode = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Pulse Rate":
                source.pulse_rate = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Number of Pulses":
                source.number_of_pulses = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning == "Number of Frames"
            ) and (
                cont.ConceptNameCodeSequence[0].CodingSchemeDesignator
                == "99PHI-IXR-XPER"
            ):
                # Philips Allura XPer systems: Private coding scheme designator: 99PHI-IXR-XPER; [number of pulses]
                source.number_of_pulses = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
                # should be a derivation thing in here for when the no. pulses is estimated
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Irradiation Duration":
                source.irradiation_duration = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Average X-Ray Tube Current"
            ):
                source.average_xray_tube_current = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Exposure Time":
                source.exposure_time = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Focal Spot Size":
                source.focal_spot_size = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Anode Target Material":
                source.anode_target_material = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Collimated Field Area":
                source.collimated_field_area = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            # TODO: xray_grid no longer exists in this table - it is a model on its own...
            # See https://bitbucket.org/openrem/openrem/issue/181
            elif cont.ConceptNameCodeSequence[0].CodeValue == "111635":  # 'X-Ray Grid'
                grid = XrayGrid.objects.create(
                    irradiation_event_xray_source_data=source
                )
                grid.xray_grid = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
                grid.save()
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Pulse Width":
                _pulsewidth(cont.MeasuredValueSequence[0].NumericValue, source)
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "KVP":
                _kvptable(cont.MeasuredValueSequence[0].NumericValue, source)
            elif (
                cont.ConceptNameCodeSequence[0].CodeValue == "113734"
            ):  # 'X-Ray Tube Current':
                _xraytubecurrent(cont.MeasuredValueSequence[0].NumericValue, source)
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Exposure":
                _exposure(cont.MeasuredValueSequence[0].NumericValue, source)
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "X-Ray Filters":
                _xrayfilters(cont.ContentSequence, source)
            # Maybe we have a Philips Xper system and we can use the privately defined information
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning == "Wedges and Shutters"
            ) and (
                cont.ConceptNameCodeSequence[0].CodingSchemeDesignator
                == "99PHI-IXR-XPER"
            ):
                # According to DICOM Conformance statement:
                # http://incenter.medical.philips.com/doclib/enc/fetch/2000/4504/577242/577256/588723/5144873/5144488/
                # 5144772/DICOM_Conformance_Allura_8.2.pdf%3fnodeid%3d10125540%26vernum%3d-2
                # "Actual shutter distance from centerpoint of collimator specified in the plane at 1 meter.
                # Unit: mm. End of run value is used."
                bottom_shutter_pos = None
                left_shutter_pos = None
                right_shutter_pos = None
                top_shutter_pos = None
                try:
                    for cont2 in cont.ContentSequence:
                        if (
                            cont2.ConceptNameCodeSequence[0].CodeMeaning
                            == "Bottom Shutter"
                        ):
                            bottom_shutter_pos = test_numeric_value(
                                cont2.MeasuredValueSequence[0].NumericValue
                            )
                        if (
                            cont2.ConceptNameCodeSequence[0].CodeMeaning
                            == "Left Shutter"
                        ):
                            left_shutter_pos = test_numeric_value(
                                cont2.MeasuredValueSequence[0].NumericValue
                            )
                        if (
                            cont2.ConceptNameCodeSequence[0].CodeMeaning
                            == "Right Shutter"
                        ):
                            right_shutter_pos = test_numeric_value(
                                cont2.MeasuredValueSequence[0].NumericValue
                            )
                        if (
                            cont2.ConceptNameCodeSequence[0].CodeMeaning
                            == "Top Shutter"
                        ):
                            top_shutter_pos = test_numeric_value(
                                cont2.MeasuredValueSequence[0].NumericValue
                            )
                    # Get distance_source_to_detector (Sdd) in meters
                    # Philips Allura XPer only notes distance_source_to_detector if it changed
                    try:
                        Sdd = (
                            float(
                                event.irradeventxraymechanicaldata_set.get()
                                .doserelateddistancemeasurements_set.get()
                                .distance_source_to_detector
                            )
                            / 1000
                        )
                    except (ObjectDoesNotExist, TypeError):
                        Sdd = None
                    if (
                        bottom_shutter_pos
                        and left_shutter_pos
                        and right_shutter_pos
                        and top_shutter_pos
                        and Sdd
                    ):
                        # calculate collimated field area, collimated Field Height and Collimated Field Width
                        # at image receptor (shutter positions are defined at 1 meter)
                        private_collimated_field_height = (
                            right_shutter_pos + left_shutter_pos
                        ) * Sdd  # in mm
                        private_collimated_field_width = (
                            bottom_shutter_pos + top_shutter_pos
                        ) * Sdd  # in mm
                        private_collimated_field_area = (
                            private_collimated_field_height
                            * private_collimated_field_width
                        ) / 1000000  # in m2
                except AttributeError:
                    pass
        except IndexError:
            pass
    _deviceparticipant(dataset, "source", source)
    try:
        source.ii_field_size = (
            fromstring(source.irradiation_event_xray_data.comment)
            .find("iiDiameter")
            .get("SRData")
        )
    except (ParseError, AttributeError, TypeError):
        logger.debug(
            "Failed in attempt to get II field size from comment (aimed at Siemens)"
        )
    if (not source.collimated_field_height) and private_collimated_field_height:
        source.collimated_field_height = private_collimated_field_height
    if (not source.collimated_field_width) and private_collimated_field_width:
        source.collimated_field_width = private_collimated_field_width
    if (not source.collimated_field_area) and private_collimated_field_area:
        source.collimated_field_area = private_collimated_field_area
    source.save()
    if not source.exposure_time and source.number_of_pulses:
        try:
            avg_pulse_width = source.pulsewidth_set.all().aggregate(Avg("pulse_width"))[
                "pulse_width__avg"
            ]
            if avg_pulse_width:
                source.exposure_time = avg_pulse_width * Decimal(
                    source.number_of_pulses
                )
                source.save()
        except ObjectDoesNotExist:
            pass
    if not source.average_xray_tube_current:
        if source.xraytubecurrent_set.all().count() > 0:
            source.average_xray_tube_current = (
                source.xraytubecurrent_set.all().aggregate(Avg("xray_tube_current"))[
                    "xray_tube_current__avg"
                ]
            )
            source.save()


def _irradiationeventxraydetectordata(dataset, event):  # TID 10003a

    detector = IrradEventXRayDetectorData.objects.create(
        irradiation_event_xray_data=event
    )
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Exposure Index":
            detector.exposure_index = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Target Exposure Index":
            detector.target_exposure_index = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Deviation Index":
            detector.deviation_index = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
    _deviceparticipant(dataset, "detector", detector)
    detector.save()


def _imageviewmodifier(dataset, event):

    modifier = ImageViewModifier.objects.create(irradiation_event_xray_data=event)
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Image View Modifier":
            modifier.image_view_modifier = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
            # TODO: Projection Eponymous Name should be in here - needs db change
    modifier.save()


def _get_patient_position_from_xml_string(event, xml_string):
    """Use XML parser to extract patient position information from Comment value in Siemens RF RDSR

    :param event: IrradEventXRayData object
    :param xml_string: Comment value
    :return:
    """

    if not xml_string:
        return
    try:
        orientation = (
            fromstring(xml_string)
            .find("PatientPosition")
            .find("Position")
            .get("SRData")
        )
        if orientation.strip().lower() == "hfs":
            event.patient_table_relationship_cid = get_or_create_cid(
                "F-10470", "headfirst"
            )
            event.patient_orientation_cid = get_or_create_cid("F-10450", "recumbent")
            event.patient_orientation_modifier_cid = get_or_create_cid(
                "F-10340", "supine"
            )
        elif orientation.strip().lower() == "hfp":
            event.patient_table_relationship_cid = get_or_create_cid(
                "F-10470", "headfirst"
            )
            event.patient_orientation_cid = get_or_create_cid("F-10450", "recumbent")
            event.patient_orientation_modifier_cid = get_or_create_cid(
                "F-10310", "prone"
            )
        elif orientation.strip().lower() == "ffs":
            event.patient_table_relationship_cid = get_or_create_cid(
                "F-10480", "feet-first"
            )
            event.patient_orientation_cid = get_or_create_cid("F-10450", "recumbent")
            event.patient_orientation_modifier_cid = get_or_create_cid(
                "F-10340", "supine"
            )
        elif orientation.strip().lower() == "ffp":
            event.patient_table_relationship_cid = get_or_create_cid(
                "F-10480", "feet-first"
            )
            event.patient_orientation_cid = get_or_create_cid("F-10450", "recumbent")
            event.patient_orientation_modifier_cid = get_or_create_cid(
                "F-10310", "prone"
            )
        else:
            event.patient_table_relationship_cid = None
            event.patient_orientation_cid = None
            event.patient_orientation_modifier_cid = None
        event.save()
    except (ParseError, AttributeError, TypeError):
        logger.debug(
            "Failed to extract patient orientation from comment string (aimed at Siemens)"
        )


def _irradiationeventxraydata(dataset, proj, fulldataset):  # TID 10003
    # TODO: review model to convert to cid where appropriate, and add additional fields

    event = IrradEventXRayData.objects.create(projection_xray_radiation_dose=proj)
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Acquisition Plane":
            event.acquisition_plane = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Irradiation Event UID":
            event.irradiation_event_uid = cont.UID
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Irradiation Event Label":
            event.irradiation_event_label = cont.TextValue
            try:
                for cont2 in cont.ContentSequence:
                    if cont.ConceptNameCodeSequence[0].CodeMeaning == "Label Type":
                        event.label_type = get_or_create_cid(
                            cont2.ConceptCodeSequence[0].CodeValue,
                            cont2.ConceptCodeSequence[0].CodeMeaning,
                        )
            except AttributeError:
                continue
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "111526"
        ):  # 'DateTime Started'
            event.date_time_started = make_date_time(cont.DateTime)
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Irradiation Event Type":
            event.irradiation_event_type = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Acquisition Protocol":
            try:
                event.acquisition_protocol = cont.TextValue
            except AttributeError:
                event.acquisition_protocol = None
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Anatomical structure":
            event.anatomical_structure = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
            try:
                for cont2 in cont.ContentSequence:
                    if cont2.ConceptNameCodeSequence[0].CodeMeaning == "Laterality":
                        event.laterality = get_or_create_cid(
                            cont2.ConceptCodeSequence[0].CodeValue,
                            cont2.ConceptCodeSequence[0].CodeMeaning,
                        )
            except AttributeError:
                pass
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Image View":
            event.image_view = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
            try:
                _imageviewmodifier(cont, event)
            except AttributeError:
                pass
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Patient Table Relationship"
        ):
            event.patient_table_relationship_cid = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Patient Orientation":
            event.patient_orientation_cid = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
            try:
                for cont2 in cont.ContentSequence:
                    if (
                        cont2.ConceptNameCodeSequence[0].CodeMeaning
                        == "Patient Orientation Modifier"
                    ):
                        event.patient_orientation_modifier_cid = get_or_create_cid(
                            cont2.ConceptCodeSequence[0].CodeValue,
                            cont2.ConceptCodeSequence[0].CodeMeaning,
                        )
            except AttributeError:
                pass
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Target Region":
            event.target_region = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
            try:
                for cont2 in cont.ContentSequence:
                    if cont2.ConceptNameCodeSequence[0].CodeMeaning == "Laterality":
                        event.laterality = get_or_create_cid(
                            cont2.ConceptCodeSequence[0].CodeValue,
                            cont2.ConceptCodeSequence[0].CodeMeaning,
                        )
            except AttributeError:
                pass
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Dose Area Product":
            try:
                event.dose_area_product = _check_dap_units(
                    cont.MeasuredValueSequence[0]
                )
            except LookupError:
                pass  # Will occur if measured value sequence is missing
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Half Value Layer":
            event.half_value_layer = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Entrance Exposure at RP":
            event.entrance_exposure_at_rp = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Reference Point Definition"
        ):
            try:
                event.reference_point_definition = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            except AttributeError:
                event.reference_point_definition_text = cont.TextValue
        if cont.ValueType == "CONTAINER":
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Mammography CAD Breast Composition"
            ):
                for cont2 in cont.ContentSequence:
                    if cont2.ConceptNamesCodes[0].CodeMeaning == "Breast Composition":
                        event.breast_composition = cont2.CodeValue
                    elif (
                        cont2.ConceptNamesCodes[0].CodeMeaning
                        == "Percent Fibroglandular Tissue"
                    ):
                        event.percent_fibroglandular_tissue = cont2.NumericValue
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Comment":
            event.comment = cont.TextValue
    event.save()
    for cont3 in fulldataset.ContentSequence:
        if cont3.ConceptNameCodeSequence[0].CodeMeaning == "Comment":
            _get_patient_position_from_xml_string(event, cont3.TextValue)
    # needs include for optional multiple person participant
    _irradiationeventxraydetectordata(dataset, event)
    _irradiationeventxraymechanicaldata(dataset, event)
    # in some cases we need mechanical data before x-ray source data
    _irradiationeventxraysourcedata(dataset, event)

    event.save()


def _calibration(dataset, accum):

    cal = Calibration.objects.create(accumulated_xray_dose=accum)
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Dose Measurement Device":
            cal.dose_measurement_device = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Calibration Date":
            try:
                cal.calibration_date = make_date_time(cont.DateTime)
            except (KeyError, AttributeError):
                pass  # Mandatory field not always present - see issue #770
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Calibration Factor":
            cal.calibration_factor = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Calibration Uncertainty":
            cal.calibration_uncertainty = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Calibration Responsible Party"
        ):
            cal.calibration_responsible_party = cont.TextValue
    cal.save()


def _accumulatedmammoxraydose(dataset, accum):  # TID 10005

    for cont in dataset.ContentSequence:
        if (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Accumulated Average Glandular Dose"
        ):
            accummammo = AccumMammographyXRayDose.objects.create(
                accumulated_xray_dose=accum
            )
            accummammo.accumulated_average_glandular_dose = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
            for cont2 in cont.ContentSequence:
                if cont2.ConceptNameCodeSequence[0].CodeMeaning == "Laterality":
                    accummammo.laterality = get_or_create_cid(
                        cont2.ConceptCodeSequence[0].CodeValue,
                        cont2.ConceptCodeSequence[0].CodeMeaning,
                    )
            accummammo.save()


def _accumulatedfluoroxraydose(dataset, accum):  # TID 10004
    # Name in DICOM standard for TID 10004 is Accumulated Fluoroscopy and Acquisition Projection X-Ray Dose
    # See http://dicom.nema.org/medical/Dicom/2017e/output/chtml/part16/sect_TID_10004.html

    accumproj = AccumProjXRayDose.objects.create(accumulated_xray_dose=accum)
    for cont in dataset.ContentSequence:
        try:
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Fluoro Dose Area Product Total"
            ):
                accumproj.fluoro_dose_area_product_total = _check_dap_units(
                    cont.MeasuredValueSequence[0]
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeValue == "113728"
            ):  # = 'Fluoro Dose (RP) Total'
                accumproj.fluoro_dose_rp_total = _check_rp_dose_units(
                    cont.MeasuredValueSequence[0]
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Total Fluoro Time":
                accumproj.total_fluoro_time = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Acquisition Dose Area Product Total"
            ):
                accumproj.acquisition_dose_area_product_total = _check_dap_units(
                    cont.MeasuredValueSequence[0]
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeValue == "113729"
            ):  # = 'Acquisition Dose (RP) Total'
                accumproj.acquisition_dose_rp_total = _check_rp_dose_units(
                    cont.MeasuredValueSequence[0]
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning == "Total Acquisition Time"
            ):
                accumproj.total_acquisition_time = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            # TODO: Remove the following four items, as they are also imported (correctly) into
            # _accumulatedtotalprojectionradiographydose
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning == "Dose Area Product Total"
            ):
                accumproj.dose_area_product_total = _check_dap_units(
                    cont.MeasuredValueSequence[0]
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeValue == "113725"
            ):  # = 'Dose (RP) Total':
                accumproj.dose_rp_total = _check_rp_dose_units(
                    cont.MeasuredValueSequence[0]
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Total Number of Radiographic Frames"
            ):
                accumproj.total_number_of_radiographic_frames = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Reference Point Definition"
            ):
                try:
                    accumproj.reference_point_definition_code = get_or_create_cid(
                        cont.ConceptCodeSequence[0].CodeValue,
                        cont.ConceptCodeSequence[0].CodeMeaning,
                    )
                except AttributeError:
                    accumproj.reference_point_definition = cont.TextValue
        except IndexError:
            pass
    if (
        accumproj.accumulated_xray_dose.projection_xray_radiation_dose.general_study_module_attributes.modality_type
        == "RF,DX"
    ):
        if accumproj.fluoro_dose_area_product_total or accumproj.total_fluoro_time:
            accumproj.accumulated_xray_dose.projection_xray_radiation_dose.general_study_module_attributes.modality_type = (
                "RF"
            )
        else:
            accumproj.accumulated_xray_dose.projection_xray_radiation_dose.general_study_module_attributes.modality_type = (
                "DX"
            )
    accumproj.save()


def _accumulatedcassettebasedprojectionradiographydose(dataset, accum):  # TID 10006

    accumcass = AccumCassetteBsdProjRadiogDose.objects.create(
        accumulated_xray_dose=accum
    )
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Detector Type":
            accumcass.detector_type = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Total Number of Radiographic Frames"
        ):
            accumcass.total_number_of_radiographic_frames = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
    accumcass.save()


def _accumulatedtotalprojectionradiographydose(dataset, accum):  # TID 10007
    # Name in DICOM standard for TID 10007 is Accumulated Total Projection Radiography Dose
    # See http://dicom.nema.org/medical/Dicom/2017e/output/chtml/part16/sect_TID_10007.html

    accumint = AccumIntegratedProjRadiogDose.objects.create(accumulated_xray_dose=accum)
    for cont in dataset.ContentSequence:
        try:
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "Dose Area Product Total":
                accumint.dose_area_product_total = _check_dap_units(
                    cont.MeasuredValueSequence[0]
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeValue == "113725"
            ):  # = 'Dose (RP) Total':
                accumint.dose_rp_total = _check_rp_dose_units(
                    cont.MeasuredValueSequence[0]
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Total Number of Radiographic Frames"
            ):
                accumint.total_number_of_radiographic_frames = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Reference Point Definition"
            ):
                try:
                    accumint.reference_point_definition_code = get_or_create_cid(
                        cont.ConceptCodeSequence[0].CodeValue,
                        cont.ConceptCodeSequence[0].CodeMeaning,
                    )
                except AttributeError:
                    accumint.reference_point_definition = cont.TextValue
        except IndexError:
            pass
    accumint.save()


def _accumulatedxraydose(dataset, proj):  # TID 10002

    accum = AccumXRayDose.objects.create(projection_xray_radiation_dose=proj)
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Acquisition Plane":
            accum.acquisition_plane = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        if cont.ValueType == "CONTAINER":
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "Calibration":
                _calibration(cont, accum)
    if proj.acquisition_device_type_cid:
        if (
            "Fluoroscopy-Guided" in proj.acquisition_device_type_cid.code_meaning
            or "Azurion"
            in proj.general_study_module_attributes.generalequipmentmoduleattr_set.get().manufacturer_model_name
        ):
            _accumulatedfluoroxraydose(dataset, accum)
    elif proj.procedure_reported and (
        "Projection X-Ray" in proj.procedure_reported.code_meaning
    ):
        _accumulatedfluoroxraydose(dataset, accum)
    if proj.procedure_reported and (
        proj.procedure_reported.code_meaning == "Mammography"
    ):
        _accumulatedmammoxraydose(dataset, accum)
    if proj.acquisition_device_type_cid:
        if (
            "Integrated" in proj.acquisition_device_type_cid.code_meaning
            or "Fluoroscopy-Guided" in proj.acquisition_device_type_cid.code_meaning
        ):
            _accumulatedtotalprojectionradiographydose(dataset, accum)
    elif proj.procedure_reported and (
        "Projection X-Ray" in proj.procedure_reported.code_meaning
    ):
        _accumulatedtotalprojectionradiographydose(dataset, accum)
    if proj.acquisition_device_type_cid:
        if "Cassette-based" in proj.acquisition_device_type_cid.code_meaning:
            _accumulatedcassettebasedprojectionradiographydose(dataset, accum)
    accum.save()


def _scanninglength(dataset, event):  # TID 10014

    scanlen = ScanningLength.objects.create(ct_irradiation_event_data=event)
    try:
        for cont in dataset.ContentSequence:
            if cont.ConceptNameCodeSequence[0].CodeMeaning.lower() == "scanning length":
                scanlen.scanning_length = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "length of reconstructable volume"
            ):
                scanlen.length_of_reconstructable_volume = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning.lower() == "exposed range":
                scanlen.exposed_range = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "top z location of reconstructable volume"
            ):
                scanlen.top_z_location_of_reconstructable_volume = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "bottom z location of reconstructable volume"
            ):
                scanlen.bottom_z_location_of_reconstructable_volume = (
                    test_numeric_value(cont.MeasuredValueSequence[0].NumericValue)
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "top z location of scanning length"
            ):
                scanlen.top_z_location_of_scanning_length = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "bottom z location of scanning length"
            ):
                scanlen.bottom_z_location_of_scanning_length = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "irradiation event uid"
            ):
                scanlen.irradiation_event_uid = cont.UID
        scanlen.save()
    except AttributeError:
        pass


def _ctxraysourceparameters(dataset, event):

    param = CtXRaySourceParameters.objects.create(ct_irradiation_event_data=event)
    for cont in dataset.ContentSequence:
        if (
            cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
            == "identification of the x-ray source"
            or cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
            == "identification number of the x-ray source"
        ):
            param.identification_of_the_xray_source = cont.TextValue
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "KVP":
            param.kvp = test_numeric_value(cont.MeasuredValueSequence[0].NumericValue)
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
            == "maximum x-ray tube current"
        ):
            param.maximum_xray_tube_current = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning.lower() == "x-ray tube current"
        ):
            param.xray_tube_current = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "113734":
            # Additional check as code meaning is wrong for Siemens Intevo see
            # https://bitbucket.org/openrem/openrem/issues/380/siemens-intevo-rdsr-have-wrong-code
            param.xray_tube_current = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning == "Exposure Time per Rotation"
        ):
            param.exposure_time_per_rotation = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
            == "x-ray filter aluminum equivalent"
        ):
            param.xray_filter_aluminum_equivalent = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
    param.save()


def _ctdosecheckdetails(dataset, dosecheckdetails, isalertdetails):  # TID 10015
    # PARTLY TESTED CODE (no DSR available that has Reason For Proceeding and/or Forward Estimate)

    if isalertdetails:
        for cont in dataset.ContentSequence:
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "DLP Alert Value Configured"
            ):
                dosecheckdetails.dlp_alert_value_configured = (
                    cont.ConceptCodeSequence[0].CodeMeaning == "Yes"
                )
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "DLP Alert Value":
                dosecheckdetails.dlp_alert_value = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "CTDIvol Alert Value Configured"
            ):
                dosecheckdetails.ctdivol_alert_value_configured = (
                    cont.ConceptCodeSequence[0].CodeMeaning == "Yes"
                )
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "CTDIvol Alert Value":
                dosecheckdetails.ctdivol_alert_value = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Accumulated DLP Forward Estimate"
            ):
                dosecheckdetails.accumulated_dlp_forward_estimate = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Accumulated CTDIvol Forward Estimate"
            ):
                dosecheckdetails.accumulated_ctdivol_forward_estimate = (
                    test_numeric_value(cont.MeasuredValueSequence[0].NumericValue)
                )
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "Reason For Proceeding":
                dosecheckdetails.alert_reason_for_proceeding = cont.TextValue
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "Person Name":
                person_participant(
                    cont, "ct_dose_check_alert", dosecheckdetails, logger
                )
    else:
        for cont in dataset.ContentSequence:
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "DLP Notification Value Configured"
            ):
                dosecheckdetails.dlp_notification_value_configured = (
                    cont.ConceptCodeSequence[0].CodeMeaning == "Yes"
                )
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "DLP Notification Value":
                dosecheckdetails.dlp_notification_value = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "CTDIvol Notification Value Configured"
            ):
                dosecheckdetails.ctdivol_notification_value_configured = (
                    cont.ConceptCodeSequence[0].CodeMeaning == "Yes"
                )
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "CTDIvol Notification Value"
            ):
                dosecheckdetails.ctdivol_notification_value = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "DLP Forward Estimate":
                dosecheckdetails.dlp_forward_estimate = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "CTDIvol Forward Estimate"
            ):
                dosecheckdetails.ctdivol_forward_estimate = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "Reason For Proceeding":
                dosecheckdetails.notification_reason_for_proceeding = cont.TextValue
            if cont.ConceptNameCodeSequence[0].CodeMeaning == "Person Name":
                person_participant(
                    cont, "ct_dose_check_notification", dosecheckdetails, logger
                )
    dosecheckdetails.save()


def _ctirradiationeventdata(dataset, ct):  # TID 10013

    event = CtIrradiationEventData.objects.create(ct_radiation_dose=ct)
    ctdosecheckdetails = None
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Acquisition Protocol":
            event.acquisition_protocol = cont.TextValue
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Target Region":
            try:
                event.target_region = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            except (AttributeError, IndexError):
                logger.info(
                    "Target Region ConceptNameCodeSequence exists, but no content. Study UID {0} from {1}, "
                    "{2}, {3}".format(
                        event.ct_radiation_dose.general_study_module_attributes.study_instance_uid,
                        event.ct_radiation_dose.general_study_module_attributes.generalequipmentmoduleattr_set.get().manufacturer,
                        event.ct_radiation_dose.general_study_module_attributes.generalequipmentmoduleattr_set.get().manufacturer_model_name,
                        event.ct_radiation_dose.general_study_module_attributes.generalequipmentmoduleattr_set.get().station_name,
                    )
                )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "CT Acquisition Type":
            event.ct_acquisition_type = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Procedure Context":
            event.procedure_context = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Irradiation Event UID":
            event.irradiation_event_uid = cont.UID
            event.save()
        if cont.ValueType == "CONTAINER":
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "CT Acquisition Parameters"
            ):
                _scanninglength(cont, event)
                try:
                    for cont2 in cont.ContentSequence:
                        if (
                            cont2.ConceptNameCodeSequence[0].CodeMeaning
                            == "Exposure Time"
                        ):
                            event.exposure_time = test_numeric_value(
                                cont2.MeasuredValueSequence[0].NumericValue
                            )
                        elif (
                            cont2.ConceptNameCodeSequence[0].CodeMeaning
                            == "Nominal Single Collimation Width"
                        ):
                            event.nominal_single_collimation_width = test_numeric_value(
                                cont2.MeasuredValueSequence[0].NumericValue
                            )
                        elif (
                            cont2.ConceptNameCodeSequence[0].CodeMeaning
                            == "Nominal Total Collimation Width"
                        ):
                            event.nominal_total_collimation_width = test_numeric_value(
                                cont2.MeasuredValueSequence[0].NumericValue
                            )
                        elif (
                            cont2.ConceptNameCodeSequence[0].CodeMeaning
                            == "Pitch Factor"
                        ):
                            event.pitch_factor = test_numeric_value(
                                cont2.MeasuredValueSequence[0].NumericValue
                            )
                        elif (
                            cont2.ConceptNameCodeSequence[0].CodeMeaning.lower()
                            == "number of x-ray sources"
                        ):
                            event.number_of_xray_sources = test_numeric_value(
                                cont2.MeasuredValueSequence[0].NumericValue
                            )
                        if cont2.ValueType == "CONTAINER":
                            if (
                                cont2.ConceptNameCodeSequence[0].CodeMeaning.lower()
                                == "ct x-ray source parameters"
                            ):
                                _ctxraysourceparameters(cont2, event)
                except AttributeError:
                    pass
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "CT Dose":
                for cont2 in cont.ContentSequence:
                    if cont2.ConceptNameCodeSequence[0].CodeMeaning == "Mean CTDIvol":
                        event.mean_ctdivol = test_numeric_value(
                            cont2.MeasuredValueSequence[0].NumericValue
                        )
                    elif (
                        cont2.ConceptNameCodeSequence[0].CodeMeaning
                        == "CTDIw Phantom Type"
                    ):
                        event.ctdiw_phantom_type = get_or_create_cid(
                            cont2.ConceptCodeSequence[0].CodeValue,
                            cont2.ConceptCodeSequence[0].CodeMeaning,
                        )
                    elif (
                        cont2.ConceptNameCodeSequence[0].CodeMeaning
                        == "CTDIfreeair Calculation Factor"
                    ):
                        event.ctdifreeair_calculation_factor = test_numeric_value(
                            cont2.MeasuredValueSequence[0].NumericValue
                        )
                    elif (
                        cont2.ConceptNameCodeSequence[0].CodeMeaning
                        == "Mean CTDIfreeair"
                    ):
                        event.mean_ctdifreeair = test_numeric_value(
                            cont2.MeasuredValueSequence[0].NumericValue
                        )
                    elif cont2.ConceptNameCodeSequence[0].CodeMeaning == "DLP":
                        event.dlp = test_numeric_value(
                            cont2.MeasuredValueSequence[0].NumericValue
                        )
                    elif (
                        cont2.ConceptNameCodeSequence[0].CodeMeaning == "Effective Dose"
                    ):
                        event.effective_dose = test_numeric_value(
                            cont2.MeasuredValueSequence[0].NumericValue
                        )
                        ## Effective dose measurement method and conversion factor
                    ## CT Dose Check Details
                    ## Dose Check Alert Details and Notifications Details can appear indepently
                    elif (
                        cont2.ConceptNameCodeSequence[0].CodeMeaning
                        == "Dose Check Alert Details"
                    ):
                        if ctdosecheckdetails is None:
                            ctdosecheckdetails = CtDoseCheckDetails.objects.create(
                                ct_irradiation_event_data=event
                            )
                        _ctdosecheckdetails(cont2, ctdosecheckdetails, True)
                    elif (
                        cont2.ConceptNameCodeSequence[0].CodeMeaning
                        == "Dose Check Notification Details"
                    ):
                        if ctdosecheckdetails is None:
                            ctdosecheckdetails = CtDoseCheckDetails.objects.create(
                                ct_irradiation_event_data=event
                            )
                        _ctdosecheckdetails(cont2, ctdosecheckdetails, False)
        if (
            cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
            == "x-ray modulation type"
        ):
            event.xray_modulation_type = cont.TextValue
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Comment":
            event.comment = cont.TextValue
    if not event.xray_modulation_type and event.comment:
        comments = event.comment.split(",")
        for comm in comments:
            if comm.lstrip().startswith("X-ray Modulation Type"):
                modulationtype = comm[(comm.find("=") + 2) :]
                event.xray_modulation_type = modulationtype

    ## personparticipant here
    _deviceparticipant(dataset, "ct_event", event)
    if ctdosecheckdetails is not None:
        ctdosecheckdetails.save()
    event.save()


def _ctaccumulateddosedata(dataset, ct):  # TID 10012

    ctacc = CtAccumulatedDoseData.objects.create(ct_radiation_dose=ct)
    for cont in dataset.ContentSequence:
        if (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Total Number of Irradiation Events"
        ):
            ctacc.total_number_of_irradiation_events = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "CT Dose Length Product Total"
        ):
            ctacc.ct_dose_length_product_total = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "CT Effective Dose Total":
            ctacc.ct_effective_dose_total = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        #
        # Reference authority code or name belongs here, followed by the effective dose details
        #
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Comment":
            ctacc.comment = cont.TextValue
    _deviceparticipant(dataset, "ct_accumulated", ctacc)

    # Berechne den maximalen CTDI-Wert aus allen Events
    max_ctdi = ct.ctirradiationeventdata_set.all().aggregate(
        Max('mean_ctdivol'))['mean_ctdivol__max']
    ctacc.maximum_ctdivol = max_ctdi
    
    ctacc.save()


def _import_varic(dataset, proj):

    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeValue == "C-200":
            accum = AccumXRayDose.objects.create(projection_xray_radiation_dose=proj)
            accum.acquisition_plane = get_or_create_cid("113622", "Single Plane")
            accum.save()
            accumint = AccumIntegratedProjRadiogDose.objects.create(
                accumulated_xray_dose=accum
            )
            try:
                accumint.total_number_of_radiographic_frames = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
            accumint.save()
        if cont.ConceptNameCodeSequence[0].CodeValue == "C-202":
            accumproj = AccumProjXRayDose.objects.create(accumulated_xray_dose=accum)
            accumproj.save()
            try:
                accumproj.total_fluoro_time = test_numeric_value(
                    cont.MeasuredValueSequence[0].NumericValue
                )
            except IndexError:
                pass
            accumproj.save()
        if cont.ConceptNameCodeSequence[0].CodeValue == "C-204":
            try:
                accumint.dose_area_product_total = (
                    test_numeric_value(cont.MeasuredValueSequence[0].NumericValue)
                    / 10000.0
                )
            except (TypeError, IndexError):
                pass
            accumint.save()
        # cumulative air kerma in dose report example is not available ("Measurement not attempted")


def projectionxrayradiationdose(dataset, g, reporttype):

    if reporttype == "projection":
        proj = ProjectionXRayRadiationDose.objects.create(
            general_study_module_attributes=g
        )
    elif reporttype == "ct":
        proj = CtRadiationDose.objects.create(general_study_module_attributes=g)
    else:
        logger.error(
            "Attempt to create ProjectionXRayRadiationDose failed as report type incorrect"
        )
        return
    equip = GeneralEquipmentModuleAttr.objects.get(general_study_module_attributes=g)
    proj.general_study_module_attributes.modality_type = (
        equip.unique_equipment_name.user_defined_modality
    )
    if proj.general_study_module_attributes.modality_type == "dual":
        proj.general_study_module_attributes.modality_type = None

    if (
        dataset.ConceptNameCodeSequence[0].CodingSchemeDesignator == "99SMS_RADSUM"
        and dataset.ConceptNameCodeSequence[0].CodeValue == "C-10"
    ):
        g.modality_type = "RF"
        g.save()
        _import_varic(dataset, proj)
    else:
        for cont in dataset.ContentSequence:
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "procedure reported"
            ):
                proj.procedure_reported = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
                if (
                    "ContentSequence" in cont
                ):  # Extra if statement to allow for non-conformant GE RDSR that don't have this mandatory field.
                    for cont2 in cont.ContentSequence:
                        if cont2.ConceptNameCodeSequence[0].CodeMeaning == "Has Intent":
                            proj.has_intent = get_or_create_cid(
                                cont2.ConceptCodeSequence[0].CodeValue,
                                cont2.ConceptCodeSequence[0].CodeMeaning,
                            )
                if "Mammography" in proj.procedure_reported.code_meaning:
                    proj.general_study_module_attributes.modality_type = "MG"
                elif (not proj.general_study_module_attributes.modality_type) and (
                    "Projection X-Ray" in proj.procedure_reported.code_meaning
                ):
                    proj.general_study_module_attributes.modality_type = "RF,DX"
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "acquisition device type"
            ):
                proj.acquisition_device_type_cid = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "start of x-ray irradiation"
            ):
                proj.start_of_xray_irradiation = make_date_time(cont.DateTime)
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning.lower()
                == "end of x-ray irradiation"
            ):
                proj.end_of_xray_irradiation = make_date_time(cont.DateTime)
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Scope of Accumulation":
                proj.scope_of_accumulation = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "X-Ray Detector Data Available"
            ):
                proj.xray_detector_data_available = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "X-Ray Source Data Available"
            ):
                proj.xray_source_data_available = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "X-Ray Mechanical Data Available"
            ):
                proj.xray_mechanical_data_available = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Comment":
                proj.comment = cont.TextValue
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Source of Dose Information"
            ):
                proj.source_of_dose_information = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            if (
                (not equip.unique_equipment_name.user_defined_modality)
                and (reporttype == "projection")
                and proj.acquisition_device_type_cid
            ):
                if (
                    "Fluoroscopy-Guided"
                    in proj.acquisition_device_type_cid.code_meaning
                    or "Azurion"
                    in proj.general_study_module_attributes.generalequipmentmoduleattr_set.get().manufacturer_model_name
                ):
                    proj.general_study_module_attributes.modality_type = "RF"
                elif any(
                    x in proj.acquisition_device_type_cid.code_meaning
                    for x in ["Integrated", "Cassette-based"]
                ):
                    proj.general_study_module_attributes.modality_type = "DX"
                else:
                    logging.error(
                        "Acquisition device type code exists, but the value wasn't matched. Study UID: {0}, "
                        "Station name: {1}, Study date, time: {2}, {3}, device type: {4} ".format(
                            proj.general_study_module_attributes.study_instance_uid,
                            proj.general_study_module_attributes.generalequipmentmoduleattr_set.get().station_name,
                            proj.general_study_module_attributes.study_date,
                            proj.general_study_module_attributes.study_time,
                            proj.acquisition_device_type_cid.code_meaning,
                        )
                    )

            proj.save()

            if cont.ConceptNameCodeSequence[0].CodeMeaning == "Observer Type":
                if reporttype == "projection":
                    obs = ObserverContext.objects.create(
                        projection_xray_radiation_dose=proj
                    )
                else:
                    obs = ObserverContext.objects.create(ct_radiation_dose=proj)
                observercontext(dataset, obs)

            if cont.ValueType == "CONTAINER":
                if (
                    cont.ConceptNameCodeSequence[0].CodeMeaning
                    == "Accumulated X-Ray Dose Data"
                ):
                    _accumulatedxraydose(cont, proj)
                if (
                    cont.ConceptNameCodeSequence[0].CodeMeaning
                    == "Irradiation Event X-Ray Data"
                ):
                    _irradiationeventxraydata(cont, proj, dataset)
                if (
                    cont.ConceptNameCodeSequence[0].CodeMeaning
                    == "CT Accumulated Dose Data"
                ):
                    if proj.general_study_module_attributes.modality_type != "NM":
                        proj.general_study_module_attributes.modality_type = "CT"
                    _ctaccumulateddosedata(cont, proj)
                if cont.ConceptNameCodeSequence[0].CodeMeaning == "CT Acquisition":
                    _ctirradiationeventdata(cont, proj)

    # Nach dem Erstellen aller Events den maximalen CTDI berechnen
    if reporttype == "ct":
        try:
            ctacc = proj.ctaccumulateddosedata_set.get()
            max_ctdi = proj.ctirradiationeventdata_set.all().aggregate(
                Max('mean_ctdivol'))['mean_ctdivol__max']
            ctacc.maximum_ctdivol = max_ctdi
            ctacc.save()
        except ObjectDoesNotExist:
            pass
