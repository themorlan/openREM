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
..  module:: extract-common
    :synopsis: Module of functions common to multiple extractor routines
"""

from datetime import datetime
from decimal import Decimal
import logging

from django.db.models import (
    ObjectDoesNotExist,
    Q,
)
from django.conf import settings
import numpy as np

from remapp.models import (
    GeneralStudyModuleAttr,
    PatientIDSettings,
    PatientModuleAttr,
    PatientStudyModuleAttr,
    StandardNames,
    PersonParticipant,
    GeneralEquipmentModuleAttr,
    UniqueEquipmentNames,
    MergeOnDeviceObserverUIDSettings,
)
from ..tools.dcmdatetime import (
    get_date,
    make_date_time,
    get_time,
    make_date,
    make_time,
)
from ..tools.get_values import (
    get_value_kw,
    get_or_create_cid,
    list_to_string,
    get_seq_code_meaning,
    get_seq_code_value,
)
from ..tools.not_patient_indicators import get_not_pt
from ..tools.hash_id import hash_id
from ..tools.check_uid import record_sop_instance_uid


def get_study_check_dup(dataset, modality="DX"):
    """
    If study exists, create new event
    :param dataset: DICOM object
    :param modality: Modality of image object being imported; 'MG' or 'DX'
    :return: Study object to proceed with
    """

    if modality == "MG":
        logger = logging.getLogger("remapp.extractors.mam")
    else:
        logger = logging.getLogger("remapp.extractors.dx")

    study_uid = get_value_kw("StudyInstanceUID", dataset)
    event_uid = get_value_kw("SOPInstanceUID", dataset)
    logger.debug(
        f"In get_study_check_dup for {modality}. Study UID {study_uid}, event UID {event_uid}"
    )
    this_study = None
    same_study_uid = GeneralStudyModuleAttr.objects.filter(
        study_instance_uid__exact=study_uid
    ).order_by("pk")
    if same_study_uid.count() > 1:
        logger.warning(
            "Duplicate DX study UID {0} in database - could be a problem! There are {1} copies.".format(
                study_uid, same_study_uid.count()
            )
        )
        # Studies are ordered by study level pk. FInd the first one that has a modality type, and replace our
        # filter_set with the study we have found.
        for study in same_study_uid.order_by("pk"):
            if study.modality_type:
                this_study = study
                logger.debug(
                    "Duplicate {3} study UID {0} - first instance (pk={1}) with modality type assigned ({2}) "
                    "selected to import new event into.".format(
                        study_uid, study.pk, study.modality_type, modality
                    )
                )
                break
        if not this_study:
            logger.warning(
                "Duplicate {1} study UID {0}, none of which have modality_type assigned!"
                " Setting first instance to DX".format(study_uid, modality)
            )
            this_study = same_study_uid[0]
            this_study.modality_type = modality
            this_study.save()
    elif same_study_uid.count() == 1:
        logger.debug("Importing event {0} into study {1}".format(event_uid, study_uid))
        this_study = same_study_uid[0]
    else:
        logger.error(
            "Attempting to add {0} event {1} to study UID {2}, but it isn't there anymore. Stopping.".format(
                modality, event_uid, study_uid
            )
        )
        return 0
    existing_sop_instance_uids = set()
    for previous_object in this_study.objectuidsprocessed_set.all():
        existing_sop_instance_uids.add(previous_object.sop_instance_uid)
    if event_uid in existing_sop_instance_uids:
        # We might get here if object has previously been rejected for being a for processing/presentation duplicate
        logger.debug(
            "{2} instance UID {0} of study UID {1} previously processed, stopping.".format(
                event_uid, study_uid, modality
            )
        )
        return 0
    # New event - record the SOP instance UID
    logger.debug(
        "{0} Event {1} we haven't seen before. Adding to list for study {2}".format(
            modality, event_uid, study_uid
        )
    )
    record_sop_instance_uid(this_study, event_uid)
    # further check required to ensure 'for processing' and 'for presentation'
    # versions of the same irradiation event don't get imported twice
    # Also check it isn't a Hologic SC tomo file
    if modality == "DX" or (
        modality == "MG" and dataset.SOPClassUID != "1.2.840.10008.5.1.4.1.1.7"
    ):
        event_time = get_value_kw("AcquisitionTime", dataset)
        if not event_time:
            event_time = get_value_kw("ContentTime", dataset)
        event_date = get_value_kw("AcquisitionDate", dataset)
        if not event_date:
            event_date = get_value_kw("ContentDate", dataset)
        event_date_time = make_date_time("{0}{1}".format(event_date, event_time))
        for (
            events
        ) in (
            this_study.projectionxrayradiationdose_set.get().irradeventxraydata_set.all()
        ):
            if event_date_time == events.date_time_started:
                logger.debug(
                    "A previous {2} object with this study UID ({0}) and time ({1}) has been imported."
                    " Stopping".format(study_uid, event_date_time.isoformat(), modality)
                )
                return 0
    # study exists, but event doesn't
    return this_study


def ct_event_type_count(g):
    """Count CT event types and record in GeneralStudyModuleAttr summary fields

    :param g: GeneralStudyModuleAttr database table
    :return: None - database is updated
    """

    g.number_of_axial = 0
    g.number_of_spiral = 0
    g.number_of_stationary = 0
    g.number_of_const_angle = 0
    try:
        events = g.ctradiationdose_set.get().ctirradiationeventdata_set.order_by("pk")
        g.number_of_axial += events.filter(
            ct_acquisition_type__code_value__exact="113804"
        ).count()
        g.number_of_spiral += events.filter(
            ct_acquisition_type__code_value__exact="116152004"
        ).count()
        g.number_of_spiral += events.filter(
            ct_acquisition_type__code_value__exact="P5-08001"
        ).count()
        g.number_of_spiral += events.filter(
            ct_acquisition_type__code_value__exact="C0860888"
        ).count()
        g.number_of_stationary += events.filter(
            ct_acquisition_type__code_value__exact="113806"
        ).count()
        g.number_of_const_angle += events.filter(
            ct_acquisition_type__code_value__exact="113805"
        ).count()
    except ObjectDoesNotExist:
        return
    g.save()


def populate_mammo_agd_summary(g):
    """Copy accumulated AGD to the GeneralStudyModuleAttr summary fields

    :param g: GeneralStudyModuleAttr database table
    :return: None - database is updated
    """

    logger = logging.getLogger("remapp.extractors")

    try:
        for breast in (
            g.projectionxrayradiationdose_set.get()
            .accumxraydose_set.get()
            .accummammographyxraydose_set.order_by("pk")
        ):
            if breast.laterality.code_value in [
                "T-04020",
                "73056007",
                "C0222600",
            ]:  # Right breast
                g.total_agd_right = breast.accumulated_average_glandular_dose
            elif breast.laterality.code_value in [
                "T-04030",
                "80248007",
                "C0222601",
            ]:  # Left breast
                g.total_agd_left = breast.accumulated_average_glandular_dose
            elif breast.laterality.code_value in [
                "T-04080",
                "63762007",
                "C0222605",
            ]:  # Left breast
                g.total_agd_both = breast.accumulated_average_glandular_dose
        g.save()
    except (ObjectDoesNotExist, AttributeError):
        g.total_agd_both = 0
        g.save()
        logger.warning(
            "Study UID {0}. Unable to set summary total_agd values - total_agd_both set to 0.".format(
                g.study_instance_uid
            )
        )


def populate_dx_rf_summary(g):
    """Copy accumulated DAP and RP dose for each plane into the GeneralStudyModuleAttr summary fields

    :param g: GeneralStudyModuleAttr database table
    :return: None - database is updated
    """

    logger = logging.getLogger("remapp.extractors")

    try:
        planes = g.projectionxrayradiationdose_set.get().accumxraydose_set.order_by(
            "pk"
        )
        accum_int_a = planes[0].accumintegratedprojradiogdose_set.get()
        g.total_dap_a = accum_int_a.dose_area_product_total
        g.total_rp_dose_a = accum_int_a.dose_rp_total
        g.total_dap_a_delta_weeks = accum_int_a.dose_area_product_total_over_delta_weeks
        g.total_rp_dose_a_delta_weeks = accum_int_a.dose_rp_total_over_delta_weeks
        if (
            g.projectionxrayradiationdose_set.get()
            .irradeventxraydata_set.all()
            .filter(acquisition_plane__code_value__exact="113620")
            .count()
        ):
            g.number_of_events_a = (
                g.projectionxrayradiationdose_set.get()
                .irradeventxraydata_set.all()
                .filter(acquisition_plane__code_value__exact="113620")
                .count()
            )
        elif (
            g.projectionxrayradiationdose_set.get()
            .irradeventxraydata_set.all()
            .filter(acquisition_plane__code_value__exact="113622")
            .count()
        ):
            g.number_of_events_a = (
                g.projectionxrayradiationdose_set.get()
                .irradeventxraydata_set.all()
                .filter(acquisition_plane__code_value__exact="113622")
                .count()
            )
        else:
            g.number_of_events_a = (
                g.projectionxrayradiationdose_set.get()
                .irradeventxraydata_set.all()
                .filter(acquisition_plane__code_value__exact="113890")
                .count()
            )
        try:
            accum_int_b = planes[1].accumintegratedprojradiogdose_set.get()
            g.total_dap_b = accum_int_b.dose_area_product_total
            g.total_rp_dose_b = accum_int_b.dose_rp_total
            g.total_dap_b_delta_weeks = (
                accum_int_b.dose_area_product_total_over_delta_weeks
            )
            g.total_rp_dose_b_delta_weeks = accum_int_b.dose_rp_total_over_delta_weeks
            g.number_of_planes = 2
            g.number_of_events_b = (
                g.projectionxrayradiationdose_set.get()
                .irradeventxraydata_set.all()
                .filter(acquisition_plane__code_value__exact="113621")
                .count()
            )
        except IndexError:
            g.number_of_planes = 1
            logger.debug(
                "Study UID {0}. No second plane when setting summary DAP/RP values".format(
                    g.study_instance_uid
                )
            )
        g.save()
        try:
            g.total_dap = g.total_dap_a + g.total_dap_b
        except TypeError:
            if g.total_dap_a is not None:
                g.total_dap = g.total_dap_a
            elif g.total_dap_b is not None:
                g.total_dap = g.total_dap_b
            else:
                g.total_dap = 0
        g.save()
    except (ObjectDoesNotExist, IndexError):
        g.number_of_events_a = 0
        g.save()
        logger.warning(
            "Study UID {0}. Unable to set summary total DAP and RP dose values. number_of_events_a set to 0.".format(
                g.study_instance_uid
            )
        )


def populate_rf_delta_weeks_summary(g):
    """Copy DAP and RP dose accumulated over delta weeks into the GeneralStudyModuleAttr summary fields

    :param g: GeneralStudyModuleAttr database table
    :return: None - database is updated
    """

    logger = logging.getLogger("remapp.extractors")

    try:
        # Both planes are added to the total, which is recorded in each plane. So only need to get the first one
        g.total_dap_delta_weeks = (
            g.projectionxrayradiationdose_set.get()
            .accumxraydose_set.order_by("pk")[0]
            .accumintegratedprojradiogdose_set.get()
            .dose_area_product_total_over_delta_weeks
        )
        g.total_rp_dose_delta_weeks = (
            g.projectionxrayradiationdose_set.get()
            .accumxraydose_set.order_by("pk")[0]
            .accumintegratedprojradiogdose_set.get()
            .dose_rp_total_over_delta_weeks
        )
        g.save()
    except (ObjectDoesNotExist, IndexError):
        logger.warning(
            "Study UID {0}. Unable to set summary delta weeks DAP and RP dose values".format(
                g.study_instance_uid
            )
        )


def patient_module_attributes(dataset, g):  # C.7.1.1
    """Get patient module attributes

    :param dataset: DICOM object
    :param g: GeneralStudyModuleAttr database table
    :return: None - database is updated
    """

    pat = PatientModuleAttr.objects.create(general_study_module_attributes=g)
    patient_birth_date = get_date("PatientBirthDate", dataset)
    pat.patient_sex = get_value_kw("PatientSex", dataset)
    pat.not_patient_indicator = get_not_pt(dataset)
    patientatt = PatientStudyModuleAttr.objects.get(general_study_module_attributes=g)
    if patient_birth_date and g.study_date:
        patientatt.patient_age_decimal = Decimal(
            (g.study_date.date() - patient_birth_date.date()).days
        ) / Decimal("365.25")
    elif patientatt.patient_age:
        if patientatt.patient_age[-1:] == "Y":
            patientatt.patient_age_decimal = Decimal(patientatt.patient_age[:-1])
        elif patientatt.patient_age[-1:] == "M":
            patientatt.patient_age_decimal = Decimal(
                patientatt.patient_age[:-1]
            ) / Decimal("12")
        elif patientatt.patient_age[-1:] == "D":
            patientatt.patient_age_decimal = Decimal(
                patientatt.patient_age[:-1]
            ) / Decimal("365.25")
    if patientatt.patient_age_decimal:
        patientatt.patient_age_decimal = patientatt.patient_age_decimal.quantize(
            Decimal(".1")
        )
    patientatt.save()

    patient_id_settings = PatientIDSettings.objects.get()
    if patient_id_settings.name_stored:
        name = get_value_kw("PatientName", dataset)
        if name and patient_id_settings.name_hashed:
            name = hash_id(name)
            pat.name_hashed = True
        pat.patient_name = name
    if patient_id_settings.id_stored:
        patid = get_value_kw("PatientID", dataset)
        if patid and patient_id_settings.id_hashed:
            patid = hash_id(patid)
            pat.id_hashed = True
        pat.patient_id = patid
    if patient_id_settings.dob_stored and patient_birth_date:
        pat.patient_birth_date = patient_birth_date
    pat.save()


def add_standard_names(g):
    """Add references to any matching standard name entries

    :param g: GeneralStudyModuleAttr database table
    :return: None - database is updated
    """

    # If the modality_type is CR or PX then override it to DX, because all CR, DX and PX studies are stored in the
    # standard_names table as DX
    modality = g.modality_type
    if modality in ["CR", "PX"]:
        modality = "DX"

    std_names = StandardNames.objects.filter(modality=modality)

    # Obtain a list of standard name IDs that match this GeneralStudyModuleAttr
    matching_std_name_ids = std_names.filter(
        (Q(study_description=g.study_description) & Q(study_description__isnull=False))
        | (
            Q(requested_procedure_code_meaning=g.requested_procedure_code_meaning)
            & Q(requested_procedure_code_meaning__isnull=False)
        )
        | (
            Q(procedure_code_meaning=g.procedure_code_meaning)
            & Q(procedure_code_meaning__isnull=False)
        )
    ).values_list("pk", flat=True)

    # Obtain a list of standard name IDs that are already associated with this GeneralStudyModuleAttr
    std_name_ids_already_in_study = g.standard_names.values_list("pk", flat=True)

    # Names that are in the new list, but not in the existing list
    std_name_ids_to_add = np.setdiff1d(
        matching_std_name_ids, std_name_ids_already_in_study
    )

    if std_name_ids_to_add.size:
        g.standard_names.add(*std_name_ids_to_add)

    # Add standard name references to the study irradiation events where the acquisition_protocol values match.
    # Some events will already exist if the new data is adding to an existing study
    try:
        if modality == "CT":
            for event in g.ctradiationdose_set.get().ctirradiationeventdata_set.all():
                # Only add matching standard name if the event doesn't already have one
                if not event.standard_protocols.values_list("pk", flat=True):
                    pk_value = list(
                        std_names.filter(
                            Q(acquisition_protocol=event.acquisition_protocol)
                            & Q(acquisition_protocol__isnull=False)
                        ).values_list("pk", flat=True)
                    )
                    event.standard_protocols.add(*pk_value)
        else:
            for (
                event
            ) in g.projectionxrayradiationdose_set.get().irradeventxraydata_set.all():
                # Only add matching standard name if the event doesn't already have one
                if not event.standard_protocols.values_list("pk", flat=True):
                    pk_value = list(
                        std_names.filter(
                            Q(acquisition_protocol=event.acquisition_protocol)
                            & Q(acquisition_protocol__isnull=False)
                        ).values_list("pk", flat=True)
                    )
                    event.standard_protocols.add(*pk_value)

    except ObjectDoesNotExist:
        pass


def observercontext(dataset, obs):  # TID 1002
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "Observer Type":
            obs.observer_type = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Device Observer UID":
            try:
                obs.device_observer_uid = cont.UID
            except AttributeError:
                obs.device_observer_uid = cont.TextValue
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Device Observer Name":
            obs.device_observer_name = cont.TextValue
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Device Observer Manufacturer"
        ):
            obs.device_observer_manufacturer = cont.TextValue
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning == "Device Observer Model Name"
        ):
            obs.device_observer_model_name = cont.TextValue
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Device Observer Serial Number"
        ):
            obs.device_observer_serial_number = cont.TextValue
        elif (
            cont.ConceptNameCodeSequence[0].CodeMeaning
            == "Device Observer Physical Location during observation"
        ):
            obs.device_observer_physical_location_during_observation = cont.TextValue
        elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Device Role in Procedure":
            obs.device_role_in_procedure = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
    obs.save()


def person_participant(dataset, event_data_type, foreign_key, logger):
    """
    Function to record people involved with study

    :param dataset: DICOM data being parsed
    :param event_data_type: Which function has called this function
    :param foreign_key: object of model this modal will link to
    :param logger: The logger that should be used by this function
    :return: None
    """

    if event_data_type == "ct_dose_check_alert":
        person = PersonParticipant.objects.create(
            ct_dose_check_details_alert=foreign_key
        )
    elif event_data_type == "ct_dose_check_notification":
        person = PersonParticipant.objects.create(
            ct_dose_check_details_notification=foreign_key
        )
    elif event_data_type == "radiopharmaceutical_administration_event_data":
        person = PersonParticipant.objects.create(
            radiopharmaceutical_administration_event_data=foreign_key
        )
    else:
        return
    try:
        person.person_name = dataset.PersonName
    except AttributeError:
        logger.debug("Person Name ConceptNameCodeSequence, but no PersonName element")
    try:
        for cont in dataset.ContentSequence:
            if (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Person Role in Procedure"
            ):
                person.person_role_in_procedure_cid = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Person ID":
                person.person_id = cont.TextValue
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Person ID Issue":
                person.person_id_issuer = cont.TextValue
            elif cont.ConceptNameCodeSequence[0].CodeMeaning == "Organization Name":
                person.organization_name = cont.TextValue
            elif (
                cont.ConceptNameCodeSequence[0].CodeMeaning
                == "Person Role in Organization"
            ):
                person.person_role_in_organization_cid = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
    except AttributeError:
        logger.debug("Person Name sequence malformed")
    person.save()


def generalequipmentmoduleattributes(dataset, study):  # C.7.5.1

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
    try:
        equip.date_of_last_calibration = get_date("DateOfLastCalibration", dataset)
        equip.time_of_last_calibration = get_time("TimeOfLastCalibration", dataset)
    except TypeError:  # DICOM supports MultiValue for those fields, but we don't
        pass
    if hasattr(dataset, "ContentSequence"):
        try:
            device_observer_uid = [
                content.UID
                for content in dataset.ContentSequence
                if (
                    content.ConceptNameCodeSequence[0].CodeValue == "121012"
                    and content.ConceptNameCodeSequence[0].CodingSchemeDesignator
                    == "DCM"
                )
            ][
                0
            ]  # 121012 = DeviceObserverUID
        except AttributeError:
            device_observer_uid = [
                content.TextValue
                for content in dataset.ContentSequence
                if (
                    content.ConceptNameCodeSequence[0].CodeValue == "121012"
                    and content.ConceptNameCodeSequence[0].CodingSchemeDesignator
                    == "DCM"
                )
            ][
                0
            ]  # 121012 = DeviceObserverUID
        except IndexError:
            device_observer_uid = None
    else:
        device_observer_uid = None

    if (
        equip.manufacturer_model_name
        in settings.IGNORE_DEVICE_OBSERVER_UID_FOR_THESE_MODELS
    ):
        device_observer_uid = None

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
        device_observer_uid=device_observer_uid,
        device_observer_uid_hash=hash_id(device_observer_uid),
    )

    if created:
        # If we have a device_observer_uid and it is desired, merge this "new" device with an existing one based on the
        # device observer uid.
        try:
            match_on_device_observer_uid = (
                MergeOnDeviceObserverUIDSettings.objects.values_list(
                    "match_on_device_observer_uid", flat=True
                )[0]
            )
        except IndexError:
            match_on_device_observer_uid = False
        if match_on_device_observer_uid and device_observer_uid:
            matched_equip_display_name = UniqueEquipmentNames.objects.filter(
                device_observer_uid=device_observer_uid
            )
            # We just inserted UniqueEquipmentName, so there should be more than one to match on another
            if len(matched_equip_display_name) > 1:
                # check if the first is the same as the just inserted. If it is, take the second object
                object_nr = 0
                if equip_display_name == matched_equip_display_name[object_nr]:
                    object_nr = 1
                equip_display_name.display_name = matched_equip_display_name[
                    object_nr
                ].display_name
                equip_display_name.user_defined_modality = matched_equip_display_name[
                    object_nr
                ].user_defined_modality
        if not equip_display_name.display_name:
            # if no display name, either new unit, existing unit with changes and match on UID False, or existing and
            # first import since match_on_uid added. Code below should then apply name from last version of unit.
            match_without_observer_uid = UniqueEquipmentNames.objects.filter(
                manufacturer_hash=hash_id(equip.manufacturer),
                institution_name_hash=hash_id(equip.institution_name),
                station_name_hash=hash_id(equip.station_name),
                institutional_department_name_hash=hash_id(
                    equip.institutional_department_name
                ),
                manufacturer_model_name_hash=hash_id(equip.manufacturer_model_name),
                device_serial_number_hash=hash_id(equip.device_serial_number),
                software_versions_hash=hash_id(equip.software_versions),
                gantry_id_hash=hash_id(equip.gantry_id),
            ).order_by("-pk")
            if match_without_observer_uid.count() > 1:
                # ordered by -pk; 0 is the new entry, 1 will be the last one before that
                equip_display_name.display_name = match_without_observer_uid[
                    1
                ].display_name
        if not equip_display_name.display_name:
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


def patientstudymoduleattributes(dataset, g):  # C.7.2.2

    patientatt = PatientStudyModuleAttr.objects.create(
        general_study_module_attributes=g
    )
    patientatt.patient_age = get_value_kw("PatientAge", dataset)
    patientatt.patient_weight = get_value_kw("PatientWeight", dataset)
    patientatt.patient_size = get_value_kw("PatientSize", dataset)
    patientatt.save()


def generalstudymoduleattributes(dataset, g, logger):  # C.7.2.1
    g.study_instance_uid = get_value_kw("StudyInstanceUID", dataset)
    g.series_instance_uid = get_value_kw("SeriesInstanceUID", dataset)
    g.study_date = get_date("StudyDate", dataset)
    if not g.study_date:
        g.study_date = get_date("ContentDate", dataset)
    if not g.study_date:
        g.study_date = get_date("SeriesDate", dataset)
    if not g.study_date:
        logger.error(
            f"Study UID {g.study_instance_uid} of modality {get_value_kw('ManufacturerModelName', dataset)} has no date"
            f" information which is needed in the interface - date has been set to 1900!"
        )
        g.study_date = make_date("19000101")
    g.study_time = get_time("StudyTime", dataset)
    g.series_time = get_time("SeriesTime", dataset)
    g.content_time = get_time("ContentTime", dataset)
    if not g.study_time:
        if g.content_time:
            g.study_time = g.content_time
        elif g.series_time:
            g.study_time = g.series_time
        else:
            logger.warning(
                "Study UID {0} of modality {1} has no time information which is needed in the interface - "
                "time has been set to midnight.".format(
                    g.study_instance_uid, get_value_kw("ManufacturerModelName", dataset)
                )
            )
            g.study_time = make_time(000000)
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
    g.physician_of_record = list_to_string(get_value_kw("PhysiciansOfRecord", dataset))
    g.name_of_physician_reading_study = list_to_string(
        get_value_kw("NameOfPhysiciansReadingStudy", dataset)
    )
    g.performing_physician_name = list_to_string(
        get_value_kw("PerformingPhysicianName", dataset)
    )
    g.operator_name = list_to_string(get_value_kw("OperatorsName", dataset))
    g.procedure_code_value = get_seq_code_value("ProcedureCodeSequence", dataset)
    g.procedure_code_meaning = get_seq_code_meaning("ProcedureCodeSequence", dataset)
    g.requested_procedure_code_value = get_seq_code_value(
        "RequestedProcedureCodeSequence", dataset
    )
    g.requested_procedure_code_meaning = get_seq_code_meaning(
        "RequestedProcedureCodeSequence", dataset
    )
    g.save()
