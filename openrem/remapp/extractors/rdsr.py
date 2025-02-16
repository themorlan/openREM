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
..  module:: rdsr.
    :synopsis: Module to extract radiation dose related data from DICOM Radiation SR objects
        or Radiopharmaceutical Radiation SR objects

..  moduleauthor:: Ed McDonagh

"""
from collections import OrderedDict
from datetime import timedelta
import logging
import os
import sys

import django
from django.db.models import Sum, ObjectDoesNotExist
import pydicom

from openrem.remapp.tools.background import (
    record_task_error_exit,
    record_task_related_query,
    record_task_info,
    run_in_background_with_limits,
)

# setup django/OpenREM.
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from ..tools.check_uid import record_sop_instance_uid
from ..tools.get_values import (
    get_value_kw,
)
from ..tools.make_skin_map import (
    make_skin_map,
    skin_dose_maps_enabled_for_xray_system,
)
from ..tools.send_high_dose_alert_emails import (
    send_rf_high_dose_alert_email,
    send_import_success_email,
)
from .extract_common import (  # pylint: disable=wrong-import-order, wrong-import-position
    ct_event_type_count,
    patient_module_attributes,
    populate_mammo_agd_summary,
    populate_dx_rf_summary,
    populate_rf_delta_weeks_summary,
    add_standard_names,
    generalstudymoduleattributes,
    generalequipmentmoduleattributes,
    patientstudymoduleattributes,
)
from .rdsr_methods import projectionxrayradiationdose
from .rrdsr_methods import _radiopharmaceuticalradiationdose
from remapp.models import (  # pylint: disable=wrong-import-order, wrong-import-position
    AccumIntegratedProjRadiogDose,
    DicomDeleteSettings,
    GeneralStudyModuleAttr,
    HighDoseMetricAlertSettings,
    PKsForSummedRFDoseStudiesInDeltaWeeks,
    SkinDoseMapCalcSettings,
)

logger = logging.getLogger(__name__)


def _rdsr_rrdsr_contents(dataset, g):
    try:
        template_identifier = dataset.ContentTemplateSequence[0].TemplateIdentifier
    except AttributeError:
        try:
            if dataset.ContentSequence[0].ConceptCodeSequence[0].CodeValue == "113704":
                template_identifier = "10001"
            elif (
                dataset.ConceptNameCodeSequence[0].CodingSchemeDesignator
                == "99SMS_RADSUM"
                and dataset.ConceptNameCodeSequence[0].CodeValue == "C-10"
            ):
                template_identifier = "10001"
            elif dataset.ConceptCodeSequence[0].CodeValue == "113500":
                template_identifier = "10021"
            else:
                logger.error(
                    "Study UID {0} of modality {1} has no template sequence - incomplete RDSR. "
                    "Aborting.".format(
                        g.study_instance_uid,
                        get_value_kw("ManufacturerModelName", dataset),
                    )
                )
                g.delete()
                return
        except AttributeError:
            logger.error(
                "Study UID {0} of modality {1} has no template sequence - incomplete RDSR. Aborting.".format(
                    g.study_instance_uid, get_value_kw("ManufacturerModelName", dataset)
                )
            )
            g.delete()
            return
    if template_identifier == "10001":
        projectionxrayradiationdose(dataset, g, "projection")
    elif template_identifier == "10011":
        projectionxrayradiationdose(dataset, g, "ct")
    elif template_identifier == "10021":
        _radiopharmaceuticalradiationdose(dataset, g)
    g.save()
    if not g.requested_procedure_code_meaning:
        if "RequestAttributesSequence" in dataset and dataset[0x40, 0x275].VM:
            # Ugly hack to prevent issues with zero length LS16 sequence
            req = dataset.RequestAttributesSequence
            g.requested_procedure_code_meaning = get_value_kw(
                "RequestedProcedureDescription", req[0]
            )
            # Sometimes the above is true, but there is no RequestedProcedureDescription in that sequence, but
            # there is a basic field as below.
            if not g.requested_procedure_code_meaning:
                g.requested_procedure_code_meaning = get_value_kw(
                    "RequestedProcedureDescription", dataset
                )
            g.save()
        else:
            g.requested_procedure_code_meaning = get_value_kw(
                "RequestedProcedureDescription", dataset
            )
            g.save()

    try:
        number_of_events_ct = (
            g.ctradiationdose_set.get().ctirradiationeventdata_set.count()
        )
    except ObjectDoesNotExist:
        number_of_events_ct = 0
    try:
        number_of_events_proj = (
            g.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
        )
    except ObjectDoesNotExist:
        number_of_events_proj = 0
    g.number_of_events = number_of_events_ct + number_of_events_proj
    g.save()
    if template_identifier == "10011":
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
    elif template_identifier == "10001":
        if g.modality_type == "MG":
            populate_mammo_agd_summary(g)
        else:
            populate_dx_rf_summary(g)


def _get_existing_event_uids(study):
    """
    Returns all event uids stored for this study
    """
    existing_event_uids = set()
    s = study.ctradiationdose_set.first()
    if s is not None:
        for event in s.ctirradiationeventdata_set.all():
            existing_event_uids.add(event.irradiation_event_uid)

    s = study.projectionxrayradiationdose_set.first()
    if s is not None:
        for event in s.irradeventxraydata_set.all():
            existing_event_uids.add(event.irradiation_event_uid)

    s = study.radiopharmaceuticalradiationdose_set.first()
    if s is not None:
        for event in s.radiopharmaceuticaladministrationeventdata_set.all():
            existing_event_uids.add(event.radiopharmaceutical_administration_event_uid)

    return existing_event_uids


def _update_recorded_objects(g, dataset, existing_sop_instance_uids=None):
    new_sop_instance_uid = dataset.SOPInstanceUID
    record_sop_instance_uid(g, new_sop_instance_uid)
    if existing_sop_instance_uids is not None:
        for sop_instance_uid in existing_sop_instance_uids:
            record_sop_instance_uid(g, sop_instance_uid)


def _get_dataset_event_uids(dataset):
    """
    Collect the uids from all events that can be in a dataset
    and that we care about
    """
    new_event_uids = set()
    for content in dataset.ContentSequence:
        if content.ValueType and content.ValueType == "CONTAINER":
            if content.ConceptNameCodeSequence[0].CodeMeaning in (
                "CT Acquisition",
                "Irradiation Event X-Ray Data",
                "Radiopharmaceutical Administration",
            ):
                for item in content.ContentSequence:
                    if item.ConceptNameCodeSequence[0].CodeMeaning in (
                        "Irradiation Event UID",
                        "Radiopharmaceutical Administration Event UID",
                    ):
                        new_event_uids.add("{0}".format(item.UID))
    return new_event_uids


def _handle_study_already_existing(
    dataset,
):
    """
    This function checks wheter there is already a study with the same instance uid as dataset.

    If yes it checks the data in the dataset and the existing study
    returning different strategies for continuing the import:

    If dataset was imported already: Abort the import
    If dataset contains subset of events of existing study: Abort the import
    If dataset has same events as existing study: Abort the import
    If dataset has more events than existing study (Events of existing are subset):
        Delete old study and reimport
    If dataset has different events than existing study: Create a second study and import
    If dataset is rrdsr and only images have been imported to the study: Complete study with data from the rrdsr

    :return: A dict with possible keys {status, existing_sop_instance_uids, study}, where
        :status: One of 'abort', 'continue' (Create the study and maybe save old instance uids to it),
            'drop_old_continue' (Delete old and reimport),
            'retry' (Wait then call this function again, there are partially imported studies), append_rrdsr'
        :existing_sop_instance_uids: The sop instance uids of all files imported for this study. For e.g.
            drop_old_continue status this should be saved onto the new study entry. (See _update_recorded_objects)
        :study: The study that was used to make the decision what to do next.
            Only added for status='append_rrdsr' or status='drop_old_continue' at the moment
    """
    existing_sop_instance_uids = set()

    study_uid = dataset.StudyInstanceUID
    existing_study_uid_match = GeneralStudyModuleAttr.objects.filter(
        study_instance_uid__exact=study_uid
    )
    if existing_study_uid_match:
        new_sop_instance_uid = dataset.SOPInstanceUID
        for existing_study in existing_study_uid_match.order_by("pk"):
            for processed_object in existing_study.objectuidsprocessed_set.all():
                existing_sop_instance_uids.add(processed_object.sop_instance_uid)
        if new_sop_instance_uid in existing_sop_instance_uids:
            # We've dealt with this object before...
            logger.debug(
                "Import match on Study Instance UID {0} and object SOP Instance UID {1}. "
                "Will not import.".format(study_uid, new_sop_instance_uid)
            )
            record_task_error_exit("Study already in db")
            return {
                "status": "abort",
                "existing_sop_instance_uids": existing_sop_instance_uids,
            }
        # Either we've not seen it before, or it wasn't recorded when we did.
        # Next find the event UIDs in the RDSR being imported
        new_event_uids = _get_dataset_event_uids(dataset)
        logger.debug(
            "Import match on StudyInstUID {0}. New RDSR event UIDs {1}".format(
                study_uid, new_event_uids
            )
        )

        # Now check which event UIDs are in the database already
        existing_event_uids = OrderedDict()
        for i, existing_study in enumerate(existing_study_uid_match.order_by("pk")):
            existing_event_uids[i] = set()
            existing_event_uids[i] = _get_existing_event_uids(existing_study)
        logger.debug(
            "Import match on StudyInstUID {0}. Existing event UIDs {1}".format(
                study_uid, existing_event_uids
            )
        )

        # Now compare the two
        for study_index, uid_list in list(existing_event_uids.items()):
            if uid_list == new_event_uids:
                # New RDSR is the same as the existing one
                logger.debug(
                    "Import match on StudyInstUID {0}. Event level match, will not import.".format(
                        study_uid
                    )
                )
                record_sop_instance_uid(
                    existing_study_uid_match[study_index], new_sop_instance_uid
                )
                record_task_error_exit("Study already in db")
                return {
                    "status": "abort",
                    "existing_sop_instance_uids": existing_sop_instance_uids,
                }
            elif new_event_uids.issubset(uid_list):
                # New RDSR has the same but fewer events than existing one
                logger.debug(
                    "Import match on StudyInstUID {0}. New RDSR events are subset of existing events. "
                    "Will not import.".format(study_uid)
                )
                record_sop_instance_uid(
                    existing_study_uid_match[study_index], new_sop_instance_uid
                )
                record_task_error_exit("Study already in db")
                return {
                    "status": "abort",
                    "existing_sop_instance_uids": existing_sop_instance_uids,
                }
            elif uid_list.issubset(new_event_uids):
                # New RDSR has the existing events and more
                # Check existing one had finished importing
                logger.debug(
                    "Import match on StudyInstUID {0}. Existing events are subset of new events. Will"
                    " import.".format(study_uid)
                )
                return {
                    "status": "drop_old_continue",
                    "existing_sop_instance_uids": existing_sop_instance_uids,
                    "study": existing_study_uid_match[study_index],
                }
            elif None in uid_list:
                # This happens for NM studies where only images have been imported so far
                # because they add a RadiopharmaceuticalAdministrationEventData Object without UID.
                logger.debug(
                    f"Import match on StudyInstUID {study_uid}. There is already a NM study for"
                    "which only Images where imported so far. Will delete the "
                    "RadiopharmaceuticalAdministrationEventData and RadiopharmaceuticalRadioationDose"
                    "and reimport, but keep the PET_Series data if present."
                )
                study = existing_study_uid_match[study_index]
                if study.modality_type == "NM" and (
                    dataset.SOPClassUID
                    == "1.2.840.10008.5.1.4.1.1.88.68"  # Radiopharmaceutical Radiation Dose SR
                    and dataset.ConceptNameCodeSequence[0].CodeValue
                    == "113500"  # Radiopharmaceutical Radiation Dose Report
                ):
                    tmp = study.radiopharmaceuticalradiationdose_set.first()
                    if tmp is not None:
                        tmp = tmp.radiopharmaceuticaladministrationeventdata_set.first()
                        if tmp is not None:
                            if tmp.radiopharmaceutical_administration_event_uid is None:
                                return {
                                    "status": "append_rrdsr",
                                    "existing_sop_instance_uids": existing_sop_instance_uids,
                                    "study": study,
                                }

    return {
        "status": "continue",
        "existing_sop_instance_uids": existing_sop_instance_uids,
    }


def _rdsr2db(dataset):
    if "StudyInstanceUID" in dataset:
        study_uid = dataset.StudyInstanceUID
        study_uid = dataset.StudyInstanceUID
        record_task_info(f"Study UID: {study_uid.replace('.', '. ')}")
        record_task_related_query(study_uid)
        existing = _handle_study_already_existing(dataset)

        if existing["status"] == "abort":
            return
        elif existing["status"] == "append_rrdsr":
            _update_recorded_objects(existing["study"], dataset)
            _rdsr_rrdsr_contents(dataset, existing["study"])
            radios = existing["study"].radiopharmaceuticalradiationdose_set.all()
            if (
                radios[0]
                .radiopharmaceuticaladministrationeventdata_set.get()
                .radiopharmaceutical_administration_event_uid
                is None
            ):
                radio_old, radio_new = (radios[0], radios[1])
            else:
                radio_new, radio_old = (radios[0], radios[1])
            radio_old.petseries_set.update(radiopharmaceutical_radiation_dose=radio_new)
            radio_old.delete()
            return
        elif existing["status"] == "drop_old_continue":
            existing["study"].delete()
        elif existing["status"] == "continue":
            pass  # We are allowed to proceed importing

    g = GeneralStudyModuleAttr.objects.create()
    if not g:  # Allows import to be aborted if no template found
        return
    g.save()
    if existing["status"] == "drop_old_continue":
        _update_recorded_objects(g, dataset, existing["existing_sop_instance_uids"])
    else:
        _update_recorded_objects(g, dataset)
    generalstudymoduleattributes(dataset, g, logger)
    generalequipmentmoduleattributes(dataset, g)
    _rdsr_rrdsr_contents(dataset, g)
    patientstudymoduleattributes(dataset, g)
    patient_module_attributes(dataset, g)

    try:
        SkinDoseMapCalcSettings.objects.get()
    except ObjectDoesNotExist:
        SkinDoseMapCalcSettings.objects.create()

    enable_skin_dose_maps = SkinDoseMapCalcSettings.objects.values_list(
        "enable_skin_dose_maps", flat=True
    )[0]
    calc_on_import = SkinDoseMapCalcSettings.objects.values_list(
        "calc_on_import", flat=True
    )[0]
    if g.modality_type == "RF" and enable_skin_dose_maps and calc_on_import:
        skin_maps_enabled = skin_dose_maps_enabled_for_xray_system(g)
        if skin_maps_enabled:
            run_in_background_with_limits(
                make_skin_map,
                "make_skin_map",
                0,
                {"make_skin_map": 1},
                g.pk,
            )

    # Calculate summed total DAP and dose at RP for studies that have this study's patient ID, going back week_delta
    # weeks in time from this study date. Only do this if activated in the fluoro alert settings (check whether
    # HighDoseMetricAlertSettings.calc_accum_dose_over_delta_weeks_on_import is True).
    if g.modality_type == "RF":

        try:
            HighDoseMetricAlertSettings.objects.get()
        except ObjectDoesNotExist:
            HighDoseMetricAlertSettings.objects.create()

        week_delta = HighDoseMetricAlertSettings.objects.values_list(
            "accum_dose_delta_weeks", flat=True
        )[0]
        calc_accum_dose_over_delta_weeks_on_import = (
            HighDoseMetricAlertSettings.objects.values_list(
                "calc_accum_dose_over_delta_weeks_on_import", flat=True
            )[0]
        )
        if calc_accum_dose_over_delta_weeks_on_import:

            all_rf_studies = GeneralStudyModuleAttr.objects.filter(
                modality_type__exact="RF"
            ).all()

            patient_id = g.patientmoduleattr_set.values_list("patient_id", flat=True)[0]
            if patient_id:
                study_date = g.study_date
                oldest_date = study_date - timedelta(weeks=week_delta)

                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # The try and except parts of this code are here because some of the studies in my database didn't have the
                # expected data in the related fields - not sure why. Perhaps an issue with the extractor routine?
                try:
                    g.projectionxrayradiationdose_set.get().accumxraydose_set.all()
                except ObjectDoesNotExist:
                    g.projectionxrayradiationdose_set.get().accumxraydose_set.create()

                for (
                    accumxraydose
                ) in g.projectionxrayradiationdose_set.get().accumxraydose_set.all():
                    try:
                        accumxraydose.accumintegratedprojradiogdose_set.get()
                    except:
                        accumxraydose.accumintegratedprojradiogdose_set.create()
                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

                for (
                    accumxraydose
                ) in g.projectionxrayradiationdose_set.get().accumxraydose_set.all():
                    accum_int_proj_pk = (
                        accumxraydose.accumintegratedprojradiogdose_set.get().pk
                    )

                    accum_int_proj_to_update = (
                        AccumIntegratedProjRadiogDose.objects.get(pk=accum_int_proj_pk)
                    )

                    included_studies = all_rf_studies.filter(
                        patientmoduleattr__patient_id__exact=patient_id,
                        study_date__range=[oldest_date, study_date],
                    )

                    bulk_entries = []
                    for pk in included_studies.values_list("pk", flat=True):
                        if not PKsForSummedRFDoseStudiesInDeltaWeeks.objects.filter(
                            general_study_module_attributes_id__exact=g.pk
                        ).filter(study_pk_in_delta_weeks__exact=pk):
                            new_entry = PKsForSummedRFDoseStudiesInDeltaWeeks()
                            new_entry.general_study_module_attributes_id = g.pk
                            new_entry.study_pk_in_delta_weeks = pk
                            bulk_entries.append(new_entry)
                    if len(bulk_entries):
                        PKsForSummedRFDoseStudiesInDeltaWeeks.objects.bulk_create(
                            bulk_entries
                        )

                    accum_totals = included_studies.aggregate(
                        Sum(
                            "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_area_product_total"
                        ),
                        Sum(
                            "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_rp_total"
                        ),
                    )
                    accum_int_proj_to_update.dose_area_product_total_over_delta_weeks = accum_totals[
                        "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_area_product_total__sum"
                    ]
                    accum_int_proj_to_update.dose_rp_total_over_delta_weeks = accum_totals[
                        "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_rp_total__sum"
                    ]
                    accum_int_proj_to_update.save()
                populate_rf_delta_weeks_summary(g)

        # Send an e-mail to all high dose alert recipients if this study is at or above threshold levels
        send_alert_emails_ref = HighDoseMetricAlertSettings.objects.values_list(
            "send_high_dose_metric_alert_emails_ref", flat=True
        )[0]
        send_alert_emails_skin = HighDoseMetricAlertSettings.objects.values_list(
            "send_high_dose_metric_alert_emails_skin", flat=True
        )[0]
        if send_alert_emails_ref and not send_alert_emails_skin:
            send_rf_high_dose_alert_email(g.pk)

    # Add standard names
    add_standard_names(g)

    # Sende Erfolgs-Email nach erfolgreichem Import
    if hasattr(dataset, 'StudyInstanceUID'):
        send_import_success_email(g.pk, dataset.StudyInstanceUID)


def _fix_toshiba_vhp(dataset):
    """
    Replace forward slash in multi-value decimal string VR with back slash
    :param dataset: DICOM dataset
    :return: Repaired DICOM dataset
    """

    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeMeaning == "CT Acquisition":
            for cont2 in cont.ContentSequence:
                if (
                    cont2.ConceptNameCodeSequence[0].CodeMeaning
                    == "Dose Reduce Parameters"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator
                    == "99TOSHIBA-TMSC"
                ):
                    for cont3 in cont2.ContentSequence:
                        if (
                            cont3.ConceptNameCodeSequence[0].CodeMeaning
                            == "Standard deviation of population"
                        ):
                            try:
                                cont3.MeasuredValueSequence[0].NumericValue
                            except ValueError:
                                vhp_sd = dict.__getitem__(
                                    cont3.MeasuredValueSequence[0], 0x40A30A
                                )
                                vhp_sd_value = vhp_sd.__getattribute__("value")
                                if "/" in vhp_sd_value:
                                    vhp_sd_value = vhp_sd_value.replace("/", "\\")
                                    new_vhp_sd = vhp_sd._replace(value=vhp_sd_value)
                                    dict.__setitem__(
                                        cont3.MeasuredValueSequence[0],
                                        0x40A30A,
                                        new_vhp_sd,
                                    )


def rdsr(rdsr_file):
    """
    Prozessiert RDSR (Radiation Dose Structured Report) Dateien
    """
    logger.info(f"Starting RDSR import for file: {rdsr_file}")
    try:
        del_settings = DicomDeleteSettings.objects.get()
        del_rdsr = del_settings.del_rdsr
    except ObjectDoesNotExist:
        del_rdsr = False

    try:
        dataset = pydicom.dcmread(rdsr_file)
    except FileNotFoundError:
        logger.warning(
            f"rdsr.py not attempting to extract from {rdsr_file}, the file does not exist"
        )
        record_task_error_exit(
            f"Not attempting to extract from {rdsr_file}, the file does not exist"
        )
        return 1

    try:
        dataset.decode()
    except ValueError as e:
        if "Invalid tag (0040, a30a): invalid literal for float()" in e.message:
            _fix_toshiba_vhp(dataset)
            dataset.decode()

    if (
        dataset.SOPClassUID
        in (
            "1.2.840.10008.5.1.4.1.1.88.67",  # X-Ray Radiation Dose SR
            "1.2.840.10008.5.1.4.1.1.88.22",  # Enhanced SR
        )
        and dataset.ConceptNameCodeSequence[0].CodeValue
        == "113701"  # X-Ray Radiation Dose Report
    ):
        logger.debug("rdsr.py extracting from {0}".format(rdsr_file))
        _rdsr2db(dataset)
    elif (
        dataset.SOPClassUID == ("1.2.840.10008.5.1.4.1.1.88.22")  # Enhanced SR
        and dataset.ConceptNameCodeSequence[0].CodingSchemeDesignator
        == "99SMS_RADSUM"  # Siemens Arcadis
        and dataset.ConceptNameCodeSequence[0].CodeValue == "C-10"
    ):
        logger.debug("rdsr.py extracting from {0}".format(rdsr_file))
        _rdsr2db(dataset)
    elif (
        dataset.SOPClassUID
        == "1.2.840.10008.5.1.4.1.1.88.68"  # Radiopharmaceutical Radiation Dose SR
        and dataset.ConceptNameCodeSequence[0].CodeValue
        == "113500"  # Radiopharmaceutical Radiation Dose Report
    ):
        logger.debug(f"rdsr.py extracting from {rdsr_file}")
        _rdsr2db(dataset)
    else:
        logger.warning(
            f"rdsr.py not attempting to extract from {rdsr_file}, not a radiation dose structured report"
        )
        record_task_error_exit(
            f"Not attempting to extract from {rdsr_file}, not an rdsr"
        )
        return 1

    if del_rdsr:
        os.remove(rdsr_file)

    logger.info("RDSR import completed successfully")
    return 0
