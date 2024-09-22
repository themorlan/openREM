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
..  module:: ct_export.
    :synopsis: Module to export database data to multi-sheet Microsoft XLSX files and single-sheet csv files

..  moduleauthor:: Ed McDonagh

"""
import logging
import datetime

import django.db
import numpy as np
import pandas as pd

from django.core.exceptions import ObjectDoesNotExist
from django.utils.translation import gettext as _
from django.conf import settings
from django.db.models import Max

from openrem.remapp.tools.background import get_or_generate_task_uuid

from remapp.models import GeneralStudyModuleAttr

from ..tools.check_standard_name_status import are_standard_names_enabled

from ..interface.mod_filters import ct_acq_filter

from .export_common_pandas import (
    get_common_data,
    common_headers,
    create_xlsx,
    create_csv,
    write_export,
    create_summary_sheet,
    abort_if_zero_studies,
    create_export_task,
    transform_to_one_row_per_exam,
    create_standard_name_df_columns,
    optimise_df_dtypes,
    write_row_to_acquisition_sheet,
    export_using_pandas,
    text_and_date_formats,
    sheet_name,
)

logger = logging.getLogger(__name__)


def ctxlsx(filterdict, pid=False, name=None, patid=None, user=None):
    """Export filtered CT database data to multi-sheet Microsoft XSLX files

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves xlsx file into Media directory for user to download
    """
    modality = "CT"

    # Exam-level integer field names and friendly names
    exam_int_fields = [
        "pk",
        "number_of_events",
    ]
    exam_int_field_names = ["pk", "Number of events"]

    # Exam-level object field names (string data, little or no repetition)
    exam_obj_fields = ["accession_number"]
    exam_obj_field_names = ["Accession number"]

    # Exam-level category field names and friendly names
    exam_cat_fields = [
        "generalequipmentmoduleattr__institution_name",
        "generalequipmentmoduleattr__manufacturer",
        "generalequipmentmoduleattr__manufacturer_model_name",
        "generalequipmentmoduleattr__station_name",
        "generalequipmentmoduleattr__unique_equipment_name__display_name",
        "operator_name",
        "patientmoduleattr__patient_sex",
        "study_description",
        "requested_procedure_code_meaning",
    ]
    exam_cat_field_names = [
        "Institution",
        "Manufacturer",
        "Model",
        "Station name",
        "Display name",
        "Operator",
        "Patient sex",
        "Study description",
        "Requested procedure",
    ]

    # Exam-level date field names and friendly name
    exam_date_fields = ["study_date"]
    exam_date_field_names = ["Study date"]

    # Exam-level time field names and friendly name
    exam_time_fields = ["study_time"]
    exam_time_field_names = ["Study time"]

    # Exam-level category value names and friendly names
    exam_val_fields = [
        "patientstudymoduleattr__patient_age_decimal",
        "patientstudymoduleattr__patient_size",
        "patientstudymoduleattr__patient_weight",
        "total_dlp",
    ]
    exam_val_field_names = [
        "Patient age",
        "Patient height (m)",
        "Patient weight (kg)",
        "Total DLP (mGy.cm)",
    ]

    acquisition_int_fields = [
        "ctradiationdose__ctirradiationeventdata__pk",
        "ctradiationdose__ctirradiationeventdata__number_of_xray_sources",
    ]
    acquisition_int_field_names = [
        "Acquisition pk",
        "Number of sources",
    ]

    acquisition_cat_fields = [
        "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
        "ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning",
        "ctradiationdose__ctirradiationeventdata__ctdiw_phantom_type__code_meaning",
        "ctradiationdose__ctirradiationeventdata__xray_modulation_type",
        "ctradiationdose__ctirradiationeventdata__ctxraysourceparameters__identification_of_the_xray_source",
    ]
    acquisition_cat_field_names = [
        "Acquisition protocol",
        "Acquisition type",
        "CTDI phantom type",
        "mA modulation type",
        "Source name",
    ]

    acquisition_cat_field_std_name = (
        "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name"
    )
    acquisition_cat_field_name_std_name = "Standard acquisition name"

    # Required acquisition-level value field names and friendly names
    acquisition_val_fields = [
        "ctradiationdose__ctirradiationeventdata__dlp",
        "ctradiationdose__ctirradiationeventdata__exposure_time",
        "ctradiationdose__ctirradiationeventdata__scanninglength__scanning_length",
        "ctradiationdose__ctirradiationeventdata__nominal_single_collimation_width",
        "ctradiationdose__ctirradiationeventdata__nominal_total_collimation_width",
        "ctradiationdose__ctirradiationeventdata__pitch_factor",
        "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
        "ctradiationdose__ctirradiationeventdata__ctxraysourceparameters__kvp",
        "ctradiationdose__ctirradiationeventdata__ctxraysourceparameters__maximum_xray_tube_current",
        "ctradiationdose__ctirradiationeventdata__ctxraysourceparameters__xray_tube_current",
        "ctradiationdose__ctirradiationeventdata__ctxraysourceparameters__exposure_time_per_rotation",
    ]
    acquisition_val_field_names = [
        "DLP (mGy.cm)",
        "Exposure time (s)",
        "Scanning length (mm)",
        "Slice thickness (mm)",
        "Total collimation (mm)",
        "Pitch",
        "CTDIvol (mGy)",
        "kVp",
        "Maximum mA",
        "mA",
        "Exposure time per rotation",
    ]

    ct_dose_check_fields = [
        "ctradiationdose__ctirradiationeventdata__ctdosecheckdetails__dlp_alert_value_configured",
        "ctradiationdose__ctirradiationeventdata__ctdosecheckdetails__dlp_alert_value",
        "ctradiationdose__ctirradiationeventdata__ctdosecheckdetails__accumulated_dlp_forward_estimate",
        "ctradiationdose__ctirradiationeventdata__ctdosecheckdetails__ctdivol_alert_value_configured",
        "ctradiationdose__ctirradiationeventdata__ctdosecheckdetails__ctdivol_alert_value",
        "ctradiationdose__ctirradiationeventdata__ctdosecheckdetails__accumulated_ctdivol_forward_estimate",
        "ctradiationdose__ctirradiationeventdata__ctdosecheckdetails__alert_reason_for_proceeding",
        "ctradiationdose__ctirradiationeventdata__ctdosecheckdetails__tid1020_alert__person_name",
    ]
    ct_dose_check_field_names = [
        "DLP alert configured",
        "DLP alert value",
        "DLP forward estimate",
        "CTDIvol alert configured",
        "CTDIvol alert value",
        "CTDIvol forward estimate",
        "Reason for proceeding",
        "Person name",
    ]

    # Fields for obtaining the acquisition protocols in the data
    fields_for_acquisition_frequency = [
        "ctradiationdose__ctirradiationeventdata__pk",
        "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
    ]
    field_names_for_acquisition_frequency = ["pk", "Acquisition protocol"]
    field_for_acquisition_frequency_std_name = (
        "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name"
    )
    field_name_for_acquisition_frequency_std_name = "Standard acquisition name"

    enable_standard_names = are_standard_names_enabled()

    datestamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality=modality,
        export_type="XLSX_export",
        date_stamp=datestamp,
        pid=bool(pid and (name or patid)),
        user=user,
        filters_dict=filterdict,
    )

    tmpxlsx, book = create_xlsx(tsk)
    if not tmpxlsx:
        exit()

    # Get the data
    study_pks = None
    study_pks = ct_acq_filter(filterdict, pid=pid).qs.values("pk")

    # The initial_qs may have filters to remove some acquisition types. For the export we want all acquisitions
    # that are part of a study to be included. To achieve this, use the pk list from initial_qs to get a
    # corresponding set of unfiltered studies:
    qs = GeneralStudyModuleAttr.objects.filter(pk__in=study_pks)

    n_entries = qs.count()
    tsk.num_records = n_entries
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    tsk.progress = "{0} studies in query.".format(tsk.num_records)
    tsk.save()

    export_using_pandas(
        acquisition_cat_field_name_std_name,
        acquisition_cat_field_names,
        acquisition_cat_field_std_name,
        acquisition_cat_fields,
        acquisition_int_field_names,
        acquisition_int_fields,
        acquisition_val_field_names,
        acquisition_val_fields,
        book,
        ct_dose_check_field_names,
        ct_dose_check_fields,
        datestamp,
        enable_standard_names,
        exam_cat_field_names,
        exam_cat_fields,
        exam_date_field_names,
        exam_date_fields,
        exam_int_field_names,
        exam_int_fields,
        exam_obj_field_names,
        exam_obj_fields,
        exam_time_field_names,
        exam_time_fields,
        exam_val_field_names,
        exam_val_fields,
        field_for_acquisition_frequency_std_name,
        field_name_for_acquisition_frequency_std_name,
        field_names_for_acquisition_frequency,
        fields_for_acquisition_frequency,
        modality,
        n_entries,
        name,
        patid,
        pid,
        qs,
        tmpxlsx,
        tsk,
    )


def ct_csv(filterdict, pid=False, name=None, patid=None, user=None):
    """Export filtered CT database data to a single-sheet CSV file.

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves csv file into Media directory for user to download
    """

    import datetime
    from django.db.models import Max
    from ..interface.mod_filters import ct_acq_filter

    datestamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="CT",
        export_type="CSV export",
        date_stamp=datestamp,
        pid=bool(pid and (name or patid)),
        user=user,
        filters_dict=filterdict,
    )

    tmpfile, writer = create_csv(tsk)
    if not tmpfile:
        exit()

    # Get the data!
    e = ct_acq_filter(filterdict, pid=pid).qs

    tsk.num_records = e.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    tsk.progress = "{0} studies in query.".format(tsk.num_records)
    tsk.save()

    headings = common_headers(pid=pid, name=name, patid=patid)
    headings += ["DLP total (mGy.cm)"]

    max_events_dict = e.aggregate(
        Max(
            "ctradiationdose__ctaccumulateddosedata__total_number_of_irradiation_events"
        )
    )
    max_events = max_events_dict[
        "ctradiationdose__ctaccumulateddosedata__total_number_of_irradiation_events__max"
    ]
    if not max_events:
        max_events = 1
    headings += _generate_all_data_headers_ct(max_events)
    writer.writerow(headings)

    tsk.progress = "CSV header row written."
    tsk.save()

    for i, exams in enumerate(e):
        tsk.progress = "{0} of {1}".format(i + 1, tsk.num_records)
        tsk.save()
        try:
            exam_data = get_common_data("CT", exams, pid, name, patid)
            for (
                s
            ) in exams.ctradiationdose_set.get().ctirradiationeventdata_set.order_by(
                "id"
            ):
                # Get series data
                exam_data += _ct_get_series_data(s)
            # Clear out any commas
            for index, item in enumerate(exam_data):
                if item is None:
                    exam_data[index] = ""
                if isinstance(item, str) and "," in item:
                    exam_data[index] = item.replace(",", ";")
            writer.writerow([str(data_string) for data_string in exam_data])
        except ObjectDoesNotExist:
            error_message = (
                "DoesNotExist error whilst exporting study {0} of {1},  study UID {2}, accession number"
                " {3} - maybe database entry was deleted as part of importing later version of same"
                " study?".format(
                    i + 1,
                    tsk.num_records,
                    exams.study_instance_uid,
                    exams.accession_number,
                )
            )
            logger.error(error_message)
            writer.writerow([error_message])

    tsk.progress = "All study data written."
    tsk.save()

    tmpfile.close()
    tsk.status = "COMPLETE"
    tsk.processtime = (datetime.datetime.now() - datestamp).total_seconds()
    tsk.save()


def _generate_all_data_headers_ct(max_events):
    """Generate the headers for CT that repeat once for each series of the exam with the most series in

    :param max_events: maximum number of times to repeat headers
    :return: list of headers
    """

    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    repeating_series_headers = []
    for h in range(int(max_events)):
        repeating_series_headers += ["E" + str(h + 1) + " Protocol"]

        if enable_standard_names:
            repeating_series_headers += [
                "E" + str(h + 1) + " Standard acquisition name"
            ]

        repeating_series_headers += [
            "E" + str(h + 1) + " Type",
            "E" + str(h + 1) + " Exposure time",
            "E" + str(h + 1) + " Scanning length",
            "E" + str(h + 1) + " Slice thickness",
            "E" + str(h + 1) + " Total collimation",
            "E" + str(h + 1) + " Pitch",
            "E" + str(h + 1) + " No. sources",
            "E" + str(h + 1) + " CTDIvol",
            "E" + str(h + 1) + " Phantom",
            "E" + str(h + 1) + " DLP",
            "E" + str(h + 1) + " S1 name",
            "E" + str(h + 1) + " S1 kVp",
            "E" + str(h + 1) + " S1 max mA",
            "E" + str(h + 1) + " S1 mA",
            "E" + str(h + 1) + " S1 Exposure time/rotation",
            "E" + str(h + 1) + " S2 name",
            "E" + str(h + 1) + " S2 kVp",
            "E" + str(h + 1) + " S2 max mA",
            "E" + str(h + 1) + " S2 mA",
            "E" + str(h + 1) + " S2 Exposure time/rotation",
            "E" + str(h + 1) + " mA Modulation type",
            "E" + str(h + 1) + " Dose check details",
            "E" + str(h + 1) + " Comments",
        ]

    return repeating_series_headers


def _ct_get_series_data(s):
    from collections import OrderedDict

    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    try:
        if s.ctdiw_phantom_type.code_value == "113691":
            phantom = "32 cm"
        elif s.ctdiw_phantom_type.code_value == "113690":
            phantom = "16 cm"
        else:
            phantom = s.ctdiw_phantom_type.code_meaning
    except AttributeError:
        phantom = None

    try:
        ct_acquisition_type = s.ct_acquisition_type.code_meaning
    except AttributeError:
        ct_acquisition_type = ""

    seriesdata = [
        s.acquisition_protocol,
    ]

    if enable_standard_names:
        try:
            standard_protocol = s.standard_protocols.first().standard_name
        except AttributeError:
            standard_protocol = ""

        if standard_protocol:
            seriesdata += [standard_protocol]
        else:
            seriesdata += [""]

    seriesdata = seriesdata + [
        ct_acquisition_type,
        s.exposure_time,
        s.scanninglength_set.get().scanning_length,
        s.nominal_single_collimation_width,
        s.nominal_total_collimation_width,
        s.pitch_factor,
        s.number_of_xray_sources,
        s.mean_ctdivol,
        phantom,
        s.dlp,
    ]
    source_parameters = OrderedDict()
    source_parameters[0] = {
        "id": None,
        "kvp": None,
        "max_current": None,
        "current": None,
        "time": None,
    }
    source_parameters[1] = {
        "id": None,
        "kvp": None,
        "max_current": None,
        "current": None,
        "time": None,
    }
    try:
        for index, source in enumerate(s.ctxraysourceparameters_set.all()):
            source_parameters[index]["id"] = source.identification_of_the_xray_source
            source_parameters[index]["kvp"] = source.kvp
            source_parameters[index]["max_current"] = source.maximum_xray_tube_current
            source_parameters[index]["current"] = source.xray_tube_current
            source_parameters[index]["time"] = source.exposure_time_per_rotation
    except (ObjectDoesNotExist, KeyError):
        logger.debug("Export: ctxraysourceparameters_set does not exist")
    for source in source_parameters:
        seriesdata += [
            source_parameters[source]["id"],
            source_parameters[source]["kvp"],
            source_parameters[source]["max_current"],
            source_parameters[source]["current"],
            source_parameters[source]["time"],
        ]
    try:
        dose_check = s.ctdosecheckdetails_set.get()
        dose_check_string = []
        if (
            dose_check.dlp_alert_value_configured
            or dose_check.ctdivol_alert_value_configured
        ):
            dose_check_string += ["Dose Check Alerts: "]
            if dose_check.dlp_alert_value_configured:
                dose_check_string += [
                    "DLP alert is configured at {0:.2f} mGy.cm with ".format(
                        dose_check.dlp_alert_value
                    )
                ]
                if dose_check.accumulated_dlp_forward_estimate:
                    dose_check_string += [
                        "an accumulated forward estimate of {0:.2f} mGy.cm. ".format(
                            dose_check.accumulated_dlp_forward_estimate
                        )
                    ]
                else:
                    dose_check_string += ["no accumulated forward estimate recorded. "]
            if dose_check.ctdivol_alert_value_configured:
                dose_check_string += [
                    "CTDIvol alert is configured at {0:.2f} mGy with ".format(
                        dose_check.ctdivol_alert_value
                    )
                ]
                if dose_check.accumulated_ctdivol_forward_estimate:
                    dose_check_string += [
                        "an accumulated forward estimate of {0:.2f} mGy. ".format(
                            dose_check.accumulated_ctdivol_forward_estimate
                        )
                    ]
                else:
                    dose_check_string += ["no accumulated forward estimate recorded. "]
            if dose_check.alert_reason_for_proceeding:
                dose_check_string += [
                    "Reason for proceeding: {0}. ".format(
                        dose_check.alert_reason_for_proceeding
                    )
                ]
            try:
                dose_check_person_alert = dose_check.tid1020_alert.get()
                if dose_check_person_alert.person_name:
                    dose_check_string += [
                        "Person authorizing irradiation: {0}. ".format(
                            dose_check_person_alert.person_name
                        )
                    ]
            except ObjectDoesNotExist:
                pass
        if (
            dose_check.dlp_notification_value_configured
            or dose_check.ctdivol_notification_value_configured
        ):
            dose_check_string += ["Dose Check Notifications: "]
            if dose_check.dlp_notification_value_configured:
                dose_check_string += [
                    "DLP notification is configured at {0:.2f} mGy.cm with ".format(
                        dose_check.dlp_notification_value
                    )
                ]
                if dose_check.dlp_forward_estimate:
                    dose_check_string += [
                        "an accumulated forward estimate of {0:.2f} mGy.cm. ".format(
                            dose_check.dlp_forward_estimate
                        )
                    ]
                else:
                    dose_check_string += ["no accumulated forward estimate recorded. "]
            if dose_check.ctdivol_notification_value_configured:
                dose_check_string += [
                    "CTDIvol notification is configured at {0:.2f} mGy with ".format(
                        dose_check.ctdivol_notification_value
                    )
                ]
                if dose_check.ctdivol_forward_estimate:
                    dose_check_string += [
                        "a forward estimate of {0:.2f} mGy. ".format(
                            dose_check.ctdivol_forward_estimate
                        )
                    ]
                else:
                    dose_check_string += ["no forward estimate recorded. "]
            if dose_check.notification_reason_for_proceeding:
                dose_check_string += [
                    "Reason for proceeding: {0}. ".format(
                        dose_check.notification_reason_for_proceeding
                    )
                ]
            try:
                dose_check_person_notification = dose_check.tid1020_notification.get()
                if dose_check_person_notification.person_name:
                    dose_check_string += [
                        "Person authorizing irradiation: {0}. ".format(
                            dose_check_person_notification.person_name
                        )
                    ]
            except ObjectDoesNotExist:
                pass
        dose_check_string = "".join(dose_check_string)
    except ObjectDoesNotExist:
        dose_check_string = ""
    seriesdata += [s.xray_modulation_type, dose_check_string, s.comment]
    return seriesdata


def ct_phe_2019(filterdict, user=None):
    """Export filtered CT database data in the format required for the 2019 Public Health England
    CT dose survey

    :param filterdict: Queryset of studies to export
    :param user:  User that has started the export
    :return: Saves Excel file into Media directory for user to download
    """

    import datetime
    from decimal import Decimal
    from ..interface.mod_filters import ct_acq_filter

    datestamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="CT",
        export_type="PHE CT 2019 export",
        date_stamp=datestamp,
        pid=False,
        user=user,
        filters_dict=filterdict,
    )

    tmp_xlsx, book = create_xlsx(tsk)
    if not tmp_xlsx:
        exit()

    # Get the data!
    exams = ct_acq_filter(filterdict, pid=False).qs

    tsk.num_records = exams.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    tsk.progress = "{0} studies in query.".format(tsk.num_records)
    tsk.save()

    headings = ["Patient No", "Age (yrs)", "Weight (kg)", "Height (cm)"]
    for x in range(4):  # pylint: disable=unused-variable
        headings += [
            "Imaged length",
            "Start position",
            "End position",
            "kV",
            "CTDI phantom",
            "Scan FOV (mm)",
            "CTDIvol (mGy)*",
            "DLP (mGy.cm)*",
        ]
    headings += ["Total DLP* (whole scan) mGy.cm", "Patient comments"]
    sheet = book.add_worksheet("PHE CT 2019")
    sheet.write_row(0, 0, headings)

    num_rows = exams.count()
    for row, exam in enumerate(exams):
        tsk.progress = "Writing study {0} of {1}".format(row + 1, num_rows)
        tsk.save()

        exam_data = []
        comments = []
        patient_age_decimal = None
        patient_size = None
        patient_weight = None
        try:
            patient_study_module = exam.patientstudymoduleattr_set.get()
            patient_age_decimal = patient_study_module.patient_age_decimal
            patient_size = patient_study_module.patient_size
            try:
                patient_size = patient_study_module.patient_size * Decimal(100.0)
            except TypeError:
                pass
            patient_weight = patient_study_module.patient_weight
        except ObjectDoesNotExist:
            logger.debug(
                "PHE CT 2019 export: patientstudymoduleattr_set object does not exist."
                " AccNum {0}, Date {1}".format(exam.accession_number, exam.study_date)
            )
        exam_data += [row + 1, patient_age_decimal, patient_weight, patient_size]
        series_index = 0
        for event in exam.ctradiationdose_set.get().ctirradiationeventdata_set.order_by(
            "id"
        ):
            try:
                ct_acquisition_type = event.ct_acquisition_type.code_meaning
                if ct_acquisition_type in "Constant Angle Acquisition":
                    continue
                comments += [ct_acquisition_type]
            except (ObjectDoesNotExist, AttributeError):
                comments += ["unknown type"]
            if series_index == 4:
                exam_data += ["", ""]
            series_index += 1
            scanning_length = None
            start_position = None
            end_position = None
            kv = None
            ctdi_phantom = None
            scan_fov = None
            try:
                scanning_length_data = event.scanninglength_set.get()
                scanning_length = scanning_length_data.scanning_length
                start_position = (
                    scanning_length_data.bottom_z_location_of_scanning_length
                )
                end_position = scanning_length_data.top_z_location_of_scanning_length
            except ObjectDoesNotExist:
                pass
            try:
                source_parameters = event.ctxraysourceparameters_set.order_by("pk")
                if source_parameters.count() == 2:
                    kv = "{0} | {1}".format(
                        source_parameters[0].kvp, source_parameters[1].kvp
                    )
                else:
                    kv = source_parameters[0].kvp
            except (ObjectDoesNotExist, IndexError):
                pass
            try:
                if event.ctdiw_phantom_type.code_value == "113691":
                    ctdi_phantom = "32 cm"
                elif event.ctdiw_phantom_type.code_value == "113690":
                    ctdi_phantom = "16 cm"
                else:
                    ctdi_phantom = event.ctdiw_phantom_type.code_meaning
            except AttributeError:
                pass
            exam_data += [
                scanning_length,
                start_position,
                end_position,
                kv,
                ctdi_phantom,
                scan_fov,
                event.mean_ctdivol,
                event.dlp,
            ]
        ct_dose_length_product_total = None
        try:
            ct_accumulated = (
                exam.ctradiationdose_set.get().ctaccumulateddosedata_set.get()
            )
            ct_dose_length_product_total = ct_accumulated.ct_dose_length_product_total
        except ObjectDoesNotExist:
            pass
        sheet.write_row(row + 1, 0, exam_data)
        sheet.write(row + 1, 36, ct_dose_length_product_total)
        patient_comment_cell = "Series types: " + ", ".join(comments)
        sheet.write(row + 1, 37, patient_comment_cell)
    book.close()
    tsk.progress = "PHE CT 2019 export complete"
    tsk.save()

    xlsxfilename = "PHE_CT2019{0}.xlsx".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))

    write_export(tsk, xlsxfilename, tmp_xlsx, datestamp)
