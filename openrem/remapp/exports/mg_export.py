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
..  module:: exportcsv.
    :synopsis: Module to export database data to single-sheet CSV files.

..  moduleauthor:: Ed McDonagh

"""

import datetime
import logging

from django.forms import CharField
from ..tools.background import get_or_generate_task_uuid

from django.core.exceptions import ObjectDoesNotExist

from remapp.models import GeneralStudyModuleAttr

from ..tools.check_standard_name_status import are_standard_names_enabled

from ..interface.mod_filters import mg_acq_filter

from django.db.models import F, Case, When, Value, TextField
from django.db.models.functions import Coalesce

from .export_common_pandas import (
    get_anode_target_material,
    get_common_data,
    common_headers,
    create_xlsx,
    create_csv,
    get_xray_filter_info,
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


def _series_headers(max_events):
    """Return a list of series headers

    :param max_events: number of series
    :return: headers as a list of strings
    """

    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    series_headers = []
    for series_number in range(max_events):
        series_headers += [
            "E" + str(series_number + 1) + " View",
            "E" + str(series_number + 1) + " View Modifier",
            "E" + str(series_number + 1) + " Laterality",
            "E" + str(series_number + 1) + " Acquisition",
        ]

        if enable_standard_names:
            series_headers += [
                "E" + str(series_number + 1) + " Standard acquisition name"
            ]

        series_headers += [
            "E" + str(series_number + 1) + " Thickness",
            "E" + str(series_number + 1) + " Radiological thickness",
            "E" + str(series_number + 1) + " Force",
            "E" + str(series_number + 1) + " Mag",
            "E" + str(series_number + 1) + " Area",
            "E" + str(series_number + 1) + " Mode",
            "E" + str(series_number + 1) + " Target",
            "E" + str(series_number + 1) + " Filter",
            "E" + str(series_number + 1) + " Filter thickness",
            "E" + str(series_number + 1) + " Focal spot size",
            "E" + str(series_number + 1) + " kVp",
            "E" + str(series_number + 1) + " mA",
            "E" + str(series_number + 1) + " ms",
            "E" + str(series_number + 1) + " uAs",
            "E" + str(series_number + 1) + " ESD",
            "E" + str(series_number + 1) + " AGD",
            "E" + str(series_number + 1) + " % Fibroglandular tissue",
            "E" + str(series_number + 1) + " Exposure mode description",
        ]
    return series_headers


def _mg_get_series_data(event):
    """Return the series level data

    :param event: event level object
    :return: series data as list of strings
    """

    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    try:
        mechanical_data = event.irradeventxraymechanicaldata_set.get()
        compression_thickness = mechanical_data.compression_thickness
        compression_force = mechanical_data.compression_force
        magnification_factor = mechanical_data.magnification_factor
    except ObjectDoesNotExist:
        compression_thickness = None
        compression_force = None
        magnification_factor = None

    try:
        radiological_thickness = (
            event.irradeventxraymechanicaldata_set.get()
            .doserelateddistancemeasurements_set.get()
            .radiological_thickness
        )
    except ObjectDoesNotExist:
        radiological_thickness = None

    try:
        source_data = event.irradeventxraysourcedata_set.get()
        collimated_field_area = source_data.collimated_field_area
        exposure_control_mode = source_data.exposure_control_mode
        anode_target_material = get_anode_target_material(source_data)
        focal_spot_size = source_data.focal_spot_size
        average_xray_tube_current = source_data.average_xray_tube_current
        exposure_time = source_data.exposure_time
        average_glandular_dose = source_data.average_glandular_dose
        try:
            filters, filter_thicknesses = get_xray_filter_info(source_data)
        except ObjectDoesNotExist:
            filters = None
            filter_thicknesses = None
        try:
            kvp = source_data.kvp_set.get().kvp
        except ObjectDoesNotExist:
            kvp = None
        try:
            exposure = source_data.exposure_set.get().exposure
        except ObjectDoesNotExist:
            exposure = None
    except ObjectDoesNotExist:
        collimated_field_area = None
        exposure_control_mode = None
        anode_target_material = None
        focal_spot_size = None
        average_xray_tube_current = None
        exposure_time = None
        average_glandular_dose = None
        filters = None
        filter_thicknesses = None
        kvp = None
        exposure = None

    if event.image_view:
        view = event.image_view.code_meaning
    else:
        view = None
    view_modifiers = event.imageviewmodifier_set.order_by("pk")
    modifier = ""
    if view_modifiers:
        for view_modifier in view_modifiers:
            try:
                modifier += f"{view_modifier.image_view_modifier.code_meaning} "
            except AttributeError:
                pass
    if event.laterality:
        laterality = event.laterality.code_meaning
    else:
        laterality = None

    series_data = [
        view,
        modifier,
        laterality,
        event.acquisition_protocol,
    ]

    if enable_standard_names:
        try:
            standard_protocol = event.standard_protocols.first().standard_name
        except AttributeError:
            standard_protocol = ""

        if standard_protocol:
            series_data += [standard_protocol]
        else:
            series_data += [""]

    series_data += [
        compression_thickness,
        radiological_thickness,
        compression_force,
        magnification_factor,
        collimated_field_area,
        exposure_control_mode,
        anode_target_material,
        filters,
        filter_thicknesses,
        focal_spot_size,
        kvp,
        average_xray_tube_current,
        exposure_time,
        exposure,
        event.entrance_exposure_at_rp,
        average_glandular_dose,
        event.percent_fibroglandular_tissue,
        event.comment,
    ]
    return series_data


def exportMG2csv(filterdict, pid=False, name=None, patid=None, user=None):
    """
    Export filtered mammography database data to a single-sheet CSV file.

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves csv file into Media directory for user to download
    """

    from remapp.models import GeneralStudyModuleAttr
    from ..interface.mod_filters import (
        MGSummaryListFilter,
        MGFilterPlusPid,
        MGFilterPlusStdNames,
        MGFilterPlusPidPlusStdNames,
    )

    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    datestamp = datetime.datetime.now()
    export_type = "CSV export"
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="MG",
        export_type=export_type,
        date_stamp=datestamp,
        pid=bool(pid and (name or patid)),
        user=user,
        filters_dict=filterdict,
    )

    tmpfile, writer = create_csv(tsk)
    if not tmpfile:
        exit()

    # Resetting the ordering key to avoid duplicates
    if isinstance(filterdict, dict):
        if (
            "o" in filterdict
            and filterdict["o"] == "-projectionxrayradiationdose__accumxraydose__"
            "accummammographyxraydose__accumulated_age_glandular_dose"
        ):
            logger.info("Replacing AGD ordering with study date to avoid duplication")
            filterdict["o"] = "-time_date"

    # Get the data!
    if pid:
        if enable_standard_names:
            df_filtered_qs = MGFilterPlusPidPlusStdNames(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="MG"
                ).distinct(),
            )
        else:
            df_filtered_qs = MGFilterPlusPid(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="MG"
                ).distinct(),
            )
    else:
        if enable_standard_names:
            df_filtered_qs = MGFilterPlusStdNames(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="MG"
                ).distinct(),
            )
        else:
            df_filtered_qs = MGSummaryListFilter(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="MG"
                ).distinct(),
            )

    studies = df_filtered_qs.qs

    tsk.progress = "Required study filter complete."
    tsk.save()

    tsk.num_records = studies.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    headings = common_headers(modality="MG", pid=pid, name=name, patid=patid)
    headings += [
        "View",
        "View Modifier",
        "Laterality",
        "Acquisition",
    ]

    if enable_standard_names:
        headings += ["Standard acquisition name"]

    headings += [
        "Thickness",
        "Radiological thickness",
        "Force",
        "Mag",
        "Area",
        "Mode",
        "Target",
        "Filter",
        "Filter thickness",
        "Focal spot size",
        "kVp",
        "mA",
        "ms",
        "uAs",
        "ESD",
        "AGD",
        "% Fibroglandular tissue",
        "Exposure mode description",
    ]

    writer.writerow(headings)

    max_events = 0
    for study_index, exam in enumerate(studies):
        tsk.progress = "{0} of {1}".format(study_index + 1, tsk.num_records)
        tsk.save()

        try:
            common_exam_data = get_common_data(
                "MG", exam, pid=pid, name=name, patid=patid
            )

            this_study_max_events = 0
            for (
                series
            ) in exam.projectionxrayradiationdose_set.get().irradeventxraydata_set.order_by(
                "id"
            ):
                this_study_max_events += 1
                if this_study_max_events > max_events:
                    max_events = this_study_max_events
                series_data = _mg_get_series_data(series)

                series_data = list(common_exam_data) + series_data
                for index, item in enumerate(series_data):
                    if item is None:
                        series_data[index] = ""
                    if isinstance(item, str) and "," in item:
                        series_data[index] = item.replace(",", ";")
                writer.writerow([str(data_string) for data_string in series_data])

        except ObjectDoesNotExist:
            error_message = (
                "DoesNotExist error whilst exporting study {0} of {1},  study UID {2}, accession number"
                " {3} - maybe database entry was deleted as part of importing later version of same"
                " study?".format(
                    study_index + 1,
                    tsk.num_records,
                    exam.study_instance_uid,
                    exam.accession_number,
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

def mgxlsx(filterdict, pid=False, name=None, patid=None, user=None):
    """Export filtered MG database data to multi-sheet Microsoft XSLX files

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves xlsx file into Media directory for user to download
    """
    modality = "MG"

    # Exam-level integer field names and friendly names
    exam_int_fields = [
        "pk",
        "number_of_events",
    ]
    exam_int_field_names = [
        "pk",
        "Number of events"
    ]

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
        "projectionxrayradiationdose__irradeventxraydata__image_view__code_meaning",
        'projectionxrayradiationdose__irradeventxraydata__imageviewmodifier__image_view_modifier__code_meaning',
        "projectionxrayradiationdose__irradeventxraydata__laterality__code_meaning",
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
        "View",
        "View Modifier",
        "Laterality",
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
        "total_agd_both",
    ]
    exam_val_field_names = [
        "Patient age",
        "Patient height (m)",
        "Patient weight (kg)",
        "Total AGD",
    ]

    acquisition_int_fields = [
        "projectionxrayradiationdose__irradeventxraydata__pk",
    ]
    acquisition_int_field_names = [
        "Acquisition pk",
    ]

    filter_materials = {
        "Aluminum": "Al",
        "Copper": "Cu",
        "Tantalum": "Ta",
        "Molybdenum": "Mo",
        "Rhodium": "Rh",
        "Silver": "Ag",
        "Niobium": "Nb",
        "Europium": "Eu",
        "Lead": "Pb"
    }

    cases = [When(projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__xrayfilters__xray_filter_material__code_meaning__icontains=material, then=Value(code))
         for material, code in filter_materials.items()]

    acquisition_cat_fields = [
        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__anode_target_material__code_meaning",
        Case(
            *cases,
            default=F('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__xrayfilters__xray_filter_material__code_meaning'),
         output_field=TextField()),
        "projectionxrayradiationdose__irradeventxraydata__comment",
    ]

    acquisition_cat_field_names = [
        "Acquisition protocol",
        "Target",
        "Filter",
        "Exposure mode description",
    ]

    acquisition_cat_field_std_name = "projectionxrayradiationdose__irradeventxraydata__standard_protocols__standard_name"
    acquisition_cat_field_name_std_name = "Standard acquisition name"

    # Required acquisition-level value field names and friendly names
    acquisition_val_fields = [
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__doserelateddistancemeasurements__radiological_thickness",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_force",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__magnification_factor",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__collimated_field_area",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure_control_mode",
        Coalesce(
            F('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__xrayfilters__xray_filter_thickness_minimum'),
            F('projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum'),
        ),
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__focal_spot_size",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_xray_tube_current",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure_time",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
        "projectionxrayradiationdose__irradeventxraydata__entrance_exposure_at_rp",
        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
        "projectionxrayradiationdose__irradeventxraydata__percent_fibroglandular_tissue",
    ]
    acquisition_val_field_names = [
        "Thickness",
        "Radiological thickness",
        "Force",
        "Mag",
        "Area",
        "Mode",
        "Filter thickness",
        "Focal spot size",
        "kVp",
        "mA",
        "ms",
        "uAs",
        "ESD",
        "AGD",
        "% Fibroglandular tissue",
    ]

    ct_dose_check_fields = [
    ]

    ct_dose_check_field_names = [
    ]

    # Fields for obtaining the acquisition protocols in the data
    fields_for_acquisition_frequency = [
        "projectionxrayradiationdose__irradeventxraydata__pk",
        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
    ]

    field_names_for_acquisition_frequency = [
        "pk",
        "Acquisition protocol"
    ]
    field_for_acquisition_frequency_std_name = "projectionxrayradiationdose__irradeventxraydata__standard_protocols__standard_name"
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
    study_pks = mg_acq_filter(filterdict, pid=pid).qs.values("pk")

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

    export_using_pandas(acquisition_cat_field_name_std_name, acquisition_cat_field_names,
                        acquisition_cat_field_std_name, acquisition_cat_fields,acquisition_int_field_names,
                        acquisition_int_fields, acquisition_val_field_names, acquisition_val_fields, book,
                        ct_dose_check_field_names, ct_dose_check_fields, datestamp, enable_standard_names,
                        exam_cat_field_names, exam_cat_fields, exam_date_field_names, exam_date_fields,
                        exam_int_field_names, exam_int_fields, exam_obj_field_names, exam_obj_fields,
                        exam_time_field_names, exam_time_fields, exam_val_field_names, exam_val_fields,
                        field_for_acquisition_frequency_std_name, field_name_for_acquisition_frequency_std_name,
                        field_names_for_acquisition_frequency, fields_for_acquisition_frequency, modality, n_entries,
                        name, patid, pid, qs, tmpxlsx, tsk)
    
def exportMG2excel(filterdict, pid=False, name=None, patid=None, user=None):
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-locals
    """
    Export filtered mammography database data to a multi sheet xlsx file.

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves csv file into Media directory for user to download
    """

    from remapp.models import GeneralStudyModuleAttr
    from ..interface.mod_filters import (
        MGSummaryListFilter,
        MGFilterPlusPid,
        MGFilterPlusStdNames,
        MGFilterPlusPidPlusStdNames,
    )

    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    datestamp = datetime.datetime.now()
    export_type = "XLSX export"
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="MG",
        export_type=export_type,
        date_stamp=datestamp,
        pid=bool(pid and (name or patid)),
        user=user,
        filters_dict=filterdict,
    )

    tmpfile, book = create_xlsx(tsk)
    if not tmpfile:
        exit()

    # Resetting the ordering key to avoid duplicates
    if isinstance(filterdict, dict):
        if (
            "o" in filterdict
            and filterdict["o"] == "-projectionxrayradiationdose__accumxraydose__"
            "accummammographyxraydose__accumulated_average_glandular_dose"
        ):
            logger.info("Replacing AGD ordering with study date to avoid duplication")
            filterdict["o"] = "-time_date"

    # Get the data!
    if pid:
        if enable_standard_names:
            df_filtered_qs = MGFilterPlusPidPlusStdNames(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="MG"
                ).distinct(),
            )
        else:
            df_filtered_qs = MGFilterPlusPid(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="MG"
                ).distinct(),
            )
    else:
        if enable_standard_names:
            df_filtered_qs = MGFilterPlusStdNames(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="MG"
                ).distinct(),
            )
        else:
            df_filtered_qs = MGSummaryListFilter(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="MG"
                ).distinct(),
            )

    studies = df_filtered_qs.qs

    tsk.progress = "Required study filter complete."
    tsk.save()

    tsk.num_records = studies.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    # Add summary sheet and all data sheet
    summarysheet = book.add_worksheet("Summary")
    wsalldata = book.add_worksheet("All data")
    book = text_and_date_formats(
        book, wsalldata, pid=pid, name=name, patid=patid, modality="MG"
    )

    headings = common_headers(modality="MG", pid=pid, name=name, patid=patid)
    all_data_headings = list(headings)
    headings += [
        "View",
        "View Modifier",
        "Laterality",
        "Acquisition",
    ]

    if enable_standard_names:
        headings += ["Standard acquisition name"]

    headings += [
        "Thickness",
        "Radiological thickness",
        "Force",
        "Mag",
        "Area",
        "Mode",
        "Target",
        "Filter",
        "Filter thickness",
        "Focal spot size",
        "kVp",
        "mA",
        "ms",
        "uAs",
        "ESD",
        "AGD",
        "% Fibroglandular tissue",
        "Exposure mode description",
    ]

    # Generate list of protocols in queryset and create worksheets for each
    tsk.progress = "Generating list of protocols in the dataset..."
    tsk.save()

    tsk.progress = "Creating an Excel safe version of protocol names and creating a worksheet for each..."
    tsk.save()

    book, sheet_list = generate_sheets(
        studies, book, headings, modality="MG", pid=pid, name=name, patid=patid
    )

    max_events = 0
    for study_index, exam in enumerate(studies):
        tsk.progress = "{0} of {1}".format(study_index + 1, tsk.num_records)
        tsk.save()

        try:
            common_exam_data = get_common_data(
                "MG", exam, pid=pid, name=name, patid=patid
            )
            all_exam_data = list(common_exam_data)

            this_study_max_events = 0
            for (
                series
            ) in exam.projectionxrayradiationdose_set.get().irradeventxraydata_set.order_by(
                "id"
            ):
                this_study_max_events += 1
                if this_study_max_events > max_events:
                    max_events = this_study_max_events
                series_data = _mg_get_series_data(series)

                all_exam_data += series_data  # For all data

                protocol = series.acquisition_protocol
                if not protocol:
                    protocol = "Unknown"
                tabtext = sheet_name(protocol)
                sheet_list[tabtext]["count"] += 1
                try:
                    sheet_list[tabtext]["sheet"].write_row(
                        sheet_list[tabtext]["count"],
                        0,
                        common_exam_data + series_data,
                    )
                except TypeError:
                    logger.error(
                        "Common is |{0}| series is |{1}|".format(
                            common_exam_data, series_data
                        )
                    )
                    exit()

                if enable_standard_names:
                    try:
                        protocol = series.standard_protocols.first().standard_name
                        if protocol:
                            tabtext = sheet_name("[standard] " + protocol)
                            sheet_list[tabtext]["count"] += 1
                            try:
                                sheet_list[tabtext]["sheet"].write_row(
                                    sheet_list[tabtext]["count"],
                                    0,
                                    common_exam_data + series_data,
                                )
                            except TypeError:
                                logger.error(
                                    "Common is |{0}| series is |{1}|".format(
                                        common_exam_data, series_data
                                    )
                                )
                                exit()
                    except AttributeError:
                        pass

            wsalldata.write_row(study_index + 1, 0, all_exam_data)

        except ObjectDoesNotExist:
            error_message = (
                "DoesNotExist error whilst exporting study {0} of {1},  study UID {2}, accession number"
                " {3} - maybe database entry was deleted as part of importing later version of same"
                " study?".format(
                    study_index + 1,
                    tsk.num_records,
                    exam.study_instance_uid,
                    exam.accession_number,
                )
            )
            logger.error(error_message)

            wsalldata.write(study_index + 1, 0, error_message)

    all_data_headings += _series_headers(max_events)
    wsalldata.write_row("A1", all_data_headings)
    numrows = studies.count()
    wsalldata.autofilter(0, 0, numrows, len(all_data_headings) - 1)
    create_summary_sheet(tsk, studies, book, summarysheet, sheet_list)

    tsk.progress = "All study data written."
    tsk.save()

    book.close()
    export_filename = f'mgexport{datestamp.strftime("%Y%m%d-%H%M%S%f")}.xlsx'
    write_export(tsk, export_filename, tmpfile, datestamp)
