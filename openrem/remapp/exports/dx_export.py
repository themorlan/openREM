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
..  module:: dx_export.
    :synopsis: Module to export radiographic data to single-sheet CSV files and to multi-sheet XLSX files.

..  moduleauthor:: David Platten and Ed McDonagh

"""
import datetime
import logging
from openrem.remapp.tools.background import get_or_generate_task_uuid

from django.core.exceptions import ObjectDoesNotExist

from remapp.models import StandardNameSettings

from .export_common import (
    text_and_date_formats,
    common_headers,
    generate_sheets,
    sheet_name,
    get_common_data,
    get_xray_filter_info,
    create_xlsx,
    create_csv,
    write_export,
    create_summary_sheet,
    get_pulse_data,
    abort_if_zero_studies,
    create_export_task,
)

logger = logging.getLogger(__name__)


def _get_source_data(series_table):
    """Return source data

    :param series_table:  irradeventxraydata_set
    :return: dict of source data
    """
    try:
        source_data = series_table.irradeventxraysourcedata_set.get()
        exposure_control_mode = source_data.exposure_control_mode
        average_xray_tube_current = source_data.average_xray_tube_current
        exposure_time = source_data.exposure_time
        pulse_data = get_pulse_data(source_data=source_data, modality="DX")
        kvp = pulse_data["kvp"]
        mas = pulse_data["mas"]
        filters, filter_thicknesses = get_xray_filter_info(source_data)
        grid_focal_distance = source_data.grid_focal_distance
    except ObjectDoesNotExist:
        exposure_control_mode = None
        average_xray_tube_current = None
        exposure_time = None
        kvp = None
        mas = None
        filters = None
        filter_thicknesses = None
        grid_focal_distance = None
    return {
        "exposure_control_mode": exposure_control_mode,
        "average_xray_tube_current": average_xray_tube_current,
        "exposure_time": exposure_time,
        "kvp": kvp,
        "mas": mas,
        "filters": filters,
        "filter_thicknesses": filter_thicknesses,
        "grid_focal_distance": grid_focal_distance,
    }


def _get_detector_data(series_table):
    """Return detector data

    :param series_table: irradeventxraydata_set
    :return: dict of detector data
    """
    try:
        detector_data = series_table.irradeventxraydetectordata_set.get()
        exposure_index = detector_data.exposure_index
        target_exposure_index = detector_data.target_exposure_index
        deviation_index = detector_data.deviation_index
        relative_xray_exposure = detector_data.relative_xray_exposure
    except ObjectDoesNotExist:
        exposure_index = None
        target_exposure_index = None
        deviation_index = None
        relative_xray_exposure = None
    return {
        "exposure_index": exposure_index,
        "target_exposure_index": target_exposure_index,
        "deviation_index": deviation_index,
        "relative_xray_exposure": relative_xray_exposure,
    }


def _get_distance_data(series_table):
    """Return distance data

    :param series_table: irradeventxraydata_set
    :return: dict of distance data
    """
    try:
        distances = (
            series_table.irradeventxraymechanicaldata_set.get().doserelateddistancemeasurements_set.get()
        )
        distance_source_to_detector = distances.distance_source_to_detector
        distance_source_to_entrance_surface = (
            distances.distance_source_to_entrance_surface
        )
        distance_source_to_isocenter = distances.distance_source_to_isocenter
        table_height_position = distances.table_height_position
    except ObjectDoesNotExist:
        distance_source_to_detector = None
        distance_source_to_entrance_surface = None
        distance_source_to_isocenter = None
        table_height_position = None
    return {
        "distance_source_to_detector": distance_source_to_detector,
        "distance_source_to_entrance_surface": distance_source_to_entrance_surface,
        "distance_source_to_isocenter": distance_source_to_isocenter,
        "table_height_position": table_height_position,
    }


def _series_headers(max_events):
    """Return the series headers common to both DX exports

    :param max_events: number of series
    :return: headers as a list of strings
    """

    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

    series_headers = []
    for series_number in range(int(max_events)):
        series_headers += ["E" + str(series_number + 1) + " Protocol"]

        if enable_standard_names:
            series_headers += [
                "E" + str(series_number + 1) + " Standard acquisition name"
            ]

        series_headers += [
            "E" + str(series_number + 1) + " Anatomy",
            "E" + str(series_number + 1) + " Image view",
            "E" + str(series_number + 1) + " Exposure control mode",
            "E" + str(series_number + 1) + " kVp",
            "E" + str(series_number + 1) + " mAs",
            "E" + str(series_number + 1) + " mA",
            "E" + str(series_number + 1) + " Exposure time (ms)",
            "E" + str(series_number + 1) + " Filters",
            "E" + str(series_number + 1) + " Filter thicknesses (mm)",
            "E" + str(series_number + 1) + " Exposure index",
            "E" + str(series_number + 1) + " Target exposure index",
            "E" + str(series_number + 1) + " Deviation index",
            "E" + str(series_number + 1) + " Relative x-ray exposure",
            "E" + str(series_number + 1) + " DAP (cGy.cm^2)",
            "E" + str(series_number + 1) + " Entrance Exposure at RP (mGy)",
            "E" + str(series_number + 1) + " SDD Detector Dist",
            "E" + str(series_number + 1) + " SPD Patient Dist",
            "E" + str(series_number + 1) + " SIsoD Isocentre Dist",
            "E" + str(series_number + 1) + " Table Height",
            "E" + str(series_number + 1) + " Comment",
        ]
    return series_headers


def _dx_get_series_data(s):
    """Return the series level data

    :param s: series
    :return: series data
    """

    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

    source_data = _get_source_data(s)
    detector_data = _get_detector_data(s)

    cgycm2 = s.convert_gym2_to_cgycm2()
    entrance_exposure_at_rp = s.entrance_exposure_at_rp

    distances = _get_distance_data(s)

    try:
        anatomical_structure = s.anatomical_structure.code_meaning
    except AttributeError:
        anatomical_structure = ""

    series_data = [s.acquisition_protocol]

    if enable_standard_names:
        try:
            standard_protocol = s.standard_protocols.first().standard_name
        except AttributeError:
            standard_protocol = ""

        if standard_protocol:
            series_data += [standard_protocol]
        else:
            series_data += [""]

    series_data += [anatomical_structure]

    try:
        series_data += [s.image_view.code_meaning]
    except AttributeError:
        series_data += [None]
    series_data += [
        source_data["exposure_control_mode"],
        source_data["kvp"],
        source_data["mas"],
        source_data["average_xray_tube_current"],
        source_data["exposure_time"],
        source_data["filters"],
        source_data["filter_thicknesses"],
        detector_data["exposure_index"],
        detector_data["target_exposure_index"],
        detector_data["deviation_index"],
        detector_data["relative_xray_exposure"],
        cgycm2,
        entrance_exposure_at_rp,
        distances["distance_source_to_detector"],
        distances["distance_source_to_entrance_surface"],
        distances["distance_source_to_isocenter"],
        distances["table_height_position"],
        s.comment,
    ]
    return series_data


def exportDX2excel(filterdict, pid=False, name=None, patid=None, user=None):
    """Export filtered DX database data to a single-sheet CSV file.

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves csv file into Media directory for user to download
    """

    from ..interface.mod_filters import dx_acq_filter

    datestamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="DX",
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
    e = dx_acq_filter(filterdict, pid=pid).qs

    tsk.progress = "Required study filter complete."
    tsk.save()

    tsk.num_records = e.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    tsk.progress = "{0} studies in query.".format(tsk.num_records)
    tsk.num_records = tsk.num_records
    tsk.save()

    headers = common_headers(pid=pid, name=name, patid=patid)
    headers += ["DAP total (cGy.cm^2)"]

    from django.db.models import Max

    max_events_dict = e.aggregate(
        Max(
            "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__"
            "total_number_of_radiographic_frames"
        )
    )
    max_events = max_events_dict[
        "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__"
        "total_number_of_radiographic_frames__max"
    ]
    if not max_events:
        max_events = 1

    headers += _series_headers(max_events)

    writer.writerow(headers)

    tsk.progress = "CSV header row written."
    tsk.save()

    for row, exams in enumerate(e):
        tsk.progress = "Writing {0} of {1} to csv file".format(row + 1, tsk.num_records)
        tsk.save()
        try:
            exam_data = get_common_data("DX", exams, pid=pid, name=name, patid=patid)
            for (
                s
            ) in exams.projectionxrayradiationdose_set.get().irradeventxraydata_set.order_by(
                "id"
            ):
                # Get series data
                exam_data += _dx_get_series_data(s)
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
                    row + 1,
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


def dxxlsx(filterdict, pid=False, name=None, patid=None, user=None):
    """Export filtered DX and CR database data to multi-sheet Microsoft XSLX files.

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves xlsx file into Media directory for user to download
    """

    from ..interface.mod_filters import dx_acq_filter

    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

    datestamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="DX",
        export_type="XLSX export",
        date_stamp=datestamp,
        pid=bool(pid and (name or patid)),
        user=user,
        filters_dict=filterdict,
    )

    tmpxlsx, book = create_xlsx(tsk)
    if not tmpxlsx:
        exit()

    e = dx_acq_filter(filterdict, pid=pid).qs

    tsk.num_records = e.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    # Add summary sheet and all data sheet
    summarysheet = book.add_worksheet("Summary")
    wsalldata = book.add_worksheet("All data")

    book = text_and_date_formats(
        book, wsalldata, pid=pid, name=name, patid=patid, modality="DX"
    )

    # Some prep
    commonheaders = common_headers(pid=pid, name=name, patid=patid)
    commonheaders += ["DAP total (cGy.cm^2)"]

    protocolheaders = commonheaders + ["Protocol"]

    if enable_standard_names:
        protocolheaders += ["Standard acquisition name"]

    protocolheaders = protocolheaders + [
        "Anatomy",
        "Image view",
        "Exposure control mode",
        "kVp",
        "mAs",
        "mA",
        "Exposure time (ms)",
        "Filters",
        "Filter thicknesses (mm)",
        "Exposure index",
        "Target exposure index",
        "Deviation index",
        "Relative x-ray exposure",
        "DAP (cGy.cm^2)",
        "Entrance exposure at RP",
        "SDD Detector Dist",
        "SPD Patient Dist",
        "SIsoD Isocentre Dist",
        "Table Height",
        "Comment",
    ]

    # Generate list of protocols in queryset and create worksheets for each
    tsk.progress = "Generating list of protocols in the dataset..."
    tsk.save()

    tsk.progress = "Creating an Excel safe version of protocol names and creating a worksheet for each..."
    tsk.save()

    book, sheet_list = generate_sheets(
        e, book, protocolheaders, modality="DX", pid=pid, name=name, patid=patid
    )

    ##################
    # All data sheet

    from django.db.models import Max

    max_events_dict = e.aggregate(
        Max(
            "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__"
            "total_number_of_radiographic_frames"
        )
    )
    max_events = max_events_dict[
        "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__"
        "total_number_of_radiographic_frames__max"
    ]
    if not max_events:
        max_events = 1

    alldataheaders = list(commonheaders)

    tsk.progress = "Generating headers for the all data sheet..."
    tsk.save()

    alldataheaders += _series_headers(max_events)
    wsalldata.write_row("A1", alldataheaders)
    numrows = e.count()
    wsalldata.autofilter(0, 0, numrows, len(alldataheaders) - 1)

    for row, exams in enumerate(e):

        tsk.progress = "Writing study {0} of {1} to All data sheet and individual protocol sheets".format(
            row + 1, numrows
        )
        tsk.save()

        try:
            common_exam_data = get_common_data(
                "DX", exams, pid=pid, name=name, patid=patid
            )
            all_exam_data = list(common_exam_data)

            for (
                s
            ) in exams.projectionxrayradiationdose_set.get().irradeventxraydata_set.order_by(
                "id"
            ):
                # Get series data
                series_data = _dx_get_series_data(s)

                # Add series to all data
                all_exam_data += series_data

                # Add series data to series tab
                protocol = s.acquisition_protocol
                if not protocol:
                    protocol = "Unknown"
                tabtext = sheet_name(protocol)
                sheet_list[tabtext]["count"] += 1
                sheet_list[tabtext]["sheet"].write_row(
                    sheet_list[tabtext]["count"], 0, common_exam_data + series_data
                )

                if enable_standard_names:
                    try:
                        protocol = s.standard_protocols.first().standard_name
                        if protocol:
                            tabtext = sheet_name("[standard] " + protocol)
                            sheet_list[tabtext]["count"] += 1
                            sheet_list[tabtext]["sheet"].write_row(
                                sheet_list[tabtext]["count"],
                                0,
                                common_exam_data + series_data,
                            )
                    except AttributeError:
                        pass

            wsalldata.write_row(row + 1, 0, all_exam_data)

        except ObjectDoesNotExist:
            error_message = (
                "DoesNotExist error whilst exporting study {0} of {1},  study UID {2}, accession number"
                " {3} - maybe database entry was deleted as part of importing later version of same"
                " study?".format(
                    row + 1, numrows, exams.study_instance_uid, exams.accession_number
                )
            )
            logger.error(error_message)
            wsalldata.write(row + 1, 0, error_message)

    create_summary_sheet(tsk, e, book, summarysheet, sheet_list)

    book.close()
    tsk.progress = "XLSX book written."
    tsk.save()

    xlsxfilename = "dxexport{0}.xlsx".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))

    write_export(tsk, xlsxfilename, tmpxlsx, datestamp)


def dx_phe_2019(filterdict, user=None, projection=True, bespoke=False):
    """Export filtered DX database data in the format for the 2019 Public Health England DX dose survey

    :param filterdict: Queryset of studies to export
    :param user:  User that has started the export
    :param projection: projection export if True, study export if False
    :param bespoke: for study export, are there more than six projections
    :return: Saves Excel file into Media directory for user to download
    """

    from .export_common import get_patient_study_data
    from ..interface.mod_filters import dx_acq_filter

    datestamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="DX",
        export_type="PHE DX 2019 export",
        date_stamp=datestamp,
        pid=False,
        user=user,
        filters_dict=filterdict,
    )

    tmp_xlsx, book = create_xlsx(tsk)
    if not tmp_xlsx:
        exit()

    exams = dx_acq_filter(filterdict, pid=False).qs

    tsk.num_records = exams.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    tsk.progress = "{0} studies in query.".format(tsk.num_records)
    tsk.save()

    columns_a_d = ["", "PHE Record No", "Contributor's record ID", "Exam date"]
    column_e_projection = ["Projection DAP dose"]
    column_e_study = ["Study DAP dose"]
    columns_f_m = [
        "DAP dose units",
        "Protocol name",
        "Patient weight",
        "",
        "",
        "Patient age",
        "Sex",
        "Height",
    ]
    study_num_projections = ["number of projections"]

    per_projection_headings = [
        "Detector used",
        "Grid used",
        "FDD",
        "Filtration in mm Al",
        "AEC used",
        "kVp",
        "mAs",
        "Patient position",
        "Detector in bucky",
        "Other projection info",
    ]
    final_columns = [
        "Additional one",
        "Additional two",
        "Additional three",
        "Additional four",
        "SNOMED CT code",
        "NICIP code",
        "Variation in dose collection",
        "Other information, comments",
    ]
    if projection:
        sheet = book.add_worksheet("PHE DX 2019 Single Projection")
        headings = (
            columns_a_d
            + column_e_projection
            + columns_f_m
            + per_projection_headings
            + final_columns
        )
    else:
        if bespoke:
            event_columns = 20
        else:
            event_columns = 6
        sheet = book.add_worksheet("PHE DX 2019 Exam")
        headings = columns_a_d + column_e_study + columns_f_m + study_num_projections
        for x in range(event_columns):
            headings += ["Projection {0} DAP".format(x + 1)]
        for x in range(event_columns):
            headings += ["Projection {0} Name".format(x + 1)]
            headings += per_projection_headings
        headings += final_columns
    sheet.write_row(0, 0, headings)

    num_rows = exams.count()
    for row, exam in enumerate(exams):
        tsk.progress = "Writing study {0} of {1}".format(row + 1, num_rows)
        tsk.save()

        try:
            projection_events = exam.projectionxrayradiationdose_set.get().irradeventxraydata_set.order_by(
                "id"
            )
        except ObjectDoesNotExist:
            logger.error(
                "Failed to export study to PHE 2019 DX as had no event data! PK={0}".format(
                    exam.pk
                )
            )
            continue

        patient_study_data = get_patient_study_data(exam)
        patient_sex = None
        try:
            patient_module = exam.patientmoduleattr_set.get()
            patient_sex = patient_module.patient_sex
        except ObjectDoesNotExist:
            logger.debug(
                "Export {0}; patientmoduleattr_set object does not exist. AccNum {1}, Date {2}".format(
                    "PHE 2019 DX", exams.accession_number, exams.study_date
                )
            )
        row_data = ["", row + 1, exam.pk, exam.study_date]
        if not projection:
            row_data += [exam.dap_total_cgycm2()]
        else:
            row_data += [projection_events[0].convert_gym2_to_cgycm2()]
        row_data += ["cGy·cm²"]

        exam_name_text = (
            f"{exam.procedure_code_meaning} | {exam.requested_procedure_code_meaning}"
            f" | {exam.study_description}"
        )
        if projection:
            exam_name_text = (
                f"{exam_name_text} | {projection_events[0].acquisition_protocol}"
            )
        row_data += [exam_name_text]

        row_data += [
            patient_study_data["patient_weight"],
            "",
            "",
            patient_study_data["patient_age_decimal"],
            patient_sex,
            patient_study_data["patient_size"],
        ]

        if not projection:
            row_data += [exam.number_of_events]
            for x in range(event_columns):
                try:
                    row_data += [projection_events[x].convert_gym2_to_cgycm2()]
                except IndexError:
                    row_data += [""]

        for event in projection_events:
            source_data = _get_source_data(event)
            if source_data["filters"] is not None:
                filters = (
                    f"{source_data['filters']} {source_data['filter_thicknesses']}"
                )
            else:
                filters = ""

            detector_data = _get_detector_data(event)
            distances = _get_distance_data(event)

            try:
                image_view = event.image_view.code_meaning
            except AttributeError:
                image_view = None
            try:
                pt_orientation = event.patient_orientation_cid.code_meaning
            except AttributeError:
                pt_orientation = None
            try:
                pt_orientation_mod = event.patient_orientation_modifier_cid.code_meaning
            except AttributeError:
                pt_orientation_mod = None
            try:
                pt_table_rel = event.patient_table_relationship_cid.code_meaning
            except AttributeError:
                pt_table_rel = None

            pt_position = ""
            if pt_orientation:
                pt_position = "{0}{1}".format(pt_position, pt_orientation)
            if pt_orientation_mod:
                pt_position = "{0}, {1}".format(pt_position, pt_orientation_mod)
            if pt_table_rel:
                pt_position = "{0}, {1}".format(pt_position, pt_table_rel)

            if not projection:
                row_data += [event.acquisition_protocol]
            sdd = ""
            if distances["distance_source_to_detector"]:
                sdd = distances["distance_source_to_detector"] / 10
            row_data += [
                "",
                source_data["grid_focal_distance"],
                sdd,
                filters,
                source_data["exposure_control_mode"],
                source_data["kvp"],
                source_data["mas"],
                pt_position,
                "",
            ]
            other_info = ""
            if detector_data["exposure_index"]:
                other_info = "EI: {0}".format(round(detector_data["exposure_index"], 2))
            if image_view:
                other_info = "{0} {1}".format(other_info, image_view)
            row_data += [other_info]
            if projection:
                break

        sheet.write_row(row + 1, 0, row_data)

    book.close()
    tsk.progress = "PHE DX 2019 export complete"
    tsk.save()

    xlsxfilename = "PHE_DX_2019_{0}.xlsx".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))

    write_export(tsk, xlsxfilename, tmp_xlsx, datestamp)
