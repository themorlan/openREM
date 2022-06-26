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
from openrem.remapp.tools.background import get_or_generate_task_uuid

from django.core.exceptions import ObjectDoesNotExist

from remapp.models import StandardNameSettings

from .export_common import (
    common_headers,
    text_and_date_formats,
    generate_sheets,
    create_summary_sheet,
    get_common_data,
    get_anode_target_material,
    get_xray_filter_info,
    create_csv,
    create_xlsx,
    write_export,
    sheet_name,
    abort_if_zero_studies,
    create_export_task,
)

logger = logging.getLogger(__name__)


def _series_headers(max_events):
    """Return a list of series headers

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
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

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


def exportMG2excel(filterdict, pid=False, name=None, patid=None, user=None, xlsx=False):
    """
    Export filtered mammography database data to a single-sheet CSV file or a multi sheet xlsx file.

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :param xlsx: Whether to export a single sheet csv or a multi sheet xlsx
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
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

    datestamp = datetime.datetime.now()
    if xlsx:
        export_type = "XLSX export"
    else:
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

    if xlsx:
        tmpfile, book = create_xlsx(tsk)
        if not tmpfile:
            exit()
    else:
        tmpfile, writer = create_csv(tsk)
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
            filterdict["o"] = "-study_date"

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

    if xlsx:
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

    if not xlsx:
        writer.writerow(headings)
    else:
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
                if not xlsx:
                    series_data = list(common_exam_data) + series_data
                    for index, item in enumerate(series_data):
                        if item is None:
                            series_data[index] = ""
                        if isinstance(item, str) and "," in item:
                            series_data[index] = item.replace(",", ";")
                    writer.writerow([str(data_string) for data_string in series_data])
                else:
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

            if xlsx:
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
            if xlsx:
                wsalldata.write(study_index + 1, 0, error_message)
            else:
                writer.writerow([error_message])

    if xlsx:
        all_data_headings += _series_headers(max_events)
        wsalldata.write_row("A1", all_data_headings)
        numrows = studies.count()
        wsalldata.autofilter(0, 0, numrows, len(all_data_headings) - 1)
        create_summary_sheet(tsk, studies, book, summarysheet, sheet_list)

    tsk.progress = "All study data written."
    tsk.save()

    if xlsx:
        book.close()
        export_filename = f'mgexport{datestamp.strftime("%Y%m%d-%H%M%S%f")}.xlsx'
        write_export(tsk, export_filename, tmpfile, datestamp)
    else:
        tmpfile.close()
        tsk.status = "COMPLETE"
        tsk.processtime = (datetime.datetime.now() - datestamp).total_seconds()
        tsk.save()
