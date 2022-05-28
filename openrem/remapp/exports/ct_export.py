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

from django.core.exceptions import ObjectDoesNotExist
from django.utils.translation import gettext as _
from openrem.remapp.tools.background import get_or_generate_task_uuid

from remapp.models import StandardNameSettings

from .export_common import (
    get_common_data,
    common_headers,
    create_xlsx,
    create_csv,
    write_export,
    create_summary_sheet,
    abort_if_zero_studies,
    create_export_task,
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

    import datetime
    from django.db.models import Max
    from .export_common import text_and_date_formats, generate_sheets, sheet_name
    from ..interface.mod_filters import ct_acq_filter

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
        modality="CT",
        export_type="XLSX_export",
        date_stamp=datestamp,
        pid=bool(pid and (name or patid)),
        user=user,
        filters_dict=filterdict,
    )

    tmpxlsx, book = create_xlsx(tsk)
    if not tmpxlsx:
        exit()

    # Get the data!
    e = ct_acq_filter(filterdict, pid=pid).qs

    tsk.num_records = e.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    # Add summary sheet and all data sheet
    summarysheet = book.add_worksheet("Summary")
    wsalldata = book.add_worksheet("All data")

    book = text_and_date_formats(
        book, wsalldata, pid=pid, name=name, patid=patid, modality="CT"
    )

    # Some prep
    commonheaders = common_headers(pid=pid, name=name, patid=patid)
    commonheaders += ["DLP total (mGy.cm)"]
    protocolheaders = commonheaders + ["Protocol"]

    if enable_standard_names:
        protocolheaders += ["Standard acquisition name"]

    protocolheaders = protocolheaders + [
        "Type",
        "Exposure time",
        "Scanning length",
        "Slice thickness",
        "Total collimation",
        "Pitch",
        "No. sources",
        "CTDIvol",
        "Phantom",
        "DLP",
        "S1 name",
        "S1 kVp",
        "S1 max mA",
        "S1 mA",
        "S1 Exposure time/rotation",
        "S2 name",
        "S2 kVp",
        "S2 max mA",
        "S2 mA",
        "S2 Exposure time/rotation",
        "mA Modulation type",
        "Dose check details",
        "Comments",
    ]

    # Generate list of protocols in queryset and create worksheets for each
    tsk.progress = "Generating list of protocols in the dataset..."
    tsk.save()

    book, sheet_list = generate_sheets(
        e, book, protocolheaders, modality="CT", pid=pid, name=name, patid=patid
    )

    max_events_dict = e.aggregate(
        Max(
            "ctradiationdose__ctaccumulateddosedata__total_number_of_irradiation_events"
        )
    )
    max_events = max_events_dict[
        "ctradiationdose__ctaccumulateddosedata__total_number_of_irradiation_events__max"
    ]

    alldataheaders = list(commonheaders)

    tsk.progress = "Generating headers for the all data sheet..."
    tsk.save()

    if not max_events:
        max_events = 1
    alldataheaders += _generate_all_data_headers_ct(max_events)

    wsalldata.write_row("A1", alldataheaders)
    numcolumns = len(alldataheaders) - 1
    numrows = e.count()
    wsalldata.autofilter(0, 0, numrows, numcolumns)

    for row, exams in enumerate(e):
        # Translators: CT xlsx export progress
        tsk.progress = _(
            "Writing study {row} of {numrows} to All data sheet and individual protocol sheets".format(
                row=row + 1, numrows=numrows
            )
        )
        # tsk.progress = f"Writing study {row + 1} of {numrows} to All data sheet and individual protocol sheets"
        tsk.save()

        try:
            common_exam_data = get_common_data("CT", exams, pid, name, patid)
            all_exam_data = list(common_exam_data)

            for (
                s
            ) in exams.ctradiationdose_set.get().ctirradiationeventdata_set.order_by(
                "id"
            ):
                # Get series data
                series_data = _ct_get_series_data(s)

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

                # Add series data to standard acquisition tab
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

    xlsxfilename = "ctexport{0}.xlsx".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))

    write_export(tsk, xlsxfilename, tmpxlsx, datestamp)


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
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

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
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

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
