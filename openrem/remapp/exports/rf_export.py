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
..  module:: rf_export.
    :synopsis: Module to export RF data from database to single sheet csv and multisheet xlsx.

..  moduleauthor:: Ed McDonagh

"""

import datetime
import logging
from openrem.remapp.tools.background import get_or_generate_task_uuid

from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Avg, Max, Min

from remapp.models import (
    GeneralStudyModuleAttr,
    IrradEventXRayData,
    StandardNameSettings,
)

from ..exports.export_common import (
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
    get_patient_study_data,
)
from ..interface.mod_filters import (
    RFSummaryListFilter,
    RFFilterPlusPid,
    RFFilterPlusStdNames,
    RFFilterPlusPidPlusStdNames,
)
from ..tools.get_values import return_for_export

logger = logging.getLogger(__name__)


def _get_accumulated_data(accumXrayDose):
    """Extract all the summary level data

    :param accumXrayDose: Accumulated x-ray radiation dose object
    :return: dict of summary level data
    """
    accum = {}
    accum["plane"] = accumXrayDose.acquisition_plane.code_meaning
    try:
        accumulated_integrated_projection_dose = (
            accumXrayDose.accumintegratedprojradiogdose_set.get()
        )
        accum[
            "dose_area_product_total"
        ] = accumulated_integrated_projection_dose.dose_area_product_total
        accum["dose_rp_total"] = accumulated_integrated_projection_dose.dose_rp_total
        accum[
            "reference_point_definition"
        ] = accumulated_integrated_projection_dose.reference_point_definition_code
        if not accum["reference_point_definition"]:
            accum[
                "reference_point_definition"
            ] = accumulated_integrated_projection_dose.reference_point_definition
    except ObjectDoesNotExist:
        accum["dose_area_product_total"] = None
        accum["dose_rp_total"] = None
        accum["reference_point_definition_code"] = None
    try:
        accumulated_projection_dose = accumXrayDose.accumprojxraydose_set.get()
        accum[
            "fluoro_dose_area_product_total"
        ] = accumulated_projection_dose.fluoro_dose_area_product_total
        accum["fluoro_dose_rp_total"] = accumulated_projection_dose.fluoro_dose_rp_total
        accum["total_fluoro_time"] = accumulated_projection_dose.total_fluoro_time
        accum[
            "acquisition_dose_area_product_total"
        ] = accumulated_projection_dose.acquisition_dose_area_product_total
        accum[
            "acquisition_dose_rp_total"
        ] = accumulated_projection_dose.acquisition_dose_rp_total
        accum[
            "total_acquisition_time"
        ] = accumulated_projection_dose.total_acquisition_time
    except ObjectDoesNotExist:
        accum["fluoro_dose_area_product_total"] = None
        accum["fluoro_dose_rp_total"] = None
        accum["total_fluoro_time"] = None
        accum["acquisition_dose_area_product_total"] = None
        accum["acquisition_dose_rp_total"] = None
        accum["total_acquisition_time"] = None

    try:
        accum["eventcount"] = int(
            accumXrayDose.projection_xray_radiation_dose.irradeventxraydata_set.filter(
                acquisition_plane__code_meaning__exact=accum["plane"]
            ).count()
        )
    except ObjectDoesNotExist:
        accum["eventcount"] = None

    return accum


def _add_plane_summary_data(exam):
    """Add plane level accumulated data to examdata

    :param exam: exam to export
    :return: list of summary data at plane level
    """
    exam_data = []
    for plane in (
        exam.projectionxrayradiationdose_set.get()
        .accumxraydose_set.all()
        .order_by("acquisition_plane__code_value")
    ):
        accum = _get_accumulated_data(plane)
        exam_data += [
            accum["dose_area_product_total"],
            accum["dose_rp_total"],
            accum["fluoro_dose_area_product_total"],
            accum["fluoro_dose_rp_total"],
            accum["total_fluoro_time"],
            accum["acquisition_dose_area_product_total"],
            accum["acquisition_dose_rp_total"],
            accum["total_acquisition_time"],
            accum["eventcount"],
        ]
        if "Single" in accum["plane"]:
            exam_data += ["", "", "", "", "", "", "", "", ""]

    return exam_data


def _get_series_data(event, filter_data):
    """Return series level data for protocol sheets

    :param event: event in question
    :return: list of data
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
        source_data = event.irradeventxraysourcedata_set.get()
        pulse_rate = source_data.pulse_rate
        ii_field_size = source_data.ii_field_size
        exposure_time = source_data.exposure_time
        dose_rp = source_data.dose_rp
        number_of_pulses = source_data.number_of_pulses
        irradiation_duration = source_data.irradiation_duration
        pulse_data = get_pulse_data(source_data=source_data, modality="RF")
        kVp = pulse_data["kvp"]
        xray_tube_current = pulse_data["xray_tube_current"]
        pulse_width = pulse_data["pulse_width"]
    except ObjectDoesNotExist:
        pulse_rate = None
        ii_field_size = None
        exposure_time = None
        dose_rp = None
        number_of_pulses = None
        irradiation_duration = None
        kVp = None
        xray_tube_current = None
        pulse_width = None
    try:
        mechanical_data = event.irradeventxraymechanicaldata_set.get()
        pos_primary_angle = mechanical_data.positioner_primary_angle
        pos_secondary_angle = mechanical_data.positioner_secondary_angle
    except ObjectDoesNotExist:
        pos_primary_angle = None
        pos_secondary_angle = None

    series_data = [
        str(event.date_time_started),
        event.irradiation_event_type.code_meaning,
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

    series_data = series_data + [
        event.acquisition_plane.code_meaning,
        ii_field_size,
        filter_data["filter_material"],
        filter_data["filter_thick"],
        kVp,
        xray_tube_current,
        pulse_width,
        pulse_rate,
        number_of_pulses,
        exposure_time,
        irradiation_duration,
        event.convert_gym2_to_cgycm2(),
        dose_rp,
        pos_primary_angle,
        pos_secondary_angle,
    ]

    return series_data


def _all_data_headers(pid=False, name=None, patid=None):
    """Compile list of column headers

    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :return: list of headers for all_data sheet and csv sheet
    """
    all_data_headers = common_headers(
        modality="RF", pid=pid, name=name, patid=patid
    ) + [
        "A DAP total (Gy.m^2)",
        "A Dose RP total (Gy)",
        "A Fluoro DAP total (Gy.m^2)",
        "A Fluoro dose RP total (Gy)",
        "A Fluoro duration total (s)",
        "A Acq. DAP total (Gy.m^2)",
        "A Acq. dose RP total (Gy)",
        "A Acq. duration total (s)",
        "A Number of events",
        "B DAP total (Gy.m^2)",
        "B Dose RP total (Gy)",
        "B Fluoro DAP total (Gy.m^2)",
        "B Fluoro dose RP total (Gy)",
        "B Fluoro duration total (s)",
        "B Acq. DAP total (Gy.m^2)",
        "B Acq. dose RP total (Gy)",
        "B Acq. duration total (s)",
        "B Number of events",
    ]
    return all_data_headers


def rfxlsx(filterdict, pid=False, name=None, patid=None, user=None):
    """Export filtered RF database data to multi-sheet Microsoft XSLX files.

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves xlsx file into Media directory for user to download
    """

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
        modality="RF",
        export_type="XLSX export",
        date_stamp=datestamp,
        pid=bool(pid and (name or patid)),
        user=user,
        filters_dict=filterdict,
    )

    tmpxlsx, book = create_xlsx(tsk)
    if not tmpxlsx:
        exit()

    # Get the data
    if pid:
        if enable_standard_names:
            df_filtered_qs = RFFilterPlusPidPlusStdNames(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="RF"
                ).distinct(),
            )
        else:
            df_filtered_qs = RFFilterPlusPid(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="RF"
                ).distinct(),
            )
    else:
        if enable_standard_names:
            df_filtered_qs = RFFilterPlusStdNames(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="RF"
                ).distinct(),
            )
        else:
            df_filtered_qs = RFSummaryListFilter(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="RF"
                ).distinct(),
            )

    e = df_filtered_qs.qs

    tsk.num_records = e.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    # Add summary sheet and all data sheet
    summarysheet = book.add_worksheet("Summary")
    wsalldata = book.add_worksheet("All data")

    book = text_and_date_formats(
        book, wsalldata, pid=pid, name=name, patid=patid, modality="RF"
    )
    tsk.progress = "Creating an Excel safe version of protocol names and creating a worksheet for each..."
    tsk.save()

    all_data_headers = _all_data_headers(pid=pid, name=name, patid=patid)

    sheet_headers = list(all_data_headers)
    protocolheaders = sheet_headers + [
        "Time",
        "Type",
        "Protocol",
    ]

    if enable_standard_names:
        protocolheaders += [
            "Standard acquisition name",
        ]

    protocolheaders += [
        "Plane",
        "Field size",
        "Filter material",
        "Mean filter thickness (mm)",
        "kVp",
        "mA",
        "Pulse width (ms)",
        "Pulse rate",
        "Number of pulses",
        "Exposure time (ms)",
        "Exposure duration (s)",
        "DAP (cGy.cm^2)",
        "Ref point dose (Gy)",
        "Primary angle",
        "Secondary angle",
    ]

    book, sheetlist = generate_sheets(
        e, book, protocolheaders, modality="RF", pid=pid, name=name, patid=patid
    )

    ##################
    # All data sheet

    num_groups_max = 0
    for row, exams in enumerate(e):

        tsk.progress = f"Writing study {row + 1} of {e.count()}"
        tsk.save()

        try:
            examdata = get_common_data("RF", exams, pid=pid, name=name, patid=patid)
            examdata += _add_plane_summary_data(exams)
            common_exam_data = list(examdata)

            angle_range = 5.0  # plus or minus range considered to be the same position

            # TODO: Check if generation of inst could be more efficient, ie start with exams?
            inst = IrradEventXRayData.objects.filter(
                projection_xray_radiation_dose__general_study_module_attributes__id__exact=exams.id
            )

            num_groups_this_exam = 0
            while (
                inst
            ):  # ie while there are events still left that haven't been matched into a group
                tsk.progress = "Writing study {0} of {1}; {2} events remaining.".format(
                    row + 1, e.count(), inst.count()
                )
                tsk.save()
                num_groups_this_exam += 1
                plane = inst[0].acquisition_plane.code_meaning
                try:
                    mechanical_data = inst[0].irradeventxraymechanicaldata_set.get()
                    anglei = mechanical_data.positioner_primary_angle
                    angleii = mechanical_data.positioner_secondary_angle
                except ObjectDoesNotExist:
                    anglei = None
                    angleii = None
                try:
                    source_data = inst[0].irradeventxraysourcedata_set.get()
                    pulse_rate = source_data.pulse_rate
                    fieldsize = source_data.ii_field_size
                    try:
                        filter_material, filter_thick = get_xray_filter_info(
                            source_data
                        )
                    except ObjectDoesNotExist:
                        filter_material = None
                        filter_thick = None
                except ObjectDoesNotExist:
                    pulse_rate = None
                    fieldsize = None
                    filter_material = None
                    filter_thick = None

                protocol = inst[0].acquisition_protocol

                standard_protocol = ""
                if enable_standard_names:
                    try:
                        standard_protocol = (
                            inst[0].standard_protocols.first().standard_name
                        )
                    except AttributeError:
                        standard_protocol = ""

                event_type = inst[0].irradiation_event_type.code_meaning

                similarexposures = inst
                if plane:
                    similarexposures = similarexposures.filter(
                        acquisition_plane__code_meaning__exact=plane
                    )
                if protocol:
                    similarexposures = similarexposures.filter(
                        acquisition_protocol__exact=protocol
                    )
                if fieldsize:
                    similarexposures = similarexposures.filter(
                        irradeventxraysourcedata__ii_field_size__exact=fieldsize
                    )
                if pulse_rate:
                    similarexposures = similarexposures.filter(
                        irradeventxraysourcedata__pulse_rate__exact=pulse_rate
                    )
                if filter_material:
                    for xray_filter in (
                        inst[0].irradeventxraysourcedata_set.get().xrayfilters_set.all()
                    ):
                        similarexposures = similarexposures.filter(
                            irradeventxraysourcedata__xrayfilters__xray_filter_material__code_meaning__exact=xray_filter.xray_filter_material.code_meaning
                        )
                        similarexposures = similarexposures.filter(
                            irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum__exact=xray_filter.xray_filter_thickness_maximum
                        )
                if anglei:
                    similarexposures = similarexposures.filter(
                        irradeventxraymechanicaldata__positioner_primary_angle__range=(
                            float(anglei) - angle_range,
                            float(anglei) + angle_range,
                        )
                    )
                if angleii:
                    similarexposures = similarexposures.filter(
                        irradeventxraymechanicaldata__positioner_secondary_angle__range=(
                            float(angleii) - angle_range,
                            float(angleii) + angle_range,
                        )
                    )
                if event_type:
                    similarexposures = similarexposures.filter(
                        irradiation_event_type__code_meaning__exact=event_type
                    )

                # Remove exposures included in this group from inst
                exposures_to_exclude = [
                    o.irradiation_event_uid for o in similarexposures
                ]
                inst = inst.exclude(irradiation_event_uid__in=exposures_to_exclude)

                angle1 = similarexposures.all().aggregate(
                    Min("irradeventxraymechanicaldata__positioner_primary_angle"),
                    Max("irradeventxraymechanicaldata__positioner_primary_angle"),
                    Avg("irradeventxraymechanicaldata__positioner_primary_angle"),
                )
                angle2 = similarexposures.all().aggregate(
                    Min("irradeventxraymechanicaldata__positioner_secondary_angle"),
                    Max("irradeventxraymechanicaldata__positioner_secondary_angle"),
                    Avg("irradeventxraymechanicaldata__positioner_secondary_angle"),
                )
                dap = similarexposures.all().aggregate(
                    Min("dose_area_product"),
                    Max("dose_area_product"),
                    Avg("dose_area_product"),
                )
                dose_rp = similarexposures.all().aggregate(
                    Min("irradeventxraysourcedata__dose_rp"),
                    Max("irradeventxraysourcedata__dose_rp"),
                    Avg("irradeventxraysourcedata__dose_rp"),
                )
                kvp = similarexposures.all().aggregate(
                    Min("irradeventxraysourcedata__kvp__kvp"),
                    Max("irradeventxraysourcedata__kvp__kvp"),
                    Avg("irradeventxraysourcedata__kvp__kvp"),
                )
                tube_current = similarexposures.all().aggregate(
                    Min("irradeventxraysourcedata__xraytubecurrent__xray_tube_current"),
                    Max("irradeventxraysourcedata__xraytubecurrent__xray_tube_current"),
                    Avg("irradeventxraysourcedata__xraytubecurrent__xray_tube_current"),
                )
                exp_time = similarexposures.all().aggregate(
                    Min("irradeventxraysourcedata__exposure_time"),
                    Max("irradeventxraysourcedata__exposure_time"),
                    Avg("irradeventxraysourcedata__exposure_time"),
                )
                pulse_width = similarexposures.all().aggregate(
                    Min("irradeventxraysourcedata__pulsewidth__pulse_width"),
                    Max("irradeventxraysourcedata__pulsewidth__pulse_width"),
                    Avg("irradeventxraysourcedata__pulsewidth__pulse_width"),
                )

                examdata += [
                    event_type,
                    protocol,
                ]

                if enable_standard_names:
                    if standard_protocol:
                        examdata += [standard_protocol]
                    else:
                        examdata += [""]

                examdata += [
                    similarexposures.count(),
                    plane,
                    pulse_rate,
                    fieldsize,
                    filter_material,
                    filter_thick,
                    kvp["irradeventxraysourcedata__kvp__kvp__min"],
                    kvp["irradeventxraysourcedata__kvp__kvp__max"],
                    kvp["irradeventxraysourcedata__kvp__kvp__avg"],
                    tube_current[
                        "irradeventxraysourcedata__xraytubecurrent__xray_tube_current__min"
                    ],
                    tube_current[
                        "irradeventxraysourcedata__xraytubecurrent__xray_tube_current__max"
                    ],
                    tube_current[
                        "irradeventxraysourcedata__xraytubecurrent__xray_tube_current__avg"
                    ],
                    pulse_width[
                        "irradeventxraysourcedata__pulsewidth__pulse_width__min"
                    ],
                    pulse_width[
                        "irradeventxraysourcedata__pulsewidth__pulse_width__max"
                    ],
                    pulse_width[
                        "irradeventxraysourcedata__pulsewidth__pulse_width__avg"
                    ],
                    exp_time["irradeventxraysourcedata__exposure_time__min"],
                    exp_time["irradeventxraysourcedata__exposure_time__max"],
                    exp_time["irradeventxraysourcedata__exposure_time__avg"],
                    dap["dose_area_product__min"],
                    dap["dose_area_product__max"],
                    dap["dose_area_product__avg"],
                    dose_rp["irradeventxraysourcedata__dose_rp__min"],
                    dose_rp["irradeventxraysourcedata__dose_rp__max"],
                    dose_rp["irradeventxraysourcedata__dose_rp__avg"],
                    angle1[
                        "irradeventxraymechanicaldata__positioner_primary_angle__min"
                    ],
                    angle1[
                        "irradeventxraymechanicaldata__positioner_primary_angle__max"
                    ],
                    angle1[
                        "irradeventxraymechanicaldata__positioner_primary_angle__avg"
                    ],
                    angle2[
                        "irradeventxraymechanicaldata__positioner_secondary_angle__min"
                    ],
                    angle2[
                        "irradeventxraymechanicaldata__positioner_secondary_angle__max"
                    ],
                    angle2[
                        "irradeventxraymechanicaldata__positioner_secondary_angle__avg"
                    ],
                ]

                if not protocol:
                    protocol = "Unknown"
                tab_text = sheet_name(protocol)
                filter_data = {
                    "filter_material": filter_material,
                    "filter_thick": filter_thick,
                }
                for exposure in similarexposures.order_by("pk"):
                    series_data = _get_series_data(exposure, filter_data)
                    sheetlist[tab_text]["count"] += 1
                    sheetlist[tab_text]["sheet"].write_row(
                        sheetlist[tab_text]["count"], 0, common_exam_data + series_data
                    )

                if enable_standard_names:
                    if standard_protocol:
                        tab_text = sheet_name("[standard] " + standard_protocol)
                        filter_data = {
                            "filter_material": filter_material,
                            "filter_thick": filter_thick,
                        }
                        for exposure in similarexposures.order_by("pk"):
                            series_data = _get_series_data(exposure, filter_data)
                            sheetlist[tab_text]["count"] += 1
                            sheetlist[tab_text]["sheet"].write_row(
                                sheetlist[tab_text]["count"],
                                0,
                                common_exam_data + series_data,
                            )

            if num_groups_this_exam > num_groups_max:
                num_groups_max = num_groups_this_exam

            wsalldata.write_row(row + 1, 0, examdata)

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
            wsalldata.write(row + 1, 0, error_message)

    tsk.progress = "Generating headers for the all data sheet..."
    tsk.save()

    for h in range(num_groups_max):
        all_data_headers += [
            "G" + str(h + 1) + " Type",
            "G" + str(h + 1) + " Protocol",
        ]

        if enable_standard_names:
            all_data_headers += ["G" + str(h + 1) + " Standard acquisition name"]

        all_data_headers += [
            "G" + str(h + 1) + " No. exposures",
            "G" + str(h + 1) + " Plane",
            "G" + str(h + 1) + " Pulse rate",
            "G" + str(h + 1) + " Field size",
            "G" + str(h + 1) + " Filter material",
            "G" + str(h + 1) + " Mean filter thickness (mm)",
            "G" + str(h + 1) + " kVp min",
            "G" + str(h + 1) + " kVp max",
            "G" + str(h + 1) + " kVp mean",
            "G" + str(h + 1) + " mA min",
            "G" + str(h + 1) + " mA max",
            "G" + str(h + 1) + " mA mean",
            "G" + str(h + 1) + " pulse width min (ms)",
            "G" + str(h + 1) + " pulse width max (ms)",
            "G" + str(h + 1) + " pulse width mean (ms)",
            "G" + str(h + 1) + " Exp time min (ms)",
            "G" + str(h + 1) + " Exp time max (ms)",
            "G" + str(h + 1) + " Exp time mean (ms)",
            "G" + str(h + 1) + " DAP min (Gy.m^2)",
            "G" + str(h + 1) + " DAP max (Gy.m^2)",
            "G" + str(h + 1) + " DAP mean (Gy.m^2)",
            "G" + str(h + 1) + " Ref point dose min (Gy)",
            "G" + str(h + 1) + " Ref point dose max (Gy)",
            "G" + str(h + 1) + " Ref point dose mean (Gy)",
            "G" + str(h + 1) + " Primary angle min",
            "G" + str(h + 1) + " Primary angle max",
            "G" + str(h + 1) + " Primary angle mean",
            "G" + str(h + 1) + " Secondary angle min",
            "G" + str(h + 1) + " Secondary angle max",
            "G" + str(h + 1) + " Secondary angle mean",
        ]
    wsalldata.write_row("A1", all_data_headers)
    num_rows = e.count()
    wsalldata.autofilter(0, 0, num_rows, len(all_data_headers) - 1)

    create_summary_sheet(tsk, e, book, summarysheet, sheetlist)

    book.close()
    tsk.progress = "XLSX book written."
    tsk.save()

    xlsxfilename = "rfexport{0}.xlsx".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))

    write_export(tsk, xlsxfilename, tmpxlsx, datestamp)


def exportFL2excel(filterdict, pid=False, name=None, patid=None, user=None):
    """Export filtered fluoro database data to a single-sheet CSV file.

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves csv file into Media directory for user to download
    """

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
        modality="RF",
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
    if pid:
        if enable_standard_names:
            df_filtered_qs = RFFilterPlusPidPlusStdNames(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="RF"
                ).distinct(),
            )
        else:
            df_filtered_qs = RFFilterPlusPid(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="RF"
                ).distinct(),
            )
    else:
        if enable_standard_names:
            df_filtered_qs = RFFilterPlusStdNames(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="RF"
                ).distinct(),
            )
        else:
            df_filtered_qs = RFSummaryListFilter(
                filterdict,
                queryset=GeneralStudyModuleAttr.objects.filter(
                    modality_type__exact="RF"
                ).distinct(),
            )

    e = df_filtered_qs.qs

    tsk.num_records = e.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    headings = _all_data_headers(pid=pid, name=name, patid=patid)
    writer.writerow(headings)
    for i, exams in enumerate(e):

        tsk.progress = "{0} of {1}".format(i + 1, tsk.num_records)
        tsk.save()

        try:
            exam_data = get_common_data("RF", exams, pid=pid, name=name, patid=patid)

            for (
                plane
            ) in exams.projectionxrayradiationdose_set.get().accumxraydose_set.all():
                accum = _get_accumulated_data(plane)
                exam_data += [
                    accum["dose_area_product_total"],
                    accum["dose_rp_total"],
                    accum["fluoro_dose_area_product_total"],
                    accum["fluoro_dose_rp_total"],
                    accum["total_fluoro_time"],
                    accum["acquisition_dose_area_product_total"],
                    accum["acquisition_dose_rp_total"],
                    accum["total_acquisition_time"],
                    accum["eventcount"],
                ]
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


def rfopenskin(studyid):
    """Export single RF study data to OpenSkin RF csv sheet.

    :param studyid: RF study database ID.
    :type studyid: int

    """

    datestamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="RF-OpenSkin",
        export_type="OpenSkin RF csv export",
        date_stamp=datestamp,
        pid=False,
        user=None,
        filters_dict={"study_id": studyid},
    )

    tmpfile, writer = create_csv(tsk)
    if not tmpfile:
        exit()

    # Get the data
    study = GeneralStudyModuleAttr.objects.get(pk=studyid)
    numevents = (
        study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
    )
    tsk.num_records = numevents
    tsk.save()

    for i, event in enumerate(
        study.projectionxrayradiationdose_set.get().irradeventxraydata_set.all()
    ):
        try:
            study.patientmoduleattr_set.get()
        except ObjectDoesNotExist:
            patient_sex = ""
        else:
            patient_sex = study.patientmoduleattr_set.get().patient_sex

        try:
            event.irradeventxraysourcedata_set.get()
        except ObjectDoesNotExist:
            reference_point_definition = ""
            dose_rp = ""
            fluoro_mode = ""
            pulse_rate = ""
            number_of_pulses = ""
            exposure_time = ""
            focal_spot_size = ""
            irradiation_duration = ""
            average_xray_tube_current = ""
        else:
            reference_point_definition = (
                event.irradeventxraysourcedata_set.get().reference_point_definition
            )
            dose_rp = event.irradeventxraysourcedata_set.get().dose_rp
            fluoro_mode = event.irradeventxraysourcedata_set.get().fluoro_mode
            pulse_rate = event.irradeventxraysourcedata_set.get().pulse_rate
            number_of_pulses = event.irradeventxraysourcedata_set.get().number_of_pulses
            exposure_time = event.irradeventxraysourcedata_set.get().exposure_time
            focal_spot_size = event.irradeventxraysourcedata_set.get().focal_spot_size
            irradiation_duration = (
                event.irradeventxraysourcedata_set.get().irradiation_duration
            )
            average_xray_tube_current = (
                event.irradeventxraysourcedata_set.get().average_xray_tube_current
            )

        try:
            event.irradeventxraymechanicaldata_set.get()
        except ObjectDoesNotExist:
            positioner_primary_angle = ""
            positioner_secondary_angle = ""
            positioner_primary_end_angle = ""
            positioner_secondary_end_angle = ""
            column_angulation = ""
        else:
            positioner_primary_angle = (
                event.irradeventxraymechanicaldata_set.get().positioner_primary_angle
            )
            positioner_secondary_angle = (
                event.irradeventxraymechanicaldata_set.get().positioner_secondary_angle
            )
            positioner_primary_end_angle = (
                event.irradeventxraymechanicaldata_set.get().positioner_primary_end_angle
            )
            positioner_secondary_end_angle = (
                event.irradeventxraymechanicaldata_set.get().positioner_secondary_end_angle
            )
            column_angulation = (
                event.irradeventxraymechanicaldata_set.get().column_angulation
            )

        xray_filter_type = ""
        xray_filter_material = ""
        xray_filter_thickness_minimum = ""
        xray_filter_thickness_maximum = ""
        try:
            for (
                filters
            ) in event.irradeventxraysourcedata_set.get().xrayfilters_set.all():
                try:
                    if "Copper" in filters.xray_filter_material.code_meaning:
                        xray_filter_type = filters.xray_filter_type
                        xray_filter_material = filters.xray_filter_material
                        xray_filter_thickness_minimum = (
                            filters.xray_filter_thickness_minimum
                        )
                        xray_filter_thickness_maximum = (
                            filters.xray_filter_thickness_maximum
                        )
                except AttributeError:
                    pass
        except ObjectDoesNotExist:
            pass

        try:
            event.irradeventxraysourcedata_set.get().kvp_set.get()
        except ObjectDoesNotExist:
            kvp = ""
        else:
            kvp = event.irradeventxraysourcedata_set.get().kvp_set.get().kvp

        try:
            event.irradeventxraysourcedata_set.get().xraytubecurrent_set.get()
        except ObjectDoesNotExist:
            xray_tube_current = ""
        else:
            xray_tube_current = (
                event.irradeventxraysourcedata_set.get()
                .xraytubecurrent_set.get()
                .xray_tube_current
            )

        try:
            event.irradeventxraysourcedata_set.get().pulsewidth_set.get()
        except ObjectDoesNotExist:
            pulse_width = ""
        else:
            pulse_width = (
                event.irradeventxraysourcedata_set.get()
                .pulsewidth_set.get()
                .pulse_width
            )

        try:
            event.irradeventxraysourcedata_set.get().exposure_set.get()
        except ObjectDoesNotExist:
            exposure = ""
        else:
            exposure = (
                event.irradeventxraysourcedata_set.get().exposure_set.get().exposure
            )

        try:
            event.irradeventxraymechanicaldata_set.get().doserelateddistancemeasurements_set.get()
        except ObjectDoesNotExist:
            distance_source_to_detector = ""
            distance_source_to_isocenter = ""
            table_longitudinal_position = ""
            table_lateral_position = ""
            table_height_position = ""
        else:
            distance_source_to_detector = (
                event.irradeventxraymechanicaldata_set.get()
                .doserelateddistancemeasurements_set.get()
                .distance_source_to_detector
            )
            distance_source_to_isocenter = (
                event.irradeventxraymechanicaldata_set.get()
                .doserelateddistancemeasurements_set.get()
                .distance_source_to_isocenter
            )
            table_longitudinal_position = (
                event.irradeventxraymechanicaldata_set.get()
                .doserelateddistancemeasurements_set.get()
                .table_longitudinal_position
            )
            table_lateral_position = (
                event.irradeventxraymechanicaldata_set.get()
                .doserelateddistancemeasurements_set.get()
                .table_lateral_position
            )
            table_height_position = (
                event.irradeventxraymechanicaldata_set.get()
                .doserelateddistancemeasurements_set.get()
                .table_height_position
            )

        acquisition_protocol = return_for_export(event, "acquisition_protocol")
        if isinstance(acquisition_protocol, str) and "," in acquisition_protocol:
            acquisition_protocol = acquisition_protocol.replace(",", ";")
        comment = event.comment
        if isinstance(comment, str) and "," in comment:
            comment = comment.replace(",", ";")

        data = [
            "Anon",
            patient_sex,
            study.study_instance_uid,
            "",
            event.acquisition_plane,
            event.date_time_started,
            event.irradiation_event_type,
            acquisition_protocol,
            reference_point_definition,
            event.irradiation_event_uid,
            event.dose_area_product,
            dose_rp,
            positioner_primary_angle,
            positioner_secondary_angle,
            positioner_primary_end_angle,
            positioner_secondary_end_angle,
            column_angulation,
            xray_filter_type,
            xray_filter_material,
            xray_filter_thickness_minimum,
            xray_filter_thickness_maximum,
            fluoro_mode,
            pulse_rate,
            number_of_pulses,
            kvp,
            xray_tube_current,
            exposure_time,
            pulse_width,
            exposure,
            focal_spot_size,
            irradiation_duration,
            average_xray_tube_current,
            distance_source_to_detector,
            distance_source_to_isocenter,
            table_longitudinal_position,
            table_lateral_position,
            table_height_position,
            event.target_region,
            comment,
        ]
        writer.writerow(data)
        tsk.progress = "{0} of {1}".format(i, numevents)
        tsk.save()
    tsk.progress = "All study data written."
    tsk.save()

    tmpfile.close()
    tsk.status = "COMPLETE"
    tsk.processtime = (datetime.datetime.now() - datestamp).total_seconds()
    tsk.save()


def rf_phe_2019(filterdict, user=None):
    """Export filtered RF database data in the format for the 2019 Public Health England IR/fluoro dose survey

    :param filterdict: Queryset of studies to export
    :param user: User that has started the export
    :return: Saves Excel file into media directory for user to download
    """

    datestamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="RF",
        export_type="PHE RF 2019 export",
        date_stamp=datestamp,
        pid=False,
        user=user,
        filters_dict=filterdict,
    )

    tmp_xlsx, book = create_xlsx(tsk)
    if not tmp_xlsx:
        exit()
    sheet = book.add_worksheet("PHE IR-Fluoro")

    exams = RFSummaryListFilter(
        filterdict,
        queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="RF"),
    ).qs
    tsk.num_records = exams.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    tsk.progress = "{0} studies in query.".format(tsk.num_records)
    tsk.save()

    row_4 = ["", "", "", "", "", "Gy·m²", "", "seconds", "Gy"]
    sheet.write_row(3, 0, row_4)

    num_rows = exams.count()
    for row, exam in enumerate(exams):
        tsk.progress = "Writing study {0} of {1}".format(row + 1, num_rows)
        tsk.save()

        row_data = ["", row + 1, exam.pk, exam.study_date, exam.total_dap]
        accum_data = []
        for plane in exam.projectionxrayradiationdose_set.get().accumxraydose_set.all():
            accum_data.append(_get_accumulated_data(plane))
        if len(accum_data) == 2:
            accum_data[0]["fluoro_dose_area_product_total"] += accum_data[1][
                "fluoro_dose_area_product_total"
            ]
            accum_data[0]["fluoro_dose_rp_total"] += accum_data[1][
                "fluoro_dose_rp_total"
            ]
            accum_data[0]["total_fluoro_time"] += accum_data[1]["total_fluoro_time"]
            accum_data[0]["acquisition_dose_area_product_total"] += accum_data[1][
                "acquisition_dose_area_product_total"
            ]
            accum_data[0]["acquisition_dose_rp_total"] += accum_data[1][
                "acquisition_dose_rp_total"
            ]
            accum_data[0]["total_acquisition_time"] += accum_data[1][
                "total_acquisition_time"
            ]
            accum_data[0]["eventcount"] += accum_data[1]["eventcount"]
        row_data += [
            accum_data[0]["fluoro_dose_area_product_total"],
            accum_data[0]["acquisition_dose_area_product_total"],
            accum_data[0]["total_fluoro_time"],
        ]

        try:
            total_rp_dose = exam.total_rp_dose_a + exam.total_rp_dose_b
        except TypeError:
            if exam.total_rp_dose_a is not None:
                total_rp_dose = exam.total_rp_dose_a
            elif exam.total_rp_dose_b is not None:
                total_rp_dose = exam.total_rp_dose_b
            else:
                total_rp_dose = 0
        row_data += [
            total_rp_dose,
            accum_data[0]["fluoro_dose_rp_total"],
            accum_data[0]["acquisition_dose_rp_total"],
            "{0} | {1} | {2}".format(
                exam.procedure_code_meaning,
                exam.requested_procedure_code_meaning,
                exam.study_description,
            ),
        ]
        patient_study_data = get_patient_study_data(exam)
        patient_sex = None
        try:
            patient_sex = exam.patientmoduleattr_set.get().patient_sex
        except ObjectDoesNotExist:
            logger.debug(
                "Export {0}; patientmoduleattr_set object does not exist. AccNum {1}, Date {2}".format(
                    "PHE 2019 RF", exams.accession_number, exams.study_date
                )
            )
        row_data += [
            patient_study_data["patient_weight"],
            "",
            patient_study_data["patient_age_decimal"],
            patient_sex,
            patient_study_data["patient_size"],
        ]

        events = IrradEventXRayData.objects.filter(
            projection_xray_radiation_dose__general_study_module_attributes__pk__exact=exam.pk
        )
        fluoro_events = events.exclude(
            irradiation_event_type__code_value__contains="11361"
        )  # acq events are 113611, 113612, 113613
        acquisition_events = events.filter(
            irradiation_event_type__code_value__contains="11361"
        )
        try:
            row_data += [
                " | ".join(
                    fluoro_events.order_by()
                    .values_list("acquisition_protocol", flat=True)
                    .distinct()
                )
            ]
        except TypeError:
            row_data += [""]
        try:
            row_data += [
                " | ".join(
                    fluoro_events.order_by()
                    .values_list(
                        "irradeventxraysourcedata__fluoro_mode__code_meaning", flat=True
                    )
                    .distinct()
                )
            ]
        except TypeError:
            row_data += [""]
        fluoro_frame_rates = (
            fluoro_events.order_by()
            .values_list("irradeventxraysourcedata__pulse_rate", flat=True)
            .distinct()
        )
        column_aq = ""
        if len(fluoro_frame_rates) > 1:
            column_aq += "Fluoro: "
            column_aq += " | ".join(
                format(x, "1.1f") for x in fluoro_frame_rates if x is not None
            )
            column_aq += " fps. "
            row_data += ["Multiple rates"]
        else:
            try:
                row_data += [fluoro_frame_rates[0]]
            except IndexError:
                row_data += [""]
        acquisition_frame_rates = (
            acquisition_events.order_by()
            .values_list("irradeventxraysourcedata__pulse_rate", flat=True)
            .distinct()
        )
        add_single = False
        if None in acquisition_frame_rates:
            if len(acquisition_frame_rates) == 1:
                acquisition_frame_rates = ["Single shot"]
            else:
                acquisition_frame_rates = acquisition_frame_rates[1:]
                add_single = True
        if len(acquisition_frame_rates) > 1:
            row_data += ["Multiple rates"]
            column_aq += "Acquisition: "
            if add_single:
                column_aq += "Single shot | "
            column_aq += " | ".join(format(x, "1.1f") for x in acquisition_frame_rates)
            column_aq += " fps. "
        else:
            try:
                row_data += [acquisition_frame_rates[0]]
            except IndexError:
                row_data += [""]
        row_data += [acquisition_events.count()]
        try:
            grid_types = (
                events.order_by()
                .values_list(
                    "irradeventxraysourcedata__xraygrid__xray_grid__code_meaning",
                    flat=True,
                )
                .distinct()
            )
            if None in grid_types:
                grid_types = grid_types[1:]
        except ObjectDoesNotExist:
            grid_types = [""]
        row_data += [" | ".join(grid_types), ""]  # AEC used - not recorded in RDSR
        patient_position = (
            events.order_by()
            .values_list(
                "patient_table_relationship_cid__code_meaning",
                "patient_orientation_cid__code_meaning",
                "patient_orientation_modifier_cid__code_meaning",
            )
            .distinct()
        )
        patient_position_str = ""
        for position_set in patient_position:
            for element in (i for i in position_set if i):
                patient_position_str += "{0}, ".format(element)
        row_data += [
            patient_position_str,
            "",  # digital subtraction
            "",  # circular field of view
        ]
        field_dimensions = events.aggregate(
            Min("irradeventxraysourcedata__collimated_field_area"),
            Max("irradeventxraysourcedata__collimated_field_area"),
            Min("irradeventxraysourcedata__collimated_field_width"),
            Max("irradeventxraysourcedata__collimated_field_width"),
            Min("irradeventxraysourcedata__collimated_field_height"),
            Max("irradeventxraysourcedata__collimated_field_height"),
        )
        rectangular_fov = ""
        if field_dimensions["irradeventxraysourcedata__collimated_field_area__min"]:
            rectangular_fov += "Area {0:.4f} to {1:.4f} m², ".format(
                field_dimensions[
                    "irradeventxraysourcedata__collimated_field_area__min"
                ],
                field_dimensions[
                    "irradeventxraysourcedata__collimated_field_area__max"
                ],
            )
        if field_dimensions["irradeventxraysourcedata__collimated_field_width__min"]:
            rectangular_fov += "Width {0:.4f} to {1:.4f} m, ".format(
                field_dimensions[
                    "irradeventxraysourcedata__collimated_field_width__min"
                ],
                field_dimensions[
                    "irradeventxraysourcedata__collimated_field_width__max"
                ],
            )
        if field_dimensions["irradeventxraysourcedata__collimated_field_height__min"]:
            rectangular_fov += "Height {0:.4f} to {1:.4f} m, ".format(
                field_dimensions[
                    "irradeventxraysourcedata__collimated_field_height__min"
                ],
                field_dimensions[
                    "irradeventxraysourcedata__collimated_field_height__max"
                ],
            )
        field_sizes = (
            events.order_by()
            .values_list("irradeventxraysourcedata__ii_field_size", flat=True)
            .distinct()
        )
        diagonal_fov = ""
        for fov in field_sizes:
            if fov:
                diagonal_fov += "{0}, ".format(fov)
        if diagonal_fov:
            diagonal_fov += " mm"
        row_data += [rectangular_fov, diagonal_fov]
        filters_al = events.filter(
            irradeventxraysourcedata__xrayfilters__xray_filter_material__code_value__exact="C-120F9"
        )
        filters_cu = events.filter(
            irradeventxraysourcedata__xrayfilters__xray_filter_material__code_value__exact="C-127F9"
        )
        filters_al_thick = filters_al.aggregate(
            Min("irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum"),
            Max("irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum"),
        )
        filters_cu_thick = filters_cu.aggregate(
            Min("irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum"),
            Max("irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum"),
        )
        filters_al_str = ""
        filters_cu_str = ""
        if filters_al_thick[
            "irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum__min"
        ]:
            filters_al_str = "{0:.2} - {1:.2} mm".format(
                filters_al_thick[
                    "irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum__min"
                ],
                filters_al_thick[
                    "irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum__max"
                ],
            )
        if filters_cu_thick[
            "irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum__min"
        ]:
            filters_cu_str = "{0:.2} - {1:.2} mm".format(
                filters_cu_thick[
                    "irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum__min"
                ],
                filters_cu_thick[
                    "irradeventxraysourcedata__xrayfilters__xray_filter_thickness_maximum__max"
                ],
            )
        row_data += ["", filters_cu_str, filters_al_str]  # filtration automated?
        row_data += ["", "", "", "", "", "", "", "", "", ""]
        row_data += [column_aq]
        sheet.write_row(row + 6, 0, row_data)

    book.close()
    tsk.progress = "PHE IR/Fluoro 2019 export complete"
    tsk.save()

    xlsxfilename = "PHE_RF_2019_{0}.xlsx".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))

    write_export(tsk, xlsxfilename, tmp_xlsx, datestamp)
