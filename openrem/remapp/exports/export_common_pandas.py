# pylint: disable=too-many-lines
# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2017  The Royal Marsden NHS Foundation Trust
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
..  module:: export_common.
    :synopsis: Module to deduplicate some of the export code

..  moduleauthor:: David Platten and Jordan Suter

"""
import codecs
import csv
import logging
import sys
from tempfile import TemporaryFile
import uuid
import datetime
from numbers import Number

import django.db
import numpy as np
import pandas as pd

from django.core.files import File

from django.conf import settings
from django.core.exceptions import (
    ObjectDoesNotExist,
    MultipleObjectsReturned,
)
from django.core.files.base import ContentFile
from django.db.models import Q, F, Avg
from django.db.models.functions import Coalesce
from xlsxwriter.workbook import Workbook

from ..models import (
    CtRadiationDose,
    Exports,
    ProjectionXRayRadiationDose,
    StandardNames,
)

from ..version import __version__

from ..tools.check_standard_name_status import are_standard_names_enabled

logger = logging.getLogger(__name__)


def text_and_date_formats(
    book, sheet, pid=False, name=None, patid=None, modality=None, headers=None
):
    """
    Function to write out the headings common to each sheet and modality and format the date, time, patient ID and
    accession number columns.
    :param book: xlsxwriter book to work on
    :param sheet: xlsxwriter sheet to work on
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param modality: modality
    :param headers: The headers used for the sheet
    :return: book

    Parameters
    ----------
    modality
    """

    textformat = book.add_format({"num_format": "@"})
    dateformat = book.add_format({"num_format": settings.XLSX_DATE})
    timeformat = book.add_format({"num_format": settings.XLSX_TIME})
    #datetimeformat = book.add_format(
    #    {"num_format": f"{settings.XLSX_DATE} {settings.XLSX_TIME}"}
    #)

    date_column = 10

    patid_column = 10
    if pid and patid:
        date_column += 1
    if pid and name:
        date_column += 1
        patid_column += 1

    # Obtain the system-level enable_standard_names setting. If the modality is nuclear medicine force standard
    # names feature off because standard names not implemented for this modality yet.
    enable_standard_names = False
    if modality != "NM":
        enable_standard_names = are_standard_names_enabled()

    if enable_standard_names:
        date_column += 3

    if modality == "RF":
        date_column += 1

    if modality == "NM":
        date_column += 1

    sheet.set_column(
        date_column, date_column, 10, dateformat
    )  # allow date to be displayed.

    sheet.set_column(
        date_column + 1, date_column + 1, None, timeformat
    )  # allow time to be displayed.

    # if pid and (name or patid):
    #    sheet.set_column(
    #        date_column + 2, date_column + 2, 10, dateformat
    #    )  # Birth date column [DJP: it isn't a date of birth column, it is a patient age column as a decimal]

    if pid and patid:
        sheet.set_column(
            patid_column, patid_column, None, textformat
        )  # make sure leading zeros are not dropped
    sheet.set_column(
        date_column - 2, date_column - 2, None, textformat
    )  # Accession number as text

    return book


def common_headers(modality=None, pid=False, name=None, patid=None):
    """
    Function to generate list of header text common to several exports
    :param modality: export modality to customise some of the columns
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :return: list of strings
    """

    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    pid_headings = []
    if pid and name:
        pid_headings += ["Patient name"]
    if pid and patid:
        pid_headings += ["Patient ID"]
    headers = pid_headings + [
        "Institution",
        "Manufacturer",
        "Model name",
        "Station name",
        "Display name",
        "Accession number",
        "Operator",
    ]
    if modality == "RF":
        headers += ["Physician"]
    headers += ["Study date", "Study time"]
    if pid and (name or patid):
        headers += ["Date of birth"]
    headers += ["Age", "Sex"]
    mammo = bool(modality == "MG")
    if not mammo:
        headers += ["Height", "Mass (kg)"]
    headers += [
        "Test patient?",
        "Study description",
    ]
    if enable_standard_names and modality != "NM":
        headers += [
            "Standard study name (study)",
        ]
    headers += [
        "Requested procedure",
    ]
    if enable_standard_names and modality != "NM":
        headers += [
            "Standard study name (request)",
        ]
    headers += [
        "Study Comments",
        "No. events",
    ]

    return headers


def sheet_name(protocol_name):
    """
    Creates Excel safe version of protocol name for sheet tab text
    :param protocol_name: string, protocol name
    :return: string, Excel safe sheet name for tab text
    """
    tab_text = protocol_name.lower().replace(" ", "_")
    translation_table = {
        ord("["): ord("("),
        ord("]"): ord(")"),
        ord(":"): ord(";"),
        ord("*"): ord("#"),
        ord("?"): ord(";"),
        ord("/"): ord("|"),
        ord("\\"): ord("|"),
    }
    tab_text = tab_text.translate(translation_table)  # remove illegal characters
    tab_text = tab_text[:31]
    return tab_text


def generate_sheets(
    studies, book, protocol_headers, modality=None, pid=False, name=None, patid=None
):
    # pylint: disable=too-many-locals
    """
    Function to generate the sheets in the xlsx book based on the protocol names
    :param studies: filtered queryset of exams
    :param book: xlsxwriter book to work on
    :param protocol_headers: list of headers to insert on each sheet
    :param modality: study modality to determine database location of acquisition_protocol
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :return: book
    """
    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    sheet_list = {}

    required_fields = []
    column_names = []
    acq_name_df = None
    if modality in ["DX", "RF", "MG"]:
        required_fields.append("irradeventxraydata__acquisition_protocol")
        column_names.append("Acquisition protocol")

        if enable_standard_names:
            required_fields.append(
                "irradeventxraydata__standard_protocols__standard_name"
            )
            column_names.append("Standard acquisition name")

        # Obtain a dataframe of all the acquisition protocols and standard acquisition names in the supplied studies
        acq_name_df = pd.DataFrame.from_records(
            data=ProjectionXRayRadiationDose.objects.filter(
                general_study_module_attributes__in=studies.values("pk")
            ).values_list(*required_fields),
            columns=column_names,
        )

    elif modality in "CT":
        required_fields.append("ctirradiationeventdata__acquisition_protocol")
        column_names.append("Acquisition protocol")

        if enable_standard_names:
            required_fields.append(
                "ctirradiationeventdata__standard_protocols__standard_name"
            )
            column_names.append("Standard acquisition name")

        # Obtain a dataframe of all the acquisition protocols and standard acquisition names in the supplied studies
        acq_name_df = pd.DataFrame.from_records(
            data=CtRadiationDose.objects.filter(
                general_study_module_attributes__in=studies.values("pk")
            ).values_list(*required_fields),
            columns=column_names,
        )

    # Obtain a list of the unique acquisition protocols. Replace any na or None values with "Unknown"
    acq_protocols = (
        acq_name_df.sort_values(by=["Acquisition protocol"])["Acquisition protocol"]
        .fillna("Unknown")
        .unique()
    )

    protocols_list = list(acq_protocols)

    if enable_standard_names:
        # Obtain a list of the unique standard acquisition names.
        # Drop any na or None values. Prepend "[standard] " to each entry
        std_acq_protocols = (
            acq_name_df.sort_values(by=["Standard acquisition name"])[
                "Standard acquisition name"
            ]
            .dropna()
            .unique()
        )
        std_acq_protocols = "[standard] " + std_acq_protocols
        protocols_list.extend(list(std_acq_protocols))

    protocols_list.sort()

    for protocol in protocols_list:
        tab_text = sheet_name(protocol)
        if tab_text not in sheet_list:
            sheet_list[tab_text] = {
                "sheet": book.add_worksheet(tab_text),
                "count": 0,
                "protocolname": [protocol],
            }
            sheet_list[tab_text]["sheet"].write_row(0, 0, protocol_headers)
            book = text_and_date_formats(
                book,
                sheet_list[tab_text]["sheet"],
                pid=pid,
                name=name,
                patid=patid,
                modality=modality,
                headers=protocol_headers,
            )
        else:
            if protocol not in sheet_list[tab_text]["protocolname"]:
                sheet_list[tab_text]["protocolname"].append(protocol)

    return book, sheet_list


def get_patient_study_data(exam):
    """Get patient study module data

    :param exam: Exam table
    :return: dict of patient study module data
    """
    patient_age_decimal = None
    patient_size = None
    patient_weight = None
    try:
        patient_study_module = exam.patientstudymoduleattr_set.get()
        patient_age_decimal = patient_study_module.patient_age_decimal
        patient_size = patient_study_module.patient_size
        patient_weight = patient_study_module.patient_weight
    except ObjectDoesNotExist:
        logger.debug(
            f"Export {exam.modality_type}; patientstudymoduleattr_set object does not exist." +
            f" AccNum {exam.accession_number}, Date {exam.study_date}"
        )
    return {
        "patient_age_decimal": patient_age_decimal,
        "patient_size": patient_size,
        "patient_weight": patient_weight,
    }


def get_common_data(modality, exams, pid=None, name=None, patid=None):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """Get the data common to several exports

    :param modality: Modality for the number of irradiation events database location
    :param exams: exam to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :return: the common data for that exam
    """

    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    patient_birth_date = None
    patient_name = None
    patient_id = None
    patient_sex = None
    not_patient_indicator = None
    try:
        patient_module = exams.patientmoduleattr_set.get()
        patient_sex = patient_module.patient_sex
        not_patient_indicator = patient_module.not_patient_indicator
        if pid and (name or patid):
            patient_birth_date = patient_module.patient_birth_date
            if name:
                patient_name = patient_module.patient_name
            if patid:
                patient_id = patient_module.patient_id
    except ObjectDoesNotExist:
        logger.debug(
            f"Export {modality}; patientmoduleattr_set object does not exist." +
            f" AccNum {exams.accession_number}, Date {exams.study_date}"
        )

    institution_name = None
    manufacturer = None
    manufacturer_model_name = None
    station_name = None
    display_name = None
    try:
        equipment_module = exams.generalequipmentmoduleattr_set.get()
        institution_name = equipment_module.institution_name
        manufacturer = equipment_module.manufacturer
        manufacturer_model_name = equipment_module.manufacturer_model_name
        station_name = equipment_module.station_name
        try:
            display_name = equipment_module.unique_equipment_name.display_name
        except AttributeError:
            logger.debug(
                f"Export {modality}; unique_equipment_name object does not exist." +
                f" AccNum {exams.accession_number}, Date {exams.study_date}"
            )

    except ObjectDoesNotExist:
        logger.debug(
            f"Export {modality}; generalequipmentmoduleattr_set object does not exist." +
            f" AccNum {exams.accession_number}, Date {exams.study_date}"
        )

    patient_study_data = get_patient_study_data(exams)

    event_count = None
    cgycm2 = None
    comment = None
    ct_dose_length_product_total = None
    if modality in "CT":
        try:
            comment = exams.ctradiationdose_set.get().comment
            ct_accumulated = (
                exams.ctradiationdose_set.get().ctaccumulateddosedata_set.get()
            )
            ct_dose_length_product_total = ct_accumulated.ct_dose_length_product_total
            try:
                event_count = int(ct_accumulated.total_number_of_irradiation_events)
            except TypeError:
                logger.debug(
                    "Export CT; couldn't get number of irradiation events." +
                    f" AccNum {exams.accession_number}, Date {exams.study_date}"
                )
        except ObjectDoesNotExist:
            logger.debug(
                "Export CT; ctradiationdose_set object does not exist." +
                f" AccNum {exams.accession_number}, Date {exams.study_date}"
            )

    elif modality in "DX":
        try:
            comment = exams.projectionxrayradiationdose_set.get().comment
            dx_accumulated = (
                exams.projectionxrayradiationdose_set.get()
                .accumxraydose_set.get()
                .accumintegratedprojradiogdose_set.get()
            )
            event_count = dx_accumulated.total_number_of_radiographic_frames
            dap_total = dx_accumulated.dose_area_product_total
            if dap_total:
                cgycm2 = dx_accumulated.convert_gym2_to_cgycm2()
            else:
                cgycm2 = None
        except ObjectDoesNotExist:
            logger.debug(
                "Export DX; projectionxrayradiationdose_set object does not exist." +
                f" AccNum {exams.accession_number}, Date {exams.study_date}"
            )
    elif modality in ["RF", "MG"]:
        try:
            event_count = (
                exams.projectionxrayradiationdose_set.get()
                .irradeventxraydata_set.all()
                .count()
            )
            comment = exams.projectionxrayradiationdose_set.get().comment
        except ObjectDoesNotExist:
            logger.debug(
                f"Export {modality}; projectionxrayradiationdose_set object does not exist."
                f" AccNum {exams.accession_number}, Date {exams.study_date}"
            )

    examdata = []
    if pid and name:
        examdata += [patient_name]
    if pid and patid:
        examdata += [patient_id]
    examdata += [
        institution_name,
        manufacturer,
        manufacturer_model_name,
        station_name,
        display_name,
        exams.accession_number,
        exams.operator_name,
    ]
    if modality in ["RF"]:
        examdata += [exams.performing_physician_name]
    examdata += [exams.study_date, exams.study_time]
    if pid and (name or patid):
        examdata += [patient_birth_date]
    examdata += [patient_study_data["patient_age_decimal"], patient_sex]
    if modality not in "MG":
        examdata += [
            patient_study_data["patient_size"],
            patient_study_data["patient_weight"],
        ]
    examdata += [
        not_patient_indicator,
        exams.study_description,
    ]

    std_name_modality = modality
    if std_name_modality in ["CR", "PX"]:
        std_name_modality = "DX"

    std_name = None
    if enable_standard_names:
        std_names = StandardNames.objects.filter(modality=std_name_modality)

        # Get standard name that matches study_description
        std_name = std_names.filter(
            Q(study_description=exams.study_description)
            & Q(study_description__isnull=False)
        ).values_list("standard_name", flat=True)

        if std_name:
            examdata += [
                list(std_name)[0],
            ]
        else:
            examdata += [
                "",
            ]

    examdata += [
        exams.requested_procedure_code_meaning,
    ]

    if enable_standard_names:
        # Get standard name that matches requested_procedure_code_meaning
        std_name = std_names.filter(
            Q(requested_procedure_code_meaning=exams.requested_procedure_code_meaning)
            & Q(requested_procedure_code_meaning__isnull=False)
        ).values_list("standard_name", flat=True)

        if std_name:
            examdata += [
                list(std_name)[0],
            ]
        else:
            examdata += [
                "",
            ]

    examdata += [
        comment,
    ]
    if modality in "CT":
        examdata += [event_count, ct_dose_length_product_total]
    elif modality in "DX":
        examdata += [event_count, cgycm2]
    elif modality in ["RF", "MG"]:
        examdata += [event_count]

    return examdata


def get_pulse_data(source_data, modality=None):
    # pylint: disable=too-many-branches
    """Get the pulse level data, which could be a single value or average, or could be per pulse data. Return average.

    :param source_data: IrradEventXRaySourceData table
    :param modality: RF or DX to limit what we look for
    :return: dict of values
    """

    try:
        kvp = source_data.kvp_set.get().kvp
    except MultipleObjectsReturned:
        kvp = (
            source_data.kvp_set.all()
            .exclude(kvp__isnull=True)
            .exclude(kvp__exact=0)
            .aggregate(Avg("kvp"))["kvp__avg"]
        )
    except ObjectDoesNotExist:
        kvp = None

    if modality == "DX":
        try:
            exposure_set = source_data.exposure_set.get()
            uas = exposure_set.exposure
            if isinstance(uas, Number):
                mas = exposure_set.convert_uAs_to_mAs()
            else:
                mas = None
        except MultipleObjectsReturned:
            mas = (
                source_data.exposure_set.all()
                .exclude(exposure__isnull=True)
                .exclude(exposure__exact=0)
                .aggregate(Avg("exposure"))["exposure__avg"]
            )
            mas = mas / 1000.0
        except ObjectDoesNotExist:
            mas = None
    else:
        mas = None

    if modality == "RF":
        try:
            xray_tube_current = source_data.xraytubecurrent_set.get().xray_tube_current
        except MultipleObjectsReturned:
            xray_tube_current = (
                source_data.xraytubecurrent_set.all()
                .exclude(xray_tube_current__isnull=True)
                .exclude(xray_tube_current__exact=0)
                .aggregate(Avg("xray_tube_current"))["xray_tube_current__avg"]
            )
        except ObjectDoesNotExist:
            xray_tube_current = None
    else:
        xray_tube_current = None

    if modality == "RF":
        try:
            pulse_width = source_data.pulsewidth_set.get().pulse_width
        except MultipleObjectsReturned:
            pulse_width = (
                source_data.pulsewidth_set.all()
                .exclude(pulse_width__isnull=True)
                .exclude(pulse_width__exact=0)
                .aggregate(Avg("pulse_width"))["pulse_width__avg"]
            )
        except ObjectDoesNotExist:
            pulse_width = None
    else:
        pulse_width = None

    return {
        "kvp": kvp,
        "mas": mas,
        "xray_tube_current": xray_tube_current,
        "pulse_width": pulse_width,
    }


def get_xray_filter_info(source):
    # pylint: disable=too-many-branches
    """Compile a string containing details of the filters, and a corresponding string of filter thicknesses

    :param source: exposure in question
    :return: two strings of filters and filter thicknesses
    """
    try:
        filters = ""
        filter_thicknesses = ""
        for current_filter in source.xrayfilters_set.all():
            xray_filter_material = str(
                current_filter.xray_filter_material.code_meaning
            ).lower()
            if "aluminum" in xray_filter_material:
                filters += "Al"
            elif "copper" in xray_filter_material:
                filters += "Cu"
            elif "tantalum" in xray_filter_material:
                filters += "Ta"
            elif "molybdenum" in xray_filter_material:
                filters += "Mo"
            elif "rhodium" in xray_filter_material:
                filters += "Rh"
            elif "silver" in xray_filter_material:
                filters += "Ag"
            elif "niobium" in xray_filter_material:
                filters += "Nb"
            elif "europium" in xray_filter_material:
                filters += "Eu"
            elif "lead" in xray_filter_material:
                filters += "Pb"
            else:
                filters += str(current_filter.xray_filter_material.code_meaning)
            filters += " | "
            thicknesses = [
                current_filter.xray_filter_thickness_minimum,
                current_filter.xray_filter_thickness_maximum,
            ]
            thick = ""
            if thicknesses[0] is not None and thicknesses[1] is not None:
                thick = sum(thicknesses) / len(thicknesses)
            elif thicknesses[0] is not None:
                thick = thicknesses[0]
            elif thicknesses[1] is not None:
                thick = thicknesses[1]
            if thick:
                thick = round(thick, 4)
            filter_thicknesses += str(thick) + " | "
        filters = filters[:-3]
        filter_thicknesses = filter_thicknesses[:-3]
    except (ObjectDoesNotExist, AttributeError):
        filters = None
        filter_thicknesses = None
    return filters, filter_thicknesses


def get_anode_target_material(source):
    """Return abbreviated version of anode target material

    :param source: x-ray source data for the exposure
    :return: string containing target material abbreviation
    """
    try:
        anode_target_material = str(source.anode_target_material.code_meaning).lower()

        if "molybdenum" in anode_target_material:
            anode = "Mo"
        elif "rhodium" in anode_target_material:
            anode = "Rh"
        elif "tungsten" in anode_target_material:
            anode = "W"
        else:
            anode = str(source.anode_target_material.code_meaning)

    except (ObjectDoesNotExist, AttributeError):
        anode = None
    return anode


def create_xlsx(task):
    """Function to create the xlsx temporary file

    :param task: Export task object
    :return: workbook, temp file
    """

    try:
        temp_xlsx = TemporaryFile()
        book = Workbook(
            temp_xlsx,
            {
                "strings_to_numbers": False,
                "remove_timezone": True,
                "constant_memory": True,
            },
        )
    except (OSError, IOError) as e:
        logger.error(f"Error saving xlsx temporary file ({e.errno}): {e.strerror}")
        return None, None
    except Exception:
        logger.error(f"Unexpected error: {sys.exc_info()[0]}")
        return None, None
    else:
        task.progress = "Workbook created"
        task.save()
        return temp_xlsx, book


def create_csv(task):
    """Function to create the csv temporary file

    :param task: Export task object
    :return: workbook, temp file
    """

    try:
        export_filename = f'{task.modality.lower()}export{task.export_date.strftime("%Y%m%d-%H%M%S%f")}.csv'
        task.filename.save(export_filename, ContentFile(codecs.BOM_UTF8))
        task.save()
        temp_csv = open(task.filename.path, "a", newline="", encoding="utf-8")
        writer = csv.writer(temp_csv, dialect="excel")
    except (OSError, IOError) as e:
        logger.error(f"Error saving csv temporary file ({e.errno}): {e.strerror}")
        return None, None
    except Exception:
        logger.error(f"Unexpected error: {sys.exc_info()[0]}")
        return None, None
    else:
        task.progress = "CSV file created"
        task.save()
        return temp_csv, writer


def write_export(task, filename, temp_file, datestamp):
    """Function to write out the exported xlsx or csv file.

    :param task: Export task object
    :param filename: Filename to use
    :param temp_file: Temporary file
    :param datestamp: dat and time export function started
    :return: Nothing
    """

    try:
        task.filename.save(filename, File(temp_file))
    except (OSError, IOError) as e:
        logger.error(f"Error saving export file ({e.errno}): {e.strerror}")

    task.status = "COMPLETE"
    task.processtime = (datetime.datetime.now() - datestamp).total_seconds()
    task.save()


def create_summary_sheet(
    task, studies, book, summary_sheet, modality=None
):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    """Create summary sheet for xlsx exports

    :param task: Export task object
    :param studies: study level object that has been exported
    :param book: xlsxwriter book to work on
    :param summary_sheet: worksheet object
    :param modality: the DICOM modality of the export
    :return: nothing
    """

    # Populate summary sheet
    task.progress = "Now populating the summary sheet..."
    task.save()

    version = __version__
    titleformat = book.add_format()
    titleformat.set_font_size = 22
    titleformat.set_font_color = "#FF0000"
    titleformat.set_bold()
    toplinestring = f"XLSX Export from OpenREM version {version} on {str(datetime.datetime.now())}"
    linetwostring = (
        "OpenREM is copyright 2019 The Royal Marsden NHS Foundation Trust, and available under the GPL. "
        "See http://openrem.org"
    )
    summary_sheet.write(0, 0, toplinestring, titleformat)
    summary_sheet.write(1, 0, linetwostring)

    # Number of exams
    summary_sheet.write(3, 0, "Total number of exams")
    summary_sheet.write(3, 1, studies.count())

    # Obtain the system-level enable_standard_names setting
    enable_standard_names = are_standard_names_enabled()

    if modality in ["NM"]:
        enable_standard_names = False

    required_fields = ["pk", "study_description", "requested_procedure_code_meaning"]
    column_names = ["pk", "Study description", "Requested procedure"]
    if enable_standard_names:
        required_fields.append("standard_names__standard_name")
        column_names.append("Standard study name")

    # DataFrame containing study description, requested procedure data and possibly standard study names.
    # Note that a single exam can have more than one standard study name associated with it because a standard name
    # can be mapped to study description, requested procedure, and also to procedure. When we are counting up study
    # description and requested procedure occurences it is important to drop private key duplicates to avoid double
    # counting.
    df = pd.DataFrame.from_records(
        data=studies.values_list(*required_fields), columns=column_names
    )

    # Get the study descriptions used and their frequency
    study_description_frequency = (
        df.drop_duplicates(subset="pk")["Study description"]
        .value_counts(dropna=False)
        .sort_index(ascending=True)
        .sort_values(ascending=False)
    )
    study_description_frequency = study_description_frequency.reset_index()
    study_description_frequency.columns = ["Study description", "Frequency"]
    study_description_frequency["Frequency"] = study_description_frequency[
        "Frequency"
    ].astype("UInt32")
    study_description_frequency["BlankCol"] = None

    # Get the requested procedures used and their frequency
    requested_procedure_frequency = (
        df.drop_duplicates(subset="pk")["Requested procedure"]
        .value_counts(dropna=False)
        .sort_index(ascending=True)
        .sort_values(ascending=False)
    )
    requested_procedure_frequency = requested_procedure_frequency.reset_index()
    requested_procedure_frequency.columns = ["Requested procedure", "Frequency"]
    requested_procedure_frequency["Frequency"] = requested_procedure_frequency[
        "Frequency"
    ].astype("UInt32")
    requested_procedure_frequency["BlankCol"] = None

    if enable_standard_names:
        # Get the standard study names used and their frequency
        standard_study_name_frequency = (
            df["Standard study name"]
            .value_counts(dropna=False)
            .sort_index(ascending=True)
            .sort_values(ascending=False)
        )
        standard_study_name_frequency = standard_study_name_frequency.reset_index()
        standard_study_name_frequency.columns = ["Standard study name", "Frequency"]
        standard_study_name_frequency["Frequency"] = standard_study_name_frequency[
            "Frequency"
        ].astype("UInt32")
        standard_study_name_frequency["BlankCol"] = None

    # Get the acquisition protocols used and their frequency
    required_fields = []
    column_names = []
    acq_df = None
    if modality in ["DX", "RF", "MG"]:
        required_fields.extend(
            ["irradeventxraydata__pk", "irradeventxraydata__acquisition_protocol"]
        )
        column_names.extend(["pk", "Acquisition protocol"])

        if enable_standard_names:
            required_fields.append(
                "irradeventxraydata__standard_protocols__standard_name"
            )
            column_names.append("Standard acquisition name")

        acq_df = pd.DataFrame.from_records(
            data=ProjectionXRayRadiationDose.objects.filter(
                general_study_module_attributes__in=studies.values("pk")
            ).values_list(*required_fields),
            columns=column_names,
        )

    elif modality in "CT":
        required_fields.extend(
            [
                "ctirradiationeventdata__pk",
                "ctirradiationeventdata__acquisition_protocol",
            ]
        )
        column_names.extend(["pk", "Acquisition protocol"])

        if enable_standard_names:
            required_fields.append(
                "ctirradiationeventdata__standard_protocols__standard_name"
            )
            column_names.append("Standard acquisition name")

        acq_df = pd.DataFrame.from_records(
            data=CtRadiationDose.objects.filter(
                general_study_module_attributes__in=studies.values("pk")
            ).values_list(*required_fields),
            columns=column_names,
        )

    acquisition_protocol_frequency = None
    if len(required_fields) != 0:

        acquisition_protocol_frequency = (
            acq_df["Acquisition protocol"]
            .value_counts(dropna=False)
            .sort_index(ascending=True)
            .sort_values(ascending=False)
            .reset_index()
        )
        acquisition_protocol_frequency.columns = ["Acquisition protocol", "Frequency"]
        acquisition_protocol_frequency["Frequency"] = acquisition_protocol_frequency[
            "Frequency"
        ].astype("UInt32")
        acquisition_protocol_frequency["BlankCol"] = None

        if enable_standard_names:
            std_acquisition_protocol_frequency = (
                acq_df["Standard acquisition name"]
                .value_counts(dropna=False)
                .sort_index(ascending=True)
                .sort_values(ascending=False)
                .reset_index()
            )
            std_acquisition_protocol_frequency.columns = [
                "Standard acquisition name",
                "Frequency",
            ]
            std_acquisition_protocol_frequency["Frequency"] = (
                std_acquisition_protocol_frequency["Frequency"].astype("UInt32")
            )
            std_acquisition_protocol_frequency["BlankCol"] = None

    # Now write the data to the worksheet
    # Write the column titles
    col_titles = [
        "Study Description",
        "Frequency",
        "",
        "Requested Procedure",
        "Frequency",
        "",
        "Acquisition protocol",
        "Frequency",
        "",
    ]
    if enable_standard_names:
        col_titles.extend(
            [
                "Standard Study Name",
                "Frequency",
                "",
                "Standard Acquisition Name",
                "Frequency",
                "",
            ]
        )

    summary_sheet.write_row(5, 0, col_titles)

    # Widen the name columns
    summary_sheet.set_column("A:A", 25)
    summary_sheet.set_column("D:D", 25)
    summary_sheet.set_column("G:G", 25)
    summary_sheet.set_column("J:J", 25)
    summary_sheet.set_column("M:M", 25)

    # Write the frequency data to the xlsx file
    combined_df = pd.concat(
        [
            study_description_frequency,
            requested_procedure_frequency,
            acquisition_protocol_frequency,
        ],
        axis=1,
    )

    if enable_standard_names:
        combined_df = pd.concat(
            [
                combined_df,
                standard_study_name_frequency,
                std_acquisition_protocol_frequency,
            ],
            axis=1,
        )

    combined_df = combined_df.where(pd.notnull(combined_df), None)

    for idx in combined_df.index:
        summary_sheet.write_row(
            idx + 6,
            0,
            [
                None if x is pd.NA or not pd.notna(x) else x
                for x in combined_df.iloc[idx].to_list()
            ],
        )


def abort_if_zero_studies(num_studies, tsk):
    """Function to update progress and status if filter is empty

    :param num_studies: study count in fiilter
    :param tsk: export task
    :return: bool - True if should abort
    """
    if not num_studies:
        tsk.status = "ERROR"
        tsk.progress = "Export aborted - zero studies in the filter!"
        tsk.save()
        return True
    else:
        tsk.progress = "Required study filter complete."
        tsk.save()
        return False


def create_export_task(
    task_id, modality, export_type, date_stamp, pid, user, filters_dict
):
    """Create export task, add filter details and Task UUID to export table to track later

    :param task_id: The id allocated for this task
    :param modality: export modality
    :param export_type: CSV, XLSX or special
    :param date_stamp: datetime export started
    :param pid: If user is permitted to and requested patient names and/or IDs
    :param user: logged in user
    :param filters_dict: filters from GET
    :return: Exports database object
    """
    if task_id is None:
        task_id = str(uuid.uuid4())

    removed_blanks = {k: v for k, v in filters_dict.items() if v}
    if removed_blanks:
        if "submit" in removed_blanks:
            del removed_blanks["submit"]
        if "csrfmiddlewaretoken" in removed_blanks:
            del removed_blanks["csrfmiddlewaretoken"]
        if "itemsPerPage" in removed_blanks:
            del removed_blanks["itemsPerPage"]
    no_plot_filters_dict = {k: v for k, v in removed_blanks.items() if "plot" not in k}

    task = Exports.objects.create(task_id=task_id)
    task.modality = modality
    task.export_type = export_type
    task.export_date = date_stamp
    task.progress = "Query filters imported, task started"
    task.status = "CURRENT"
    task.includes_pid = pid
    task.export_user_id = user
    try:
        task.export_summary = "<br/>".join(
            ": ".join(_) for _ in no_plot_filters_dict.items()
        )
    except TypeError:
        task.export_summary = no_plot_filters_dict
    task.save()

    return task


def transform_to_one_row_per_exam(
    df,
    acquisition_cat_field_names,
    acquisition_int_field_names,
    acquisition_val_field_names,
    exam_cat_field_names,
    exam_date_field_names,
    exam_int_field_names,
    exam_obj_field_names,
    exam_time_field_names,
    exam_val_field_names,
    all_field_names,
):
    # pylint: disable=too-many-locals
    """Transform a DataFrame with one acquisition per row into a DataFrame with
    one exam per row, including all acquisitions for that exam.
    """

    if settings.DEBUG:
        print("Initial DataFrame created")
        df.info()

    exam_cat_f_names = exam_cat_field_names[:]

    # Make DataFrame columns category type where appropriate
    cat_field_names = exam_cat_f_names + acquisition_cat_field_names
    df[cat_field_names] = df[cat_field_names].astype("category")

    # Make DataFrame columns datetime type where appropriate
    for date_field in exam_date_field_names:
        df[date_field] = pd.to_datetime(df[date_field], format="%Y-%m-%d")

    # Make DataFrame columns float32 type where appropriate
    val_field_names = exam_val_field_names + acquisition_val_field_names
    df[val_field_names] = df[val_field_names].astype("float32")

    # Make DataFrame columns UInt32 type where appropriate
    int_field_names = exam_int_field_names + acquisition_int_field_names
    df[int_field_names] = df[int_field_names].astype("UInt32")

    if settings.DEBUG:
        print("DataFrame column types changed to reduce memory consumption")
        df.info()

    if "Standard study name" in df.columns:
        df = create_standard_name_df_columns(df)

        # Remove the original standard study name column from the list of exam category field names and then
        # add the three standard name columns
        exam_cat_f_names.remove("Standard study name")
        exam_cat_f_names.extend(
            ["Standard study name 1", "Standard study name 2", "Standard study name 3"]
        )

        # Make the exam_cat_f_names a categorical column (saves server memory)
        df[exam_cat_f_names] = df[exam_cat_f_names].astype("category")

    # Drop any duplicate acquisition pk rows
    df.drop_duplicates(subset="Acquisition pk", inplace=True)

    # Reformat the DataFrame so that we have one row per exam, with sets of columns for each acquisition data
    g = df.groupby("pk").cumcount().add(1)
    exam_field_names = (
        exam_cat_f_names
        + exam_obj_field_names
        + exam_date_field_names
        + exam_time_field_names
        + exam_int_field_names
        + exam_val_field_names
    )
    exam_field_names.append(g)
    df = df.set_index(exam_field_names).unstack().sort_index(axis=1, level=1)
    df.columns = ["E{} {}".format(b, a) for a, b in df.columns]  # pylint: disable=consider-using-f-string
    df = df.reset_index()

    # Set datatypes of the exam-level integer and value fields again because the reformat undoes the earlier changes
    df[exam_int_field_names] = df[exam_int_field_names].astype("UInt32")
    df[exam_val_field_names] = df[exam_val_field_names].astype("float32")

    # Drop all pk columns
    pk_list = [i for i in df.columns if "pk" in i]
    df = df.drop(pk_list, axis=1)

    if settings.DEBUG:
        print("DataFrame reformatted")
        df.info()

    # Sort date by descending date and time
    df.sort_values(
        by=["Study date", "Study time"], ascending=[False, False], inplace=True
    )

    return df


def create_standard_name_df_columns(df):
    num_std_names = len(df["Standard study name"].unique().categories)

    if num_std_names:
        std_name_df = (
            df.groupby("pk")["Standard study name"]
            .apply(lambda x: pd.Series(list(x.unique())))
            .unstack()
        )
        num_std_name_cols = len(std_name_df.columns)
        std_name_df.columns = [
            "Standard study name {}".format(a + 1) for a in std_name_df.columns  # pylint: disable=consider-using-f-string
        ]
        std_name_df = std_name_df.reset_index()

        # Join the std_name_df to df using Study ID as an index
        df = df.join(std_name_df.set_index(["pk"]), on=["pk"])

        # Now move the columns so they are next to the original "Standard Name" column
        std_name_col_idx = df.columns.get_loc("Standard study name")
        if num_std_name_cols >= 1:
            col = df.pop("Standard study name 1")
            df.insert(std_name_col_idx, col.name, col)

        if num_std_name_cols >= 2:
            col = df.pop("Standard study name 2")
            df.insert(std_name_col_idx + 1, col.name, col)
        else:
            df["Standard study name 2"] = ""
            col = df.pop("Standard study name 2")
            df.insert(std_name_col_idx + 1, col.name, col)

        if num_std_name_cols >= 3:
            col = df.pop("Standard study name 3")
            df.insert(std_name_col_idx + 2, col.name, col)
        else:
            df["Standard study name 3"] = ""
            col = df.pop("Standard study name 3")
            df.insert(std_name_col_idx + 2, col.name, col)

    else:
        # There were no standard name entries
        df["Standard study name 1"] = ""
        df["Standard study name 2"] = ""
        df["Standard study name 3"] = ""

    # Then drop the original standard name column
    df.drop(columns=["Standard study name"], inplace=True)

    return df


def optimise_df_dtypes(
    df,
    acquisition_cat_field_names,
    acquisition_int_field_names,
    acquisition_val_field_names,
    exam_cat_field_names,
    exam_date_field_names,
    exam_int_field_names,
    exam_val_field_names,
):
    # Optimise the data frame types to minimise memory usage
    # Make DataFrame columns category type where appropriate
    cat_field_names = exam_cat_field_names + acquisition_cat_field_names
    df[cat_field_names] = df[cat_field_names].astype("category")
    # Make DataFrame columns datetime type where appropriate
    for date_field in exam_date_field_names:
        df[date_field] = pd.to_datetime(df[date_field], format="%Y-%m-%d")
    # Make DataFrame columns float32 type where appropriate
    val_field_names = exam_val_field_names + acquisition_val_field_names
    df[val_field_names] = df[val_field_names].astype("float32")
    # Make DataFrame columns UInt32 type where appropriate
    int_field_names = exam_int_field_names + acquisition_int_field_names
    df[int_field_names] = df[int_field_names].astype("UInt32")


def write_row_to_acquisition_sheet(acq_df, acquisition, book, worksheet_log, modality):
    # Make the acquisition name safe for an Excel sheet name
    acquisition = sheet_name(acquisition)
    sheet = book.get_worksheet_by_name(acquisition)
    sheet_row = worksheet_log[acquisition]

    # Drop any pk columns
    pk_cols = [i for i in acq_df.columns if "pk" in i]
    acq_df = acq_df.drop(pk_cols, axis=1)

    if sheet_row == 0:
        sheet.write_row(0, 0, acq_df.columns)
        sheet_row = 1
        worksheet_log[acquisition] = sheet_row

    for _, row in acq_df.iterrows():
        sheet.write_row(sheet_row, 0, row.fillna(""))
        sheet_row = sheet_row + 1

    if modality in ["NM"]:
        transform_nm_datetime_columns(book, sheet, acq_df)

    transform_datetime_columns(book, sheet, acq_df)

    worksheet_log[acquisition] = sheet_row


def replace_long_target_with_short(x):
    x = str(x)
    if "molybdenum" in x.lower():
        return "Mo"
    elif "rhodium" in x.lower():
        return "Rh"
    elif "tungsten" in x.lower():
        return "W"
    return x


def replace_long_filter_with_short(x):
    # pylint: disable=too-many-return-statements
    x = str(x)
    if "aluminum" in x.lower():
        return "Al"
    elif "copper" in x.lower():
        return "Cu"
    elif "tantalum" in x.lower():
        return "Ta"
    elif "molybdenum" in x.lower():
        return "Mo"
    elif "rhodium" in x.lower():
        return "Rh"
    elif "silver" in x.lower():
        return "Ag"
    elif "niobium" in x.lower():
        return "Nb"
    elif "europium" in x.lower():
        return "Eu"
    elif "lead" in x.lower():
        return "Pb"
    else:
        return x


def export_using_pandas(
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
):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    # Add summary sheet and all data sheet
    summarysheet = book.add_worksheet("Summary")
    wsalldata = book.add_worksheet("All data")

    # ====================================================================================
    # Write the all data sheet
    # This code is taken from the ct_csv method...
    qs_chunk_size = 10000

    if pid and name:
        exam_obj_fields.append("patientmoduleattr__patient_name")
        exam_obj_field_names.append("Patient name")

    if pid and patid:
        exam_obj_fields.append("patientmoduleattr__patient_id")
        exam_obj_field_names.append("Patient ID")

    if pid and (patid or name):
        exam_obj_fields.append("patientmoduleattr__patient_birth_date")
        exam_obj_field_names.append("Patient date of birth")

    if enable_standard_names:
        exam_cat_fields.append("standard_names__standard_name")
        exam_cat_field_names.append("Standard study name")
        acquisition_cat_fields.append(acquisition_cat_field_std_name)
        acquisition_cat_field_names.append(acquisition_cat_field_name_std_name)

    exam_fields = (
        exam_cat_fields
        + exam_obj_fields
        + exam_date_fields
        + exam_time_fields
        + exam_int_fields
        + exam_val_fields
    )
    acquisition_fields = (
        acquisition_int_fields + acquisition_cat_fields + acquisition_val_fields
    )
    all_fields = exam_fields + acquisition_fields
    exam_field_names = (
        exam_cat_field_names
        + exam_obj_field_names
        + exam_date_field_names
        + exam_time_field_names
        + exam_int_field_names
        + exam_val_field_names
    )
    acquisition_field_names = (
        acquisition_int_field_names
        + acquisition_cat_field_names
        + acquisition_val_field_names
    )
    all_field_names = exam_field_names + acquisition_field_names

    # Create a series of DataFrames by chunking the queryset into groups of accession numbers.
    # Chunking saves server memory at the expense of speed.

    # Create a work sheet for each acquisition protocol present in the data in alphabetical order
    # and a dictionary to hold the number of rows that have been written to each protocol sheet
    # Get the acquisition protocols used and their frequency
    required_fields = fields_for_acquisition_frequency
    column_names = field_names_for_acquisition_frequency

    if enable_standard_names:
        required_fields.append(field_for_acquisition_frequency_std_name)
        column_names.append(field_name_for_acquisition_frequency_std_name)

    acq_df = pd.DataFrame.from_records(
        data=qs.values_list(*required_fields), columns=column_names
    )
    acq_df["Acquisition protocol"] = acq_df["Acquisition protocol"].astype("category")

    if enable_standard_names:
        acq_df["Standard acquisition name"] = acq_df[
            "Standard acquisition name"
        ].astype("category")

    required_sheets = acq_df.sort_values("Acquisition protocol")[
        "Acquisition protocol"
    ].unique()

    if enable_standard_names:
        std_name_sheets = (
            acq_df.sort_values("Standard acquisition name")["Standard acquisition name"]
            .dropna()
            .unique()
        )
        std_name_sheets = "[standard] " + std_name_sheets.categories
        required_sheets = np.concatenate((required_sheets, std_name_sheets))

    worksheet_log = {}
    for current_name in required_sheets:
        if current_name in (None, np.nan, ""):
            current_name = "Unknown"

        current_name = sheet_name(current_name)

        if current_name not in book.sheetnames.keys():
            book.add_worksheet(current_name)
            worksheet_log[current_name] = 0

    current_row = 1

    # Generate a list of non-null accession numbers
    filtering_criteria = "accession_number__in"
    values_for_filtering = [
        x[0]
        for x in qs.order_by("-study_date", "-study_time")
        .filter(accession_number__isnull=False)
        .values_list("accession_number")
    ]
    values_field_list = all_fields + ct_dose_check_fields
    field_names_list = all_field_names + ct_dose_check_field_names
    n_entries = len(values_for_filtering)
    extra_msg_txt = ""

    current_row = write_out_data_as_chunks(
        acquisition_cat_field_names,
        acquisition_int_field_names,
        acquisition_val_field_names,
        all_field_names,
        book,
        ct_dose_check_field_names,
        current_row,
        enable_standard_names,
        exam_cat_field_names,
        exam_date_field_names,
        exam_int_field_names,
        exam_obj_field_names,
        exam_time_field_names,
        exam_val_field_names,
        extra_msg_txt,
        field_names_list,
        filtering_criteria,
        modality,
        n_entries,
        qs,
        qs_chunk_size,
        tsk,
        values_field_list,
        values_for_filtering,
        worksheet_log,
        wsalldata,
    )

    # Now write out any None accession number data if any such data is present
    fields_for_none_accession = all_fields
    field_names_for_non_accession = all_field_names
    if modality in ["CT"]:
        fields_for_none_accession.extend(ct_dose_check_fields)
        field_names_for_non_accession.extend(ct_dose_check_field_names)

    # Generate a list of non-null private keys
    filtering_criteria = "pk__in"
    values_for_filtering = [
        x[0]
        for x in qs.order_by("-study_date", "-study_time")
        .filter(accession_number__isnull=True)
        .values_list("pk")
    ]
    values_field_list = fields_for_none_accession
    field_names_list = field_names_for_non_accession
    n_entries = len(values_for_filtering)
    extra_msg_txt = "no "

    current_row = write_out_data_as_chunks(
        acquisition_cat_field_names,
        acquisition_int_field_names,
        acquisition_val_field_names,
        all_field_names,
        book,
        ct_dose_check_field_names,
        current_row,
        enable_standard_names,
        exam_cat_field_names,
        exam_date_field_names,
        exam_int_field_names,
        exam_obj_field_names,
        exam_time_field_names,
        exam_val_field_names,
        extra_msg_txt,
        field_names_list,
        filtering_criteria,
        modality,
        n_entries,
        qs,
        qs_chunk_size,
        tsk,
        values_field_list,
        values_for_filtering,
        worksheet_log,
        wsalldata,
    )

    # Now create the summary sheet
    create_summary_sheet(tsk, qs, book, summarysheet, modality=modality)

    tsk.progress = "Finished populating the summary sheet"
    tsk.save()
    book.close()
    tsk.progress = "XLSX book written."
    tsk.save()
    xlsxfilename = f"{modality.lower()}export{datestamp.strftime('%Y%m%d-%H%M%S%f')}.xlsx"
    write_export(tsk, xlsxfilename, tmpxlsx, datestamp)


def write_out_data_as_chunks(
    acquisition_cat_field_names,
    acquisition_int_field_names,
    acquisition_val_field_names,
    all_field_names,
    book,
    ct_dose_check_field_names,
    current_row,
    enable_standard_names,
    exam_cat_field_names,
    exam_date_field_names,
    exam_int_field_names,
    exam_obj_field_names,
    exam_time_field_names,
    exam_val_field_names,
    extra_msg_txt,
    field_names_list,
    filtering_criteria,
    modality,
    n_entries,
    qs,
    qs_chunk_size,
    tsk,
    values_field_list,
    values_for_filtering,
    worksheet_log,
    wsalldata,
):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-branches
    for chunk_min_idx in range(0, n_entries, qs_chunk_size):

        chunk_max_idx = chunk_min_idx + qs_chunk_size
        if chunk_max_idx > n_entries:
            chunk_max_idx = n_entries

        tsk.progress = f"Working on {chunk_min_idx + 1} to {chunk_max_idx} with {extra_msg_txt} accession numbers"
        tsk.save()

        filter_dict = {
            filtering_criteria: values_for_filtering[chunk_min_idx:chunk_max_idx]
        }
        data = qs.order_by().filter(**filter_dict).values_list(*(values_field_list))

        if data.exists():
            # Clear the query cache
            django.db.reset_queries()

            df_unprocessed = pd.DataFrame.from_records(
                data=data,
                columns=(field_names_list),
                coerce_float=True,
            )

            if modality in ["NM"]:
                df_unprocessed = create_nm_columns(qs, filter_dict, df_unprocessed)

            if modality in ["CT"]:
                fields_to_remove = [
                    "Dose check alerts",
                    "S1 Source name",
                    "S2 Source name",
                ]
                for field_name in fields_to_remove:
                    if field_name in acquisition_cat_field_names:
                        acquisition_cat_field_names.remove(field_name)

                fields_to_add = ["Source name"]
                for field_name in fields_to_add:
                    if field_name not in acquisition_cat_field_names:
                        acquisition_cat_field_names.append(field_name)

                fields_to_remove = [
                    "S1 kVp",
                    "S1 mA",
                    "S1 Maximum mA",
                    "S1 Exposure time per rotation",
                    "S2 kVp",
                    "S2 mA",
                    "S2 Maximum mA",
                    "S2 Exposure time per rotation",
                ]
                for field_name in fields_to_remove:
                    if field_name in acquisition_val_field_names:
                        acquisition_val_field_names.remove(field_name)

                fields_to_add = [
                    "kVp",
                    "mA",
                    "Maximum mA",
                    "Exposure time per rotation",
                ]
                for field_name in fields_to_add:
                    if field_name not in acquisition_val_field_names:
                        acquisition_val_field_names.append(field_name)

            if modality in ["DX", "MG"]:
                fields_to_remove = ["Filter thicknesses (mm)"]
                for field_name in fields_to_remove:
                    if field_name in acquisition_cat_field_names:
                        acquisition_cat_field_names.remove(field_name)

            optimise_df_dtypes(
                df_unprocessed,
                acquisition_cat_field_names,
                acquisition_int_field_names,
                acquisition_val_field_names,
                exam_cat_field_names,
                exam_date_field_names,
                exam_int_field_names,
                exam_val_field_names,
            )

            transform_dap_uas_units(df_unprocessed)

            if modality in ["CT"]:
                df_unprocessed = create_dose_check_and_source_columns(
                    acquisition_cat_field_names,
                    acquisition_val_field_names,
                    ct_dose_check_field_names,
                    df_unprocessed,
                )

            if modality in ["MG"]:
                df_unprocessed = create_image_view_modifier_column(df_unprocessed)
                df_unprocessed = create_mg_pulse_columns(df_unprocessed)
                df_unprocessed = create_target_column(df_unprocessed)

            if modality in ["DX", "MG"]:
                df_unprocessed = create_filter_columns(
                    acquisition_cat_field_names, df_unprocessed
                )

                fields_to_remove = ["Filter thickness min", "Filter thickness max"]
                for field_name in fields_to_remove:
                    if field_name in acquisition_val_field_names:
                        acquisition_val_field_names.remove(field_name)
                    if field_name in acquisition_int_field_names:
                        acquisition_int_field_names.remove(field_name)

            df = transform_to_one_row_per_exam(
                df_unprocessed,
                acquisition_cat_field_names,
                acquisition_int_field_names,
                acquisition_val_field_names,
                exam_cat_field_names,
                exam_date_field_names,
                exam_int_field_names,
                exam_obj_field_names,
                exam_time_field_names,
                exam_val_field_names,
                all_field_names,
            )

            # Write the headings to the sheet (over-writing each time, but this ensures we'll include the study
            # with the most events without doing anything complicated to generate the headings)
            wsalldata.write_row(0, 0, df.columns)

            # Write the DataFrame to the all data sheet
            for _, row in df.iterrows():
                wsalldata.write_row(current_row, 0, row.fillna(""))
                current_row = current_row + 1

            if modality in ["NM"]:
                transform_nm_datetime_columns(book, wsalldata, df)

            transform_datetime_columns(book, wsalldata, df)

            # Write out data to the acquisition protocol sheets
            df = df_unprocessed

            df = create_standard_name_columns(df, exam_cat_field_names)

            # Drop any duplicate acquisition pk rows
            df.drop_duplicates(subset="Acquisition pk", inplace=True)

            # Sort the data by descending date and time
            df.sort_values(
                by=["Study date", "Study time"], ascending=[False, False], inplace=True
            )

            write_acquisition_data(book, df, worksheet_log, modality)

            write_standard_acquisition_data(
                book, df, enable_standard_names, worksheet_log, modality
            )

    return current_row


def create_nm_columns(qs, filter_dict, df_unprocessed):
    # pylint: disable=too-many-locals

    # person_participant
    person_participant_val_fields = [  # pylint: disable=line-too-long
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__personparticipant__person_name",
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__personparticipant__person_role_in_procedure_cid__code_meaning",  # pylint: disable=line-too-long
    ]

    person_participant_val_fields_names = [
        "Person participant name",
        "Person participant role",
    ]

    person_participant_int_fields = [
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__pk",
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__personparticipant__pk",
    ]

    person_participant_int_fields_names = [
        "Acquisition pk",
        "Person Participant pk",
    ]

    person_participant_values_field_list = (
        person_participant_int_fields + person_participant_val_fields
    )
    person_participant_field_names_list = (
        person_participant_int_fields_names + person_participant_val_fields_names
    )

    df_unprocessed = create_multiindex_columns(
        qs,
        filter_dict,
        df_unprocessed,
        person_participant_values_field_list,
        person_participant_field_names_list,
    )

    # organ_dose
    organ_dose_val_fields = [  # pylint: disable=line-too-long
        Coalesce(
            F(
                "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__organdose__reference_authority_code_id__code_meaning"  # pylint: disable=line-too-long
            ),
            F(
                "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__organdose__reference_authority_text"  # pylint: disable=line-too-long
            ),
        ),
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__organdose__finding_site__code_meaning",  # pylint: disable=line-too-long
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__organdose__laterality__code_meaning",  # pylint: disable=line-too-long
    ]

    organ_dose_val_fields_names = [
        "Organ Dose Reference Authority",
        "Organ Dose Finding Site",
        "Organ Dose Laterality",
    ]

    organ_dose_int_fields = [
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__pk",
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__organdose__pk",
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__organdose__organ_dose",
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__organdose__mass",
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__organdose__measurement_method",
    ]

    organ_dose_int_fields_names = [
        "Acquisition pk",
        "Organ Dose pk",
        "Organ Dose (mGy)",
        "Organ Dose Mass (g)",
        "Organ Dose Measurement Method",
    ]

    organ_dose_values_field_list = organ_dose_int_fields + organ_dose_val_fields
    organ_dose_field_names_list = (
        organ_dose_int_fields_names + organ_dose_val_fields_names
    )

    df_unprocessed = create_multiindex_columns(
        qs,
        filter_dict,
        df_unprocessed,
        organ_dose_values_field_list,
        organ_dose_field_names_list,
    )

    # patient_state
    patient_state_val_fields = [  # pylint: disable=line-too-long
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__patientstate__patient_state__code_meaning",  # pylint: disable=line-too-long
    ]

    patient_state_val_fields_names = [
        "Patient state",
    ]

    patient_state_int_fields = [
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__pk",
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__patientstate__pk",
    ]

    patient_state_int_fields_names = [
        "Acquisition pk",
        "Organ Dose pk",
    ]

    patient_state_values_field_list = (
        patient_state_int_fields + patient_state_val_fields
    )
    patient_state_field_names_list = (
        patient_state_int_fields_names + patient_state_val_fields_names
    )

    df_unprocessed = create_multiindex_columns(
        qs,
        filter_dict,
        df_unprocessed,
        patient_state_values_field_list,
        patient_state_field_names_list,
    )

    # glomerular
    glomerular_val_fields = [  # pylint: disable=line-too-long
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__glomerularfiltrationrate__measurement_method__code_meaning",  # pylint: disable=line-too-long
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__glomerularfiltrationrate__equivalent_meaning_of_concept_name__code_meaning",  # pylint: disable=line-too-long
    ]

    glomerular_val_fields_names = [
        "Measurement Method",
        "Equivalent meaning of concept name",
    ]

    glomerular_int_fields = [  # pylint: disable=line-too-long
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__pk",
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__glomerularfiltrationrate__pk",  # pylint: disable=line-too-long
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__glomerularfiltrationrate__glomerular_filtration_rate",  # pylint: disable=line-too-long
    ]

    glomerular_int_fields_names = [
        "Acquisition pk",
        "Organ Dose pk",
        "Glomerular Filtration Rate (ml/min/1.73m^2)",
    ]

    glomerular_values_field_list = glomerular_int_fields + glomerular_val_fields
    glomerular_field_names_list = (
        glomerular_int_fields_names + glomerular_val_fields_names
    )

    df_unprocessed = create_multiindex_columns(
        qs,
        filter_dict,
        df_unprocessed,
        glomerular_values_field_list,
        glomerular_field_names_list,
    )

    # pet_series
    pet_series_val_fields = [
        "radiopharmaceuticalradiationdose__petseries__series_datetime",
        "radiopharmaceuticalradiationdose__petseries__reconstruction_method",
        "radiopharmaceuticalradiationdose__petseries__scan_progression_direction",
    ]

    pet_series_val_fields_names = [
        "PET Series Datetime",
        "Reconstruction Method",
        "Scan Progression Direction",
    ]

    pet_series_int_fields = [
        "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__pk",
        "radiopharmaceuticalradiationdose__petseries__pk",
        "radiopharmaceuticalradiationdose__petseries__number_of_slices",
        "radiopharmaceuticalradiationdose__petseries__coincidence_window_width",
        "radiopharmaceuticalradiationdose__petseries__energy_window_lower_limit",
        "radiopharmaceuticalradiationdose__petseries__energy_window_upper_limit",
        "radiopharmaceuticalradiationdose__petseries__number_of_rr_intervals",
        "radiopharmaceuticalradiationdose__petseries__number_of_time_slots",
        "radiopharmaceuticalradiationdose__petseries__number_of_time_slices",
    ]

    pet_series_int_fields_names = [
        "Acquisition pk",
        "Organ Dose pk",
        "Number Of Slices",
        "Coincidence Window Width",
        "Energy Window Lower Limit",
        "Energy Window Upper Limit",
        "Number Of RR Intervals",
        "Number Of Time Slots",
        "Number Of Time Slices",
    ]

    pet_series_values_field_list = pet_series_int_fields + pet_series_val_fields
    pet_series_field_names_list = (
        pet_series_int_fields_names + pet_series_val_fields_names
    )

    df_unprocessed = create_multiindex_columns(
        qs,
        filter_dict,
        df_unprocessed,
        pet_series_values_field_list,
        pet_series_field_names_list,
    )

    return df_unprocessed


def create_multiindex_columns(
    qs,
    filter_dict,
    df_unprocessed,
    secondary_values_field_list,
    secondary_field_names_list,
):
    organ_dose_data = (
        qs.order_by().filter(**filter_dict).values_list(*(secondary_values_field_list))
    )
    df_unprocessed_organ_dose = pd.DataFrame.from_records(
        data=organ_dose_data,
        columns=(secondary_field_names_list),
        coerce_float=True,
    )

    df_pivot = df_unprocessed_organ_dose.pivot_table(
        index="Acquisition pk",
        fill_value="",
        columns=df_unprocessed_organ_dose.groupby("Acquisition pk").cumcount() + 1,
        values=secondary_field_names_list,
        aggfunc="first",
    )

    df_pivot.columns = [f"{a} {b}" for a, b in df_pivot.columns]
    sorted_columns = sorted(df_pivot.columns, key=lambda x: int(x.rsplit(" ", 1)[1]))
    df_pivot = df_pivot[sorted_columns]
    df_unprocessed = df_unprocessed.merge(df_pivot, on="Acquisition pk", how="left")

    return df_unprocessed


def transform_datetime_columns(book, sheet, df):
    dateformat = book.add_format({"num_format": settings.XLSX_DATE})
    timeformat = book.add_format({"num_format": settings.XLSX_TIME})
    conditionaldateformat = {"type": "no_blanks", "format": dateformat}
    conditionaltimeformat = {"type": "no_blanks", "format": timeformat}
    maxrows = 1048575

    for column in df.columns:
        if " date" in column.lower():
            date_column = df.columns.get_loc(column)
            sheet.conditional_format(
                1, date_column, maxrows, date_column, conditionaldateformat
            )
        if "study time" in column.lower():
            date_column = df.columns.get_loc(column)
            sheet.conditional_format(
                1, date_column, maxrows, date_column, conditionaltimeformat
            )


def transform_nm_datetime_columns(book, sheet, df):
    datetimeformat = book.add_format(
        {"num_format": f"{settings.XLSX_DATE} {settings.XLSX_TIME}"}
    )
    conditionaldatetimeformat = {"type": "no_blanks", "format": datetimeformat}
    maxrows = 1048575
    datetimecolumns = [
        "Radiopharmaceutical Start Time",
        "Radiopharmaceutical Stop Time",
        "PET Series Datetime",
    ]

    for column in df.columns:
        for datetimecolumn in datetimecolumns:
            if datetimecolumn in column:
                date_column = df.columns.get_loc(column)
                sheet.conditional_format(
                    1, date_column, maxrows, date_column, conditionaldatetimeformat
                )


def create_image_view_modifier_column(df_unprocessed):
    df_unprocessed["View Modifier"] = (  # pylint disable=unnecessary-lambda
        df_unprocessed[df_unprocessed["View Modifier"].notnull()]
        .drop_duplicates(["Acquisition pk", "View Modifier pk"])
        .sort_values(by=["Acquisition pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["View Modifier"]
        .transform(lambda x: ",".join(x))  # pylint disable=unnecessary-lambda
    )

    return df_unprocessed


def create_mg_pulse_columns(df_unprocessed):
    df_unprocessed["kVp Concatenated"] = (  # pylint disable=unnecessary-lambda
        df_unprocessed[df_unprocessed["kVp"].notnull()]
        .drop_duplicates(["Acquisition pk", "kVp pk"])
        .sort_values(by=["kVp pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["kVp"]
        .transform(lambda x: " | ".join(map(str, x)))  # pylint disable=unnecessary-lambda
    )
    df_unprocessed["uAs Concatenated"] = (  # pylint disable=unnecessary-lambda
        df_unprocessed[df_unprocessed["uAs"].notnull()]
        .drop_duplicates(["Acquisition pk", "exposure pk"])
        .sort_values(by=["exposure pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["uAs"]
        .transform(lambda x: " | ".join(map(str, x)))  # pylint disable=unnecessary-lambda
    )
    df_unprocessed["ms Concatenated"] = (  # pylint disable=unnecessary-lambda
        df_unprocessed[df_unprocessed["ms"].notnull()]
        .drop_duplicates(["Acquisition pk", "pulsewidth pk"])
        .sort_values(by=["pulsewidth pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["ms"]
        .transform(lambda x: " | ".join(map(str, x)))  # pylint disable=unnecessary-lambda
    )
    df_unprocessed["mA Concatenated"] = (  # pylint disable=unnecessary-lambda
        df_unprocessed[df_unprocessed["mA"].notnull()]
        .drop_duplicates(["Acquisition pk", "xraytubecurrent pk"])
        .sort_values(by=["xraytubecurrent pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["mA"]
        .transform(lambda x: " | ".join(map(str, x)))  # pylint disable=unnecessary-lambda
    )

    df_unprocessed["kVp Mean"] = (
        df_unprocessed[df_unprocessed["kVp"].notnull()]
        .drop_duplicates(["Acquisition pk", "kVp pk"])
        .sort_values(by=["kVp pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["kVp"]
        .transform("mean")
    )
    df_unprocessed["uAs Mean"] = (
        df_unprocessed[df_unprocessed["uAs"].notnull()]
        .drop_duplicates(["Acquisition pk", "exposure pk"])
        .sort_values(by=["exposure pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["uAs"]
        .transform("mean")
    )
    df_unprocessed["ms Mean"] = (
        df_unprocessed[df_unprocessed["ms"].notnull()]
        .drop_duplicates(["Acquisition pk", "pulsewidth pk"])
        .sort_values(by=["pulsewidth pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["ms"]
        .transform("mean")
    )
    df_unprocessed["mA Mean"] = (
        df_unprocessed[df_unprocessed["mA"].notnull()]
        .drop_duplicates(["Acquisition pk", "xraytubecurrent pk"])
        .sort_values(by=["xraytubecurrent pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["mA"]
        .transform("mean")
    )

    df_unprocessed["kVp"] = (
        df_unprocessed[df_unprocessed["kVp"].notnull()]
        .drop_duplicates(["Acquisition pk", "kVp pk"])
        .sort_values(by=["kVp pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["kVp"]
        .transform("first")
    )
    df_unprocessed["uAs"] = (
        df_unprocessed[df_unprocessed["uAs"].notnull()]
        .drop_duplicates(["Acquisition pk", "exposure pk"])
        .sort_values(by=["exposure pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["uAs"]
        .transform("first")
    )
    df_unprocessed["ms"] = (
        df_unprocessed[df_unprocessed["ms"].notnull()]
        .drop_duplicates(["Acquisition pk", "pulsewidth pk"])
        .sort_values(by=["pulsewidth pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["ms"]
        .transform("first")
    )
    df_unprocessed["mA"] = (
        df_unprocessed[df_unprocessed["mA"].notnull()]
        .drop_duplicates(["Acquisition pk", "xraytubecurrent pk"])
        .sort_values(by=["xraytubecurrent pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["mA"]
        .transform("first")
    )

    return df_unprocessed


def create_target_column(df_unprocessed):
    df_unprocessed["Target"] = df_unprocessed["Target"].apply(
        replace_long_target_with_short
    )
    return df_unprocessed


def create_filter_columns(acquisition_cat_field_names, df_unprocessed):
    # Replace the long text filter material description with a short one
    df_unprocessed["Filters"] = df_unprocessed["Filters"].apply(
        replace_long_filter_with_short
    )

    # Calculate the mean filter thickness and put the result in a new column called "Filter thicknesses (mm)", then
    # drop the min and max columns, and then set any nan values to None in the "Filter thicknesses (mm)" column.
    df_unprocessed["Filter thicknesses (mm)"] = df_unprocessed[
        ["Filter thickness min", "Filter thickness max"]
    ].mean(axis=1)
    df_unprocessed.drop(
        ["Filter thickness min", "Filter thickness max"], axis=1, inplace=True
    )

    # Combine the "Filters" text in any rows that have matching "Acquisition pk" and the "Filters" text is not null
    df_unprocessed["Filters"] = (  # pylint disable=unnecessary-lambda
        df_unprocessed[df_unprocessed["Filters"].notnull()]
        .drop_duplicates(["Acquisition pk", "Filter pk"])
        .sort_values(by=["Filter pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["Filters"]
        .transform(lambda x: " | ".join(x))  # pylint disable=unnecessary-lambda
    )

    # Combine the "Filter thicknesses (mm)" values in any rows that have matching "Acquisition pk".
    #   1 Any np.nan values are replaced with None values
    #   2 The float values are converted to strings with 4 decimal places; any None values are set to an empty string
    #   3 The data is grouped by "Acquisition pk" and all "Filter thickness (mm)" strings joined with a " | " between.
    # Steps 1 and 2 are carried out by the first line of code below. Step 3 is carried out by the second line of code.
    df_unprocessed["Filter thicknesses (mm)"] = (
        df_unprocessed["Filter thicknesses (mm)"]
        .replace(np.nan, None)
        .apply(lambda x: f"{x:.4f}" if (x is not None) else "")
    )
    df_unprocessed["Filter thicknesses (mm)"] = (  # pylint disable=unnecessary-lambda
        df_unprocessed.drop_duplicates(["Acquisition pk", "Filter pk"])
        .sort_values(by=["Filter pk"], ascending=[True], inplace=False)
        .groupby(["Acquisition pk"])["Filter thicknesses (mm)"]
        .transform(lambda x: " | ".join(x))  # pylint disable=unnecessary-lambda
    )

    if "Filter thicknesses (mm)" not in acquisition_cat_field_names:
        acquisition_cat_field_names.append("Filter thicknesses (mm)")

    return df_unprocessed


def create_dose_check_and_source_columns(
    acquisition_cat_field_names,
    acquisition_val_field_names,
    ct_dose_check_field_names,
    df_unprocessed,
):
    # Add the Dose check alert column to the acquisition category field names if it isn't already there
    if "Dose check alerts" not in acquisition_cat_field_names:
        acquisition_cat_field_names.append("Dose check alerts")
    # Create the CT dose check column
    df_unprocessed = create_ct_dose_check_column(
        ct_dose_check_field_names, df_unprocessed
    )
    df_unprocessed["Dose check alerts"] = df_unprocessed["Dose check alerts"].astype(
        "category"
    )
    df_unprocessed = create_ct_source_columns(
        acquisition_cat_field_names, acquisition_val_field_names, df_unprocessed
    )
    return df_unprocessed


def write_standard_acquisition_data(
    book, df, enable_standard_names, worksheet_log, modality
):
    # Write out all standard acquisition name data to the sheets
    if enable_standard_names:
        all_std_acquisitions_in_df = df["Standard acquisition name"].dropna().unique()

        for acquisition in all_std_acquisitions_in_df:
            acq_df = df[df["Standard acquisition name"] == acquisition]

            acquisition = "[standard] " + acquisition

            write_row_to_acquisition_sheet(
                acq_df, acquisition, book, worksheet_log, modality
            )


def write_acquisition_data(book, df, worksheet_log, modality):
    # Obtain a list of unique acquisition protocols
    all_acquisitions_in_df = df["Acquisition protocol"].unique()
    for acquisition in all_acquisitions_in_df:

        acq_df = df[df["Acquisition protocol"] == acquisition]

        if acquisition in (None, np.nan, ""):
            acquisition = "Unknown"
            acq_df = df[df["Acquisition protocol"].isnull()]

        write_row_to_acquisition_sheet(
            acq_df, acquisition, book, worksheet_log, modality
        )


def create_standard_name_columns(df, exam_cat_field_names):
    if "Standard study name" in df.columns:
        df = create_standard_name_df_columns(df)

        # Make the exam_cat_field_names a categorical column (saves server memory)
        exam_cat_f_names = exam_cat_field_names[:]
        exam_cat_f_names.remove("Standard study name")
        exam_cat_f_names.extend(
            ["Standard study name 1", "Standard study name 2", "Standard study name 3"]
        )
        df[exam_cat_f_names] = df[exam_cat_f_names].astype("category")
    return df


def transform_dap_uas_units(df_unprocessed):
    # Transform DAP and uAs values into the required units
    if "Total DAP (cGy·cm²)" in df_unprocessed.columns:
        df_unprocessed["Total DAP (cGy·cm²)"] = (
            df_unprocessed["Total DAP (cGy·cm²)"] * 1000000
        )
    if "DAP (cGy·cm²)" in df_unprocessed.columns:
        df_unprocessed["DAP (cGy·cm²)"] = df_unprocessed["DAP (cGy·cm²)"] * 1000000
    if "mAs" in df_unprocessed.columns:
        df_unprocessed["mAs"] = df_unprocessed["mAs"] / 1000


def create_ct_source_columns(
    acquisition_cat_field_names, acquisition_val_field_names, df_unprocessed
):
    # ----------------------------------------
    # Create columns for two possible sources
    df_unprocessed[
        [
            "S1 Source name",
            "S1 kVp",
            "S1 mA",
            "S1 Maximum mA",
            "S1 Exposure time per rotation",
        ]
    ] = 5 * [np.nan]
    df_unprocessed[
        [
            "S2 Source name",
            "S2 kVp",
            "S2 mA",
            "S2 Maximum mA",
            "S2 Exposure time per rotation",
        ]
    ] = 5 * [np.nan]
    # Where "Source name" equals "A" copy source data fields to S1
    (
        df_unprocessed["S1 Source name"],
        df_unprocessed["S1 kVp"],
        df_unprocessed["S1 mA"],
        df_unprocessed["S1 Maximum mA"],
        df_unprocessed["S1 Exposure time per rotation"],
    ) = np.where(
        (df_unprocessed["Number of sources"] == 2)
        & (df_unprocessed["Source name"] == "A"),
        [
            df_unprocessed["Source name"],
            df_unprocessed["kVp"],
            df_unprocessed["mA"],
            df_unprocessed["Maximum mA"],
            df_unprocessed["Exposure time per rotation"],
        ],
        None,
    )
    # Where "Source name" equals "B" copy source data fields to S2
    (
        df_unprocessed["S2 Source name"],
        df_unprocessed["S2 kVp"],
        df_unprocessed["S2 mA"],
        df_unprocessed["S2 Maximum mA"],
        df_unprocessed["S2 Exposure time per rotation"],
    ) = np.where(
        (df_unprocessed["Number of sources"] == 2)
        & (df_unprocessed["Source name"] == "B"),
        [
            df_unprocessed["Source name"],
            df_unprocessed["kVp"],
            df_unprocessed["mA"],
            df_unprocessed["Maximum mA"],
            df_unprocessed["Exposure time per rotation"],
        ],
        None,
    )
    # Where "Number of sources" is not 2 copy source data fields to S1, but leave any non-matching ones as the
    # existing values, otherwise the writing of S1 data for the dual source entries will be over-written. Some
    # of my CT scanners have NA for the "Number of sources" value, so need to replace these with 0 to ensure
    # the != 2 works.
    df_unprocessed["Number of sources"] = df_unprocessed["Number of sources"].fillna(0)
    (
        df_unprocessed["S1 Source name"],
        df_unprocessed["S1 kVp"],
        df_unprocessed["S1 mA"],
        df_unprocessed["S1 Maximum mA"],
        df_unprocessed["S1 Exposure time per rotation"],
    ) = np.where(
        df_unprocessed["Number of sources"] != 2,
        [
            df_unprocessed["Source name"],
            df_unprocessed["kVp"],
            df_unprocessed["mA"],
            df_unprocessed["Maximum mA"],
            df_unprocessed["Exposure time per rotation"],
        ],
        [
            df_unprocessed["S1 Source name"],
            df_unprocessed["S1 kVp"],
            df_unprocessed["S1 mA"],
            df_unprocessed["S1 Maximum mA"],
            df_unprocessed["S1 Exposure time per rotation"],
        ],
    )
    # Drop the original columns
    df_unprocessed.drop(
        ["Source name", "kVp", "mA", "Maximum mA", "Exposure time per rotation"],
        axis=1,
        inplace=True,
    )
    # For any dual-source data we now have two rows per acquisition: one with source A data, one with source B.
    # We need to merge these into one row per acquisition.
    source_a_df = df_unprocessed.loc[df_unprocessed["S1 Source name"] == "A"]
    source_a_df.reset_index(drop=True, inplace=True)
    source_b_df = df_unprocessed.loc[df_unprocessed["S2 Source name"] == "B"]
    source_b_df.reset_index(drop=True, inplace=True)
    source_ab_df = source_a_df.combine_first(source_b_df)
    # Concatenate the dual source data with the single source data
    df_unprocessed = pd.concat(
        [source_ab_df, df_unprocessed.loc[df_unprocessed["Number of sources"] != 2]]
    )
    # Update the acquisition_cat_field_names entries to reflect the changes
    acquisition_cat_field_names[acquisition_cat_field_names.index("Source name")] = (
        "S1 Source name"
    )
    acquisition_cat_field_names.append("S2 Source name")
    # Update the acquisition_val_field_names entries to reflect the changes
    acquisition_val_field_names[acquisition_val_field_names.index("kVp")] = "S1 kVp"
    acquisition_val_field_names[acquisition_val_field_names.index("mA")] = "S1 mA"
    acquisition_val_field_names[acquisition_val_field_names.index("Maximum mA")] = (
        "S1 Maximum mA"
    )
    acquisition_val_field_names[
        acquisition_val_field_names.index("Exposure time per rotation")
    ] = "S1 Exposure time per rotation"
    new_fields = ["S2 kVp", "S2 mA", "S2 Maximum mA", "S2 Exposure time per rotation"]
    acquisition_val_field_names.extend(new_fields)
    # ----------------------------------------
    return df_unprocessed


def create_ct_dose_check_column(ct_dose_check_field_names, df):
    if df.empty:
        return None

    # Combine the dose alert fields
    # The title if either DLP or CTDIvol alerts are configured
    indices = df[
        (df["DLP alert configured"] == True) | (df["CTDIvol alert configured"] == True)
    ].index
    df.loc[indices, "Dose check alerts"] = "Dose check alerts:"
    # The DLP alert value
    indices = df[(df["DLP alert configured"] == True)].index
    df.loc[indices, "Dose check alerts"] = (
        df.loc[indices, "Dose check alerts"]
        + "\nDLP alert is configured at "
        + df.loc[indices, "DLP alert value"].astype("str")
        + " mGy.cm"
    )
    # The DLP forward estimate
    indices = df[(df["DLP forward estimate"].notnull())].index
    df.loc[indices, "Dose check alerts"] = (
        df.loc[indices, "Dose check alerts"]
        + "\nwith an accumulated forward estimate of "
        + df.loc[indices, "DLP forward estimate"].astype("str")
        + " mGy.cm"
    )
    # The CTDIvol alert value
    indices = df[(df["CTDIvol alert configured"] == True)].index
    df.loc[indices, "Dose check alerts"] = (
        df.loc[indices, "Dose check alerts"]
        + "\nCTDIvol alert is configured at "
        + df.loc[indices, "CTDIvol alert value"].astype("str")
        + " mGy"
    )
    # The CTDIvol forward estimate
    indices = df[(df["CTDIvol forward estimate"].notnull())].index
    df.loc[indices, "Dose check alerts"] = (
        df.loc[indices, "Dose check alerts"]
        + "\nwith an accumulated forward estimate of "
        + df.loc[indices, "CTDIvol forward estimate"].astype("str")
        + " mGy"
    )
    # The reason for proceeding
    indices = df[(df["Reason for proceeding"].notnull())].index
    df.loc[indices, "Dose check alerts"] = (
        df.loc[indices, "Dose check alerts"]
        + "\nReason for proceeding: "
        + df.loc[indices, "Reason for proceeding"]
    )
    # The person authorizing the exposure
    indices = df[(df["Person name"].notnull())].index
    df.loc[indices, "Dose check alerts"] = (
        df.loc[indices, "Dose check alerts"]
        + "\nPerson authorizing irradiation: "
        + df.loc[indices, "Person name"]
    )
    # Remove the individual dose check columns from the dataframe
    df = df.drop(columns=ct_dose_check_field_names)
    return df
