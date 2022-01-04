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
import datetime
import logging
import os

import django.db
import pandas as pd
import pkg_resources

from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from celery import shared_task
from zipfile import (
    ZipFile,
    ZIP_DEFLATED,
)

from .export_common import (
    create_xlsx,
    create_csv,
    write_export,
    abort_if_zero_studies,
    create_export_task,
    transform_to_one_row_per_exam,
)

logger = logging.getLogger(__name__)


@shared_task
def ctxlsx(filterdict, pid=False, name=None, patid=None, user=None):
    """Export filtered CT database data to multi-sheet Microsoft XSLX files

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves xlsx file into Media directory for user to download
    """

    from django.db.models import Max
    from .export_common import text_and_date_formats, sheet_name
    from ..interface.mod_filters import ct_acq_filter

    datestamp = datetime.datetime.now()
    tsk = create_export_task(
        celery_uuid=ctxlsx.request.id,
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
    qs = ct_acq_filter(filterdict, pid=pid).qs

    tsk.num_records = qs.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    # Add summary sheet and all data sheet
    summarysheet = book.add_worksheet("Summary")
    wsalldata = book.add_worksheet("All data")

    book = text_and_date_formats(
        book, wsalldata, pid=pid, name=name, patid=patid, modality="CT"
    )

    textformat = book.add_format({"num_format": "@"})
    dateformat = book.add_format({"num_format": settings.XLSX_DATE})
    timeformat = book.add_format({"num_format": settings.XLSX_TIME})

    wsalldata.set_column(11, 11, 10, dateformat)  # Study date
    wsalldata.set_column(12, 12, 10, timeformat)  # Study time

    #====================================================================================
    # Write the summary data sheet
    vers = pkg_resources.require("openrem")[0].version
    version = vers
    titleformat = book.add_format()
    titleformat.set_font_size = 22
    titleformat.set_font_color = "#FF0000"
    titleformat.set_bold()
    toplinestring = "XLSX Export from OpenREM version {0} on {1}".format(
        version, str(datetime.datetime.now())
    )
    linetwostring = (
        "OpenREM is copyright 2019 The Royal Marsden NHS Foundation Trust, and available under the GPL. "
        "See http://openrem.org"
    )
    summarysheet.write(0, 0, toplinestring, titleformat)
    summarysheet.write(1, 0, linetwostring)

    # Number of exams
    summarysheet.write(3, 0, "Total number of exams")
    summarysheet.write(3, 1, qs.count())

    # DataFrame containing study description and requested procedure data
    df = pd.DataFrame.from_records(data=qs.values_list("pk", "study_description", "requested_procedure_code_meaning"),
                                   columns=["pk", "Study description", "Requested procedure"])

    study_description_frequency = df["Study description"].value_counts(dropna=False)
    study_description_frequency = study_description_frequency.reset_index()
    study_description_frequency.columns = ["Study description", "Frequency"]
    study_description_frequency["Frequency"] = study_description_frequency["Frequency"].astype("UInt32")

    requested_procedure_frequency = df["Requested procedure"].value_counts(dropna=False)
    requested_procedure_frequency = requested_procedure_frequency.reset_index()
    requested_procedure_frequency.columns = ["Requested procedure", "Frequency"]
    requested_procedure_frequency["Frequency"] = requested_procedure_frequency["Frequency"].astype("UInt32")

    acquisition_protocol_frequency = pd.DataFrame.from_records(
        data=qs.values_list("ctradiationdose__ctirradiationeventdata__pk",
                            "ctradiationdose__ctirradiationeventdata__acquisition_protocol"),
        columns=["pk", "Acquisition protocol"])["Acquisition protocol"].value_counts(dropna=False).reset_index()
    acquisition_protocol_frequency.columns = ["Acquisition protocol", "Frequency"]
    acquisition_protocol_frequency["Frequency"] = acquisition_protocol_frequency["Frequency"].astype("UInt32")

    # Write the column titles
    col_titles = [
        "Study Description", "Frequency", "",
        "Requested Procedure", "Frequency", "",
        "Acquisition protocol", "Frequency"
    ]
    summarysheet.write_row(5, 0, col_titles)

    # Widen the name columns
    summarysheet.set_column("A:A", 25)
    summarysheet.set_column("D:D", 25)
    summarysheet.set_column("G:G", 25)

    # Write the frequency data to the xlsx file
    study_description_frequency["BlankCol"] = None
    requested_procedure_frequency["BlankCol"] = None
    combined_df = pd.concat([
        study_description_frequency,
        requested_procedure_frequency,
        acquisition_protocol_frequency
    ], axis=1)

    combined_df = combined_df.where(pd.notnull(combined_df), None)

    for idx in combined_df.index:
        summarysheet.write_row(
            idx + 6, 0, [None if x is pd.NA or not pd.notna(x) else x for x in combined_df.iloc[idx].to_list()]
        )
    #====================================================================================


    #====================================================================================
    # Write the all data sheet
    # This code is taken from the ct_csv method...
    qs_chunk_size=10000

    # Exam-level integer field names
    exam_int_fields = [
        "pk",
        "number_of_events",
    ]

    # Friendly exam-level integer field names
    exam_int_field_names = [
        "pk",
        "Number of events"
    ]

    # Exam-level object field names (string data, little or no repetition)
    exam_obj_fields = [
        "accession_number",
    ]
    if pid and name:
        exam_obj_fields.append("patientmoduleattr__patient_name")
    if pid and patid:
        exam_obj_fields.append("patientmoduleattr__patient_id")

    # Friendly exam-level object field names
    exam_obj_field_names = [
        "Accession number",
    ]
    if pid and name:
        exam_obj_field_names.append("Patient name")
    if pid and patid:
        exam_obj_field_names.append("Patient ID")

    # Exam-level category field names
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

    # Friendly exam-level category field names
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

    # Exam-level date field names
    exam_date_fields = ["study_date"]

    # Friendly exam-level date field names
    exam_date_field_names = ["Study date"]

    # Exam-level time field names
    exam_time_fields = ["study_time"]

    # Friendly exam-level time field names
    exam_time_field_names = ["Study time"]

    # Exam-level category value names
    exam_val_fields = [
        "patientstudymoduleattr__patient_age_decimal",
        "patientstudymoduleattr__patient_size",
        "patientstudymoduleattr__patient_weight",
        "total_dlp"
    ]

    # Friendly exam-level value field names
    exam_val_field_names = [
        "Patient age",
        "Patient height (m)",
        "Patient weight (kg)",
        "Total DLP (mGy.cm)"
    ]

    # Required acquisition-level integer field names
    acquisition_int_fields = [
        "ctradiationdose__ctirradiationeventdata__pk",
        "ctradiationdose__ctirradiationeventdata__number_of_xray_sources",
    ]

    # Friendly acquisition-level integer field names
    acquisition_int_field_names = [
        "acq_pk",
        "Number of sources",
    ]

    # Required acquisition-level category field names
    acquisition_cat_fields = [
        "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
        "ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning",
        "ctradiationdose__ctirradiationeventdata__ctdiw_phantom_type__code_meaning",
        "ctradiationdose__ctirradiationeventdata__xray_modulation_type",
        "ctradiationdose__ctirradiationeventdata__ctxraysourceparameters__identification_of_the_xray_source",
    ]

    # Friendly acquisition-level category field names
    acquisition_cat_field_names = [
        "Acquisition protocol",
        "Acquisition type",
        "CTDI phantom type",
        "mA modulation type",
        "Source name"
    ]

    # Required acquisition-level value field names
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

    # Friendly acquisition-level value field names
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

    exam_fields = exam_int_fields + exam_obj_fields + exam_cat_fields + exam_date_fields + exam_time_fields + exam_val_fields
    acquisition_fields = acquisition_int_fields + acquisition_cat_fields + acquisition_val_fields
    all_fields = exam_fields + acquisition_fields

    exam_field_names = exam_int_field_names + exam_obj_field_names + exam_cat_field_names + exam_date_field_names + exam_time_field_names + exam_val_field_names
    acquisition_field_names = acquisition_int_field_names + acquisition_cat_field_names + acquisition_val_field_names
    all_field_names = exam_field_names + acquisition_field_names

    # Add the Dose check alert column to the acquisition category field names
    acquisition_cat_field_names.append("Dose check alerts")

    # Create a series of DataFrames by chunking the queryset into groups of accession numbers.
    # Chunking saves server memory at the expense of speed.
    n_entries = qs.count()
    tsk.num_records = n_entries
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    tsk.progress = "{0} studies in query.".format(tsk.num_records)
    tsk.save()

    write_headers = True

    # Generate a list of non-null accession numbers
    accession_numbers = [x[0] for x in qs.order_by("-study_date", "-study_time").filter(accession_number__isnull=False).values_list("accession_number")]

    # Create a work sheet for each acquisition protocol present in the data in alphabetical order
    # and a dictionary to hold the number of rows that have been written to each protocol sheet
    required_sheets = qs.values_list("ctradiationdose__ctirradiationeventdata__acquisition_protocol", flat=True).order_by("ctradiationdose__ctirradiationeventdata__acquisition_protocol").distinct()
    worksheet_log = {}
    for name in required_sheets:
        if name is None:
            name = "Unknown"

        name = sheet_name(name)

        if name not in book.sheetnames.keys():
            new_sheet = book.add_worksheet(name)
            new_sheet.set_column(11, 11, 10, dateformat)  # Study date
            new_sheet.set_column(12, 12, 10, timeformat)  # Study time

            worksheet_log[name] = 0

    current_row = 1

    for chunk_min_idx in range(0, n_entries, qs_chunk_size):

        chunk_max_idx = chunk_min_idx + qs_chunk_size
        if chunk_max_idx > n_entries:
            chunk_max_idx = n_entries

        tsk.progress = "Working on entries {0} to {1}".format(chunk_min_idx + 1, chunk_max_idx)
        tsk.save()

        data = qs.order_by().filter(accession_number__in=accession_numbers[chunk_min_idx:chunk_max_idx]).values_list(*(all_fields + ct_dose_check_fields))

        # Clear the query cache
        django.db.reset_queries()

        df = pd.DataFrame.from_records(
            data=data,
            columns=(all_field_names + ct_dose_check_field_names), coerce_float=True,
        )

        # Create the CT dose check column
        df = create_ct_dose_check_column(ct_dose_check_field_names, df)


        df = transform_to_one_row_per_exam(
            df,
            acquisition_cat_field_names, acquisition_int_field_names, acquisition_val_field_names,
            exam_cat_field_names, exam_date_field_names, exam_int_field_names,
            exam_obj_field_names, exam_time_field_names, exam_val_field_names,
            all_field_names)

        # Write the headings to the sheet (over-writing each time, but this ensures we'll include the study
        # with the most events without doing anything complicated to generate the headings)
        wsalldata.write_row(0, 0, df.columns)

        # Write the DataFrame to the all data sheet
        for idx, row in df.iterrows():
            wsalldata.write_row(current_row, 0, row.fillna(""))
            current_row = current_row + 1


        # Write out data to the acquisition protocol sheets
        df = pd.DataFrame.from_records(data=data, columns=(all_field_names + ct_dose_check_field_names), coerce_float=True)

        # Create the CT dose check column
        df = create_ct_dose_check_column(ct_dose_check_field_names, df)

        all_acquisitions_in_df = df["Acquisition protocol"].unique()

        for acquisition in all_acquisitions_in_df:

            acq_df = df[df["Acquisition protocol"] == acquisition]

            if acquisition == None:
                acquisition = "Unknown"
                acq_df = df[df["Acquisition protocol"].isnull()]

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

            for idx, row in acq_df.iterrows():
                sheet.write_row(sheet_row, 0, row.fillna(""))
                sheet_row = sheet_row + 1

            worksheet_log[acquisition] = sheet_row


    # Now write out any None accession number data if any such data is present
    n_entries = qs.filter(accession_number__isnull=True).count()
    data = qs.order_by().filter(accession_number__isnull=True).values_list(*(all_fields + ct_dose_check_fields))

    df = pd.DataFrame.from_records(
        data=data,
        columns=(all_field_names + ct_dose_check_field_names), coerce_float=True,
    )

    # Create the CT dose check column
    df = create_ct_dose_check_column(ct_dose_check_field_names, df)

    # Clear the query cache
    django.db.reset_queries()

    if data:
        tsk.progress = "Working on {0} entries with blank accession numbers".format(n_entries)
        tsk.save()

        df = transform_to_one_row_per_exam(
            df,
            acquisition_cat_field_names, acquisition_int_field_names, acquisition_val_field_names,
            exam_cat_field_names, exam_date_field_names, exam_int_field_names,
            exam_obj_field_names, exam_time_field_names, exam_val_field_names,
            all_field_names)

        # Write the headings to the sheet (over-writing each time, but this ensures we'll include the study
        # with the most events without doing anything complicated to generate the headings)
        wsalldata.write_row(0, 0, df.columns)

        # Write the DataFrame to the all data sheet
        for idx, row in df.iterrows():
            wsalldata.write_row(current_row, 0, row.fillna(""))
            current_row = current_row + 1


        # Write out data to the acquisition protocol sheets
        df = pd.DataFrame.from_records(data=data, columns=(all_field_names + ct_dose_check_field_names), coerce_float=True)
        all_acquisitions_in_df = df["Acquisition protocol"].unique()

        for acquisition in all_acquisitions_in_df:

            acq_df = df[df["Acquisition protocol"] == acquisition]

            if acquisition == None:
                acquisition = "Unknown"
                acq_df = df[df["Acquisition protocol"].isnull()]

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

            for idx, row in acq_df.iterrows():
                sheet.write_row(sheet_row, 0, row.fillna(""))
                sheet_row = sheet_row + 1

            worksheet_log[acquisition] = sheet_row


    #====================================================================================
    # Close the book
    book.close()
    tsk.progress = "XLSX book written."
    tsk.save()

    xlsxfilename = "ctexport{0}.xlsx".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))

    write_export(tsk, xlsxfilename, tmpxlsx, datestamp)
    #====================================================================================


def create_ct_dose_check_column(ct_dose_check_field_names, df):
    if df.empty:
        return None

    # Combine the dose alert fields
    # The title if either DLP or CTDIvol alerts are configured
    indices = df[(df["DLP alert configured"] == True) | (df["CTDIvol alert configured"] == True)].index
    df.loc[indices, "Dose check alerts"] = "Dose check alerts:"
    # The DLP alert value
    indices = df[(df["DLP alert configured"] == True)].index
    df.loc[indices, "Dose check alerts"] = (
            df.loc[indices, "Dose check alerts"] +
            "\nDLP alert is configured at " +
            df.loc[indices, "DLP alert value"].astype("str") +
            " mGy.cm"
    )
    # The DLP forward estimate
    indices = df[(df["DLP forward estimate"].notnull())].index
    df.loc[indices, "Dose check alerts"] = (
            df.loc[indices, "Dose check alerts"] +
            "\nwith an accumulated forward estimate of " +
            df.loc[indices, "DLP forward estimate"].astype("str") +
            " mGy.cm"
    )
    # The CTDIvol alert value
    indices = df[(df["CTDIvol alert configured"] == True)].index
    df.loc[indices, "Dose check alerts"] = (
            df.loc[indices, "Dose check alerts"] +
            "\nCTDIvol alert is configured at " +
            df.loc[indices, "CTDIvol alert value"].astype("str") +
            " mGy"
    )
    # The CTDIvol forward estimate
    indices = df[(df["CTDIvol forward estimate"].notnull())].index
    df.loc[indices, "Dose check alerts"] = (
            df.loc[indices, "Dose check alerts"] +
            "\nwith an accumulated forward estimate of " +
            df.loc[indices, "CTDIvol forward estimate"].astype("str") +
            " mGy"
    )
    # The reason for proceeding
    indices = df[(df["Reason for proceeding"].notnull())].index
    df.loc[indices, "Dose check alerts"] = (
            df.loc[indices, "Dose check alerts"] +
            "\nReason for proceeding: " +
            df.loc[indices, "Reason for proceeding"]
    )
    # The person authorizing the exposure
    indices = df[(df["Person name"].notnull())].index
    df.loc[indices, "Dose check alerts"] = (
            df.loc[indices, "Dose check alerts"] +
            "\nPerson authorizing irradiation: " +
            df.loc[indices, "Person name"]
    )
    # Remove the individual dose check columns from the dataframe
    df = df.drop(columns=ct_dose_check_field_names)
    return df


@shared_task
def ct_csv(filterdict, pid=False, name=None, patid=None, user=None):
    """Export filtered CT database data to a single-sheet CSV file.

    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves csv file into Media directory for user to download
    """

    from django.db.models import Max
    from ..interface.mod_filters import ct_acq_filter

    datestamp = datetime.datetime.now()
    tsk = create_export_task(
        celery_uuid=ct_csv.request.id,
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
    qs = ct_acq_filter(filterdict, pid=pid).qs

    qs_chunk_size=10000

    # Exam-level integer field names
    exam_int_fields = [
        "pk",
        "number_of_events",
    ]

    # Friendly exam-level integer field names
    exam_int_field_names = [
        "pk",
        "Number of events"
    ]

    # Exam-level object field names (string data, little or no repetition)
    exam_obj_fields = [
        "accession_number",
    ]
    if pid and name:
        exam_obj_fields.append("patientmoduleattr__patient_name")
    if pid and patid:
        exam_obj_fields.append("patientmoduleattr__patient_id")

    # Friendly exam-level object field names
    exam_obj_field_names = [
        "Accession number",
    ]
    if pid and name:
        exam_obj_field_names.append("Patient name")
    if pid and patid:
        exam_obj_field_names.append("Patient ID")

    # Exam-level category field names
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

    # Friendly exam-level category field names
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

    # Exam-level date field names
    exam_date_fields = ["study_date"]

    # Friendly exam-level date field names
    exam_date_field_names = ["Study date"]

    # Exam-level time field names
    exam_time_fields = ["study_time"]

    # Friendly exam-level time field names
    exam_time_field_names = ["Study time"]

    # Exam-level category value names
    exam_val_fields = [
        "patientstudymoduleattr__patient_age_decimal",
        "patientstudymoduleattr__patient_size",
        "patientstudymoduleattr__patient_weight",
        "total_dlp"
    ]

    # Friendly exam-level value field names
    exam_val_field_names = [
        "Patient age",
        "Patient height (m)",
        "Patient weight (kg)",
        "Total DLP (mGy.cm)"
    ]

    # Required acquisition-level integer field names
    acquisition_int_fields = [
        "ctradiationdose__ctirradiationeventdata__number_of_xray_sources",
    ]

    # Friendly acquisition-level integer field names
    acquisition_int_field_names = [
        "Number of sources",
    ]

    # Required acquisition-level category field names
    acquisition_cat_fields = [
        "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
        "ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning",
        "ctradiationdose__ctirradiationeventdata__ctdiw_phantom_type__code_meaning",
        "ctradiationdose__ctirradiationeventdata__xray_modulation_type",
        "ctradiationdose__ctirradiationeventdata__ctxraysourceparameters__identification_of_the_xray_source",
    ]

    # Friendly acquisition-level category field names
    acquisition_cat_field_names = [
        "Acquisition protocol",
        "Acquisition type",
        "CTDI phantom type",
        "mA modulation type",
        "Source name"
    ]

    # Required acquisition-level value field names
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

    # Friendly acquisition-level value field names
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

    all_fields = exam_int_fields + exam_obj_fields + exam_cat_fields + exam_date_fields + exam_time_fields + exam_val_fields + acquisition_int_fields + acquisition_cat_fields + acquisition_val_fields
    all_field_names = exam_int_field_names + exam_obj_field_names + exam_cat_field_names + exam_date_field_names + exam_time_field_names + exam_val_field_names + acquisition_int_field_names + acquisition_cat_field_names + acquisition_val_field_names

    # Add the Dose check alert column to the acquisition category field names
    acquisition_cat_field_names.append("Dose check alerts")

    # Create a series of DataFrames by chunking the queryset into groups of accession numbers.
    # Chunking saves server memory at the expense of speed.
    n_entries = qs.count()
    tsk.num_records = n_entries
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    tsk.progress = "{0} studies in query.".format(tsk.num_records)
    tsk.save()

    write_headers = True

    # Generate a list of non-null accession numbers
    accession_numbers = [x[0] for x in qs.order_by("-study_date", "-study_time").filter(accession_number__isnull=False).values_list("accession_number")]

    for chunk_min_idx in range(0, n_entries, qs_chunk_size):

        chunk_max_idx = chunk_min_idx + qs_chunk_size
        if chunk_max_idx > n_entries:
            chunk_max_idx = n_entries

        tsk.progress = "Working on entries {0} to {1}".format(chunk_min_idx + 1, chunk_max_idx)
        tsk.save()

        data = qs.order_by().filter(accession_number__in=accession_numbers[chunk_min_idx:chunk_max_idx]).values_list(*(all_fields + ct_dose_check_fields))

        # Clear the query cache
        django.db.reset_queries()

        df = pd.DataFrame.from_records(
            data=data,
            columns=(all_field_names + ct_dose_check_field_names), coerce_float=True,
        )

        # Create the CT dose check column
        df = create_ct_dose_check_column(ct_dose_check_field_names, df)

        df = transform_to_one_row_per_exam(
            df,
            acquisition_cat_field_names, acquisition_int_field_names, acquisition_val_field_names,
            exam_cat_field_names, exam_date_field_names, exam_int_field_names,
            exam_obj_field_names, exam_time_field_names, exam_val_field_names,
            all_field_names)

        # Write the DataFrame to a csv file
        df.to_csv(tmpfile, index=False, mode="a", header=write_headers)
        write_headers = False

        # Flush the csv data to the file
        tsk.progress = "Writing entries {0} to {1} to csv file".format(chunk_min_idx + 1, chunk_max_idx)
        tsk.save()
        tmpfile.flush()

    # Now write out any None accession number data if any such data is present
    n_entries = qs.filter(accession_number__isnull=True).count()
    data = qs.order_by().filter(accession_number__isnull=True).values_list(*(all_fields + ct_dose_check_fields))

    df = pd.DataFrame.from_records(
        data=data,
        columns=(all_field_names + ct_dose_check_field_names), coerce_float=True,
    )

    # Create the CT dose check column
    df = create_ct_dose_check_column(ct_dose_check_field_names, df)

    # Clear the query cache
    django.db.reset_queries()

    if data:
        tsk.progress = "Working on {0} entries with blank accession numbers".format(n_entries)
        tsk.save()

        df = transform_to_one_row_per_exam(
            df,
            acquisition_cat_field_names, acquisition_int_field_names, acquisition_val_field_names,
            exam_cat_field_names, exam_date_field_names, exam_int_field_names,
            exam_obj_field_names, exam_time_field_names, exam_val_field_names,
            all_field_names)

        # Write the None values to the csv file
        df.to_csv(tmpfile, index=False, mode="a", header=write_headers)

        # Flush the csv data to the file
        tsk.progress = "Writing {0} entries with blank accession numbers to csv file".format(n_entries)
        tsk.save()
        tmpfile.flush()

    # Close the csv file - we've finished writing data
    tmpfile.close()

    tsk.progress = "All study data written. Zipping file to save space"
    tsk.save()

    # Zip up the csv results file to save server space, and delete the uncompressed csv file
    if os.path.exists(tsk.filename.path):
        with ZipFile(tsk.filename.path + ".zip", "w", compression=ZIP_DEFLATED, compresslevel=9) as myzip:
            myzip.write(tsk.filename.path, arcname=os.path.split(tsk.filename.path)[1])
            myzip.close()

    # Remove the original csv file
    os.remove(tsk.filename.path)

    # Update the task filename to be the zip file
    tsk.filename.name = tsk.filename.name + ".zip"

    tsk.status = "COMPLETE"
    tsk.processtime = (datetime.datetime.now() - datestamp).total_seconds()
    tsk.save()


def _generate_all_data_headers_ct(max_events):
    """Generate the headers for CT that repeat once for each series of the exam with the most series in

    :param max_events: maximum number of times to repeat headers
    :return: list of headers
    """

    repeating_series_headers = []
    for h in range(int(max_events)):
        repeating_series_headers += [
            "E" + str(h + 1) + " Protocol",
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


@shared_task
def ct_phe_2019(filterdict, user=None):
    """Export filtered CT database data in the format required for the 2019 Public Health England
    CT dose survey

    :param filterdict: Queryset of studies to export
    :param user:  User that has started the export
    :return: Saves Excel file into Media directory for user to download
    """

    from decimal import Decimal
    from ..interface.mod_filters import ct_acq_filter

    datestamp = datetime.datetime.now()
    tsk = create_export_task(
        celery_uuid=ct_phe_2019.request.id,
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
