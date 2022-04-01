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
..  module:: nm_export
    :synopsis: Module to export database data to xlsx and csv files

..  moduleauthor:: Jannis Widmer

"""
import logging

from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Max, Count
from celery import shared_task
import datetime
from ..interface.mod_filters import nm_filter
import traceback
import sys

from .export_common import (
    get_common_data,
    common_headers,
    create_xlsx,
    create_csv,
    text_and_date_formats,
    write_export,
    abort_if_zero_studies,
    create_export_task,
)

logger = logging.getLogger(__name__)


def _exit_proc(task, date_stamp, error_msg=None, force_exit=True):
    if error_msg is not None:
        task.status = "ERROR"
        task.progress = error_msg
    else:
        task.status = "COMPLETE"
    task.processtime = (datetime.datetime.now() - date_stamp).total_seconds()
    task.save()
    if force_exit:
        exit(0)


def unknown_error(task, date_stamp):
    etype, evalue, _ = sys.exc_info()
    logger.error(
        f"Failed to export NM with error: \n {''.join(traceback.format_exc())}"
    )
    _exit_proc(task, date_stamp, traceback.format_exception_only(etype, evalue)[-1])


def _get_data(filterdict, pid, task):
    data = nm_filter(filterdict, pid).qs

    task.num_records = data.count()
    if abort_if_zero_studies(task.num_records, task):
        _exit_proc(task, datetime.datetime.now(), "Zero studies marked for export")
    task.progress = f"{task.num_records} studies in query"
    task.save()

    #num_person_participants, num_organ_doses,
    #num_patient_state, num_glomerular_filtration_rate
    statistics = data.annotate(
        num_person_participants=Count(
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__"
            "personparticipant", distinct=True
        ),
        num_organ_doses=Count(
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__"
            "organdose", distinct=True
        ),
        num_patient_state=Count(
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__"
            "patientstate", distinct=True
        ),
        num_glomerular_filtration_rate=Count(
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__"
            "glomerularfiltrationrate", distinct=True
        ),
    ).aggregate(
        max_person_participants=Max("num_person_participants"),
        max_organ_doses=Max("num_organ_doses"),
        max_patient_state=Max("num_patient_state"),
        max_glomerular_filtration_rate=Max("num_glomerular_filtration_rate"),
    )

    return (data, statistics)

def _nm_headers(pid, name, patid, statistics):
    headings = common_headers("NM", pid, name, patid)
    headings.remove("No. events")  # There is always just one event for a study
    headings += [ 
        "Radiopharmaceutical Agent",
        "Radionuclide",
        "Radionuclide Half Live",
        "Administered activity (MBq)",
        "Associated Procedure",
        "Radiopharmaceutical Start Time",
        "Radiopharmaceutical Stop Time",
        "Route of Administration",
        "Route of Administration Laterality",
    ]
    for i in range(1, statistics["max_person_participants"]+1):
        headings += [f"Person participant name {i}",
                    f"Person participant role {i}"]
    headings += ["Comment"]
    for i in range(1, statistics["max_organ_doses"]+1):
        headings += [
            f"Organ Dose Finding Site {i}",
            f"Organ Laterality {i}",
            f"Organ Dose {i} (mGy)",
            f"Organ Mass {i}(g)",
            f"Organ Dose Measurement Method {i}",
            f"Organ Dose Reference Authority {i}"
        ]
    for i in range(1, statistics["max_patient_state"]+1):
        headings += [
            f"Patient state {i}"
        ]
    headings += [
        "Body Surface Area (m^2)",
        "Body Surface Area Formula",
        "Body Mass Index (kg/m^2)",
        "Body Mass Index Equation",
        "Glucose (mmol/l)",
        "Fasting Duration (hours)",
        "Hydration Volume (ml)",
        "Recent Physical Activity",
        "Serum Creatinine (mg/dl)",
    ]
    for i in range(1, statistics["max_glomerular_filtration_rate"]+1):
        headings += [
            f"Glomerular Filtration Rate {i} (ml/min/1.73m^2)",
            f"Measurement Method {i}",
            f"Equivalent meaning of concept name {i}"
        ]

    return headings

def _get_code_not_none(code):
    if code is None:
        return None
    else:
        return code.code_meaning

def _extract_study_data(exams, pid, name, patid):
    exam_data = get_common_data("NM", exams, pid, name, patid)

    try:
        radiopharm = exams.radiopharmaceuticalradiationdose_set.get()
        radiopharm_admin = radiopharm.radiopharmaceuticaladministrationeventdata_set.get()
        patient_charac = radiopharm.radiopharmaceuticaladministrationpatientcharacteristics_set.get()
        person_participants = radiopharm_admin.personparticipant_set.all()
        organ_doses = radiopharm_admin.organdose_set.all()
        patient_states = patient_charac.patientstate_set.all()
        glomerular_filtration_rates = patient_charac.glomerularfiltrationrate_set.all()
    except ObjectDoesNotExist:
        raise  # We handle this on the level of the export function

    exam_data += [
        _get_code_not_none(radiopharm_admin.radiopharmaceutical_agent),
        _get_code_not_none(radiopharm_admin.radionuclide),
        radiopharm_admin.radionuclide_half_life,
        radiopharm_admin.administered_activity,
        _get_code_not_none(radiopharm.associated_procedure),
        radiopharm_admin.radiopharmaceutical_start_datetime,
        radiopharm_admin.radiopharmaceutical_stop_datetime,
        _get_code_not_none(radiopharm_admin.route_of_administration),
        _get_code_not_none(radiopharm_admin.laterality),
    ]
    for person_participant in person_participants:
        exam_data += [
            person_participant.person_name,
            _get_code_not_none(person_participant.person_role_in_procedure_cid),
        ]
    exam_data += [radiopharm.comment]
    for organ_dose in organ_doses:
        if organ_dose.reference_authority_code is not None:
            organ_dose_reference_authority = organ_dose.reference_authority_code.code_meaning
        else:
            organ_dose_reference_authority = organ_dose.reference_authority_text
        exam_data += [
            _get_code_not_none(organ_dose.finding_site),
            _get_code_not_none(organ_dose.laterality),
            organ_dose.organ_dose,
            organ_dose.mass,
            organ_dose.measurement_method,
            organ_dose_reference_authority,
        ]
    for patient_state in patient_states:
        exam_data += [
            _get_code_not_none(patient_state.patient_state)
        ]
    exam_data += [
        patient_charac.body_surface_area,
        _get_code_not_none(patient_charac.body_surface_area_formula),
        patient_charac.body_mass_index,
        _get_code_not_none(patient_charac.equation),
        patient_charac.glucose,
        patient_charac.fasting_duration,
        patient_charac.hydration_volume,
        patient_charac.recent_physical_activity,
        patient_charac.serum_creatinine,
    ]
    for glomerular in glomerular_filtration_rates:
        exam_data += [
            glomerular.glomerular_filtration_rate,
            _get_code_not_none(glomerular.measurement_method),
            _get_code_not_none(glomerular.equivalent_meaning_of_concept_name)
        ]
    
    for i, item in enumerate(exam_data):
        if item is None:
            exam_data[i] = ""
        if isinstance(item, str):
            exam_data[i] = item.replace(",", ";")

    return exam_data


@shared_task
def exportNM2csv(filterdict, pid=False, name=None, patid=None, user=None):
    logger.debug("Started csv export task for NM")

    date_stamp = datetime.datetime.now()
    task = create_export_task(
        celery_uuid=exportNM2csv.request.id,
        modality="NM",
        export_type="CSV export",
        date_stamp=date_stamp,
        pid=bool(pid and (name or patid)),
        user=user,
        filters_dict=filterdict,
    )

    try:
        tmpfile, writer = create_csv(task)
        if not tmpfile:
            _exit_proc(task, date_stamp, "Failed to create the export file")

        data, statistics = _get_data(filterdict, pid, task)
        headings = _nm_headers(pid, name, patid, statistics)
        writer.writerow(headings)

        task.progress = "CSV header row written."
        task.save()

        for i, exam in enumerate(data):
            try:
                exam_data = _extract_study_data(exam, pid, name, patid)
                writer.writerow(exam_data)
            except ObjectDoesNotExist:
                error_message = (
                    f"DoesNotExist error whilst exporting study {i + 1} of {task.num_records},  study UID {exam.study_instance_uid}, accession number"
                    f" {exam.accession_number} - maybe database entry was deleted as part of importing later version of same"
                    " study?"
                )
                logger.error(error_message)
                writer.writerow([error_message])
            task.progress = f"{i+1} of {task.num_records} written."
            task.save()
    except Exception:
        unknown_error(task, date_stamp)

    tmpfile.close()
    task.progress = "All data written."
    _exit_proc(task, date_stamp, force_exit=False)


@shared_task
def exportNM2excel(filterdict, pid=False, name=None, patid=None, user=None):
    logger.debug("Started XLSX export task for NM")

    date_stamp = datetime.datetime.now()
    task = create_export_task(
        celery_uuid=exportNM2excel.request.id,
        modality="NM",
        export_type="XLSX export",
        date_stamp=date_stamp,
        pid=bool(pid and (name or patid)),
        user=user,
        filters_dict=filterdict,
    )

    try:
        tmpxlsx, book = create_xlsx(task)
        if not tmpxlsx:
            _exit_proc(task, date_stamp, "Failed to create file")

        data, statistics = _get_data(filterdict, pid, task)
        headings = _nm_headers(pid, name, patid, statistics)

        all_data = book.add_worksheet("All data")
        book = text_and_date_formats(book, all_data, pid, name, patid)

        all_data.write_row(0, 0, headings)
        numcolumns = len(headings) - 1
        numrows = data.count()
        all_data.autofilter(0, 0, numrows, numcolumns)

        for i, exam in enumerate(data):
            try:
                exam_data = _extract_study_data(exam, pid, name, patid)
                all_data.write_row(i + 1, 0, exam_data)
            except ObjectDoesNotExist:
                error_message = (
                    f"DoesNotExist error whilst exporting study {i + 1} of {task.num_records},  study UID {exam.study_instance_uid}, accession number"
                    f" {exam.accession_number} - maybe database entry was deleted as part of importing later version of same"
                    " study?"
                )
                logger.error(error_message)
                all_data.write_row(i + 1, 0, [error_message])

            task.progress = f"{i+1} of {task.num_records} written."
            task.save()

        book.close()
    except Exception:
        unknown_error(task, date_stamp)

    xlsxfilename = "nmexport{0}.xlsx".format(date_stamp.strftime("%Y%m%d-%H%M%S%f"))
    write_export(
        task, xlsxfilename, tmpxlsx, date_stamp
    )  # Does nearly the same as _exit_proc, so it's used to leave the process
