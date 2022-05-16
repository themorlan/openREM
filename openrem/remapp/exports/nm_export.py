#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2022 The Royal Marsden NHS Foundation Trust
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
import datetime
import traceback
import sys

from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Max, Count

from openrem.remapp.tools.background import (
    get_or_generate_task_uuid,
    record_task_error_exit,
)

from ..interface.mod_filters import nm_filter
from .export_common import (
    create_summary_sheet,
    get_common_data,
    common_headers,
    create_xlsx,
    create_csv,
    text_and_date_formats,
    write_export,
    abort_if_zero_studies,
    create_export_task,
    sheet_name,
)

logger = logging.getLogger(__name__)


def _exit_proc(task, date_stamp, error_msg=None):
    if error_msg is not None:
        task.status = "ERROR"
        task.progress = error_msg
        record_task_error_exit(error_msg)
    else:
        task.status = "COMPLETE"
    task.processtime = (datetime.datetime.now() - date_stamp).total_seconds()
    task.save()
    if error_msg is not None:
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

    # num_person_participants, num_organ_doses,
    # num_patient_state, num_glomerular_filtration_rate
    statistics = data.annotate(
        num_person_participants=Count(
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__"
            "personparticipant",
            distinct=True,
        ),
        num_organ_doses=Count(
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__"
            "organdose",
            distinct=True,
        ),
        num_patient_state=Count(
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__"
            "patientstate",
            distinct=True,
        ),
        num_glomerular_filtration_rate=Count(
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationpatientcharacteristics__"
            "glomerularfiltrationrate",
            distinct=True,
        ),
        num_pet_series=Count(
            "radiopharmaceuticalradiationdose__petseries",
            distinct=True,
        ),
    ).aggregate(
        max_person_participants=Max("num_person_participants"),
        max_organ_doses=Max("num_organ_doses"),
        max_patient_states=Max("num_patient_state"),
        max_glomerular_filtration_rates=Max("num_glomerular_filtration_rate"),
        max_pet_series=Max("num_pet_series"),
    )

    return (data, statistics)


def _build_block_header(
    headings, get_header_block, header_block_name, header_lengths, statistics=None
):
    tmp = get_header_block(1)
    header_lengths[header_block_name] = len(tmp)
    if statistics is None:
        until = 2  # 1 + 1
    else:
        until = statistics[header_block_name] + 1
    for i in range(1, until):
        headings += get_header_block(i)
    return headings, header_lengths


def _nm_headers(pid, name, patid, statistics):
    headings = common_headers("NM", pid, name, patid)
    header_lengths = {}
    headings.remove("No. events")  # There is always just one event for a study
    header_lengths["common_begin"] = len(headings)
    headings += [
        "Radiopharmaceutical Agent",
        "Radionuclide",
        "Radionuclide Half Live",
        "Administered activity (MBq)",
        "Effective dose (mSv)",
        "Associated Procedure",
        "Radiopharmaceutical Start Time",
        "Radiopharmaceutical Stop Time",
        "Route of Administration",
        "Route of Administration Laterality",
    ]
    headings, header_lengths = _build_block_header(
        headings,
        lambda i: [f"Person participant name {i}", f"Person participant role {i}"],
        "max_person_participants",
        header_lengths,
        statistics,
    )
    headings += ["Comment"]
    headings, header_lengths = _build_block_header(
        headings,
        lambda i: [
            f"Organ Dose Finding Site {i}",
            f"Organ Laterality {i}",
            f"Organ Dose {i} (mGy)",
            f"Organ Mass {i}(g)",
            f"Organ Dose Measurement Method {i}",
            f"Organ Dose Reference Authority {i}",
        ],
        "max_organ_doses",
        header_lengths,
        statistics,
    )
    headings, header_lengths = _build_block_header(
        headings,
        lambda i: [f"Patient state {i}"],
        "max_patient_states",
        header_lengths,
        statistics,
    )
    headings, header_lengths = _build_block_header(
        headings,
        lambda i: [
            "Body Surface Area (m^2)",
            "Body Surface Area Formula",
            "Body Mass Index (kg/m^2)",
            "Body Mass Index Equation",
            "Glucose (mmol/l)",
            "Fasting Duration (hours)",
            "Hydration Volume (ml)",
            "Recent Physical Activity",
            "Serum Creatinine (mg/dl)",
        ],
        "max_patient_charac_header",
        header_lengths,
        None,
    )
    headings, header_lengths = _build_block_header(
        headings,
        lambda i: [
            f"Glomerular Filtration Rate {i} (ml/min/1.73m^2)",
            f"Measurement Method {i}",
            f"Equivalent meaning of concept name {i}",
        ],
        "max_glomerular_filtration_rates",
        header_lengths,
        statistics,
    )
    headings, header_lengths = _build_block_header(
        headings,
        lambda i: [
            f"Series {i} date",
            f"Series {i} number of slices",
            f"Series {i} reconstruction method",
            f"Series {i} coincidence window width",
            f"Series {i} energy window lower limit",
            f"Series {i} energy window upper limit",
            f"Series {i} scan procession",
            f"Series {i} number of RR intervals",
            f"Series {i} number of time slots",
            f"Series {i} number of time slices",
        ],
        "max_pet_series",
        header_lengths,
        statistics,
    )

    return (headings, header_lengths)


def _get_code_not_none(code):
    if code is None:
        return None
    else:
        return code.code_meaning


def _array_to_match_maximum(len_current, len_max, current_header_length):
    return [
        None
        for _ in range(
            len_max * current_header_length - len_current * current_header_length
        )
    ]


def _extract_study_data(exams, pid, name, patid, statistics, header_lengths):
    exam_data = get_common_data("NM", exams, pid, name, patid)

    try:
        radiopharm = exams.radiopharmaceuticalradiationdose_set.get()
        radiopharm_admin = (
            radiopharm.radiopharmaceuticaladministrationeventdata_set.get()
        )
        patient_charac = (
            radiopharm.radiopharmaceuticaladministrationpatientcharacteristics_set.first()
        )
        person_participants = radiopharm_admin.personparticipant_set.all()
        organ_doses = radiopharm_admin.organdose_set.all()
        if patient_charac is not None:
            patient_states = patient_charac.patientstate_set.all()
            glomerular_filtration_rates = (
                patient_charac.glomerularfiltrationrate_set.all()
            )
        else:
            patient_states, glomerular_filtration_rates = ([], [])
        pet_series = radiopharm.petseries_set.all()
    except ObjectDoesNotExist:
        raise  # We handle this on the level of the export function
    if radiopharm_admin.radiopharmaceutical_agent is None:
        radiopharmaceutical_agent = radiopharm_admin.radiopharmaceutical_agent_string
    else:
        radiopharmaceutical_agent = _get_code_not_none(
            radiopharm_admin.radiopharmaceutical_agent
        )
    exam_data += [
        radiopharmaceutical_agent,
        _get_code_not_none(radiopharm_admin.radionuclide),
        radiopharm_admin.radionuclide_half_life,
        radiopharm_admin.administered_activity,
        radiopharm_admin.effective_dose,
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
    exam_data += _array_to_match_maximum(
        len(person_participants),
        statistics["max_person_participants"],
        header_lengths["max_person_participants"],
    )
    exam_data += [radiopharm.comment]
    for organ_dose in organ_doses:
        if organ_dose.reference_authority_code is not None:
            organ_dose_reference_authority = (
                organ_dose.reference_authority_code.code_meaning
            )
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
    exam_data += _array_to_match_maximum(
        len(organ_doses),
        statistics["max_organ_doses"],
        header_lengths["max_organ_doses"],
    )
    for patient_state in patient_states:
        exam_data += [_get_code_not_none(patient_state.patient_state)]
    exam_data += _array_to_match_maximum(
        len(patient_states),
        statistics["max_patient_states"],
        header_lengths["max_patient_states"],
    )
    if patient_charac is not None:
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
    else:
        exam_data += _array_to_match_maximum(
            0, 1, header_lengths["max_patient_charac_header"]
        )
    for glomerular in glomerular_filtration_rates:
        exam_data += [
            glomerular.glomerular_filtration_rate,
            _get_code_not_none(glomerular.measurement_method),
            _get_code_not_none(glomerular.equivalent_meaning_of_concept_name),
        ]
    exam_data += _array_to_match_maximum(
        len(glomerular_filtration_rates),
        statistics["max_glomerular_filtration_rates"],
        header_lengths["max_glomerular_filtration_rates"],
    )
    for series in pet_series:
        exam_data += [
            series.series_datetime,
            series.number_of_slices,
            series.reconstruction_method,
            series.coincidence_window_width,
            series.energy_window_lower_limit,
            series.energy_window_upper_limit,
            series.scan_progression_direction,
            series.number_of_rr_intervals,
            series.number_of_time_slots,
            series.number_of_time_slices,
        ]
    exam_data += _array_to_match_maximum(
        len(pet_series), statistics["max_pet_series"], header_lengths["max_pet_series"]
    )

    for i, item in enumerate(exam_data):
        if item is None:
            exam_data[i] = ""
        if isinstance(item, str):
            exam_data[i] = item.replace(",", ";")

    return exam_data


def exportNM2csv(filterdict, pid=False, name=None, patid=None, user=None):
    """
    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves csv file Media directory for user to download
    """
    logger.debug("Started csv export task for NM")

    date_stamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    task = create_export_task(
        task_id=task_id,
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
        headings, header_lengths = _nm_headers(pid, name, patid, statistics)
        writer.writerow(headings)

        task.progress = "CSV header row written."
        task.save()

        for i, exam in enumerate(data):
            try:
                exam_data = _extract_study_data(
                    exam, pid, name, patid, statistics, header_lengths
                )
                writer.writerow(exam_data)
            except ObjectDoesNotExist:
                error_message = (
                    f"DoesNotExist error whilst exporting study {i + 1} of {task.num_records}, "
                    f"study UID {exam.study_instance_uid}, accession number  {exam.accession_number} "
                    f"- maybe database entry was deleted as part of importing later version of same "
                    "study?"
                )
                logger.error(error_message)
                writer.writerow([error_message])
            task.progress = f"{i+1} of {task.num_records} written."
            task.save()
    except Exception:  # pylint: disable=broad-except
        unknown_error(task, date_stamp)

    tmpfile.close()
    task.progress = "All data written."
    _exit_proc(task, date_stamp)


def _write_nm_excel_sheet(
    task,
    sheet,
    data,
    pid,
    name,
    patid,
    headings,
    statistics,
    header_lengths,
    sheet_index=1,
    sheet_total=1,
):
    sheet.write_row(0, 0, headings)
    numcolumns = len(headings) - 1
    if isinstance(data, list):
        numrows = len(data)
    else:
        numrows = data.count()
    sheet.autofilter(0, 0, numrows, numcolumns)

    for i, exam in enumerate(data):
        try:
            exam_data = _extract_study_data(
                exam, pid, name, patid, statistics, header_lengths
            )
            sheet.write_row(i + 1, 0, exam_data)
        except ObjectDoesNotExist:
            error_message = (
                f"DoesNotExist error whilst exporting study {i + 1} of {task.num_records},"
                f" study UID {exam.study_instance_uid}, accession number {exam.accession_number} - maybe database "
                "entry was deleted as part of importing later version of same study?"
            )
            logger.error(error_message)
            sheet.write_row(i + 1, 0, [error_message])

        task.progress = f"{i+1} of {task.num_records} written on sheet {sheet_index} of {sheet_total}"
        task.save()


def exportNM2excel(filterdict, pid=False, name=None, patid=None, user=None):
    """
    :param filterdict: Queryset of studies to export
    :param pid: does the user have patient identifiable data permission
    :param name: has patient name been selected for export
    :param patid: has patient ID been selected for export
    :param user: User that has started the export
    :return: Saves xlsx file into Media directory for user to download
    """
    logger.debug("Started XLSX export task for NM")

    date_stamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    task = create_export_task(
        task_id=task_id,
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
        headings, header_lengths = _nm_headers(pid, name, patid, statistics)

        summary = book.add_worksheet("Summary")
        create_summary_sheet(task, data, book, summary, None, False)

        # We create the detail sheets per study description, other than for other modalities
        study_descriptions = {}
        for exam in data:
            if exam.study_description:
                w = study_descriptions.setdefault(exam.study_description, [])
            else:
                w = study_descriptions.setdefault("Unknown", [])
            w.append(exam)
        study_descriptions = list(study_descriptions.items())
        study_descriptions.sort(key=lambda x: x[0])
        sheet_count = len(study_descriptions) + 1

        all_data = book.add_worksheet("All data")
        book = text_and_date_formats(book, all_data, pid, name, patid, "NM", headings)
        _write_nm_excel_sheet(
            task,
            all_data,
            data,
            pid,
            name,
            patid,
            headings,
            statistics,
            header_lengths,
            1,
            sheet_count,
        )

        for i, study_description in enumerate(study_descriptions):
            study_description, current_data = study_description
            current_sheet = book.add_worksheet(sheet_name(study_description))
            book = text_and_date_formats(
                book, current_sheet, pid, name, patid, "NM", headings
            )
            _write_nm_excel_sheet(
                task,
                current_sheet,
                current_data,
                pid,
                name,
                patid,
                headings,
                statistics,
                header_lengths,
                i,
                sheet_count,
            )

        book.close()
    except Exception:  # pylint: disable=broad-except
        unknown_error(task, date_stamp)

    xlsxfilename = "nmexport{0}.xlsx".format(date_stamp.strftime("%Y%m%d-%H%M%S%f"))
    write_export(
        task, xlsxfilename, tmpxlsx, date_stamp
    )  # Does nearly the same as _exit_proc, so it's used to leave the process
