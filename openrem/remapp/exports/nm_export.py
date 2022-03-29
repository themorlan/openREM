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
from django.utils.translation import gettext as _
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
    create_summary_sheet,
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
    logger.error(f"Failed to export NM with error: \n {''.join(traceback.format_exc())}")
    _exit_proc(task, date_stamp, traceback.format_exception_only(etype, evalue)[-1])

def _nm_headers(pid, name, patid):
    headings = common_headers("NM", pid, name, patid)
    headings.remove("No. events") # There is always just one event for a study
    headings += ["Radiopharmaceutical Agent", "Radionuclide", "Radionuclide Half Live", 
        "Start Time", "Stop Time", "Administered activity (MBq)", "Radiopharmaceutical Volume (cm^3)"]

    return headings

def _get_data(filterdict, pid, task):
    data = nm_filter(filterdict, pid).qs

    task.num_records = data.count()
    if abort_if_zero_studies(task.num_records, task):
        _exit_proc(task, datetime.datetime.now(), "Zero studies marked for export")
    task.progress = f"{task.num_records} studies in query"
    task.save()

    return data

def _extract_study_data(exams, pid, name, patid):
    exam_data = get_common_data("NM", exams, pid, name, patid)

    try:
        radiopharm = exams.radiopharmaceuticalradiationdose_set.get()
        radiopharm_admin = radiopharm.radiopharmaceuticaladministrationeventdata_set.get()
        radiopharm_agent = radiopharm_admin.radiopharmaceutical_agent.code_meaning
        radiopharm_radionuclide = radiopharm_admin.radionuclide.code_meaning
        radiopharm_radionuclide_half_life = radiopharm_admin.radionuclide_half_life
        radiopharm_start = radiopharm_admin.radiopharmaceutical_start_datetime
        radiopharm_stop = radiopharm_admin.radiopharmaceutical_stop_datetime
        radiopharm_activity = radiopharm_admin.administered_activity
        radiopharm_volume = radiopharm_admin.radiopharmaceutical_volume
    except ObjectDoesNotExist:
        raise # We handle this on the level of the export

    exam_data += [radiopharm_agent, radiopharm_radionuclide, 
        radiopharm_radionuclide_half_life, radiopharm_start,
        radiopharm_stop, radiopharm_activity, radiopharm_volume]

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
        
        data = _get_data(filterdict, pid, task)
        headings = _nm_headers(pid, name, patid)
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
        
        data = _get_data(filterdict, pid, task)
        headings = _nm_headers(pid, name, patid)
        
        all_data = book.add_worksheet("All data")
        book = text_and_date_formats(book, all_data, pid, name, patid)

        all_data.write_row(0, 0, headings)
        numcolumns = len(headings) - 1
        numrows = data.count()
        all_data.autofilter(0, 0, numrows, numcolumns)

        for i, exam in enumerate(data):
            try:
                exam_data = _extract_study_data(exam, pid, name, patid)
                all_data.write_row(i+1, 0, exam_data)
            except ObjectDoesNotExist:
                error_message = (
                    f"DoesNotExist error whilst exporting study {i + 1} of {task.num_records},  study UID {exam.study_instance_uid}, accession number"
                    f" {exam.accession_number} - maybe database entry was deleted as part of importing later version of same"
                    " study?"
                )
                logger.error(error_message)
                all_data.write_row(i+1, 0, [error_message])
            
            task.progress = f"{i+1} of {task.num_records} written."
            task.save()
        
        book.close()
    except Exception:
        unknown_error(task, date_stamp)
    
    xlsxfilename = "nmexport{0}.xlsx".format(date_stamp.strftime("%Y%m%d-%H%M%S%f"))
    write_export(task, xlsxfilename, tmpxlsx, date_stamp) # Does nearly the same as _exit_proc, so it's used to leave the process