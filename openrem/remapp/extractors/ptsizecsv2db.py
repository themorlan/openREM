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
..  module:: ptsizecsv2db.
    :synopsis: Use to import height and weight data from csv file to existing studies in the database.

..  moduleauthor:: Ed McDonagh

"""
import argparse
import csv
import datetime
from decimal import Decimal
import logging
import os
import sys
import uuid

import django
from django import db
from django.core.exceptions import ObjectDoesNotExist
from django.core.files.base import ContentFile
from openrem.remapp.tools.background import (
    get_or_generate_task_uuid,
    record_task_error_exit,
)


logger = logging.getLogger(__name__)


basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()


# Absolute import path to prevent issues with script
from remapp.models import (
    GeneralStudyModuleAttr,
    SizeUpload,
)  # pylint: disable=wrong-import-position


def _patientstudymoduleattributes(
    size_upload=None, exam=None, size_dict=None
):  # C.7.2.2

    log_file = size_upload.logfile
    try:
        patient_attributes = exam.patientstudymoduleattr_set.get()
    except ObjectDoesNotExist:
        error = (
            f"Attempt to import pt size info for study UID {exam.study_instance_uid}/acc. number"
            f" {exam.accession_number} failed due to a failed import"
        )
        logger.error(error)
        record_task_error_exit(error)
        log_file.file.open("a")
        log_file.write(
            "\r\n    ********* Failed to insert size - database entry incomplete *********"
        )
        log_file.file.close()
        if size_dict["verbose"]:
            print(
                "    ********* Failed to insert size - database entry incomplete *********"
            )
        return

    if size_dict["height"]:
        if not patient_attributes.patient_size:
            patient_attributes.patient_size = Decimal(size_dict["height"]) / Decimal(
                100.0
            )
            log_file.file.open("a")
            log_file.write(f"\r\n    Inserted height of {size_dict['height']} cm")
            log_file.file.close()
            if size_dict["verbose"]:
                print(f"    Inserted height of {size_dict['height']} cm")
        elif size_upload.overwrite:
            existing_height = patient_attributes.patient_size * Decimal(100.0)
            patient_attributes.patient_size = Decimal(size_dict["height"]) / Decimal(
                100.0
            )
            log_file.file.open("a")
            log_file.write(
                f"\r\n    Inserted height of {size_dict['height']} cm replacing {existing_height:.0f} cm"
            )
            log_file.file.close()
            if size_dict["verbose"]:
                print(
                    f"    Inserted height of {size_dict['height']} cm replacing {existing_height:.0f} cm"
                )
        else:
            existing_height = patient_attributes.patient_size * Decimal(100.0)
            log_file.file.open("a")
            log_file.write(
                f"\r\n    Height of {size_dict['height']} cm not inserted as {existing_height:.0f} cm "
                f"already in the database"
            )
            log_file.file.close()
            if size_dict["verbose"]:
                print(
                    f"    Height of {size_dict['height']} cm not inserted as {existing_height:.0f} cm already in "
                    f"the database"
                )

    if size_dict["weight"]:
        if not patient_attributes.patient_weight:
            patient_attributes.patient_weight = size_dict["weight"]
            log_file.file.open("a")
            log_file.write(f"\r\n    Inserted weight of {size_dict['weight']} kg")
            log_file.file.close()
            if size_dict["verbose"]:
                print(f"    Inserted weight of {size_dict['weight']} kg")
        elif size_upload.overwrite:
            existing_weight = patient_attributes.patient_weight
            patient_attributes.patient_weight = size_dict["weight"]
            log_file.file.open("a")
            log_file.write(
                f"\r\n    Inserted weight of {size_dict['weight']} kg replacing {existing_weight:.1f} kg"
            )
            log_file.file.close()
            if size_dict["verbose"]:
                print(
                    f"    Inserted weight of {size_dict['weight']} kg replacing {existing_weight:.1f} kg"
                )
        else:
            log_file.file.open("a")
            log_file.write(
                f"\r\n    Weight of {size_dict['weight']} kg not inserted as "
                f"{patient_attributes.patient_weight:.1f} kg already in the database"
            )
            log_file.file.close()
            if size_dict["verbose"]:
                print(
                    f"    Weight of {size_dict['weight']} kg not inserted as "
                    f"{patient_attributes.patient_weight:.1f} kg already in the database"
                )

    patient_attributes.save()


def _ptsizeinsert(size_upload=None, size_dict=None):

    log_file = size_upload.logfile
    if (size_dict["height"] or size_dict["weight"]) and size_dict["acc_no"]:
        if not size_dict["si_uid"]:
            exams = GeneralStudyModuleAttr.objects.filter(
                accession_number__exact=size_dict["acc_no"]
            )
        else:
            exams = GeneralStudyModuleAttr.objects.filter(
                study_instance_uid__exact=size_dict["acc_no"]
            )
        if exams:
            for exam in exams:
                log_file.file.open("a")
                log_file.write(f"\r\n{size_dict['acc_no']}:")
                log_file.file.close()
                if size_dict["verbose"]:
                    print(size_dict["acc_no"])
                _patientstudymoduleattributes(
                    size_upload=size_upload, exam=exam, size_dict=size_dict
                )

    db.reset_queries()


def websizeimport(csv_pk=None):
    """Task to import patient size data from the OpenREM web interface.

    :param csv_pk: Database index key for the import record, containing
        the path to the import csv file and the field header details.

    """

    if csv_pk:
        size_upload = SizeUpload.objects.all().filter(id__exact=csv_pk)[0]
        size_upload.task_id = get_or_generate_task_uuid()
        datestamp = datetime.datetime.now()
        size_upload.import_date = datestamp
        size_upload.progress = "Patient size data import started"
        size_upload.status = "CURRENT"
        size_upload.save()
        if (
            size_upload.id_type
            and size_upload.id_field
            and size_upload.height_field
            and size_upload.weight_field
        ):
            si_uid = False
            verbose = False
            if size_upload.id_type == "si-uid":
                si_uid = True

            log_file = "pt_size_import_log_{0}.txt".format(
                datestamp.strftime("%Y%m%d-%H%M%S%f")
            )
            headerrow = ContentFile(
                f"Patient size import from {size_upload.sizefile.name}\r\n"
            )

            try:
                size_upload.logfile.save(log_file, headerrow)
            except OSError as e:
                error = (
                    "Error saving export file - please contact an administrator. "
                    "Error({0}): {1}".format(e.errno, e.strerror)
                )
                size_upload.progress = error
                size_upload.status = "ERROR"
                size_upload.save()
                record_task_error_exit(error)
                return
            except:
                error = (
                    "Unexpected error saving export file - please contact an "
                    "administrator: {0}".format(sys.exc_info()[0])
                )
                size_upload.progress = error
                size_upload.status = "ERROR"
                size_upload.save()
                record_task_error_exit(error)
                return

            log_file = size_upload.logfile
            log_file.file.close()
            # Method used for opening and writing to file as per https://code.djangoproject.com/ticket/13809

            size_upload.sizefile.open(mode="r")
            csv_file = size_upload.sizefile.readlines()
            size_upload.num_records = len(csv_file) - 1
            size_upload.save()
            try:
                dataset = csv.DictReader(csv_file)
                for i, line in enumerate(dataset):
                    size_upload.progress = (
                        f"Processing row {i + 1} of {size_upload.num_records}"
                    )
                    size_upload.save()
                    size_dict = {
                        "acc_no": line[size_upload.id_field],
                        "height": line[size_upload.height_field],
                        "weight": line[size_upload.weight_field],
                        "si_uid": si_uid,
                        "verbose": verbose,
                    }
                    _ptsizeinsert(size_upload=size_upload, size_dict=size_dict)
            finally:
                size_upload.sizefile.delete()
                size_upload.processtime = (
                    datetime.datetime.now() - datestamp
                ).total_seconds()
                size_upload.status = "COMPLETE"
                size_upload.save()


def _create_parser():
    parser = argparse.ArgumentParser(
        description="Import height and weight data from a CSV file into an OpenREM database. If either height or "
        "weight is missing just add a blank column with an appropriate title."
    )
    parser.add_argument(
        "-u",
        "--si-uid",
        action="store_true",
        help="use Study Instance UID instead of Accession Number",
    )
    parser.add_argument(
        "-v", "--verbose", help="also print log to shell", action="store_true"
    )
    parser.add_argument(
        "-o", "--overwrite", help="overwrite existing values", action="store_true"
    )
    parser.add_argument(
        "csvfile", help="csv file with height, weight and study identifier"
    )
    parser.add_argument(
        "id", help="column title for the accession number or study instance UID"
    )
    parser.add_argument(
        "height", help="column title for the patient height, values in cm"
    )
    parser.add_argument(
        "weight", help="column title for the patient weight, values in kg"
    )

    return parser


def csv2db():
    """Import patient height and weight data from csv RIS exports. Called from ``openrem_ptsizecsv.py`` script

    :param args: sys.argv from the command line call

    Example::

        openrem_ptsizecsv.py -s MyRISExport.csv StudyInstanceUID height weight

    """

    args = _create_parser().parse_args()

    with open(args.csvfile) as csv_file:
        dataset = csv.DictReader(csv_file)
        fieldnames = dataset.fieldnames
        arg_headers = [args.id, args.height, args.weight]
        if not all(header in fieldnames for header in arg_headers):
            msg = f"Error: one or more of {arg_headers} not found in csv file"
            print(msg)
            record_task_error_exit(msg)
            return
        size_upload = SizeUpload()
        date_stamp = datetime.datetime.now()
        size_upload.import_date = date_stamp
        size_upload.task_id = uuid.uuid4()
        size_upload.progress = "Patient size data import from shell started"
        size_upload.status = "CURRENT"
        size_upload.overwrite = args.overwrite
        size_upload.save()
        log_file_name = "pt_size_import_log_{0}.txt".format(
            date_stamp.strftime("%Y%m%d-%H%M%S%f")
        )
        log_header_row = ContentFile(f"Patient size import from {args.csvfile}\r\n")
        size_upload.logfile.save(log_file_name, log_header_row)
        size_upload.save()
        log_file = size_upload.logfile
        log_file.file.close()
        size_upload.num_records = len(csv_file.readlines())
        size_upload.save()
        csv_file.seek(0)

        for line in dataset:
            size_dict = {
                "acc_no": line[args.id],
                "height": line[args.height],
                "weight": line[args.weight],
                "si_uid": args.si_uid,
                "verbose": args.verbose,
            }
            _ptsizeinsert(size_upload=size_upload, size_dict=size_dict)

    size_upload.processtime = (datetime.datetime.now() - date_stamp).total_seconds()
    size_upload.status = "COMPLETE"
    size_upload.save()
