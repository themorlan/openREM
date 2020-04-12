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

from celery import shared_task
import django
from django import db
from django.core.exceptions import ObjectDoesNotExist
from django.core.files.base import ContentFile


logger = logging.getLogger(__name__)


basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'
django.setup()


from remapp.models import (GeneralStudyModuleAttr, SizeUpload)  # Absolute import path to prevent issues with script


def _patientstudymoduleattributes(size_upload=None, exam=None, height=None, weight=None, verbose=False):  # C.7.2.2

    log_file = size_upload.logfile
    try:
        patient_attributes = exam.patientstudymoduleattr_set.get()
    except ObjectDoesNotExist:
        logger.error(u"Attempt to import pt size info for study UID {0}/acc. number {1} failed due to a "
                     u"failed import".format(exam.study_instance_uid, exam.accession_number))
        log_file.file.open("a")
        log_file.write("\r\n    ********* Failed to insert size - database entry incomplete *********")
        log_file.file.close()
        if verbose:
            print("    ********* Failed to insert size - database entry incomplete *********")
        return

    if height:
        if not patient_attributes.patient_size:
            patient_attributes.patient_size = Decimal(height) / Decimal(100.)
            log_file.file.open("a")
            log_file.write(f"\r\n    Inserted height of {height} cm")
            log_file.file.close()
            if verbose:
                print(f"    Inserted height of {height} cm")
        elif size_upload.overwrite:
            existing_height = patient_attributes.patient_size * Decimal(100.)
            patient_attributes.patient_size = Decimal(height) / Decimal(100.)
            log_file.file.open("a")
            log_file.write(f"\r\n    Inserted height of {height} cm replacing {existing_height:.0f} cm")
            log_file.file.close()
            if verbose:
                print(f"    Inserted height of {height} cm replacing {existing_height:.0f} cm")
        else:
            existing_height = patient_attributes.patient_size * Decimal(100.)
            log_file.file.open("a")
            log_file.write(f"\r\n    Height of {height} cm not inserted as {existing_height:.0f} cm already in "
                          f"the database")
            log_file.file.close()
            if verbose:
                print(f"    Height of {height} cm not inserted as {existing_height:.0f} cm already in the database")

    if weight:
        if not patient_attributes.patient_weight:
            patient_attributes.patient_weight = weight
            log_file.file.open("a")
            log_file.write("\r\n    Inserted weight of {0} kg".format(weight))
            log_file.file.close()
            if verbose:
                print(f"    Inserted weight of {weight} kg")
        elif size_upload.overwrite:
            existing_weight = patient_attributes.patient_weight
            patient_attributes.patient_weight = weight
            log_file.file.open("a")
            log_file.write(f"\r\n    Inserted weight of {weight} kg replacing {existing_weight:.1f} kg")
            log_file.file.close()
            if verbose:
                print(f"    Inserted weight of {weight} kg replacing {existing_weight:.1f} kg")
        else:
            log_file.file.open("a")
            log_file.write(
                f"\r\n    Weight of {weight} kg not inserted as "
                f"{patient_attributes.patient_weight:.1f} kg already in the database")
            log_file.file.close()
            if verbose:
                print(f"    Weight of {weight} kg not inserted as "
                      f"{patient_attributes.patient_weight:.1f} kg already in the database")

    patient_attributes.save()


def _ptsizeinsert(size_upload=None, accno=None, height=None, weight=None, siuid=False, verbose=False):

    log_file = size_upload.logfile
    if (height or weight) and accno:
        if not siuid:
            exams = GeneralStudyModuleAttr.objects.filter(accession_number__exact=accno)
        else:
            exams = GeneralStudyModuleAttr.objects.filter(study_instance_uid__exact=accno)
        if exams:
            for exam in exams:
                log_file.file.open('a')
                log_file.write(f'\r\n{accno}:')
                log_file.file.close()
                if verbose:
                    print(accno)
                _patientstudymoduleattributes(size_upload=size_upload, exam=exam,
                                              height=height, weight=weight, verbose=verbose)

    db.reset_queries()


@shared_task
def websizeimport(csv_pk=None):
    """Task to import patient size data from the OpenREM web interface.

    :param csv_pk: Database index key for the import record, containing
        the path to the import csv file and the field header details.

    """

    if csv_pk:
        csvrecord = SizeUpload.objects.all().filter(id__exact=csv_pk)[0]
        csvrecord.task_id = websizeimport.request.id
        datestamp = datetime.datetime.now()
        csvrecord.import_date = datestamp
        csvrecord.progress = 'Patient size data import started'
        csvrecord.status = 'CURRENT'
        csvrecord.save()
        if csvrecord.id_type and csvrecord.id_field and csvrecord.height_field and csvrecord.weight_field:
            si_uid = False
            verbose = False
            if csvrecord.id_type == "si-uid":
                si_uid = True

            log_file = "pt_size_import_log_{0}.txt".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))
            headerrow = ContentFile("Patient size import from {0}\r\n".format(csvrecord.sizefile.name))

            try:
                csvrecord.logfile.save(log_file, headerrow)
            except OSError as e:
                csvrecord.progress = "Error saving export file - please contact an administrator. " \
                                     "Error({0}): {1}".format(e.errno, e.strerror)
                csvrecord.status = 'ERROR'
                csvrecord.save()
                return
            except:
                csvrecord.progress = "Unexpected error saving export file - please contact an " \
                                     "administrator: {0}".format(sys.exc_info()[0])
                csvrecord.status = 'ERROR'
                csvrecord.save()
                return

            log_file = csvrecord.logfile
            log_file.file.close()
            # Method used for opening and writing to file as per https://code.djangoproject.com/ticket/13809

            csvrecord.sizefile.open(mode='r')
            f = csvrecord.sizefile.readlines()
            csvrecord.num_records = len(f) - 1
            csvrecord.save()
            try:
                dataset = csv.DictReader(f)
                for i, line in enumerate(dataset):
                    csvrecord.progress = "Processing row {0} of {1}".format(i + 1, csvrecord.num_records)
                    csvrecord.save()
                    _ptsizeinsert(size_upload=csvrecord, accno=line[csvrecord.id_field],
                                  height=line[csvrecord.height_field], weight=line[csvrecord.weight_field],
                                  siuid=si_uid, verbose=verbose)
            finally:
                csvrecord.sizefile.delete()
                csvrecord.processtime = (datetime.datetime.now() - datestamp).total_seconds()
                csvrecord.status = 'COMPLETE'
                csvrecord.save()


def _create_parser():
    parser = argparse.ArgumentParser(
        description="Import height and weight data from a CSV file into an OpenREM database. If either height or "
                    "weight is missing just add a blank column with an appropriate title.")
    parser.add_argument("-u", "--si-uid", action="store_true",
                        help="use Study Instance UID instead of Accession Number")
    parser.add_argument("-v", "--verbose", help="also print log to shell", action="store_true")
    parser.add_argument("-o", "--overwrite", help="overwrite existing values", action="store_true")
    parser.add_argument("csvfile", help="csv file with height, weight and study identifier")
    parser.add_argument("id", help="column title for the accession number or study instance UID")
    parser.add_argument("height", help="column title for the patient height, values in cm")
    parser.add_argument("weight", help="column title for the patient weight, values in kg")

    return parser


def csv2db():
    """ Import patient height and weight data from csv RIS exports. Called from ``openrem_ptsizecsv.py`` script

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
            print(f'Error: one or more of {arg_headers} not found in csv file')
            return
        size_upload = SizeUpload()
        date_stamp = datetime.datetime.now()
        size_upload.progress = 'Patient zie data import from shell started'
        size_upload.status = 'CURRENT'
        size_upload.overwrite = args.overwrite
        size_upload.save()
        log_file_name = "pt_size_import_log_{0}.txt".format(date_stamp.strftime("%Y%m%d-%H%M%S%f"))
        log_header_row = ContentFile(f"Patient size import from {args.csvfile}\r\n")
        size_upload.logfile.save(log_file_name, log_header_row)
        size_upload.save()
        log_file = size_upload.logfile
        log_file.file.close()

        for line in dataset:
            _ptsizeinsert(
                size_upload=size_upload, accno=line[args.id], height=line[args.height], weight=line[args.weight],
                siuid=args.si_uid, verbose=args.verbose)

    size_upload.processtime = (datetime.datetime.now() - date_stamp).total_seconds()
    size_upload.status = 'COMPLETE'
    size_upload.save()