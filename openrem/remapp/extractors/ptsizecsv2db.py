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
import logging
import os
import sys

from celery import shared_task
import django
from django.core.exceptions import ObjectDoesNotExist

logger = logging.getLogger(__name__)


# setup django/OpenREM
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'
django.setup()


def _patientstudymoduleattributes(exam, height, weight, verbose, imp_log=None):  # C.7.2.2
    from decimal import Decimal

    try:
        patient_attributes = exam.patientstudymoduleattr_set.get()
    except ObjectDoesNotExist:
        logger.error(u"Attempt to import pt size info for study UID {0}/acc. number {1} failed due to a "
                     u"failed import".format(exam.study_instance_uid, exam.accession_number))
        if imp_log:
            imp_log.file.open("ab")
            imp_log.write("\r\n    ********* Failed to insert size - database entry incomplete *********")
            imp_log.file.close()
        else:
            print(u"    ********* Failed to insert size - database entry incomplete *********")
        return
    if height:
        if not patient_attributes.patient_size:
            patient_attributes.patient_size = Decimal(height) / Decimal(100.)
            if verbose:
                if imp_log:
                    imp_log.file.open("ab")
                    imp_log.write(f"\r\n    Inserted height of {height} cm")
                    imp_log.file.close()
                else:
                    print(u"    Inserted height of {0}".format(height))
        elif verbose:
            existing_height = patient_attributes.patient_size * Decimal(100.)
            if imp_log:
                imp_log.file.open("ab")
                imp_log.write(f"\r\n    Height of {height} cm not inserted as {existing_height:.2f} cm already in "
                              f"the database")
                imp_log.file.close()
            else:
                print(f"    Height of {height} cm not inserted as {existing_height:.2f} cm already in the database")

    if weight:
        if not patient_attributes.patient_weight:
            patient_attributes.patient_weight = weight
            if verbose:
                if imp_log:
                    imp_log.file.open("ab")
                    imp_log.write("\r\n    Inserted weight of {0} kg".format(weight))
                    imp_log.file.close()
                else:
                    print(u"    Inserted weight of {0}".format(weight))
        elif verbose:
            if imp_log:
                imp_log.file.open("ab")
                imp_log.write(
                    "\r\n    Weight of {0} kg not inserted as {1:.1f} kg already in the "
                    "database".format(weight, patient_attributes.patient_weight))
                imp_log.file.close()
            else:
                print(u"    Weight of {0} kg not inserted as {1:.1f} kg already "
                      u"in the database".format(weight, patient_attributes.patient_weight))
    patient_attributes.save()


def _ptsizeinsert(accno, height, weight, siuid, verbose, **kwargs):
    from ..models import GeneralStudyModuleAttr
    from django import db

    imp_log = None
    if 'imp_log' in kwargs:
        imp_log = kwargs['imp_log']

    if (height or weight) and accno:
        if not siuid:
            exams = GeneralStudyModuleAttr.objects.filter(accession_number__exact=accno)
        else:
            exams = GeneralStudyModuleAttr.objects.filter(study_instance_uid__exact=accno)
        if exams:
            for exam in exams:
                if verbose:
                    if imp_log:
                        imp_log.file.open("ab")
                        imp_log.write("\r\n{0}:".format(accno))
                        imp_log.file.close()
                    else:
                        print(u"{0}:".format(accno))
                _patientstudymoduleattributes(exam, height, weight, verbose, imp_log=imp_log)

    db.reset_queries()


@shared_task
def websizeimport(csv_pk=None):
    """Task to import patient size data from the OpenREM web interface.

    :param csv_pk: Database index key for the import record, containing
        the path to the import csv file and the field header details.

    """

    import csv
    import datetime

    from django.core.files.base import ContentFile

    from ..models import SizeUpload

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
            verbose = True
            if csvrecord.id_type == "si-uid":
                si_uid = True

            logfile = "pt_size_import_log_{0}.txt".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))
            headerrow = ContentFile("Patient size import from {0}\r\n".format(csvrecord.sizefile.name))

            try:
                csvrecord.logfile.save(logfile, headerrow)
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

            csvrecord.sizefile.open(mode='rb')
            f = csvrecord.sizefile.readlines()
            csvrecord.num_records = len(f)
            csvrecord.save()
            try:
                dataset = csv.DictReader(f)
                for i, line in enumerate(dataset):
                    csvrecord.progress = "Processing row {0} of {1}".format(i + 1, csvrecord.num_records)
                    csvrecord.save()
                    _ptsizeinsert(line[csvrecord.id_field], line[csvrecord.height_field], line[csvrecord.weight_field],
                                  si_uid, verbose, imp_log=log_file)
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
    parser.add_argument("csvfile", help="csv file with height, weight and study identifier")
    parser.add_argument("id", help="column title for the accession number or study instance UID")
    parser.add_argument("height", help="column title for the patient height, values in cm")
    parser.add_argument("weight", help="column title for the patient weight, values in kg")

    return parser


def csv2db(args):
    """ Import patient height and weight data from csv RIS exports. Called from ``openrem_ptsizecsv.py`` script

    :param args: sys.argv from the command line call

    Example::

        openrem_ptsizecsv.py -s MyRISExport.csv StudyInstanceUID height weight

    """

    args = _create_parser().parse_args()

    f = open(args.csvfile, 'rb')
    try:
        dataset = csv.DictReader(f)
        for line in dataset:
            _ptsizeinsert(line[args.id], line[args.height], line[args.weight], args.si_uid, args.verbose)
    finally:
        f.close()
