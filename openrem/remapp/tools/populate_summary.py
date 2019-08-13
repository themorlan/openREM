# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2012-2019  The Royal Marsden NHS Foundation Trust
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
..  module:: populate_summary.
    :synopsis: Populates new summary study-level fields on upgrade to 0.10.

..  moduleauthor:: Ed McDonagh

"""

from celery import shared_task
from django.core.exceptions import ObjectDoesNotExist
import logging
from remapp.models import GeneralStudyModuleAttr, SummaryFields

logger = logging.getLogger(__name__)


@shared_task
def populate_summary_study_level(modality, study_pk):
    """Enables the summary level data to be sent as a task at study level

    :param modality: Modality type
    :param study_pk: GeneralStudyModuleAttr database object primary key
    :return:
    """
    from remapp.extractors.extract_common import populate_mammo_agd_summary, populate_dx_rf_summary, \
        populate_rf_delta_weeks_summary, ct_event_type_count

    try:
        study = GeneralStudyModuleAttr.objects.get(pk__exact=study_pk)
    except ObjectDoesNotExist:
        logger.error(u"Attempt to get {0} study with pk {1} failed - presumably deleted?".format(modality, study_pk))
        return
    try:
        if modality in ['DX', 'RF']:
            study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
            study.save()
            populate_dx_rf_summary(study)
            if modality in 'RF':
                populate_rf_delta_weeks_summary(study)
        elif 'MG' in modality:
            study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
            study.save()
            populate_mammo_agd_summary(study)
        elif modality in 'CT':
            study.number_of_events = study.ctradiationdose_set.get().ctirradiationeventdata_set.count()
            study.total_dlp = study.ctradiationdose_set.get().ctaccumulateddosedata_set.get(
                ).ct_dose_length_product_total
            study.save()
            ct_event_type_count(study)
    except ObjectDoesNotExist:
        logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
            study.modality_type, study.pk, study.study_instance_uid))


@shared_task
def populate_summary_ct():
    """Populate the CT summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """

    try:
        task = SummaryFields.objects.get(modality_type__exact='CT')
    except ObjectDoesNotExist:
        task = SummaryFields.objects.create(modality_type__exact='CT')
    all_ct = GeneralStudyModuleAttr.objects.filter(modality_type__exact='CT').order_by('pk')
    task.total_studies = all_ct.count()
    to_process_ct = all_ct.filter(number_of_const_angle__isnull=True)
    task.current_study = task.total_studies - to_process_ct.count()
    task.save()
    logger.debug(u"Starting migration of CT to summary fields")
    for study in to_process_ct:
        populate_summary_study_level.delay('CT', study.pk)
        task.current_study += 1
        task.save()


@shared_task
def populate_summary_mg():
    """Populate the MG summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """

    try:
        task = SummaryFields.objects.get(modality_type__exact='MG')
    except ObjectDoesNotExist:
        task = SummaryFields.objects.create(modality_type='MG')
    all_mg = GeneralStudyModuleAttr.objects.filter(modality_type__exact='MG').order_by('pk')
    task.total_studies = all_mg.count()
    to_process_mg = all_mg.filter(total_agd_right__isnull=True).filter(
        total_agd_left__isnull=True).filter(
        total_agd_both__isnull=True)
    task.current_study = task.total_studies - to_process_mg.count()
    task.save()
    logger.debug(u"Starting migration of MG to summary fields")
    for study in to_process_mg:
        populate_summary_study_level('MG', study.pk)
        task.save()


@shared_task
def populate_summary_dx():
    """Populate the DX summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """
    from django.db.models import Q

    try:
        task = SummaryFields.objects.get(modality_type__exact='DX')
    except ObjectDoesNotExist:
        task = SummaryFields.objects.create(modality_type='DX')
    all_dx = GeneralStudyModuleAttr.objects.filter(
        Q(modality_type__exact='DX') | Q(modality_type__exact='CR')).order_by('pk')
    task.total_studies = all_dx.count()
    to_process_dx = all_dx.filter(number_of_events_a__isnull=True)
    task.current_study = task.total_studies - to_process_dx.count()
    task.save()
    logger.debug(u"Starting migration of DX to summary fields")
    for study in to_process_dx:
        populate_summary_study_level('DX', study.pk)
        task.current_study += 1
        task.save()


@shared_task
def populate_summary_rf():
    """Populate the RF summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """

    try:
        task = SummaryFields.objects.get(modality_type__exact='RF')
    except ObjectDoesNotExist:
        task = SummaryFields.objects.create(modality_type='RF')
    all_rf = GeneralStudyModuleAttr.objects.filter(modality_type__exact='RF').order_by('pk')
    task.total_studies = all_rf.count()
    to_process_rf = all_rf.filter(number_of_events_a__isnull=True)
    task.current_study = task.total_studies - to_process_rf.count()
    task.save()
    logger.debug(u"Starting migration of RF to summary fields")
    for study in to_process_rf:
        populate_summary_study_level('RF', study.pk)
        task.current_study += 1
        task.save()
