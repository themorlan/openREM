from celery import shared_task
from django.core.exceptions import ObjectDoesNotExist
import logging
from remapp.models import GeneralStudyModuleAttr, SummaryFields

logger = logging.getLogger(__name__)


@shared_task
def populate_summary():
    """Populate the summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """
    from django.db.models import Q
    from remapp.extractors.extract_common import ct_event_type_count, populate_mammo_agd_summary, \
        populate_dx_rf_summary, populate_rf_delta_weeks_summary

    task = SummaryFields.get_solo()
    all_ct = GeneralStudyModuleAttr.objects.filter(modality_type__exact='CT')
    logger.debug(u"Starting migration of CT to summary fields")
    for study in all_ct:
        try:
            study.number_of_events = study.ctradiationdose_set.get().ctirradiationeventdata_set.count()
            study.total_dlp = study.ctradiationdose_set.get().ctaccumulateddosedata_set.get(
                ).ct_dose_length_product_total
            study.save()
            ct_event_type_count(study)
        except ObjectDoesNotExist:
            logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
                study.modality_type, study.pk, study.study_instance_uid))
    logger.debug(u"Completed migration of CT to summary fields")
    all_mg = GeneralStudyModuleAttr.objects.filter(modality_type__exact='MG')
    logger.debug(u"Starting migration of MG to summary fields")
    for study in all_mg:
        try:
            study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
            study.save()
            populate_mammo_agd_summary(study)
        except ObjectDoesNotExist:
            logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
                study.modality_type, study.pk, study.study_instance_uid))
    logger.debug(u"Completed migration of MG to summary fields")
    all_dx = GeneralStudyModuleAttr.objects.filter(Q(modality_type__exact='DX') | Q(modality_type__exact='CR'))
    logger.debug(u"Starting migration of DX to summary fields")
    for study in all_dx:
        try:
            study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
            study.save()
            populate_dx_rf_summary(study)
        except ObjectDoesNotExist:
            logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
                study.modality_type, study.pk, study.study_instance_uid))
    logger.debug(u"Completed migration of DX to summary fields")
    all_rf = GeneralStudyModuleAttr.objects.filter(modality_type__exact='RF')
    logger.debug(u"Starting migration of RF to summary fields")
    for study in all_rf:
        try:
            study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
            study.save()
            populate_dx_rf_summary(study)
            populate_rf_delta_weeks_summary(study)
        except ObjectDoesNotExist:
            logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
                study.modality_type, study.pk, study.study_instance_uid))
    logger.debug(u"Completed migration of RF to summary fields")
    task.complete = True
    task.save()
