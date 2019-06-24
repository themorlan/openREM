from celery import shared_task
from django.core.exceptions import ObjectDoesNotExist
import logging
from remapp.models import GeneralStudyModuleAttr, SummaryFields

logger = logging.getLogger(__name__)


@shared_task
def populate_summary_ct():
    """Populate the CT summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """
    from remapp.extractors.extract_common import ct_event_type_count

    try:
        task = SummaryFields.objects.get(modality_type__exact='CT')
    except ObjectDoesNotExist:
        task = SummaryFields.objects.create(modality_type='CT')
    all_ct = GeneralStudyModuleAttr.objects.filter(modality_type__exact='CT')
    task.total_studies = all_ct.count()
    task.current_study = 1
    task.save()
    logger.debug(u"Starting migration of CT to summary fields")
    for study in all_ct:
        if not study.number_of_events:
            try:
                study.number_of_events = study.ctradiationdose_set.get().ctirradiationeventdata_set.count()
                study.total_dlp = study.ctradiationdose_set.get().ctaccumulateddosedata_set.get(
                    ).ct_dose_length_product_total
                study.save()
                ct_event_type_count(study)
            except ObjectDoesNotExist:
                logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
                    study.modality_type, study.pk, study.study_instance_uid))
        task.current_study += 1
        task.save()
    logger.debug(u"Completed migration of CT to summary fields")
    task.complete = True
    task.save()


@shared_task
def populate_summary_mg():
    """Populate the MG summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """
    from remapp.extractors.extract_common import populate_mammo_agd_summary

    try:
        task = SummaryFields.objects.get(modality_type__exact='MG')
    except ObjectDoesNotExist:
        task = SummaryFields.objects.create(modality_type='CT')
    all_mg = GeneralStudyModuleAttr.objects.filter(modality_type__exact='MG')
    task.total_studies = all_mg.count()
    task.current_study = 1
    task.save()
    logger.debug(u"Starting migration of MG to summary fields")
    for study in all_mg:
        if not study.number_of_events:
            try:
                study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
                study.save()
                populate_mammo_agd_summary(study)
            except ObjectDoesNotExist:
                logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
                    study.modality_type, study.pk, study.study_instance_uid))
        task.current_study += 1
        task.save()
    logger.debug(u"Completed migration of MG to summary fields")
    task.complete = True
    task.save()


@shared_task
def populate_summary_dx():
    """Populate the DX summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """
    from django.db.models import Q
    from remapp.extractors.extract_common import populate_dx_rf_summary

    try:
        task = SummaryFields.objects.get(modality_type__exact='DX')
    except ObjectDoesNotExist:
        task = SummaryFields.objects.create(modality_type='DX')
    all_dx = GeneralStudyModuleAttr.objects.filter(Q(modality_type__exact='DX') | Q(modality_type__exact='CR'))
    task.total_studies = all_dx.count()
    task.current_study = 1
    task.delete()
    logger.debug(u"Starting migration of DX to summary fields")
    for study in all_dx:
        if not study.number_of_events:
            try:
                study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
                study.save()
                populate_dx_rf_summary(study)
            except ObjectDoesNotExist:
                logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
                    study.modality_type, study.pk, study.study_instance_uid))
        task.current_study += 1
        task.save()
    logger.debug(u"Completed migration of DX to summary fields")
    task.complete = True
    task.save()


@shared_task
def populate_summary_rf():
    """Populate the RF summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """
    from remapp.extractors.extract_common import populate_dx_rf_summary, populate_rf_delta_weeks_summary

    try:
        task = SummaryFields.objects.get(modality_type__exact='RF')
    except ObjectDoesNotExist:
        task = SummaryFields.objects.create(modality_type='RF')
    all_rf = GeneralStudyModuleAttr.objects.filter(modality_type__exact='RF')
    task.total_studies = all_rf.count()
    task.current_study = 1
    task.save()
    logger.debug(u"Starting migration of RF to summary fields")
    for study in all_rf:
        if not study.number_of_events:
            try:
                study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
                study.save()
                populate_dx_rf_summary(study)
                populate_rf_delta_weeks_summary(study)
            except ObjectDoesNotExist:
                logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
                    study.modality_type, study.pk, study.study_instance_uid))
        task.current_study += 1
        task.save()
    logger.debug(u"Completed migration of RF to summary fields")
    task.complete = True
    task.save()
