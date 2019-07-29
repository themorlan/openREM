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
    # from remapp.extractors.extract_common import ct_event_type_count

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
        # try:
        #     study.number_of_events = study.ctradiationdose_set.get().ctirradiationeventdata_set.count()
        #     study.total_dlp = study.ctradiationdose_set.get().ctaccumulateddosedata_set.get(
        #         ).ct_dose_length_product_total
        #     study.save()
        #     ct_event_type_count(study)
        # except ObjectDoesNotExist:
        #     logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
        #         study.modality_type, study.pk, study.study_instance_uid))
        task.current_study += 1
        task.save()
    # logger.debug(u"Completed migration of CT to summary fields")
    # task.complete = True
    # task.save()


@shared_task
def populate_summary_mg():
    """Populate the MG summary fields in GeneralStudyModuleAttr table for existing studies

    :return:
    """
    from remapp.extractors.extract_common import populate_mammo_agd_summary

    try:
        task = SummaryFields.objects.get(modality_type__exact='MG')
    except ObjectDoesNotExist:
        task = SummaryFields.objects.create(modality_type='MG')
    all_mg = GeneralStudyModuleAttr.objects.filter(modality_type__exact='MG').order_by('pk')
    task.total_studies = all_mg.count()
    to_process_mg = all_mg.exclude(number_of_events__gt=0)
    task.current_study = task.total_studies - to_process_mg.count()
    task.save()
    logger.debug(u"Starting migration of MG to summary fields")
    for study in to_process_mg:
        populate_summary_study_level('MG', study.pk)
        # try:
        #     study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
        #     study.save()
        #     populate_mammo_agd_summary(study)
        # except ObjectDoesNotExist:
        #     logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
        #         study.modality_type, study.pk, study.study_instance_uid))
        task.current_study += 1
        task.save()
    # logger.debug(u"Completed migration of MG to summary fields")
    # task.complete = True
    # task.save()


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
    all_dx = GeneralStudyModuleAttr.objects.filter(
        Q(modality_type__exact='DX') | Q(modality_type__exact='CR')).order_by('pk')
    task.total_studies = all_dx.count()
    to_process_dx = all_dx.exclude(number_of_events__gt=0)
    task.current_study = task.total_studies - to_process_dx.count()
    task.save()
    logger.debug(u"Starting migration of DX to summary fields")
    for study in to_process_dx:
        populate_summary_study_level('DX', study.pk)
        # try:
        #     study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
        #     study.save()
        #     populate_dx_rf_summary(study)
        # except ObjectDoesNotExist:
        #     logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
        #         study.modality_type, study.pk, study.study_instance_uid))
        task.current_study += 1
        task.save()
    # logger.debug(u"Completed migration of DX to summary fields")
    # task.complete = True
    # task.save()


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
    all_rf = GeneralStudyModuleAttr.objects.filter(modality_type__exact='RF').order_by('pk')
    task.total_studies = all_rf.count()
    to_process_rf = all_rf.exclude(number_of_events__gt=0)
    task.current_study = task.total_studies - to_process_rf.count()
    task.save()
    logger.debug(u"Starting migration of RF to summary fields")
    for study in to_process_rf:
        populate_summary_study_level('RF', study.pk)
        # try:
        #     study.number_of_events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.count()
        #     study.save()
        #     populate_dx_rf_summary(study)
        #     populate_rf_delta_weeks_summary(study)
        # except ObjectDoesNotExist:
        #     logger.warning(u"{0} {1} with study UID {2}: unable to set summary data.".format(
        #         study.modality_type, study.pk, study.study_instance_uid))
        task.current_study += 1
        task.save()
    # logger.debug(u"Completed migration of RF to summary fields")
    # task.complete = True
    # task.save()
