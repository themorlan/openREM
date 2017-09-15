import sys
import logging
from hl7settings import *
from celery import shared_task

logger = logging.getLogger('hl7.updater')  # Explicitly named so that it is still handled when using __main__


def create_dicom_agestring(decimal_age):
    """Convert a decimal age to a DICOM age string.

    The following (arbitrary) logic is used:
        1. If age < 6 month use days
        2. Else if age < 6 years use months
        3. Else use years

    :param decimal_age: age as decimal number
    :return: age as DICOM age sting [VR: AS]
    """
    import decimal
    from decimal import Decimal

    if Decimal(decimal_age) < Decimal('0.5'):
        return '{:0>3.0f}D'.format((decimal_age * Decimal('365.25'))
                                   .quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))
    elif Decimal(decimal_age) < Decimal('6'):
        return '{:0>3.0f}M'.format((decimal_age * Decimal('12'))
                                   .quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))
    else:
        return '{:0>3.0f}Y'.format(decimal_age.quantize(Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def get_openrem_modality(modality):
    try:
        return HL7_MODALITY_2_OPENREM_MODALITY[modality]
    except KeyError:
        return ''


def update_patient(mapping, patient=None):
    """update patient information in database (if configured to do so in hl7settings.py)

    Possible updates: Patient Name, Birth date, Sex
    NOTE: patient size and weight are updated at study level (size / weight at the time of the study)

    :param mapping: new patient information based on hl7-message
    :param patient: specific patient record, if patient is updated in conjunction with study update
    :return: return 0 if successful otherwise negative number
    """

    from remapp.models import PatientStudyModuleAttr, PatientModuleAttr, GeneralStudyModuleAttr
    from remapp.tools.hash_id import hash_id
    from django.db.models import Q
    from datetime import datetime
    from decimal import Decimal

    if HL7_PERFORM_PAT_UPDATE:
        if mapping.patient_id:
            pat_id = mapping.patient_id
            hash_pat_id = hash_id(pat_id)
            # get the right patient
            if patient is None:
                patient_qs = PatientModuleAttr.objects.filter((Q(patient_id=pat_id) & Q(id_hashed=False)) |
                                                              (Q(patient_id=hash_pat_id) & Q(id_hashed=True)))
            else:
                patient_qs = patient
            for patient in patient_qs:
                if mapping.patient_name:
                    if patient.name_hashed:
                        patient.patient_name = hash_id(mapping.patient_name)
                    else:
                        patient.patient_name = mapping.patient_name
                if mapping.patient_birthdate:
                    patient_birthdate = datetime.strptime(mapping.patient_birthdate, "%Y%m%d").date()
                    if patient.patient_birth_date != patient_birthdate:
                        patient.patient_birth_date = patient_birthdate

                        patientstudy_record = PatientStudyModuleAttr.objects.filter(
                            general_study_module_attributes_id=patient.general_study_module_attributes_id)[0]
                        study_record = GeneralStudyModuleAttr.objects.filter(
                            id=patient.general_study_module_attributes_id)[0]
                        age = Decimal((study_record.study_date - patient_birthdate).days) / Decimal('365.25')
                        patientstudy_record.patient_age_decimal = age.quantize(Decimal('.1'))
                        patientstudy_record.patient_age = create_dicom_agestring(age)
                        patientstudy_record.save()
                patient.patient_sex = mapping.patient_sex if mapping.patient_sex else patient.patient_sex
                patient.save()
                logger.info('patient information for patient {0} updated.'.format(pat_id))
        else:
            logger.info('Patient update is not possible, PatientID not found in hl7-message')
            return -11
    else:
        logger.debug('System is not configured to update patient information')
    return 0


def merge_patient(mapping):
    """Merge patients in database (if configured to do so in hl7settings.py)

    Effectively patientIDs are set the same

    :param mapping: patient information based on hl7-message
    :return: return 0 if successful otherwise negative number
    """
    from remapp.models import PatientModuleAttr
    from remapp.tools.hash_id import hash_id
    from django.db.models import Q

    if HL7_PERFORM_PAT_MERGE:
        if mapping.patient_id and mapping.patient_mrg_id:
            old_id = mapping.patient_mrg_id
            hash_old_id = hash_id(old_id)
            logger.debug('starting patient merge, old id: {0}, new id: {1}'.format(mapping.patient_mrg_id,
                                                                                   mapping.patient_id))

            patient_qs = PatientModuleAttr.objects.filter((Q(patient_id=old_id) & Q(id_hashed=False)) |
                                                          (Q(patient_id=hash_old_id) & Q(id_hashed=True)))
            logger.debug('query created')

            for patient in patient_qs:
                if patient.id_hashed:
                    patient.patient_id = hash_id(mapping.patient_id)
                else:
                    patient.patient_id = mapping.patient_id
                patient.save()
                logger.debug('patient {0} merged with patient {1}.'.format(old_id, mapping.patient_id))

            if HL7_UPDATE_PATINFO_ON_MERGE:
                return update_patient(mapping)
            else:
                return 0
        else:
            logger.error('Merge not possible, patient_id or patient_id to merge not found in hl7-message')
            return -21
    else:
        logger.debug('Merge message received, but merging not configured.')
        return 0


def update_study(mapping):
    """update study information in database (if configured to do so in hl7settings.py)

    Possible updates: Study date, time, referring physician name, accession number, study id, physician of record,
                      performing physician, operator, modality, procedure code value/meaning, requested procedure
                      code value/meaning, patientstudy size(length) and weight.

    :param mapping: new study information based on hl7-message
    :return: return 0 if successful otherwise negative number
    """
    from remapp.models import GeneralStudyModuleAttr, PatientStudyModuleAttr, \
        UniqueEquipmentNames, GeneralEquipmentModuleAttr, PatientModuleAttr, Hl7Message
    from remapp.tools.hash_id import hash_id
    from django.db.models import Q
    from datetime import datetime, timedelta
    from decimal import Decimal
    from hl7settings import HL7_STUDY_DATE_OFFSET

    if HL7_PERFORM_STUDY_UPDATE:
        hash_acc_nr = hash_id(mapping.study_accession_number)
        if mapping.study_instance_uid:
            study_qs = GeneralStudyModuleAttr.objects.filter(study_instance_uid=mapping.study_instance_uid)
        elif mapping.study_accession_number and HL7_MATCH_STUDY_ON_ACCNR:
            study_qs = GeneralStudyModuleAttr.objects.filter((Q(accession_number=mapping.study_accession_number) &
                                                              Q(accession_hashed=False)) |
                                                             (Q(accession_number=hash_acc_nr) &
                                                              Q(accession_hashed=True)))
        else:
            study_qs = GeneralStudyModuleAttr.objects.none()
        nr_of_studies = study_qs.count()
        if nr_of_studies == 1:
            study = study_qs.get()
            study_datetime = datetime.combine(study.study_date, study.study_time)
            msg_datetime = datetime.strptime(mapping.message.MSH.MSH_7.to_er7()[:14], "%Y%m%d%H%M%S")
            add_only = True
            if (not HL7_ADD_STUDY_ONLY) and (msg_datetime > study_datetime):
                add_only = False
            patientstudy = PatientStudyModuleAttr.objects.filter(general_study_module_attributes=study).get()
            patient = PatientModuleAttr.objects.filter(general_study_module_attributes=study).get()
            if (study.study_date and not add_only) or not study.study_date:
                if mapping.study_date:
                    study_date = datetime.strptime(mapping.study_date, "%Y%m%d").date()
                    if study.study_date != study_date:
                        study.study_date = study_date
                        if study_date and patient.patient_birth_date:
                            age = Decimal((study_date - patient.patient_birth_date).days) / Decimal('365.25')
                            patientstudy.patient_age_decimal = age.quantize(Decimal('.1'))
                            patientstudy.patient_age = create_dicom_agestring(age)
            if (study.study_time and not add_only) or not study.study_time:
                if mapping.study_time and ('.' in mapping.study_time):
                    study.study_time = datetime.strptime(mapping.study_time, "%H%M%S.%f")
                else:
                    study.study_time = datetime.strptime(mapping.study_time, "%H%M%S") if \
                        mapping.study_time else study.study_time
            if (study.referring_physician_name and not add_only) or not study.referring_physician_name:
                study.referring_physician_name = mapping.study_referring_physician_name if \
                    mapping.study_referring_physician_name else study.referring_physician_name
            if (study.referring_physician_identification and not add_only) \
               or not study.referring_physician_identification:
                study.referring_physician_identification = mapping.study_referring_physician_id if \
                mapping.study_referring_physician_id else study.referring_physician_identification
            if (study.study_id and not add_only) or not study.study_id:
                study.study_id = mapping.study_id if mapping.study_id else study.study_id
            if (study.accession_number and not add_only) or not study.accession_number:
                if study.accession_hashed:
                    study.accession_number = hash_acc_nr if \
                        mapping.study_accession_number else study.accession_number
                else:
                    study.accession_number = mapping.study_accession_number if \
                        mapping.study_accession_number else study.accession_number
                # study[0].accession_hashed
            if (study.study_description and not add_only) or not study.study_description:
                study.study_description = mapping.study_description if \
                    mapping.study_description else study.study_description
            if (study.physician_of_record and not add_only) or not study.physician_of_record:
                study.physician_of_record = mapping.study_physisician_of_record if \
                    mapping.study_physician_of_record else study.physician_of_record
            if (study.name_of_physician_reading_study and not add_only) or not study.name_of_physician_reading_study:
                study.name_of_physician_reading_study = mapping.study_name_of_physician_reading_study if \
                    mapping.study_name_of_physician_reading_study else study.name_of_physician_reading_study
            if (study.performing_physician_name and not add_only) or not study.performing_physician_name:
                study.performing_physician_name = mapping.study_performing_physician if \
                    mapping.study_performing_physician else study.performing_physician_name
            if (study.operator_name and not add_only) or not study.operator_name:
                study.operator_name = mapping.study_operator if \
                    mapping.study_operator else study.operator_name
            if (study.modality_type and not add_only) or not study.modality_type:
                if mapping.study_modality:
                    # check if user_defined_modality is set, if this is the case, don't update
                    equipment_module = GeneralEquipmentModuleAttr.objects.get(general_study_module_attributes=study)
                    equipment_name = UniqueEquipmentNames.objects.get(id=equipment_module.unique_equipment_name_id)
                    if not equipment_name.user_defined_modality:
                        modality = get_openrem_modality(mapping.study_modality)
                        if modality:
                            study.modality_type = modality
            if (study.procedure_code_value and not add_only) or not study.procedure_code_value:
                study.procedure_code_value = mapping.study_procedure_code_value if \
                    mapping.study_procedure_code_value else study.procedure_code_value
            if (study.procedure_code_meaning and not add_only) or not study.procedure_code_meaning:
                study.procedure_code_meaning = mapping.study_procedure_code_meaning if \
                    mapping.study_procedure_code_meaning else study.procedure_code_meaning
            if (study.requested_procedure_code_value and not add_only) or not study.requested_procedure_code_value:
                study.requested_procedure_code_value = mapping.study_requested_procedure_code_value if \
                    mapping.study_requested_procedure_code_value else study.requested_procedure_code_value
            if (study.requested_procedure_code_meaning and not add_only) or not study.requested_procedure_code_meaning:
                study.requested_procedure_code_meaning = mapping.study_requested_procedure_code_meaning if \
                    mapping.study_requested_procedure_code_meaning else study.requested_procedure_code_meaning
            study.save()

            if (patientstudy.patient_size and not add_only) or not patientstudy.patient_size:
                patientstudy.patient_size = mapping.study_patient_size if mapping.study_patient_size else \
                    patientstudy.patient_size
            if (patientstudy.patient_weight and not add_only) or not patientstudy.patient_weight:
                patientstudy.patient_weight = mapping.study_patient_weight if mapping.study_patient_weight else \
                    patientstudy.patient_weight
            patientstudy.save()

            if not HL7_UPDATE_PATIENT_ON_ADT_ONLY:
                return update_patient(mapping, [patient])

        elif nr_of_studies == 0:
            # no studies found, if studydate is larger than or equal to today + offset, save data for later use
            if mapping.study_date >= (datetime.today() - timedelta(days=HL7_STUDY_DATE_OFFSET)).strftime('%Y%m%d'):
                try:
                    message = Hl7Message(receive_datetime=datetime.now(),
                                         patient_id=mapping.patient_id,
                                         accession_number=mapping.study_accession_number,
                                         study_instance_uid=mapping.study_instance_uid,
                                         message_type=mapping.message.name,
                                         study_date=mapping.study_date,
                                         message=mapping.message.to_er7())
                except:
                    logger.fatal('Unexpected fatal error while parsing hl-7 message: {0}'.format(sys.exc_info()[0]))
                    return -1, None
                message.save()
                logger.info(
                    'Study update not possible. Study with studyUID / accession number ({0} / {1}) not found.'.format(
                        mapping.study_instance_uid, mapping.study_accession_number) +
                    ' Message saved for later usage.')
            else:
                logger.info(
                    'Study update not possible. Study with studyUID / accession number ({0} / {1}) not found.'.format(
                        mapping.study_instance_uid, mapping.study_accession_number))
            return 0
        else:
            logger.error(
                'Multiple studies found with studyUID / accession number ({0} / {1}), no update performed.'.format(
                    mapping.study_instance_uid, mapping.study_accession_number))
            return -31
    else:
        logger.debug('Study message received, but study update not configured.')
        return 0


def parse_hl7(str_message):
    """parse string as hl7-message

    :param str_message: er7 string
    :return: error-code (0=success) and hl7-message structure
    """
    from hl7apy import parser
    from hl7apy.exceptions import HL7apyException

    # replace \n if exists by \r. hl7apy can't handle \r\n or \n only as end of line marks
    str_message = str_message.replace('\n', '\r')
    # try removing hl7 wrapping chars "0x0B" and "0x1C", shouldn't be there, but just to be sure
    str_message = str_message.replace(b'\x0b', '').replace(b'\x1c', '')
    try:
        m = parser.parse_message(str_message, find_groups=False)
        # for example library doesn't know about ADT_A40 (patient merge)
        if not m.name:
            # TODO: what kind of side-effects does this have?
            m.name = parser.get_message_info(str_message)[1]
    except HL7apyException as e:
        logger.fatal('Fatal error while parsing hl-7 message: {0}'.format(e))
        return -1, None
    except:
        logger.fatal('Unexpected fatal error while parsing hl-7 message: {0}'.format(sys.exc_info()[0]))
        return -1, None
    return 0, m


def apply_hl7(message):
    """Apply hl7 message to database

    :param message: hl7 message structure
    :return: 0 if successful, otherwise negative number
    """
    from hl7mapping import HL7Mapping
    from hl7apy.exceptions import ChildNotFound
    from savehtmlreport import save_html_report
    import os
    from datetime import datetime

    if HL7_SAVE_HL7_MESSAGE:
        try:
            if message.MSH.MSH_7:
                filename = message.MSH.MSH_7.to_er7() + "_" + datetime.now().strftime('%Y%m%d%H%M%S') + '.hl7'
            else:
                filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.hl7'
        except ChildNotFound:
            filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.hl7'

        logger.debug('save hl7-message to {0}'.format(os.path.join(HL7_MESSAGE_LOCATION, filename)))
        try:
            with open(os.path.join(HL7_MESSAGE_LOCATION, filename), 'w') as filepointer:
                filepointer.write(message.to_er7().replace('\r', '\r\n').encode(HL7_MESSAGE_ENCODING))
        except:
            logger.debug(
                'failed writing hl7-message to disk. filename:{0}'.format(os.path.join(HL7_MESSAGE_LOCATION, filename)))
            logger.debug('exception: {0}'.format(sys.exc_info()[0]))
            logger.debug('exception-info: {0}'.format(sys.exc_info()[1]))
            logger.debug('exception-info: {0}'.format(sys.exc_info()[2]))

    if HL7_SAVE_HTML_REPORT:
        save_html_report(message)
        logger.debug('html report written')

    hl7_mapping = HL7Mapping(message)

    if 'ADT_A40' in message.name:
        logger.debug('start processing ADT Merge message (A40)')
        result_id = merge_patient(hl7_mapping)
    elif 'ADT' in message.name:
        logger.debug('start processing ADT message')
        result_id = update_patient(hl7_mapping)
    elif ('ORM' in message.name) or ('ORU' in message.name) or ('OMI' in message.name) \
            or ('OMG' in message.name):
        logger.debug('start processing order message')
        logger.debug('accession_number{0}'.format(hl7_mapping.study_accession_number))
        result_id = update_study(hl7_mapping)
    else:
        return -1, 'hl7-message of type {0} is not supported.'.format(message.name)

    if result_id == 0:
        logger.info('processed {0} message successfully.'.format(message.name))
    else:
        logger.error('processing {0} message failed, result: {1}.'.format(message.name, result_id))

    return result_id


def find_message_and_apply(patient_id, accession_number, study_instance_uid):
    from remapp.models import Hl7Message

    logger.info('Applying saved HL7-messages to study-information.')
    messages = Hl7Message.objects.filter(study_instance_uid=study_instance_uid).order_by('receive_datetime')
    if (not messages.exists()) and HL7_MATCH_STUDY_ON_ACCNR:
        messages = Hl7Message.objects.filter(patient_id=patient_id, accession_number=accession_number) \
            .order_by('receive_datetime')
    if messages.exists():
        for message in messages:
            result, hl7_message = parse_hl7(message.message)
            apply_hl7(hl7_message)
    else:
        logger.info('No HL7-messages found.')


@shared_task(name='hl7.hl7updater.del_old_messages_from_db')
def del_old_messages_from_db():
    from remapp.models import Hl7Message
    from datetime import datetime, timedelta

    logger.info('Start searching for old HL7-messages in database.')
    if Hl7Message.objects.exclude(study_date__gte=(datetime.today()-timedelta(days=HL7_STUDY_DATE_OFFSET))
                                 .strftime('%Y%m%d')).exists():
        logger.info('Messages found to delete.')
        Hl7Message.objects.exclude(study_date__gte=(datetime.today()-timedelta(days=HL7_STUDY_DATE_OFFSET))
                                  .strftime('%Y%m%d')).delete()
    else:
        logger.info('No old messages found.')


if __name__ == '__main__':
    from logging import config
    logging.config.fileConfig('./logging.conf')

    # This part is necessary if django is not started via "manage.py"
    # Also add basepath to python-path. It is excepted that this file is in ....\openrem\hl7\
    import django
    import os
    basepath = os.path.dirname(__file__)
    basepath = os.path.join(basepath, '..')
    sys.path.append(basepath)
    os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'
    django.setup()

#    if len(sys.argv) != 2:
#        sys.exit('Error: Supply exactly one argument - the hl7 message file')

#    hl7_message = open(sys.argv[1], 'r').read()
#    resultid, msg = parse_hl7(hl7_message)
#    apply_hl7(msg)
    find_message_and_apply('1234', '1235','1.2.3.4.5.6.7.8.9.0')
