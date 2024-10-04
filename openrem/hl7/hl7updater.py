import sys
import os
import io
import logging
from datetime import datetime, timedelta
import decimal
from decimal import Decimal
import django
from django.db.models import Q
from hl7apy import parser
from hl7apy.exceptions import HL7apyException, ChildNotFound
from .hl7settings import *  # .hl7settings
from .hl7mapping import HL7Mapping  # .hl7mapping
from .savehtmlreport import save_html_report  # .savehtmlreport
from remapp.tools.hash_id import hash_id

# Setup django. This is required on windows, because process is created via spawn and
# django will not be initialized anymore then (On Linux this will only be executed once)
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from huey import crontab
from huey.contrib.djhuey import db_periodic_task

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
        return None


def update_patient(mapping, patient=None):
    """update patient information in database (if configured to do so in hl7settings.py)

    Possible updates: Patient Name, Birth date, Sex
    NOTE: patient size and weight are updated at study level (size / weight at the time of the study)

    :param mapping: new patient information based on hl7-message
    :param patient: specific patient record, if patient is updated in conjunction with study update
    :return: return 0 if successful otherwise negative number
    """
    logger.debug('Entering update_patient')
    from remapp.models import PatientStudyModuleAttr, PatientModuleAttr, GeneralStudyModuleAttr

    if HL7_PERFORM_PAT_UPDATE:
        if mapping.patient_id:
            logger.debug('Updating mapping.patient_id')
            pat_id = mapping.patient_id
            hash_pat_id = hash_id(pat_id)
            # get the right patient
            if patient is None:
                patient_qs = PatientModuleAttr.objects.filter((Q(patient_id=pat_id) & Q(id_hashed=False)) |
                                                              (Q(patient_id=hash_pat_id) & Q(id_hashed=True)))
            else:
                patient_qs = patient
            patient_updated = False
            for patient in patient_qs:
                if mapping.patient_name:
                    if patient.name_hashed:
                        patient.patient_name = hash_id(mapping.patient_name)
                    else:
                        patient.patient_name = mapping.patient_name
                if mapping.patient_birthdate:
                    patient_birthdate = datetime.strptime(mapping.patient_birthdate, "%Y%m%d").date()
                    if patient.patient_birth_date != patient_birthdate:
                        # update patient birth date, but also patient age at study
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
                patient_updated = True
            if patient_updated:
                logger.info(f'Patient information for patient {pat_id} updated.')
            else:
                logger.debug(f'patient {pat_id} not found in database.')
        else:
            logger.info('Patient update is not possible, no PatientID not found in HL7-message')
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
    logger.debug('Entering merge_patient')
    from remapp.models import PatientModuleAttr

    if HL7_PERFORM_PAT_MERGE:
        if mapping.patient_id and mapping.patient_mrg_id:
            old_id = mapping.patient_mrg_id
            hash_old_id = hash_id(old_id)
            logger.debug('Starting patient merge, old id: {0}, new id: {1}'.format(mapping.patient_mrg_id,
                                                                                   mapping.patient_id))

            patient_qs = PatientModuleAttr.objects.filter((Q(patient_id=old_id) & Q(id_hashed=False)) |
                                                          (Q(patient_id=hash_old_id) & Q(id_hashed=True)))

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
        logger.debug('Merge message received, but merging of patients is not configured.')
        return 0


def update_study(mapping):
    """update study information in database (if configured to do so in hl7settings.py)

    Possible updates: Study date, time, referring physician name, accession number, study id, physician of record,
                      performing physician, operator, modality, procedure code value/meaning, requested procedure
                      code value/meaning, patientstudy size(length) and weight.

    :param mapping: new study information based on hl7-message
    :return: return 0 if successful otherwise negative number
    """
    logger.debug('Entering update_study')
    from remapp.models import GeneralStudyModuleAttr, PatientStudyModuleAttr, \
        UniqueEquipmentNames, GeneralEquipmentModuleAttr, PatientModuleAttr, Hl7Message

    if HL7_PERFORM_STUDY_UPDATE:
        logger.debug(f'Getting accessionnumber from HL7 message {mapping.study_accession_number}')
        hash_acc_nr = hash_id(mapping.study_accession_number)
        if mapping.study_instance_uid:
            logger.info(f'Get study based on study_instance_uid {mapping.study_instance_uid} ')
            study_qs = GeneralStudyModuleAttr.objects.filter(study_instance_uid=mapping.study_instance_uid)
        elif mapping.study_accession_number and HL7_MATCH_STUDY_ON_ACCNR:
            logger.info(f'Get study based on accession_number {mapping.study_accession_number} ')
            study_qs = GeneralStudyModuleAttr.objects.filter((Q(accession_number=mapping.study_accession_number) &
                                                              Q(accession_hashed=False)) |
                                                             (Q(accession_number=hash_acc_nr) &
                                                              Q(accession_hashed=True)))
        else:
            study_qs = GeneralStudyModuleAttr.objects.none()
        nr_of_studies = study_qs.count()
        logger.debug(f'number of studies found: {nr_of_studies}')
        if nr_of_studies == 1:
            study = study_qs.get()
            logger.debug(f'Got study {study}.')

            study_datetime = datetime.combine(study.study_date, study.study_time)
            msg_datetime = datetime.strptime(mapping.message.MSH.MSH_7.to_er7()[:14], "%Y%m%d%H%M%S")

            add_only = True
            if (not HL7_ADD_STUDY_ONLY) and (msg_datetime > study_datetime):
                add_only = False
            try:
                patientstudy = PatientStudyModuleAttr.objects.filter(general_study_module_attributes=study).get()
            except:
                t, v, tb = sys.exc_info()
                logger.fatal(f'Unexpected fatal error while getting patient study from DB: {t}, {v}, {tb}')
                return -1

            logger.debug(f'got patient-study {patientstudy}')
            patient = PatientModuleAttr.objects.filter(general_study_module_attributes=study).get()
            logger.debug(f'got patient {patient}')
            if (study.study_date and not add_only) or not study.study_date:
                if mapping.study_date:
                    study_date = datetime.strptime(mapping.study_date, "%Y%m%d").date()
                    if study.study_date != study_date:
                        # update studydate and patient age at studydate
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
            logger.debug('11')
            if (study.modality_type and not add_only) or not study.modality_type:
                if mapping.study_modality:
                    # check if user_defined_modality is set, if this is the case, don't update
                    equipment_module = GeneralEquipmentModuleAttr.objects.get(general_study_module_attributes=study)
                    equipment_name = UniqueEquipmentNames.objects.get(id=equipment_module.unique_equipment_name_id)
                    if not equipment_name.user_defined_modality:
                        modality = get_openrem_modality(mapping.study_modality)
                        if modality:
                            study.modality_type = modality
            logger.debug('12')
            if (study.procedure_code_value and not add_only) or not study.procedure_code_value:
                study.procedure_code_value = mapping.study_procedure_code_value if \
                    mapping.study_procedure_code_value else study.procedure_code_value
            logger.debug('13')
            if (study.procedure_code_meaning and not add_only) or not study.procedure_code_meaning:
                study.procedure_code_meaning = mapping.study_procedure_code_meaning if \
                    mapping.study_procedure_code_meaning else study.procedure_code_meaning
            logger.debug('14')
            if (study.requested_procedure_code_value and not add_only) or not study.requested_procedure_code_value:
                study.requested_procedure_code_value = mapping.study_requested_procedure_code_value if \
                    mapping.study_requested_procedure_code_value else study.requested_procedure_code_value
            logger.debug('15')
            if (study.requested_procedure_code_meaning and not add_only) or not study.requested_procedure_code_meaning:
                study.requested_procedure_code_meaning = mapping.study_requested_procedure_code_meaning if \
                    mapping.study_requested_procedure_code_meaning else study.requested_procedure_code_meaning
            logger.debug('study information set, trying to save data')
            try:
                study.save()
            except Exception as e:
                t, v, tb = sys.exc_info()
                logger.fatal(f'Unexpected fatal error while saving study-data: {t}, {v}, {tb}')
                raise e

            logger.debug('Study data saved')

            if (patientstudy.patient_size and not add_only) or not patientstudy.patient_size:
                patientstudy.patient_size = mapping.study_patient_size if mapping.study_patient_size else \
                    patientstudy.patient_size
            if (patientstudy.patient_weight and not add_only) or not patientstudy.patient_weight:
                patientstudy.patient_weight = mapping.study_patient_weight if mapping.study_patient_weight else \
                    patientstudy.patient_weight
            logger.debug('Patient-study information set, trying to save')
            patientstudy.save()
            logger.debug('patient-study data saved')

            logger.info(
                f'Study information for studyUID / accession number ({mapping.study_instance_uid} / {mapping.study_accession_number}) updated.')

            if not HL7_UPDATE_PATIENT_ON_ADT_ONLY:
                logger.info('Update patient information on basis of order message.')
                return update_patient(mapping, [patient])

            logger.info('Just for debugging purposes, save the HL7-message')
            try:
                message = Hl7Message(receive_datetime=datetime.now(),
                                     patient_id=mapping.patient_id,
                                     accession_number=mapping.study_accession_number,
                                     study_instance_uid=mapping.study_instance_uid,
                                     message_type=mapping.message.name,
                                     study_date=mapping.study_date,
                                     message=mapping.message.to_er7())
                message.save()
            except:
                t, v, tb = sys.exc_info()
                logger.fatal(f'Unexpected fatal error while parsing HL7 message: {t}, {v}, {tb}')

        elif nr_of_studies == 0:
            # no studies found, if studydate is larger than or equal to today + offset, save data for later use
            logger.debug('study-date: {0}, keep studies from: {1}'.format(mapping.study_date,
                                                                          (datetime.today() - timedelta(
                                                                              days=HL7_STUDY_DATE_OFFSET)).strftime(
                                                                              '%Y%m%d')))
            logger.debug(f'orderstatus: {mapping.order_status}, keep orderstatus {HL7_KEEP_FROM_ORDER_STATUS}')

            if ((mapping.study_date >=
                 (datetime.today() - timedelta(days=HL7_STUDY_DATE_OFFSET)).strftime('%Y%m%d')) and \
                (HL7_ORDER_STATUSSES.index(mapping.order_status) >=
                 HL7_ORDER_STATUSSES.index(HL7_KEEP_FROM_ORDER_STATUS) if
                mapping.order_status in HL7_ORDER_STATUSSES else False)) or \
                    (('SIU' in mapping.message.name) and mapping.study_accession_number):
                try:
                    message = Hl7Message(receive_datetime=datetime.now(),
                                         patient_id=mapping.patient_id,
                                         accession_number=mapping.study_accession_number,
                                         study_instance_uid=mapping.study_instance_uid,
                                         message_type=mapping.message.name,
                                         study_date=mapping.study_date,
                                         message=mapping.message.to_er7())
                    message.save()
                    logger.info(
                        'Study update not possible. Study with studyUID / accession number ({0} / {1}) not found. Message saved for later usage.'.format(
                            mapping.study_instance_uid, mapping.study_accession_number))
                except:
                    logger.fatal('Unexpected fatal error while saving HL7 message in database: {0}, {1}'.format(
                        sys.exc_info()[:2]))
                    return -1
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
    # replace \n if exists by \r. hl7apy can't handle \r\n or \n only as end of line marks
    str_message = str_message.replace('\n', '\r')
    # try removing hl7 wrapping chars "0x0B" and "0x1C", shouldn't be there, but just to be sure
    str_message = str_message.replace(str(b'\x0b'), '').replace(str(b'\x1c'), '')
    logger.debug('replaced hl7 wrapping chars')
    try:
        m = parser.parse_message(str_message, find_groups=False)
        logger.debug('Parsed HL7-message')
        # for example library doesn't know about ADT_A40 (patient merge)
        if not m.name:
            # TODO: what kind of side-effects does this have?
            m.name = parser.get_message_info(str_message)[1]
    except HL7apyException as e:
        logger.fatal(f'Fatal error while parsing hl-7 message: {e}')
        return -1
    except:
        logger.fatal(f'Unexpected fatal error while parsing hl-7 message: {sys.exc_info()[0]}')
        return -1
    return 0, m


def save_not_processed(message):
    if HL7_SAVE_FAILED_MSG:
        try:
            filename = message.MSH.MSH_9.to_er7().replace('^', '_') + "_" + \
                       datetime.now().strftime('%Y%m%d%H%M%S') + '.hl7'
        except ChildNotFound:
            filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.hl7'
        try:
            with io.open(os.path.join(HL7_FAILED_MSG_PATH, filename), 'w',
                         encoding=HL7_MESSAGE_ENCODING) as filepointer:
                filepointer.write(message.to_er7())
        except:
            pass


def apply_hl7(message):
    """Apply hl7 message to database

    :param message: hl7 message structure
    :return: 0 if successful, otherwise negative number
    """
    logger.debug('start applying HL7 message')

    if HL7_SAVE_HL7_MESSAGE:
        logger.debug('Will save HL7 message')
        try:
            filename = message.MSH.MSH_9.to_er7().replace('^', '_')
            if message.MSH.MSH_7:
                filename = filename + "_" + message.MSH.MSH_7.to_er7() + "_" + datetime.now().strftime(
                    '%Y%m%d%H%M%S') + '.hl7'
            else:
                filename = filename + "_" + datetime.now().strftime('%Y%m%d%H%M%S') + '.hl7'
        except ChildNotFound:
            filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.hl7'

        logger.debug(f'Save HL7-message to {os.path.join(HL7_MESSAGE_LOCATION, filename)}')
        try:
            with io.open(os.path.join(HL7_MESSAGE_LOCATION, filename), 'w',
                         encoding=HL7_MESSAGE_ENCODING) as filepointer:
                filepointer.write(message.to_er7())
        except:
            logger.debug(f'Failed writing hl7-message to disk. filename:{os.path.join(HL7_MESSAGE_LOCATION, filename)}')
            logger.debug('exception: {0}'.format(sys.exc_info()[0]))
            logger.debug('exception-info: {0}'.format(sys.exc_info()[1]))
            logger.debug('exception-info: {0}'.format(sys.exc_info()[2]))

    if HL7_SAVE_HTML_REPORT:
        logger.debug('Starting to write hl7 message to html')
        save_html_report(message)
        logger.debug('Html report written')

    hl7_mapping = HL7Mapping(message)

    if 'ADT_A40' in message.name:
        logger.debug('Start processing ADT Merge message (A40)')
        try:
            result_id = merge_patient(hl7_mapping)
        except:
            result_id = -1

    elif 'ADT' in message.name:
        logger.debug('Start processing ADT message')
        try:
            result_id = update_patient(hl7_mapping)
        except:
            logger.error('Processing failed: {0}, {1}'.format(sys.exc_info()[0], sys.exc_info()[1]))
            result_id = -1
    elif ('ORM' in message.name) or ('ORU' in message.name) or ('OMI' in message.name) \
            or ('OMG' in message.name):
        logger.debug('Start processing order message')
        try:
            result_id = update_study(hl7_mapping)
        except:
            logger.error(f'processing failed: {sys.exc_info()[0]}, {sys.exc_info()[1]}')
            result_id = -1
    elif 'SIU' in message.name:
        logger.debug('Start processing appointment message')
        try:
            result_id = update_study(hl7_mapping)
        except:
            logger.error(f'processing failed: {sys.exc_info()[0]}, {sys.exc_info()[1]}')
            result_id = -1
    else:
        return -1, f'hl7-message of type {message.name} is not supported.'

    if result_id == 0:
        logger.info(f'processed {message.name} message successfully.')
    else:
        logger.error(f'processing {message.name} message failed, result: {result_id}.')
        save_not_processed(message)

    return result_id


def find_message_and_apply(patient_id, accession_number, study_instance_uid):
    from remapp.models import Hl7Message

    logger.info(
        'Applying saved HL7-messages to study-information for study with studyUID / accession number ({0} / {1})'.format(
            study_instance_uid, accession_number))
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


# for testing purposes
if __name__ == '__main__':
    from logging import config
    import django
    import codecs

    from remapp.models import PatientStudyModuleAttr, PatientModuleAttr, GeneralStudyModuleAttr, \
        UniqueEquipmentNames, GeneralEquipmentModuleAttr, Hl7Message

    if len(sys.argv) != 2:
        sys.exit('Error: Supply exactly one argument - the hl7 message file')

    hl7_message = codecs.open(sys.argv[1], 'r', encoding='utf-8').read()
    resultid, msg = parse_hl7(hl7_message)
    apply_hl7(msg)
    # if apply_hl7(msg) == 0:
    #     os.remove(sys.argv[1])
    # find_message_and_apply('1136321', '1118125.01257398', '1.2.840.114350.2.357.2.798268.2.125151199.1')
    # del_old_messages_from_db()
