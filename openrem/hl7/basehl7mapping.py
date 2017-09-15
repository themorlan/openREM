import logging
from hl7apy.exceptions import ChildNotFound

logger = logging.getLogger(__name__)


class BaseHL7Mapping:
    """
    Default Hl7 to DICOM mapping based on IHE Radiology Technical Framework Volume 2 IHE RAD TF-2 Transactions
    Some values are explicitly not taken at ADT level (i.e. patient weight / height) as these could change
    study information with current values that were not applicable at the time of the study.
    """

    def __init__(self, msg):
        """
        initialization of BaseHL7Mapping

        :param msg: The HL7- message that should be mapped
        """
        self._msg = msg

    # Static methods
    @staticmethod
    def _date_part_only(datetime):
        """
        Return only the date part of the hl7 datetime argument

        :param datetime: hl7 datetime value
        :return: date part of hl7 datetime value
        """
        return datetime[:8]

    @staticmethod
    def _time_part_only(datetime):
        """
        Return only the time part of the hl7 datetime argument

        :param datetime: hl7 datetime value
        :return: time part of hl7 datetime value
        """
        return datetime[8:]

    def _get_element_value(self, element_path):
        """
        Get the value of the element given in hl7 path

        :param element_path: path of the element in the hl7 message which value should be returned
        :return: er7 value of the hl7_path given, if not found return empty string
        """
        from hl7apy.exceptions import ChildNotFound
        if len(element_path) < 3:
            return ''

        hl7_path_list = element_path.split('.')

        msg_element = self._msg
        for path in hl7_path_list:
            try:
                if '[' in path:
                    path_part = path[:path.index('[')]
                    index_part = int(path[path.index('[')+1:path.index(']')])
                    msg_element = msg_element.children.get(path_part)[index_part]
                else:
                    msg_element = msg_element.children.get(path)
            except ChildNotFound:
                logger.debug('part ({0}) of path {1} not found.'.format(path, element_path))
                return ''
        if msg_element.to_er7():
            return msg_element.to_er7()
        else:
            return ''

    @property
    def message(self):
        """
        return the message

        :return: hl7-message
        """
        return self._msg

    # Default mapping properties for patient information (methods)
    # refer to PatientModuleAttr in remapp\models.py
    @property
    def patient_name(self):
        """
        Return patient name

        Mapping of patient name according Note 10, Table B-1, IHE RAD TF-2 Transactions

        :return: patient name
        """

        # Family name complex^given name complex^middle name^name prefix^name suffix
        return (self._get_element_value('PID.PID_5.XPN_1') + '^'
                + self._get_element_value('PID.PID_5.XPN_2') + '^'
                + self._get_element_value('PID.PID_5.XPN_3') + '^'
                + (self._get_element_value('PID.PID_5.XPN_4') + ' '
                   + self._get_element_value('PID.PID_5.XPN_6')).strip() + '^'
                + self._get_element_value('PID.PID_5.XPN_5')).strip('^')

    @property
    def patient_id(self):
        """
        Return patient ID

        :return: patient ID
        """
        return self._date_part_only(self._get_element_value('PID.PID_3.CX_1'))

    @property
    def patient_birthdate(self):
        """
        Return patient birth date

        :return: patient birth date
        """
        return self._date_part_only(self._get_element_value('PID.PID_7.TS_1'))

    @property
    def patient_sex(self):
        """
        Return patient sex

        Mapping of patient sex according Note 11, Table B-1, IHE RAD TF-2 Transactions

        :return: patient sex
        """
        result = self._get_element_value('PID.PID_8').upper()
        if result == 'M' or result == 'F' or result is '':
            return result
        elif result == 'U':
            return ''
        else:
            return 'O'

    @property
    def patient_other_ids(self):
        """
        Return other patient ids

        Other patient ids mapping is not defined by IHE RAD TF-2
        Specific hospital implementation should be implemented in child class

        :return: empty string
        """
        return ''

    @property
    def patient_mrg_id(self):
        """
        Return merge id

        Return patient id that should be merged to the patient id in PID segment

        :return: patient id that should be merged
        """
        return self._get_element_value('MRG.MRG_1.CX_1')

    # Default mapping properties for study information (methods)
    # refer to PatientStudyModuleAttr in remapp\models
    # update of patient age should be triggered on change of birth-date or study-date
    @property
    def study_patient_weight(self):
        """
        Return patient weight

        Return patient weight only in OBX-segments to be sure it is linked to a study (order)

        :return: patient weight
        """
        try:
            result = [obx_segment for obx_segment in self._msg.OBX.list
                      if obx_segment.OBX_3.CE_2.to_er7() == 'BODY WEIGHT']
            if result:
                result = result[0].OBX_5
        except ChildNotFound:
            result = ''
        if result and result.to_er7():
            return result.to_er7()
        else:
            return ''

    @property
    def study_patient_size(self):
        """
        Return patient size (height)

        Return patient size (height) only in OBX-segments to be sure it is linked to a study (order)

        :return: patient size
        """
        try:
            result = [obx_segment for obx_segment in self._msg.OBX.list
                      if obx_segment.OBX_3.CE_1.to_er7() == 'BODY HEIGHT']
            if result:
                result = result[0].OBX_5
        except ChildNotFound:
            result = ''
        if result and result.to_er7():
            return result.to_er7()
        else:
            return ''

    # Default mapping properties for study information (methods)
    # Refer to GeneralStudyModuleAttr in remapp\models
    @property
    def study_instance_uid(self):
        """
        Return study instance UID

        :return: Study instance UID
        """
        result = self._get_element_value('IPC.IPC_3')
        if not result:
            result = self._get_element_value('ZDS.ZDS_1.ST.ST[0]') if 'APPLICATION^DICOM' \
                                                                     in self._get_element_value('ZDS.ZDS_1') else None
        return result

    @property
    def study_date(self):
        """
        Return study date

        :return: Study date
        """
        result = self._get_element_value('TQ1.TQ1_7')
        if not result:
            result = self._get_element_value('ORC.ORC_7.TQ_4')
        if not result:
            result = self._get_element_value('OBR.OBR_27.TQ_4')
        return self._date_part_only(result)

    @property
    def study_time(self):
        """
        Return study time

        :return: Study time
        """
        result = self._get_element_value('TQ1.TQ1_7')
        if not result:
            result = self._get_element_value('ORC.ORC_7.TQ_4')
        if not result:
            result = self._get_element_value('OBR.OBR_27.TQ_4')
        return self._time_part_only(result)

    @property
    def study_referring_physician_name(self):
        """
        Return referring physician name

        :return: Referring physician name
        """
        if ('ORM' in self._msg.name) or ('OMI' in self._msg.name) or ('OMG' in self._msg.name):
            # Family name complex^given name complex^middle name^name prefix^name suffix
            return (self._get_element_value('PV1.PV1_8.XCN_2') + '^'
                    + self._get_element_value('PV1.PV1_8.XCN_3') + '^'
                    + self._get_element_value('PV1.PV1_8.XCN_4') + '^'
                    + (self._get_element_value('PV1.PV1_8.XCN_5') + ' '
                       + self._get_element_value('PV1.PV1_8.XCN_7')).strip() + '^'
                    + self._get_element_value('PV1.PV1_8.XCN_6')).strip('^')
        else:
            return ''

    @property
    def study_referring_physician_id(self):
        """
        Return referring physician ID

        :return: Referring physician ID
        """
        if ('ORM' in self._msg.name) or ('OMI' in self._msg.name) or ('OMG' in self._msg.name):
            return self._get_element_value('PV1.PV1_8.XCN_1')
        else:
            return ''

    @property
    def study_id(self):
        """
        Return study ID

        Mapping Hl7 to DICOM for study ID is not defined in IHE RAD TF-2 Transactions, but the value of
        Requested Procedure ID is recommended.

        :return: Study ID
        """
        result = self._get_element_value('IPC.IPC_2')
        if not result:
            result = self._get_element_value('OBR.OBR_19')
        return result

    @property
    def study_accession_number(self):
        """
        Return accession number

        :return: Accession number
        """
        result = self._get_element_value('IPC.IPC_1')
        if not result:
            result = self._get_element_value('OBR.OBR_18')
        return result

    @property
    def study_description(self):
        """
        Return study description

        Mapping Hl7 to DICOM for study description is not defined in IHE RAD TF-2 Transactions, but the value of
        (Requested) Procedure Description seems to fulfill the need. Is also suggested that the same values for
        (Performed) Procedure step description and study description are used.

        :return: Study description
        """
        return self.study_requested_procedure_code_meaning

    @property
    def study_physician_of_record(self):
        """
        Return physician of record

        Mapping Hl7 to DICOM for physician of record is not defined in IHE RAD TF-2 Transactions.
        Might be best to map 'attending doctor' to physician of record.

        :return: empty string
        """
        return ''

    @property
    def study_name_of_physician_reading_study(self):
        """
        Return name of physician reading study

        Mapping Hl7 to DICOM for name of physician reading study is not defined in IHE RAD TF-2 Transactions.
        Due to the fact ORU is not defined for reporting in IHE.
        Principal Result Interpreter seems to be the most logical HL7-field.

        :return: Name of physician reading study (Principal Result Interpreter)
        """
        if 'ORU' in self._msg.name:
            # Family name complex^given name complex^middle name^name prefix^name suffix
            return (self._get_element_value('OBR.OBR_32.NDL_2') + '^'
                    + self._get_element_value('OBR.OBR_32.NDL_3') + '^'
                    + self._get_element_value('OBR.OBR_32.NDL_4') + '^'
                    + (self._get_element_value('OBR.OBR_32.NDL_5') + ' '
                       + self._get_element_value('OBR.OBR_32.NDL_7')).strip() + '^'
                    + self._get_element_value('OBR.OBR_32.NDL_6')).strip('^')
        else:
            return ''

    @property
    def study_performing_physician(self):
        """
        Return performing physician

        OBR-34 is used per IHE RAD TF-2 Transactions. In practise OBR-34 might be filled with Technician (as described
        by the HL7 standard).

        :return: Performing physician
        """
        if ('ORM' in self._msg.name) or ('OMI' in self._msg.name) or ('OMG' in self._msg.name):
            # Family name complex^given name complex^middle name^name prefix^name suffix
            # Very likely that this HL7 field is used for radiographer / technician
            return (self._get_element_value('OBR.OBR_34.NDL_2') + '^'
                    + self._get_element_value('OBR.OBR_34.NDL_3')
                    + self._get_element_value('OBR.OBR_34.NDL_4') + '^'
                    + (self._get_element_value('OBR.OBR_34.NDL_5') + ' '
                       + self._get_element_value('OBR.OBR_34.NDL_7')).strip() + '^'
                    + self._get_element_value('OBR.OBR_34.NDL_6')).strip('^')
        else:
            return ''

    @property
    def study_operator(self):
        """
        Return Operator

        Mapping from HL7 to DICOM for Operator is not defined in IHE RAD TF-2 Transactions.
        In practise OBR-34 might be filled with Technician (as described by the HL7 standard: Technician).

        :return: Operator
        """
        return ''

    @property
    def study_modality(self):
        """
        Return modality

        :return: modality
        """
        result = self._get_element_value('IPC.IPC_16')
        if not result:
            result = self._get_element_value('OBR.OBR_24')
        return result

    @property
    def study_procedure_code_value(self):
        """
        Return Procedure code value

        Mapping Hl7 to DICOM for Procedure code is not defined in IHE RAD TF-2 Transactions.
        It might be valid to map OBR-44 in the ORU to the procedure code (implemented at the moment)

        :return: Procedure Code value
        """
        if 'ORU' in self._msg.name:
            return self._get_element_value('OBR.OBR_44.CE_1')

    @property
    def study_procedure_code_meaning(self):
        """
        Return Procedure code value

        Mapping Hl7 to DICOM for Procedure code is not defined in IHE RAD TF-2 Transactions.
        It might be valid to map OBR-44 in the ORU to the procedure code (implemented at the moment)

        :return: Procedure Code value
        """
        if 'ORU' in self._msg.name:
            return self._get_element_value('OBR.OBR_44.CE_2')

    @property
    def study_requested_procedure_code_value(self):
        """
        Return requested procedure code value

        :return: Requested procedure Code value
        """
        if ('ORM' in self._msg.name) or ('OMI' in self._msg.name) or ('OMG' in self._msg.name):
            return self._get_element_value('OBR.OBR_44.CE_1')
        else:
            return ''

    @property
    def study_requested_procedure_code_meaning(self):
        """
        Return requested procedure code meaning

        :return: Requested procedure Code meaing
        """
        if ('ORM' in self._msg.name) or ('OMI' in self._msg.name) or ('OMG' in self._msg.name):
            return self._get_element_value('OBR.OBR_44.CE_2')
        else:
            return ''
