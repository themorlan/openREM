# Note: This file should not be overridden with updates

import logging
from hl7apy.exceptions import ChildNotFound
from basehl7mapping import BaseHL7Mapping

logger = logging.getLogger(__name__)


class HL7Mapping(BaseHL7Mapping):
    """
    HL7 mapping used to map data from HL7 messages to DICOM fields for a specific implementation
    Overrides mappings defined in BaseHL7Mapping
    Only differences from BaseHL7Mapping should be implemented here
    """

    def __init__(self, msg):
        """
        This function is actually not necessary, but something should be overridden.
        So this function is always overriden

        :param msg: The HL7- message that should be mapped
        """
        BaseHL7Mapping.__init__(self, msg)

    @property
    def patient_name(self):
        """
        Return patient name

        Specific mapping of patient name for Radboud UMC (Nijmegen)

        :return: patient name
        """
        if self._get_element_value('PID.PID_5.XPN_1.FN_2'):
            family_name = self._get_element_value('PID.PID_5.XPN_1.FN_3') + ', ' \
                          + self._get_element_value('PID.PID_5.XPN_1.FN_2')
        else:
            family_name = self._get_element_value('PID.PID_5.XPN_1.FN_1')
        return (family_name + '^'
                + self._get_element_value('PID.PID_5.XPN_2') + '^'
                + self._get_element_value('PID.PID_5.XPN_3') + '^'
                + (self._get_element_value('PID.PID_5.XPN_4') + ' '
                   + self._get_element_value('PID.PID_5.XPN_5')).strip() + '^'
                + self._get_element_value('PID.PID_5.XPN_6')).strip('^')

    @property
    def study_operator(self):
        """
        Return Operator

        Specific mapping of Operator for Radboud UMC (Nijmegen)

        :return: Operator
        """
        if ('ORM' in self._msg.name) or ('ORU' in self._msg.name):
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
    def study_performing_physician(self):
        """
        Return performing physician

        Specific mapping of Operator for Radboud UMC (Nijmegen)

        :return: Performing physician
        """
        if ('ORM' in self._msg.name) or ('OMI' in self._msg.name) or ('OMG' in self._msg.name):
            # Family name complex^given name complex^middle name^name prefix^name suffix
            # Very likely that this HL7 field is used for radiographer / technician
            return (self._get_element_value('OBR.OBR_32.NDL_2') + '^'
                    + self._get_element_value('OBR.OBR_32.NDL_3')
                    + self._get_element_value('OBR.OBR_32.NDL_4') + '^'
                    + (self._get_element_value('OBR.OBR_32.NDL_5') + ' '
                       + self._get_element_value('OBR.OBR_32.NDL_7')).strip() + '^'
                    + self._get_element_value('OBR.OBR_32.NDL_6')).strip('^')
        else:
            return ''

    @property
    def study_patient_weight(self):
        """
        Return patient weight

        Specific mapping of Operator for Radboud UMC (Nijmegen)

        :return: patient weight
        """
        try:
            result = [obx_segment for obx_segment in self._msg.OBX.list
                      if obx_segment.OBX_3.CE_1.to_er7() == 'GEWICHT']
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
        Return patient length

        Specific mapping of Operator for Radboud UMC (Nijmegen)

        :return: patient length
        """
        try:
            result = [obx_segment for obx_segment in self._msg.OBX.list
                      if obx_segment.OBX_3.CE_1.to_er7() == 'LENGTE']
            if result:
                result = result[0].OBX_5
        except ChildNotFound:
            result = ''

        if result and result.to_er7():
            return result.to_er7()
        else:
            return ''

    @property
    def study_accession_number(self):
        """
        Return accession number

        :return: Accession number
        """
        result = self._get_element_value('IPC.IPC_1')
        if not result:
            result = self._get_element_value('OBR.OBR_19')
        return result
