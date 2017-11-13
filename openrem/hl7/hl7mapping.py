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
