import logging
import sys
from hl7apy.exceptions import ChildNotFound
from .basehl7mapping import BaseHL7Mapping  # .
from .hl7settings import HL7_MESSAGE_ENCODING  # .

logger = logging.getLogger(__name__)


class HL7Mapping(BaseHL7Mapping):
    """
    HL7 mapping used to map data from HL7 messages to DICOM fields for a specific implementation
    Overrides mappings defined in BaseHL7Mapping
    Only differences from BaseHL7Mapping should be implemented here
    """
    pass