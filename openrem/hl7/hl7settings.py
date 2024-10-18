# Server settings
HL7_SERVER_HOSTNAME = u'localhost'  # hostname of local receiving server
HL7_SERVER_PORT = 1234  # port of local receiving server
HL7_RESPONSE_UNSUPPORTED_MESSAGE = u'AR'  # AA:Application Accept; AE:Application Error; AR:Application Reject
HL7_RESPONSE_ALWAYS_ACCEPT = True  # True: Always response Application Accept
HL7_DEFAULT_VERSION = u'2.4'  # Default hl7-version to use (if not found in MSH)
HL7_MESSAGE_ENCODING = 'latin_1'  # The Character encoding of the HL7 message (utf-8, latin_1, etc...)
HL7_KEEP_CONNECTION_ALIVE = True  # Most HL7 applications (like Cloverleaf) expect an open connection
HL7_STUDY_DATE_OFFSET = 15  # Number of days to keep hl7 order messages for applying on receive of RDSR
HL7_KEEP_FROM_ORDER_STATUS = 'IP'  # Keep hl7 order messages with orc-5 status as given or 'higher'
HL7_ACCEPT_SIU = True

# What operations should we perform1
HL7_PERFORM_PAT_MERGE = True  # perform patient merge operation on receive of ADT^A40
HL7_PERFORM_PAT_UPDATE = True  # perform update of patient information on receive of ADT-messages
HL7_PERFORM_STUDY_UPDATE = True  # perform update of study information on receive of order-messages
HL7_UPDATE_PATIENT_ON_ADT_ONLY = False  # False: update also on ORM/OMI/ORU messages; True: only update on ADT
HL7_UPDATE_PATINFO_ON_MERGE = True  # True: update patientname / birthdate on a patient merge.
# True: if studyIUID is not available, match on accessionnumber; False: only match on studyIUID.
HL7_MATCH_STUDY_ON_ACCNR = True
HL7_ADD_STUDY_ONLY = False  # True: only add information, False: Also update information if newer than study-date/time.
HL7_MODALITY_2_OPENREM_MODALITY = {
    'CT': 'CT',
    'DX': 'DX',
    'MG': 'MG',
    'XA': 'RF',
    'CR': 'DX',
    'PT': 'CT',
    'NM': 'CT',
}  # If we get rdsr information from a PET-CT or SPECT-CT, it must be from the CT part

# HL7 order statuses (ORC-5): please adapt to your local settings. Order is important
HL7_ORDER_STATUSES = [
    'CA',  # Order Canceled
    'SC',  # Scheduled
    'RP',  # Order has been replaced
    'HD',  # Order is on hold
    'PA',  # Patient Arrived (not an official status)
    'IP',  # In process
    'DC',  # Order discontinued
    'CM',  # Completed
    'A',   # Some but not all results available
]

HL7_SAVE_FAILED_MSG = True  # True: save failed messages to file, False: Don't save
HL7_FAILED_MSG_PATH = u'D:/OpenREM/PRD/HL7/notProcessed/'  # Path for messages that aren't processed correctly

# For testing
HL7_SAVE_HTML_REPORT = False  # True: prints result of hl7-reading as html-report.
HL7_HTML_FILENAME = u'D:/OpenREM/PRD/HL7/hl7report.html'  # Name of html report file.
HL7_SAVE_HL7_MESSAGE = False  # True: Save the hl7-message to disk
HL7_MESSAGE_LOCATION = u'D:/OpenREM/PRD/HL7'  # Location to save Hl7 messages.
