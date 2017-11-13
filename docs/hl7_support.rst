###########
HL7 support
###########

* HL7 makes it possible to keep information in the OpenREM database up to
  date and add extra information to the database.

* HL7 messages are almost always sent by the ZIS/EMR system and/or the RIS
  system.

* OpenREM can be configured to receive HL7-messages and add and/or update
  information in its database

* Only HL7 version 2 messages are supported. HL7 version 2.7 is not
  supported (due to the extra seperator defined in this version)

* Please read the hints before starting the mllp-server and applying the
  information of the messages to the database

*****************************************
Functionality that is currently available
*****************************************

* Update / add patient information:

  * Patient name
  * Patient date of birth
  * Patient sex
  * Patient other IDs
  * Patient merge

* Update / add study information:

  * Patient weight (is study based information)
  * Patient height (is study based information)
  * study date
  * study time
  * Accession number
  * study id
  * study description
  * referring physician
  * physician of record
  * reading physician
  * performing physician
  * operator (radiographer / technician)
  * modality
  * study procedure code

************
Installation
************

Prerequisites

* HL7Apy should be installed: git-hub fork and adaptations created by LuukOost

--> Are we gonna include this in OpenREM or what would be the best way of
distribution

Installation:

* Include it in OpenREM or make it an add-on package?

************************
Starting the MLLP-server
************************

The MLLP-server receives the HL7-messages and starts the handling of the
messages. If python is in your path, it can be started by:

python <openrempath>\\hl7\\mllpserver.py

*************
Configuration
*************

The main problem with HL7 messages is that there is a wide variety of what
information is in what fields. The file openrem\hl7\basehl7mapping.py contains
the base mapping of hl7 fields to database fields based on
`IHE Radiology Technical Framework Volume 2 IHE RAD TF-2 Transactions`_. 
To change the mapping copy the file hl7mapping.py.example to hl7mapping.py and
edit it to your needs.

In the file hl7settings.py a lot of configuration settings are available:

HL7_SERVER_HOSTNAME
  The hostname or IP-address of the system that runs OpenREM

HL7_SERVER_PORT
  The portnumber to receive HL7-messages

HL7_RESPONSE_UNSUPPORTED_MESSAGE
  The answer to the sender on HL7-messages that are not supported.
  AR: Application reject (reject message)
  AA: Application accept (just pretend accept)
  AE: Application error (raise an error)

HL7_RESPONSE_ALWAYS_ACCEPT:
  Always respons 'AA' if this parameter is set to True

HL7_DEFAULT_VERSION
  Default HL7-version to use (if not found in MSH-segment)

HL7_MESSAGE_ENCODING
  The Character encoding of the received HL7 message
  (e.g. utf-8, latin_1, etc...)

HL7_KEEP_CONNECTION_ALIVE
  If set to True, keep HL7-connection open. Most HL7 applications
  (like Cloverleaf) expect the connection to be kept open.

HL7_STUDY_DATE_OFFSET
  Number of days to keep hl7 order messages after
  studydate for applying on receive of RDSR

HL7_KEEP_FROM_ORDER_STATUS
  Keep hl7 order messages only if orc-5 order status is the given status
  or 'higher'

HL7_PERFORM_PAT_MERGE
  If set to True, perform merge operation on receive of ADT^A40

HL7_PERFORM_PAT_UPDATE 
  If set to True, perform update of patient information on receive of 
  ADT-messages

HL7_PERFORM_STUDY_UPDATE 
  If set to True, perform update of study information on receive of 
  order-messages

HL7_UPDATE_PATIENT_ON_ADT_ONLY 
  If set to True, only ADT messages will change patient information, 
  otherwise also study messages will change patient information

HL7_UPDATE_PATINFO_ON_MERGE
  If True, update patient information on patient merge message

HL7_MATCH_STUDY_ON_ACCNR
  If True, studies are matched on accession number. Otherwise (better) on 
  study instance UID

HL7_ADD_STUDY_ONLY 
  If True, only add study information, otherwise also update/change information

HL7_MODALITY_2_OPENREM_MODALITY
  dictionary for mapping modalities given in the HL7-message to modalities used 
  in OpenREM (the only modalities used in OpenREM are 'CT', 'DX', 'RF' and 
  'MG')

HL7_ORDER_STATUSSES 
  List of order statusses that are available.

HL7_SAVE_HTML_REPORT 
  If True, prints result of hl7-reading as html-report. For debugging / testing 
  purposes

HL7_HTML_FILENAME 
  Path of the html report file.

HL7_SAVE_HL7_MESSAGE
  If True, Save the received hl7-messages to disk. For debugging / testing 
  purposes

HL7_MESSAGE_LOCATION 
  Directory to save Hl7 messages

*****
Hints
*****

* First start with applying the HL7 messages to false, set the following 
  parameters to false:
  
  * HL7_PERFORM_PAT_MERGE
  * HL7_PERFORM_PAT_UPDATE
  * HL7_PERFORM_STUDY_UPDATE

* Set HL7_SAVE_HTML_REPORT to True and see if all information is correctly
  retrieved by inspecting the hl7 html report

* Repeat the above step for multiple patient, order and report messages.

* Adapt the hl7mapping.py if information is retrieved incorrectly (and restart
  mllpserver.py)

* If you are sure that a certain operation (merge, patient update,
  study update) is correct, you can set it to True. But better be safe and
  make a backup of your database

* Check after applying the first messages if the database is updated
  correctly

* A test environment is the best way to go.

.. _`IHE Radiology Technical Framework Volume 2 IHE RAD TF-2 Transactions`: http://www.ihe.net/uploadedFiles/Documents/Radiology/IHE_RAD_TF_Vol2.pdf