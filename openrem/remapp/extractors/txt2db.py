#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2012,2013  The Royal Marsden NHS Foundation Trust
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    Additional permission under section 7 of GPLv3:
#    You shall not make any use of the name of The Royal Marsden NHS
#    Foundation trust in connection with this Program in any press or
#    other public announcement without the prior written consent of
#    The Royal Marsden NHS Foundation Trust.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
..  module:: txt2db.
    :synopsis: Use to import data in a txt file into the database.

..  moduleauthor:: AC Chamberlain based on ptsizecsv.py Ed McDonagh

"""
import argparse
import csv
import datetime
from dateutil.parser import parse
from decimal import Decimal
import logging
import os
import sys
import uuid
from openrem.remapp.version import __version__, __docs_version__

import django
from django import db
from django.core.exceptions import ObjectDoesNotExist
from django.core.files.base import ContentFile
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from openrem.remapp.tools.background import (
    get_or_generate_task_uuid,
    record_task_error_exit,
)
from openrem.remapp.extractors.rdsr import _rdsr2db as rdsr2db
from ..tools.hash_id import hash_id

from remapp.models import (  # pylint: disable=wrong-import-order, wrong-import-position
    CtAccumulatedDoseData,
    CtIrradiationEventData,
    CtRadiationDose,
    CtXRaySourceParameters,
    DicomDeleteSettings,
    GeneralEquipmentModuleAttr,
    GeneralStudyModuleAttr,
    PatientIDSettings,
    PatientStudyModuleAttr,
    PatientModuleAttr,
    ScanningLength,
    UniqueEquipmentNames,
)

logger = logging.getLogger(__name__)


basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from remapp.models import (
    GeneralStudyModuleAttr,
    SizeUpload,
)  # pylint: disable=wrong-import-position


def dcmformatdate(datestr:str) -> str:
    # Convert an arbitrary date string into the DT 'yyyymmdd' DICOM format.
    thedate = parse(datestr, dayfirst=True)  # dayfirst would have to be an option somewhere
    return thedate.strftime('%Y%m%d')


def dcmformattime(timestr:str) -> str:
    # Convert time in 'HH:MM' into the TM 'HHMM' DICOM format.
    thetime = parse(timestr)
    return thetime.strftime('%H%M')


def createuid() -> str:
    hexuuid = uuid.uuid1()
    decuuid = str(int(hexuuid.hex, base=16))
    return '2.25.' + decuuid


def txt2dcm(header, csvline):
    # Produced by pydicom codify utility script

    # File meta info data elements
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 210
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.88.22'
    file_meta.MediaStorageSOPInstanceUID = '1.2.840.113619.2.278.3.2831171125.139.1641461471.292'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'
    file_meta.ImplementationClassUID = '1.2.826.0.1.3680043.2.60.0.1'
    file_meta.ImplementationVersionName = 'Text Import'
    file_meta.SourceApplicationEntityTitle = 'OpenREM'

    # Main data elements
    ds = Dataset()
    ds.add_new((0x0008, 0x0000), 'UL', 412)
    ds.SpecificCharacterSet = 'ISO_IR 100'
    ds.InstanceCreationDate = dcmformatdate(csvline[header.study_date])
    ds.InstanceCreationTime = dcmformattime(csvline[header.exposure_time])
    ds.InstanceCreatorUID = ''
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.88.22'
    ds.SOPInstanceUID = createuid()
    ds.StudyDate = dcmformatdate(csvline[header.study_date])
    ds.ContentDate = dcmformatdate(csvline[header.study_date])
    ds.StudyTime = dcmformattime(csvline[header.exposure_time])
    ds.ContentTime = dcmformattime(csvline[header.exposure_time])
    ds.AccessionNumber = csvline[header.patient_id]
    ds.Modality = 'SR'
    ds.Manufacturer = csvline[header.make]
    ds.InstitutionName = csvline[header.institution_name]
    ds.InstitutionAddress = ''
    ds.ReferringPhysicianName = ''
    ds.StationName = ''
    ds.StudyDescription = csvline[header.study_type]
    ds.SeriesDescription = 'Dose Record'
    ds.InstitutionalDepartmentName = ''
    ds.ManufacturerModelName = csvline[header.model]

    # Referenced Study Sequence
    refd_study_sequence = Sequence()
    ds.ReferencedStudySequence = refd_study_sequence

    # Referenced Performed Procedure Step Sequence
    refd_performed_procedure_step_sequence = Sequence()
    ds.ReferencedPerformedProcedureStepSequence = refd_performed_procedure_step_sequence

    ds.add_new((0x0010, 0x0000), 'UL', 138)
    ds.PatientName = csvline[header.patient_name]
    ds.PatientID = csvline[header.patient_id]
    ds.IssuerOfPatientID = ''
    ds.PatientBirthDate = ''
    ds.PatientBirthTime = ''
    ds.PatientSex = 'O'
    ds.OtherPatientIDs = ''
    ds.PatientAge = ''
    ds.PatientSize = None
    ds.PatientWeight = None
    ds.EthnicGroup = ''
    ds.Occupation = ''
    ds.AdditionalPatientHistory = ''
    ds.PatientComments = ''
    ds.add_new((0x0018, 0x0000), 'UL', 26)
    ds.DeviceSerialNumber = '*'
    ds.SoftwareVersions = 'OpenREM version ' + __version__
    ds.add_new((0x0020, 0x0000), 'UL', 160)
    ds.StudyInstanceUID = ds.SOPInstanceUID + '.1'
    ds.SeriesInstanceUID = ds.StudyInstanceUID + '.1'
    ds.StudyID = csvline[header.study_type]
    ds.SeriesNumber = '1'
    ds.InstanceNumber = '1'
    ds.add_new((0x0040, 0x0000), 'UL', 11988)

    # Request Attributes Sequence
    request_attributes_sequence = Sequence()
    ds.RequestAttributesSequence = request_attributes_sequence

    ds.ObservationDateTime = '20220107124131'
    ds.ValueType = 'CONTAINER'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    ds.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 50)
    concept_name_code1.CodeValue = '113701'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'X-Ray Radiation Dose Report'
    concept_name_code_sequence.append(concept_name_code1)

    ds.ContinuityOfContent = 'SEPARATE'

    # Performed Procedure Code Sequence
    performed_procedure_code_sequence = Sequence()
    ds.PerformedProcedureCodeSequence = performed_procedure_code_sequence

    ds.CompletionFlag = 'COMPLETE'
    ds.CompletionFlagDescription = ''
    ds.VerificationFlag = 'UNVERIFIED'

    # Content Template Sequence
    content_template_sequence = Sequence()
    ds.ContentTemplateSequence = content_template_sequence

    # Content Template Sequence: Content Template 1
    content_template1 = Dataset()
    content_template1.add_new((0x0008, 0x0000), 'UL', 0)
    content_template1.MappingResource = 'DCMR'
    content_template1.add_new((0x0040, 0x0000), 'UL', 2)
    content_template1.TemplateIdentifier = '10011'
    content_template_sequence.append(content_template1)

    # Content Sequence
    content_sequence = Sequence()
    # ds.ContentSequence = content_sequence

    # Content Sequence: Content 1
    content1 = Dataset()
    content1.add_new((0x0040, 0x0000), 'UL', 512)
    content1.RelationshipType = 'HAS CONCEPT MOD'
    content1.ValueType = 'CODE'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content1.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 40)
    concept_name_code1.CodeValue = '121058'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Procedure reported'
    concept_name_code_sequence.append(concept_name_code1)

    # Concept Code Sequence
    concept_code_sequence = Sequence()
    content1.ConceptCodeSequence = concept_code_sequence

    # Concept Code Sequence: Concept Code 1
    concept_code1 = Dataset()
    concept_code1.add_new((0x0008, 0x0000), 'UL', 50)
    concept_code1.CodeValue = 'P5-08000'
    concept_code1.CodingSchemeDesignator = 'SRT'
    concept_code1.CodeMeaning = 'Computed Tomography X-Ray'
    concept_code_sequence.append(concept_code1)

    # Content Sequence
    content_sequence1 = Sequence()
    content1.ContentSequence = content_sequence1

    # Content Sequence: Content 1-1
    content1_1 = Dataset()
    content1_1.add_new((0x0040, 0x0000), 'UL', 218)
    content1_1.RelationshipType = 'HAS CONCEPT MOD'
    content1_1.ValueType = 'CODE'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content1_1.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 32)
    concept_name_code1.CodeValue = 'G-C0E8'
    concept_name_code1.CodingSchemeDesignator = 'SRT'
    concept_name_code1.CodeMeaning = 'Has Intent'
    concept_name_code_sequence.append(concept_name_code1)

    # Concept Code Sequence
    concept_code_sequence = Sequence()
    content1_1.ConceptCodeSequence = concept_code_sequence

    # Concept Code Sequence: Concept Code 1
    concept_code1 = Dataset()
    concept_code1.add_new((0x0008, 0x0000), 'UL', 42)
    concept_code1.CodeValue = 'R-408C3'
    concept_code1.CodingSchemeDesignator = 'SRT'
    concept_code1.CodeMeaning = 'Diagnostic Intent'
    concept_code_sequence.append(concept_code1)
    content_sequence1.append(content1_1)
    content_sequence.append(content1)

    # Content Sequence: Content 2
    content2 = Dataset()
    content2.add_new((0x0040, 0x0000), 'UL', 208)
    content2.RelationshipType = 'HAS OBS CONTEXT'
    content2.ValueType = 'CODE'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content2.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 36)
    concept_name_code1.CodeValue = '121005'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Observer Type'
    concept_name_code_sequence.append(concept_name_code1)

    # Concept Code Sequence
    concept_code_sequence = Sequence()
    content2.ConceptCodeSequence = concept_code_sequence

    # Concept Code Sequence: Concept Code 1
    concept_code1 = Dataset()
    concept_code1.add_new((0x0008, 0x0000), 'UL', 28)
    concept_code1.CodeValue = '121007'
    concept_code1.CodingSchemeDesignator = 'DCM'
    concept_code1.CodeMeaning = 'Device'
    concept_code_sequence.append(concept_code1)
    content_sequence.append(content2)

    # Content Sequence: Content 3
    content3 = Dataset()
    content3.add_new((0x0040, 0x0000), 'UL', 156)
    content3.RelationshipType = 'HAS OBS CONTEXT'
    content3.ValueType = 'UIDREF'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content3.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 42)
    concept_name_code1.CodeValue = '121012'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Device Observer UID'
    concept_name_code_sequence.append(concept_name_code1)

    content3.UID = '1.2.840.113619.6.278'
    content_sequence.append(content3)

    # Content Sequence: Content 4
    content4 = Dataset()
    content4.add_new((0x0040, 0x0000), 'UL', 146)
    content4.RelationshipType = 'HAS OBS CONTEXT'
    content4.ValueType = 'TEXT'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content4.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 42)
    concept_name_code1.CodeValue = '121013'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Device Observer Name'
    concept_name_code_sequence.append(concept_name_code1)

    content4.TextValue = csvline[header.model]
    content_sequence.append(content4)

    # Content Sequence: Content 5
    content5 = Dataset()
    content5.add_new((0x0040, 0x0000), 'UL', 164)
    content5.RelationshipType = 'HAS OBS CONTEXT'
    content5.ValueType = 'TEXT'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content5.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 50)
    concept_name_code1.CodeValue = '121014'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Device Observer Manufacturer'
    concept_name_code_sequence.append(concept_name_code1)

    content5.TextValue = csvline[header.make]
    content_sequence.append(content5)

    # Content Sequence: Content 6
    content6 = Dataset()
    content6.add_new((0x0040, 0x0000), 'UL', 156)
    content6.RelationshipType = 'HAS OBS CONTEXT'
    content6.ValueType = 'TEXT'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content6.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 48)
    concept_name_code1.CodeValue = '121015'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Device Observer Model Name'
    concept_name_code_sequence.append(concept_name_code1)

    content6.TextValue = csvline[header.model]
    content_sequence.append(content6)

    # Content Sequence: Content 7
    content7 = Dataset()
    content7.add_new((0x0040, 0x0000), 'UL', 150)
    content7.RelationshipType = 'HAS OBS CONTEXT'
    content7.ValueType = 'TEXT'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content7.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 52)
    concept_name_code1.CodeValue = '121016'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Device Observer Serial Number'
    concept_name_code_sequence.append(concept_name_code1)

    content7.TextValue = '*'
    content_sequence.append(content7)

    # Content Sequence: Content 8
    content8 = Dataset()
    content8.add_new((0x0040, 0x0000), 'UL', 230)
    content8.RelationshipType = 'HAS OBS CONTEXT'
    content8.ValueType = 'CODE'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content8.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 46)
    concept_name_code1.CodeValue = '113876'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Device Role in Procedure'
    concept_name_code_sequence.append(concept_name_code1)

    # Concept Code Sequence
    concept_code_sequence = Sequence()
    content8.ConceptCodeSequence = concept_code_sequence

    # Concept Code Sequence: Concept Code 1
    concept_code1 = Dataset()
    concept_code1.add_new((0x0008, 0x0000), 'UL', 40)
    concept_code1.CodeValue = '113859'
    concept_code1.CodingSchemeDesignator = 'DCM'
    concept_code1.CodeMeaning = 'Irradiating Device'
    concept_code_sequence.append(concept_code1)
    content_sequence.append(content8)

    # Content Sequence: Content 9
    content9 = Dataset()
    content9.add_new((0x0040, 0x0000), 'UL', 222)
    content9.RelationshipType = 'HAS OBS CONTEXT'
    content9.ValueType = 'CODE'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content9.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 46)
    concept_name_code1.CodeValue = '113876'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Device Role in Procedure'
    concept_name_code_sequence.append(concept_name_code1)

    # Concept Code Sequence
    concept_code_sequence = Sequence()
    content9.ConceptCodeSequence = concept_code_sequence

    # Concept Code Sequence: Concept Code 1
    concept_code1 = Dataset()
    concept_code1.add_new((0x0008, 0x0000), 'UL', 32)
    concept_code1.CodeValue = '121097'
    concept_code1.CodingSchemeDesignator = 'DCM'
    concept_code1.CodeMeaning = 'Recording'
    concept_code_sequence.append(concept_code1)
    content_sequence.append(content9)

    # Content Sequence: Content 10
    content10 = Dataset()
    content10.add_new((0x0040, 0x0000), 'UL', 158)
    content10.RelationshipType = 'HAS OBS CONTEXT'
    content10.ValueType = 'DATETIME'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content10.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 48)
    concept_name_code1.CodeValue = '113809'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Start of X-Ray Irradiation'
    concept_name_code_sequence.append(concept_name_code1)

    content10.DateTime = ds.ContentDate + ds.ContentTime
    content_sequence.append(content10)

    # Content Sequence: Content 11
    content11 = Dataset()
    content11.add_new((0x0040, 0x0000), 'UL', 156)
    content11.RelationshipType = 'HAS OBS CONTEXT'
    content11.ValueType = 'DATETIME'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content11.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 46)
    concept_name_code1.CodeValue = '113810'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'End of X-Ray Irradiation'
    concept_name_code_sequence.append(concept_name_code1)

    content11.DateTime = ''
    content_sequence.append(content11)

    # Content Sequence: Content 12
    content12 = Dataset()
    content12.add_new((0x0040, 0x0000), 'UL', 460)
    content12.RelationshipType = 'HAS OBS CONTEXT'
    content12.ValueType = 'CODE'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content12.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 44)
    concept_name_code1.CodeValue = '113705'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Scope of Accumulation'
    concept_name_code_sequence.append(concept_name_code1)

    # Concept Code Sequence
    concept_code_sequence = Sequence()
    content12.ConceptCodeSequence = concept_code_sequence

    # Concept Code Sequence: Concept Code 1
    concept_code1 = Dataset()
    concept_code1.add_new((0x0008, 0x0000), 'UL', 28)
    concept_code1.CodeValue = '113014'
    concept_code1.CodingSchemeDesignator = 'DCM'
    concept_code1.CodeMeaning = 'Study'
    concept_code_sequence.append(concept_code1)

    # Content Sequence
    content_sequence1 = Sequence()
    content12.ContentSequence = content_sequence1

    # Content Sequence: Content 1
    content12_1 = Dataset()
    content12_1.add_new((0x0040, 0x0000), 'UL', 184)
    content12_1.RelationshipType = 'HAS PROPERTIES'
    content12_1.ValueType = 'UIDREF'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content12_1.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 40)
    concept_name_code1.CodeValue = '110180'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Study Instance UID'
    concept_name_code_sequence.append(concept_name_code1)

    content12_1.UID = ds.StudyInstanceUID
    content_sequence1.append(content12_1)
    content_sequence.append(content12)

    # Content Sequence: Content 13
    content13 = Dataset()
    content13.add_new((0x0040, 0x0000), 'UL', 800)
    content13.RelationshipType = 'CONTAINS'
    content13.ValueType = 'CONTAINER'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content13.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 46)
    concept_name_code1.CodeValue = '113811'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'CT Accumulated Dose Data'
    concept_name_code_sequence.append(concept_name_code1)

    content13.ContinuityOfContent = 'CONTINUOUS'

    # Content Sequence
    content_sequence1 = Sequence()
    content13.ContentSequence = content_sequence1

    # Content Sequence: Content 1
    content13_1 = Dataset()
    content13_1.add_new((0x0040, 0x0000), 'UL', 280)
    content13_1.RelationshipType = 'CONTAINS'
    content13_1.ValueType = 'NUM'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content13_1.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 56)
    concept_name_code1.CodeValue = '113812'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'Total Number of Irradiation Events'
    concept_name_code_sequence.append(concept_name_code1)

    # Measured Value Sequence
    measured_value_sequence = Sequence()
    content13_1.MeasuredValueSequence = measured_value_sequence

    # Measured Value Sequence: Measured Value 1
    measured_value1 = Dataset()
    measured_value1.add_new((0x0040, 0x0000), 'UL', 88)

    # Measurement Units Code Sequence
    measurement_units_code_sequence = Sequence()
    measured_value1.MeasurementUnitsCodeSequence = measurement_units_code_sequence

    # Measurement Units Code Sequence: Measurement Units Code 1
    measurement_units_code1 = Dataset()
    measurement_units_code1.add_new((0x0008, 0x0000), 'UL', 30)
    measurement_units_code1.CodeValue = '{events}'
    measurement_units_code1.CodingSchemeDesignator = 'UCUM'
    measurement_units_code1.CodeMeaning = 'events'
    measurement_units_code_sequence.append(measurement_units_code1)

    measured_value1.NumericValue = '1.0'
    measured_value_sequence.append(measured_value1)
    content_sequence1.append(content13_1)

    # Content Sequence: Content 2
    content13_2 = Dataset()
    content13_2.add_new((0x0040, 0x0000), 'UL', 274)
    content13_2.RelationshipType = 'CONTAINS'
    content13_2.ValueType = 'NUM'

    # Concept Name Code Sequence
    concept_name_code_sequence = Sequence()
    content13_2.ConceptNameCodeSequence = concept_name_code_sequence

    # Concept Name Code Sequence: Concept Name Code 1
    concept_name_code1 = Dataset()
    concept_name_code1.add_new((0x0008, 0x0000), 'UL', 50)
    concept_name_code1.CodeValue = '113813'
    concept_name_code1.CodingSchemeDesignator = 'DCM'
    concept_name_code1.CodeMeaning = 'CT Dose Length Product Total'
    concept_name_code_sequence.append(concept_name_code1)

    # Measured Value Sequence
    measured_value_sequence = Sequence()
    content13_2.MeasuredValueSequence = measured_value_sequence

    # Measured Value Sequence: Measured Value 1
    measured_value1 = Dataset()
    measured_value1.add_new((0x0040, 0x0000), 'UL', 88)

    # Measurement Units Code Sequence
    measurement_units_code_sequence = Sequence()
    measured_value1.MeasurementUnitsCodeSequence = measurement_units_code_sequence

    # Measurement Units Code Sequence: Measurement Units Code 1
    measurement_units_code1 = Dataset()
    measurement_units_code1.add_new((0x0008, 0x0000), 'UL', 28)
    measurement_units_code1.CodeValue = 'mGy.cm'
    measurement_units_code1.CodingSchemeDesignator = 'UCUM'
    measurement_units_code1.CodeMeaning = 'mGy.cm'
    measurement_units_code_sequence.append(measurement_units_code1)

    measured_value1.NumericValue = float(csvline[header.reading])
    measured_value_sequence.append(measured_value1)
    content_sequence1.append(content13_2)
    content_sequence.append(content13)

    ds.file_meta = file_meta
    ds.ContentSequence = content_sequence
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.save_as(r'rdsr_from_codify.dcm', write_like_original=False)
    return ds


def _general_equipment_module_attributes(header, csvline, study):
    equip = GeneralEquipmentModuleAttr.objects.create(general_study_module_attributes=study)
    equip.manufacturer = csvline[header.make]
    equip.institution_name = csvline[header.institution_name]
    equip.manufacturer_model_name = csvline[header.model]
    equip.save()


def _patient_module_attributes(header, csvline, study):
    pat = PatientModuleAttr.objects.create(general_study_module_attributes=study)
    patient_id_settings = PatientIDSettings.objects.get()
    if patient_id_settings.name_stored:
        name = csvline[header.patient_name]
        if name and patient_id_settings.name_hashed:
            name = hash_id(name)
            pat.name_hashed = True
        pat.patient_name = name
    if patient_id_settings.id_stored:
        patid = csvline[header.patient_id]
        if patid and patient_id_settings.id_hashed:
            patid = hash_id(patid)
            pat.id_hashed = True
        pat.patient_id = patid
    pat.save()


def extract_to_db(header, csvline):
    study = GeneralStudyModuleAttr.objects.create()
    study.study_date = dcmformatdate(csvline[header.study_date])
    study.study_time = dcmformattime(csvline[header.exposure_time])
    study.modality_type = "CT"  # for now
    study.save()
    _general_equipment_module_attributes(header, csvline, study)
    _patient_module_attributes(header, csvline, study)


def _create_parser():
    parser = argparse.ArgumentParser(
        description="Import patient and study data from a text file into an OpenREM database. "
    )
    parser.add_argument(
        "-u",
        "--si-uid",
        action="store_true",
        help="use Study Instance UID instead of Accession Number",
    )
    parser.add_argument(
        "-v", "--verbose", help="also print log to shell", action="store_true"
    )
    parser.add_argument(
        "-o", "--overwrite", help="overwrite existing values", action="store_true"
    )
    parser.add_argument(
        "-f", "--header-file", help="specify file container column headers", action="store_true"
    )
    parser.add_argument(
        "txtfile", help="text file with patient name, patient ID, institution name, equipment make, equipment model, study date, study type, procedure, DAP or DLP reading"
    )
    parser.add_argument(
        "patient_name", help="column title for the patient name"
    )
    parser.add_argument(
        "patient_id", help="column title for the patient ID"
    )
    parser.add_argument(
        "institution_name", help="column title for the institution name, values in cm"
    )
    parser.add_argument(
        "make", help="column title for the equipment make"
    )
    parser.add_argument(
        "model", help="column title for the equipment model"
    )
    parser.add_argument(
        "study_date", help="column title for the study date"
    )
    parser.add_argument(
        "study_type", help="column title for the study type"
    )
    parser.add_argument(
        "exposure_time", help="column title for the exposure time"
    )
    parser.add_argument(
        "procedure", help="column title for procedure type"
    )
    parser.add_argument(
        "reading", help="column title for the dose readings"
    )

    return parser


def txt2db():
    """Import patient dose data from text file. Called from ``openrem_txt.py`` script

    :param args: sys.argv from the command line call

    Example::

        openrem_txt.py -s MyRISExport.csv StudyInstanceUID height weight

    """

    args = _create_parser().parse_args()

    with open(args.txtfile) as txt_file:
        dataset = csv.DictReader(txt_file)
        fieldnames = dataset.fieldnames
        arg_headers = [args.patient_name, args.patient_id, args.institution_name, args.make, args.model,
                       args.study_date, args.study_type, args.procedure, args.reading]
        if not all(header in fieldnames for header in arg_headers):
            msg = f"Error: one or more of {arg_headers} not found in txt file"
            print(msg)
            record_task_error_exit(msg)
            return
        size_upload = SizeUpload()
        date_stamp = datetime.datetime.now()
        size_upload.import_date = date_stamp
        size_upload.task_id = uuid.uuid4()
        size_upload.progress = "Data import from shell started"
        size_upload.status = "CURRENT"
        size_upload.overwrite = args.overwrite
        size_upload.save()
        log_file_name = "txt_import_log_{0}.txt".format(
            date_stamp.strftime("%Y%m%d-%H%M%S%f")
        )
        log_header_row = ContentFile(f"Text data import from {args.txtfile}\r\n")
        size_upload.logfile.save(log_file_name, log_header_row)
        size_upload.save()
        log_file = size_upload.logfile
        log_file.file.close()
        size_upload.num_records = len(txt_file.readlines())
        size_upload.save()
        txt_file.seek(0)

        # build DICOM rsdr and give to rsdr.py to handle. More difficult but expandable.
        for line in dataset:
            # skip column header line
            if not all(header in fieldnames for header in list(line.values())):
                # rdsr2db(txt2dcm(args, line))
                extract_to_db(args, line)
    size_upload.processtime = (datetime.datetime.now() - date_stamp).total_seconds()
    size_upload.status = "COMPLETE"
    size_upload.save()
