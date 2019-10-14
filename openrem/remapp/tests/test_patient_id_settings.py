# test_get_values.py

import os
from django.test import TestCase
from remapp.extractors import rdsr
from remapp.models import GeneralStudyModuleAttr, PatientIDSettings


class ImportPatientIDSettings(TestCase):
    def test_import_with_ids(self):
        """
        Imports a known RDSR file derived from a Siemens Definition Flash and tests that patient IDs are stored when
        requested.
        """
        import datetime
        pid = PatientIDSettings.objects.create()
        pid.name_stored = True
        pid.name_hashed = False
        pid.id_stored = True
        pid.id_hashed = False
        pid.dob_stored = True
        pid.save()
        self.assertEqual(pid.name_stored, True)

        dicom_file = "test_files/CT-RDSR-Siemens_Flash-TAP-SS.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.all()[0]

        # Test that patient identifiable data is stored in plain text
        self.assertEqual(study.patientmoduleattr_set.get().patient_name, 'SMITH^JOHN')
        self.assertEqual(study.patientmoduleattr_set.get().patient_id, '123456')
        dob = datetime.date(1929, 5, 19)
        self.assertEqual(study.patientmoduleattr_set.get().patient_birth_date, dob)


    def test_import_with_ids_hashed(self):
        """
        Imports a known RDSR file derived from a Siemens Definition Flash and tests that patient IDs are stored when
        requested, in the form of a hash
        """
        import hashlib
        pid = PatientIDSettings.objects.create()
        pid.name_stored = True
        pid.name_hashed = True
        pid.id_stored = True
        pid.id_hashed = True
        pid.accession_hashed = True
        pid.save()
        self.assertEqual(pid.name_stored, True)

        dicom_file = "test_files/CT-RDSR-Siemens_Flash-TAP-SS.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.all()[0]

        # name = hashlib.sha256('SMITH^JOHN'.encode('utf-8')).hexdigest()
        name = '6339a06c6eee552c54459ad004204009343fa78f12d7e77e9a012cf9c077c047'
        # pat_id = hashlib.sha256('123456'.encode('utf-8')).hexdigest()
        pat_id = '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92'
        # acc = hashlib.sha256('ACC12345601'.encode('utf-8')).hexdigest()
        acc = '91410632c6b5d798ba71ba34540e9044912ea65b9f4bf143fa9b53f328916be4'

        # Test that patient identifiable data is stored in plain text
        self.assertEqual(study.patientmoduleattr_set.get().patient_name, name)
        self.assertEqual(study.patientmoduleattr_set.get().patient_id, pat_id)
        self.assertEqual(study.accession_number, acc)
