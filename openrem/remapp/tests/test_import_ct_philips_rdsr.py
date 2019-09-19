# This Python file uses the following encoding: utf-8
# test_test_import_ct_philips_rdsr.py

import os
from decimal import Decimal
from django.test import TestCase
from remapp.extractors import rdsr
from remapp.models import GeneralStudyModuleAttr, PatientIDSettings


class ImportCTRDSRPhilips(TestCase):
    def test_import_ct_rdsr_bigbore(self):
        """
        Imports known Philips BigBore 4DCT RDSR where Target Region ConceptNameCodeSequence exists, but no content.
        Test imports correctly after fix for ref #765
        """
        pid = PatientIDSettings.objects.create()
        pid.name_stored = True
        pid.name_hashed = False
        pid.id_stored = True
        pid.id_hashed = False
        pid.dob_stored = True
        pid.save()

        bigbore = "test_files/CT-RDSR-Philips_BigBore4DCT.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        bigbore_path = os.path.join(root_tests, bigbore)

        rdsr.rdsr(bigbore_path)
        studies = GeneralStudyModuleAttr.objects.order_by('id')

        # Test that one study has been imported
        self.assertEqual(studies.count(), 1)

        # Test we have some data at event level
        self.assertAlmostEqual(
            studies[0].ctradiationdose_set.get().ctirradiationeventdata_set.order_by('id')[0].mean_ctdivol,
            Decimal(23.7))
