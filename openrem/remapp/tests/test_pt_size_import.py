# test_pt_size_import.py

from decimal import Decimal
import os

from django.contrib.auth.models import User, Group
from django.test import TestCase
from mock import patch

from ..extractors.dx import dx
from ..extractors.ptsizecsv2db import (csv2db, _create_parser, websizeimport, _ptsizeinsert,
                                       _patientstudymoduleattributes)
from ..extractors.rdsr import rdsr
from ..models import (GeneralStudyModuleAttr, PatientIDSettings)


class CommandLineUse(TestCase):
    def test_args(self):
        parser = _create_parser()
        parsed_args = parser.parse_args(['/home/user/sizefile.csv', 'Acc no.', 'height', 'weight'])

        self.assertEqual(parsed_args.csvfile, '/home/user/sizefile.csv')
        self.assertEqual(parsed_args.id, 'Acc no.')
        self.assertEqual(parsed_args.height, 'height')
        self.assertEqual(parsed_args.weight, 'weight')
        self.assertFalse(parsed_args.verbose)
        self.assertFalse(parsed_args.si_uid)

    def test_options(self):
        parser = _create_parser()
        parsed_args = parser.parse_args(['/home/user/sizefile.csv', 'Acc no.', '-u', 'height', 'weight', '-v'])

        self.assertTrue(parsed_args.verbose)
        self.assertTrue(parsed_args.si_uid)


class CSVUpdateHeightWeight(TestCase):
    def setUp(self):

        self.user = User.objects.create_user(
            username='jacob', email='jacob@…', password='top_secret')
        eg = Group(name="exportgroup")
        eg.save()
        eg.user_set.add(self.user)
        eg.save()

        pid = PatientIDSettings.objects.create()
        pid.name_stored = True
        pid.name_hashed = False
        pid.id_stored = True
        pid.id_hashed = False
        pid.dob_stored = True
        pid.save()

    def test_min_args(self):

        root_tests = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(root_tests, "test_files", "pt_size_import.csv")
        height_and_weight = os.path.join(root_tests, "test_files", 'CT-RDSR-Siemens_Flash-TAP-SS.dcm')
        weight_only = os.path.join(root_tests, "test_files", "CT-RDSR-Siemens_Flash-QA-DS.dcm")
        no_size = os.path.join(root_tests, "test_files", "DX-Im-GE_XR220-1.dcm")

        rdsr(height_and_weight)
        rdsr(weight_only)
        dx(no_size)

        studies = GeneralStudyModuleAttr.objects.order_by('id')
        self.assertAlmostEqual(studies[0].patientstudymoduleattr_set.get().patient_size, Decimal(1.86))
        self.assertAlmostEqual(studies[0].patientstudymoduleattr_set.get().patient_weight, Decimal(87))
        self.assertIsNone(studies[1].patientstudymoduleattr_set.get().patient_size)
        self.assertAlmostEqual(studies[1].patientstudymoduleattr_set.get().patient_weight, Decimal(75))
        self.assertIsNone(studies[2].patientstudymoduleattr_set.get().patient_size)
        self.assertIsNone(studies[2].patientstudymoduleattr_set.get().patient_weight)

        _ptsizeinsert('ACC12345601', '170',	'90', False, False)
        _ptsizeinsert('74624646290', '165', '80.6', False, False)
        _ptsizeinsert('00938475', '198', '102.3', False, False)

        studies = GeneralStudyModuleAttr.objects.order_by('id')
        self.assertAlmostEqual(studies[0].patientstudymoduleattr_set.get().patient_size, Decimal(1.86))
        self.assertAlmostEqual(studies[0].patientstudymoduleattr_set.get().patient_weight, Decimal(87))
        self.assertAlmostEqual(studies[1].patientstudymoduleattr_set.get().patient_size, Decimal(1.65))
        self.assertAlmostEqual(studies[1].patientstudymoduleattr_set.get().patient_weight, Decimal(75))
        self.assertAlmostEqual(studies[2].patientstudymoduleattr_set.get().patient_size, Decimal(1.98))
        self.assertAlmostEqual(studies[2].patientstudymoduleattr_set.get().patient_weight, Decimal(102.3))



