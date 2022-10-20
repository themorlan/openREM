# test_pt_size_import.py

import codecs
import csv
from decimal import Decimal
import os
import tempfile

from django.contrib.auth.models import User, Group
from django.core.files.base import ContentFile
from django.test import TestCase, override_settings

from ..extractors.dx import dx
from ..extractors.ptsizecsv2db import (
    _create_parser,
    websizeimport,
    _ptsizeinsert,
)
from ..extractors.rdsr import rdsr
from ..models import GeneralStudyModuleAttr, PatientIDSettings, SizeUpload


class CommandLineUse(TestCase):
    def test_args(self):
        parser = _create_parser()
        parsed_args = parser.parse_args(
            ["/home/user/sizefile.csv", "Acc no.", "height", "weight"]
        )

        self.assertEqual(parsed_args.csvfile, "/home/user/sizefile.csv")
        self.assertEqual(parsed_args.id, "Acc no.")
        self.assertEqual(parsed_args.height, "height")
        self.assertEqual(parsed_args.weight, "weight")
        self.assertFalse(parsed_args.verbose)
        self.assertFalse(parsed_args.si_uid)

    def test_options(self):
        parser = _create_parser()
        parsed_args = parser.parse_args(
            ["/home/user/sizefile.csv", "Acc no.", "-u", "height", "weight", "-v"]
        )

        self.assertTrue(parsed_args.verbose)
        self.assertTrue(parsed_args.si_uid)


class CSVUpdateHeightWeight(TestCase):
    def setUp(self):

        self.user = User.objects.create_user(
            username="jacob", email="jacob@…", password="top_secret"
        )
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
        height_and_weight = os.path.join(
            root_tests, "test_files", "CT-RDSR-Siemens_Flash-TAP-SS.dcm"
        )
        weight_only = os.path.join(
            root_tests, "test_files", "CT-RDSR-Siemens_Flash-QA-DS.dcm"
        )
        no_size = os.path.join(root_tests, "test_files", "DX-Im-GE_XR220-1.dcm")

        rdsr(height_and_weight)
        rdsr(weight_only)
        dx(no_size)

        studies = GeneralStudyModuleAttr.objects.order_by("id")
        self.assertAlmostEqual(
            studies[0].patientstudymoduleattr_set.get().patient_size, Decimal(1.86)
        )
        self.assertAlmostEqual(
            studies[0].patientstudymoduleattr_set.get().patient_weight, Decimal(87)
        )
        self.assertIsNone(studies[1].patientstudymoduleattr_set.get().patient_size)
        self.assertAlmostEqual(
            studies[1].patientstudymoduleattr_set.get().patient_weight, Decimal(75)
        )
        self.assertIsNone(studies[2].patientstudymoduleattr_set.get().patient_size)
        self.assertIsNone(studies[2].patientstudymoduleattr_set.get().patient_weight)

        size_upload = SizeUpload()
        log_file_name = "test_pt_size_import_log.txt"
        log_header_row = ContentFile(f"Patient size import for testing\r\n")
        size_upload.logfile.save(log_file_name, log_header_row)
        size_upload.save()
        log_file = size_upload.logfile
        log_file.file.close()
        size_dict_height_and_weight = {
            "acc_no": "ACC12345601",
            "height": "170",
            "weight": "90",
            "si_uid": False,
            "verbose": False,
        }
        size_dict_weight_only = {
            "acc_no": "74624646290",
            "height": "165",
            "weight": "80.6",
            "si_uid": False,
            "verbose": False,
        }
        size_dict_no_size = {
            "acc_no": "00938475",
            "height": "198",
            "weight": "102.3",
            "si_uid": False,
            "verbose": False,
        }
        _ptsizeinsert(size_upload=size_upload, size_dict=size_dict_height_and_weight)
        _ptsizeinsert(size_upload=size_upload, size_dict=size_dict_weight_only)
        _ptsizeinsert(size_upload=size_upload, size_dict=size_dict_no_size)

        studies = GeneralStudyModuleAttr.objects.order_by("id")
        self.assertAlmostEqual(
            studies[0].patientstudymoduleattr_set.get().patient_size, Decimal(1.86)
        )
        self.assertAlmostEqual(
            studies[0].patientstudymoduleattr_set.get().patient_weight, Decimal(87)
        )
        self.assertAlmostEqual(
            studies[1].patientstudymoduleattr_set.get().patient_size, Decimal(1.65)
        )
        self.assertAlmostEqual(
            studies[1].patientstudymoduleattr_set.get().patient_weight, Decimal(75)
        )
        self.assertAlmostEqual(
            studies[2].patientstudymoduleattr_set.get().patient_size, Decimal(1.98)
        )
        self.assertAlmostEqual(
            studies[2].patientstudymoduleattr_set.get().patient_weight, Decimal(102.3)
        )

    @override_settings(MEDIA_ROOT=tempfile.gettempdir())
    def test_web_import(self):
        root_tests = os.path.dirname(os.path.abspath(__file__))
        height_and_weight = os.path.join(
            root_tests, "test_files", "CT-RDSR-Siemens_Flash-TAP-SS.dcm"
        )
        weight_only = os.path.join(
            root_tests, "test_files", "CT-RDSR-Siemens_Flash-QA-DS.dcm"
        )
        no_size = os.path.join(root_tests, "test_files", "DX-Im-GE_XR220-1.dcm")

        rdsr(height_and_weight)
        rdsr(weight_only)
        dx(no_size)

        record = SizeUpload.objects.create()
        record.sizefile.save("test_csv.csv", ContentFile(""))
        record.height_field = "height"
        record.weight_field = "weight"
        record.id_field = "Acc num"
        record.id_type = "acc-no"
        record.save()
        temp_csv = open(record.sizefile.path, "a", newline="", encoding="utf-8")
        writer = csv.writer(temp_csv, dialect="excel")
        header_row = [
            record.id_field,
            record.height_field,
            record.weight_field,
            "file",
            "Existing h",
            "existing w",
        ]
        writer.writerow(header_row)
        writer.writerow(
            [
                "ACC12345601",
                "170",
                "90",
                "CT-RDSR-Siemens_Flash-TAP-SS.dcm",
                "186",
                "87",
            ]
        )
        writer.writerow(
            ["74624646290", "165", "80.6", "CT-RDSR-Siemens_Flash-QA-DS.dcm", "-", "75"]
        )
        writer.writerow(["00938475", "198", "102.3", "DX-Im-GE_XR220-1.dcm", "-", "-"])
        temp_csv.close()

        studies = GeneralStudyModuleAttr.objects.order_by("id")
        self.assertAlmostEqual(
            studies[0].patientstudymoduleattr_set.get().patient_size, Decimal(1.86)
        )
        self.assertAlmostEqual(
            studies[0].patientstudymoduleattr_set.get().patient_weight, Decimal(87)
        )
        self.assertIsNone(studies[1].patientstudymoduleattr_set.get().patient_size)
        self.assertAlmostEqual(
            studies[1].patientstudymoduleattr_set.get().patient_weight, Decimal(75)
        )
        self.assertIsNone(studies[2].patientstudymoduleattr_set.get().patient_size)
        self.assertIsNone(studies[2].patientstudymoduleattr_set.get().patient_weight)

        csv_pk = SizeUpload.objects.last().pk
        websizeimport(csv_pk=csv_pk)

        studies = GeneralStudyModuleAttr.objects.order_by("id")
        self.assertAlmostEqual(
            studies[0].patientstudymoduleattr_set.get().patient_size, Decimal(1.86)
        )
        self.assertAlmostEqual(
            studies[0].patientstudymoduleattr_set.get().patient_weight, Decimal(87)
        )
        self.assertAlmostEqual(
            studies[1].patientstudymoduleattr_set.get().patient_size, Decimal(1.65)
        )
        self.assertAlmostEqual(
            studies[1].patientstudymoduleattr_set.get().patient_weight, Decimal(75)
        )
        self.assertAlmostEqual(
            studies[2].patientstudymoduleattr_set.get().patient_size, Decimal(1.98)
        )
        self.assertAlmostEqual(
            studies[2].patientstudymoduleattr_set.get().patient_weight, Decimal(102.3)
        )
