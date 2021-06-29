# test_openskin.py

from decimal import Decimal
import gzip
import pickle
import os

from django.contrib.auth.models import User
from django.conf import settings
from django.test import TestCase

from .test_files.skin_map_zee import ZEE_SKIN_MAP
from ..extractors import rdsr
from ..models import PatientIDSettings, GeneralStudyModuleAttr, OpenSkinSafeList
from ..tools.make_skin_map import make_skin_map


class OpenSkinBlackBox(TestCase):
    """Test openSkin as a black box - known study in, known skin map file out"""

    # Load safelist fixture to allow RDSR to have skin map calculated
    fixtures = ["openskin_safelist.json"]

    def setUp(self):
        """
        Load in all the rf objects
        """
        PatientIDSettings.objects.create()
        User.objects.create_user("temporary", "temporary@gmail.com", "temporary")

        rf1 = "test_files/RF-RDSR-Siemens-Zee.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        path_rf1 = os.path.join(root_tests, rf1)

        rdsr.rdsr(path_rf1)

    def test_skin_map_zee(self):
        """Test known Siemens Zee RDSR"""
        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        make_skin_map(study.pk)
        study_date = study.study_date
        skin_map_path = os.path.join(
            settings.MEDIA_ROOT,
            "skin_maps",
            "{0:0>4}".format(study_date.year),
            "{0:0>2}".format(study_date.month),
            "{0:0>2}".format(study_date.day),
            "skin_map_" + str(study.pk) + ".p",
        )
        with gzip.open(skin_map_path, "rb") as f:
            existing_skin_map_data = pickle.load(f)

            self.assertAlmostEqual(existing_skin_map_data["width"], 90)
            self.assertAlmostEqual(existing_skin_map_data["height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_width"], 34)
            self.assertAlmostEqual(existing_skin_map_data["phantom_height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_depth"], 20)
            self.assertAlmostEqual(existing_skin_map_data["phantom_flat_dist"], 14)
            self.assertAlmostEqual(existing_skin_map_data["phantom_curved_dist"], 31)
            self.assertAlmostEqual(existing_skin_map_data["patient_height"], 178.6)
            self.assertAlmostEqual(existing_skin_map_data["patient_mass"], 73.2)
            self.assertEqual(existing_skin_map_data["patient_orientation"], "HFS")
            self.assertEqual(existing_skin_map_data["patient_height_source"], "assumed")
            self.assertEqual(existing_skin_map_data["patient_mass_source"], "assumed")
            self.assertEqual(
                existing_skin_map_data["patient_orientation_source"], "extracted"
            )
            self.assertEqual(existing_skin_map_data["skin_map_version"], "0.8")
            self.assertEqual(existing_skin_map_data["skin_map"], ZEE_SKIN_MAP)

        os.remove(skin_map_path)

    def test_version_not_matched(self):
        """Set software version to not match, ensure skin map is not created"""
        safe_list_entry = OpenSkinSafeList.objects.get()
        safe_list_entry.software_version = "No Match"
        safe_list_entry.save()

        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        make_skin_map(study.pk)
        study_date = study.study_date
        skin_map_path = os.path.join(
            settings.MEDIA_ROOT,
            "skin_maps",
            "{0:0>4}".format(study_date.year),
            "{0:0>2}".format(study_date.month),
            "{0:0>2}".format(study_date.day),
            "skin_map_" + str(study.pk) + ".p",
        )
        with gzip.open(skin_map_path, "rb") as f:
            existing_skin_map_data = pickle.load(f)
            self.assertEqual(existing_skin_map_data["skin_map"], [0, 0])

        os.remove(skin_map_path)

    def test_version_match(self):
        """Set software version to match, ensure skin map is created"""
        safe_list_entry = OpenSkinSafeList.objects.get()
        safe_list_entry.software_version = "VC14J 150507"
        safe_list_entry.save()

        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        make_skin_map(study.pk)
        study_date = study.study_date
        skin_map_path = os.path.join(
            settings.MEDIA_ROOT,
            "skin_maps",
            "{0:0>4}".format(study_date.year),
            "{0:0>2}".format(study_date.month),
            "{0:0>2}".format(study_date.day),
            "skin_map_" + str(study.pk) + ".p",
        )
        with gzip.open(skin_map_path, "rb") as f:
            existing_skin_map_data = pickle.load(f)

            self.assertAlmostEqual(existing_skin_map_data["width"], 90)
            self.assertAlmostEqual(existing_skin_map_data["height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_width"], 34)
            self.assertAlmostEqual(existing_skin_map_data["phantom_height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_depth"], 20)
            self.assertAlmostEqual(existing_skin_map_data["phantom_flat_dist"], 14)
            self.assertAlmostEqual(existing_skin_map_data["phantom_curved_dist"], 31)
            self.assertAlmostEqual(existing_skin_map_data["patient_height"], 178.6)
            self.assertAlmostEqual(existing_skin_map_data["patient_mass"], 73.2)
            self.assertEqual(existing_skin_map_data["patient_orientation"], "HFS")
            self.assertEqual(existing_skin_map_data["patient_height_source"], "assumed")
            self.assertEqual(existing_skin_map_data["patient_mass_source"], "assumed")
            self.assertEqual(
                existing_skin_map_data["patient_orientation_source"], "extracted"
            )
            self.assertEqual(existing_skin_map_data["skin_map_version"], "0.8")
            self.assertEqual(existing_skin_map_data["skin_map"], ZEE_SKIN_MAP)

        os.remove(skin_map_path)

    def test_system_not_matched(self):
        """Set manufacturer to not match, ensure skin map is not created"""
        safe_list_entry = OpenSkinSafeList.objects.get()
        safe_list_entry.software_version = ""
        safe_list_entry.manufacturer = "Not Siemens"
        safe_list_entry.save()

        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        make_skin_map(study.pk)
        study_date = study.study_date
        skin_map_path = os.path.join(
            settings.MEDIA_ROOT,
            "skin_maps",
            "{0:0>4}".format(study_date.year),
            "{0:0>2}".format(study_date.month),
            "{0:0>2}".format(study_date.day),
            "skin_map_" + str(study.pk) + ".p",
        )
        with gzip.open(skin_map_path, "rb") as f:
            existing_skin_map_data = pickle.load(f)
            self.assertEqual(existing_skin_map_data["skin_map"], [0, 0])

        os.remove(skin_map_path)
