# test_openskin.py

import gzip
import os
import pickle

from django.conf import settings
from django.contrib.auth.models import User
from django.test import TestCase, tag

from .test_files.skin_map_alphenix import ALPHENIX_SKIN_MAP
from .test_files.skin_map_zee import ZEE_SKIN_MAP
from ..extractors import rdsr
from ..models import PatientIDSettings, GeneralStudyModuleAttr, OpenSkinSafeList
from ..tools.make_skin_map import make_skin_map


class OpenSkinBlackBox(TestCase):
    """Test openSkin as a black box - known study in, known skin map file out"""

    def setUp(self):
        """
        Load in all the rf objects
        """
        PatientIDSettings.objects.create()
        User.objects.create_user("temporary", "temporary@gmail.com", "temporary")

        root_tests = os.path.dirname(os.path.abspath(__file__))

        rf1 = "test_files/RF-RDSR-Siemens-Zee.dcm"
        path_rf1 = os.path.join(root_tests, rf1)
        rdsr.rdsr(path_rf1)

        rf2 = "test_files/RF-RDSR-Canon-Alphenix-rotational.dcm"
        path_rf2 = os.path.join(root_tests, rf2)
        rdsr.rdsr(path_rf2)

        # Create entries in the OpenSkinSafeList table
        pk1 = OpenSkinSafeList(
            id=1,
            manufacturer="Siemens",
            manufacturer_model_name="AXIOM-Artis",
            software_version="VC14J 150507",
        )
        pk1.save()

        pk2 = OpenSkinSafeList(
            id=2,
            manufacturer="Siemens",
            manufacturer_model_name="AXIOM-Artis",
            software_version="test 1",
        )
        pk2.save()

        pk3 = OpenSkinSafeList(
            id=3,
            manufacturer="Siemens",
            manufacturer_model_name="AXIOM-Artis",
            software_version="test 2",
        )
        pk3.save()

        pk4 = OpenSkinSafeList(
            id=4,
            manufacturer="Siemens",
            manufacturer_model_name="AXIOM-Artis",
            software_version="test 3",
        )
        pk4.save()

        pk5 = OpenSkinSafeList(
            id=6,
            manufacturer="CANON_MEC",
            manufacturer_model_name="DFP-8000D",
            software_version="",
        )
        pk5.save()

    def test_skin_map_zee(self):
        """Set software version to match, ensure skin map is created"""
        safe_list_entry = OpenSkinSafeList.objects.all().filter(id=1)[0]
        safe_list_entry.software_version = "VC14J 150507"
        safe_list_entry.save()

        """Test known Siemens Zee RDSR"""
        study = GeneralStudyModuleAttr.objects.filter(
            generalequipmentmoduleattr__manufacturer__exact="Siemens"
        )[0]

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

            self.assertEqual(existing_skin_map_data["width"], 90)
            self.assertEqual(existing_skin_map_data["height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_width"], 33.735, 3)
            self.assertEqual(existing_skin_map_data["phantom_height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_head_height"], 24.0)
            self.assertAlmostEqual(
                existing_skin_map_data["phantom_head_radius"], 9.231, 3
            )
            self.assertAlmostEqual(existing_skin_map_data["phantom_depth"], 19.735, 3)
            self.assertAlmostEqual(existing_skin_map_data["phantom_flat_dist"], 14.0)
            self.assertAlmostEqual(existing_skin_map_data["phantom_curved_dist"], 31.0)
            self.assertAlmostEqual(existing_skin_map_data["patient_height"], 178.6)
            self.assertAlmostEqual(existing_skin_map_data["patient_mass"], 73.2)
            self.assertEqual(existing_skin_map_data["patient_orientation"], "HFS")
            self.assertEqual(existing_skin_map_data["patient_height_source"], "assumed")
            self.assertEqual(existing_skin_map_data["patient_mass_source"], "assumed")
            self.assertEqual(
                existing_skin_map_data["patient_orientation_source"], "extracted"
            )
            self.assertAlmostEqual(existing_skin_map_data["fraction_DAP"], 1.0)
            self.assertEqual(existing_skin_map_data["skin_map_version"], "0.9.0")
            self.assertEqual(existing_skin_map_data["skin_map"], ZEE_SKIN_MAP)

        os.remove(skin_map_path)

    def test_version_not_matched(self):
        """Set software version to not match, ensure skin map is not created"""
        safe_list_entry = OpenSkinSafeList.objects.all().filter(id=1)[0]
        safe_list_entry.software_version = "No Match"
        safe_list_entry.save()

        study = GeneralStudyModuleAttr.objects.filter(
            generalequipmentmoduleattr__manufacturer__exact="Siemens"
        )[0]

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
        safe_list_entry = OpenSkinSafeList.objects.all().filter(id=1)[0]
        safe_list_entry.software_version = "VC14J 150507"
        safe_list_entry.save()

        study = GeneralStudyModuleAttr.objects.filter(
            generalequipmentmoduleattr__manufacturer__exact="Siemens"
        )[0]

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

            self.assertEqual(existing_skin_map_data["width"], 90)
            self.assertEqual(existing_skin_map_data["height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_width"], 33.735, 3)
            self.assertEqual(existing_skin_map_data["phantom_height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_head_height"], 24.0)
            self.assertAlmostEqual(
                existing_skin_map_data["phantom_head_radius"], 9.231, 3
            )
            self.assertAlmostEqual(existing_skin_map_data["phantom_depth"], 19.735, 3)
            self.assertAlmostEqual(existing_skin_map_data["phantom_flat_dist"], 14.0)
            self.assertAlmostEqual(existing_skin_map_data["phantom_curved_dist"], 31.0)
            self.assertAlmostEqual(existing_skin_map_data["patient_height"], 178.6)
            self.assertAlmostEqual(existing_skin_map_data["patient_mass"], 73.2)
            self.assertEqual(existing_skin_map_data["patient_orientation"], "HFS")
            self.assertEqual(existing_skin_map_data["patient_height_source"], "assumed")
            self.assertEqual(existing_skin_map_data["patient_mass_source"], "assumed")
            self.assertEqual(
                existing_skin_map_data["patient_orientation_source"], "extracted"
            )
            self.assertAlmostEqual(existing_skin_map_data["fraction_DAP"], 1.0)
            self.assertEqual(existing_skin_map_data["skin_map_version"], "0.9.0")
            self.assertEqual(existing_skin_map_data["skin_map"], ZEE_SKIN_MAP)
        os.remove(skin_map_path)

    def test_system_not_matched(self):
        """Set manufacturer to not match, ensure skin map is not created"""
        safe_list_entry = OpenSkinSafeList.objects.all().filter(id=1)[0]
        safe_list_entry.software_version = ""
        safe_list_entry.manufacturer = "Not Siemens"
        safe_list_entry.save()

        study = GeneralStudyModuleAttr.objects.filter(
            generalequipmentmoduleattr__manufacturer__exact="Siemens"
        )[0]

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

    @tag("slow")
    def test_rotational_exposure(self):
        study = GeneralStudyModuleAttr.objects.filter(
            generalequipmentmoduleattr__manufacturer__exact="CANON_MEC"
        )[0]

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

            self.assertEqual(existing_skin_map_data["width"], 94)
            self.assertEqual(existing_skin_map_data["height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_width"], 35.372, 3)
            self.assertEqual(existing_skin_map_data["phantom_height"], 70)
            self.assertAlmostEqual(existing_skin_map_data["phantom_head_height"], 24.0)
            self.assertAlmostEqual(
                existing_skin_map_data["phantom_head_radius"], 9.231, 3
            )
            self.assertAlmostEqual(existing_skin_map_data["phantom_depth"], 20.372, 3)
            self.assertAlmostEqual(existing_skin_map_data["phantom_flat_dist"], 15.0)
            self.assertAlmostEqual(existing_skin_map_data["phantom_curved_dist"], 32.0)
            self.assertAlmostEqual(existing_skin_map_data["patient_height"], 179.0)
            self.assertAlmostEqual(existing_skin_map_data["patient_mass"], 77.0)
            self.assertEqual(existing_skin_map_data["patient_orientation"], "HFS")
            self.assertEqual(
                existing_skin_map_data["patient_height_source"], "extracted"
            )
            self.assertEqual(existing_skin_map_data["patient_mass_source"], "extracted")
            self.assertEqual(
                existing_skin_map_data["patient_orientation_source"], "supine assumed"
            )
            self.assertAlmostEqual(existing_skin_map_data["fraction_DAP"], 0.9999, 4)
            self.assertEqual(existing_skin_map_data["skin_map_version"], "0.9.0")
            self.assertEqual(existing_skin_map_data["skin_map"], ALPHENIX_SKIN_MAP)

        os.remove(skin_map_path)
