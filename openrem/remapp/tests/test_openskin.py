# test_openskin.py

from decimal import Decimal
import gzip
import pickle
import os

from django.contrib.auth.models import User
from django.conf import settings
from django.test import TestCase

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
        current_skin_map_data = make_skin_map(study.pk, return_structure_for_testing=True)

        reference_skin_map_pickle_file = "test_files/RF-RDSR-Siemens-Zee.skin-map.p"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        path_reference_skin_map_pickle_file = os.path.join(root_tests, reference_skin_map_pickle_file)

        with gzip.open(path_reference_skin_map_pickle_file, "rb") as f:
            reference_skin_map_data = pickle.load(f)

            for key in reference_skin_map_data.keys():
                type_of_value = type(reference_skin_map_data[key])
                if type_of_value == "float":
                    self.assertAlmostEqual(current_skin_map_data[key], reference_skin_map_data[key])
                elif type_of_value == "list":
                    self.assertListEqual(current_skin_map_data[key], reference_skin_map_data[key])
                else:
                    self.assertEqual(current_skin_map_data[key], reference_skin_map_data[key])

    def test_vesion_not_matched(self):
        """Set software version to not match, ensure skin map is not created"""
        safe_list_entry = OpenSkinSafeList.objects.get()
        safe_list_entry.software_version = "No Match"
        safe_list_entry.save()

        study = GeneralStudyModuleAttr.objects.order_by("id")[0]
        current_skin_map_data  = make_skin_map(study.pk, return_structure_for_testing=True)

        self.assertEqual(current_skin_map_data["skin_map"], [0, 0])

    def test_version_match(self):
        """Set software version to match, ensure skin map is created"""
        safe_list_entry = OpenSkinSafeList.objects.get()
        safe_list_entry.software_version = "VC14J 150507"
        safe_list_entry.save()

        study = GeneralStudyModuleAttr.objects.order_by("id")[0]
        current_skin_map_data =  make_skin_map(study.pk, return_structure_for_testing=True)

        reference_skin_map_pickle_file = "test_files/RF-RDSR-Siemens-Zee.skin-map.p"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        path_reference_skin_map_pickle_file = os.path.join(root_tests, reference_skin_map_pickle_file)

        with gzip.open(path_reference_skin_map_pickle_file, "rb") as f:
            reference_skin_map_data = pickle.load(f)

            for key in reference_skin_map_data.keys():
                type_of_value = type(reference_skin_map_data[key])
                if type_of_value == "float":
                    self.assertAlmostEqual(current_skin_map_data[key], reference_skin_map_data[key])
                elif type_of_value == "list":
                    self.assertListEqual(current_skin_map_data[key], reference_skin_map_data[key])
                else:
                    self.assertEqual(current_skin_map_data[key], reference_skin_map_data[key])

    def test_system_not_matched(self):
        """Set manufacturer to not match, ensure skin map is not created"""
        safe_list_entry = OpenSkinSafeList.objects.get()
        safe_list_entry.software_version = ""
        safe_list_entry.manufacturer = "Not Siemens"
        safe_list_entry.save()

        study = GeneralStudyModuleAttr.objects.order_by("id")[0]
        current_skin_map_data  = make_skin_map(study.pk, return_structure_for_testing=True)

        self.assertEqual(current_skin_map_data["skin_map"], [0, 0])
