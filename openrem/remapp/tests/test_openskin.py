# test_openskin.py

from decimal import Decimal
import gzip
import pickle
import os

from django.contrib.auth.models import User
from django.test import TestCase

from .test_files.skin_map_zee import (SKIN_MAP_HFS, SKIN_MAP_HFP, SKIN_MAP_FFS)
from ..extractors import rdsr
from ..models import PatientIDSettings, GeneralStudyModuleAttr
from ..tools.make_skin_map import make_skin_map
from openremproject.settings import MEDIA_ROOT


class OpenSkinBlackBox(TestCase):
    """Test openSkin as a black box - known study in, known skin map file out"""
    @classmethod
    def setUpTestData(cls):
        """
        Load in all the rf objects
        """
        PatientIDSettings.objects.create()
        User.objects.create_user('temporary', 'temporary@gmail.com', 'temporary')

        root_tests = os.path.dirname(os.path.abspath(__file__))
        hfs = os.path.join(root_tests, 'test_files', 'RF-RDSR-Siemens_Zee-HFS.dcm')
        hfp = os.path.join(root_tests, 'test_files', 'RF-RDSR-Siemens_Zee-HFP.dcm')
        ffs = os.path.join(root_tests, 'test_files', 'RF-RDSR-Siemens_Zee-FFS.dcm')

        rdsr.rdsr(hfs)
        rdsr.rdsr(hfp)
        rdsr.rdsr(ffs)

    def test_skin_map_hfs(self):
        """
        Test known Siemens Zee RDSR.
        Patient orientation Head First Supine. Events were:
        Head first - Supine
        Ev  Tube    Tube    Table H     Tble Z      Tble x  Fld sz
        1   0       0       -15         75          3       22      pt neck
        2   0       0       -29         75          3       22
        3   25      0       -15         75          3       22      tube is pt right
        4   25      0       -15         60          3       22      pt abdomen
        5   0       0       -15         60          3       22
        6   -45     0       -15         60          3       22      tube is pt left
        7   0       15      -15         60          3       22      tube is coming from feet
        8   0       0       -15         68          10      22      tube moved to pt left

        """
        study = GeneralStudyModuleAttr.objects.order_by('id')[0]
        make_skin_map(study.pk)
        study_date = study.study_date
        skin_map_path = os.path.join(MEDIA_ROOT, 'skin_maps', "{0:0>4}".format(study_date.year),
                                     "{0:0>2}".format(study_date.month), "{0:0>2}".format(study_date.day),
                                     'skin_map_' + str(study.pk) + '.p')
        with gzip.open(skin_map_path, 'rb') as f:
            existing_skin_map_data = pickle.load(f)

            self.assertAlmostEqual(existing_skin_map_data['width'], 90)
            self.assertAlmostEqual(existing_skin_map_data['height'], 70)
            self.assertAlmostEqual(existing_skin_map_data['phantom_width'], 34)
            self.assertAlmostEqual(existing_skin_map_data['phantom_height'], 70)
            self.assertAlmostEqual(existing_skin_map_data['phantom_depth'], 20)
            self.assertAlmostEqual(existing_skin_map_data['phantom_flat_dist'], 14)
            self.assertAlmostEqual(existing_skin_map_data['phantom_curved_dist'], 31)
            self.assertAlmostEqual(existing_skin_map_data['patient_height'], 178.6)
            self.assertAlmostEqual(existing_skin_map_data['patient_mass'], 73.2)
            self.assertEqual(existing_skin_map_data['patient_orientation'], 'HFS')
            self.assertEqual(existing_skin_map_data['patient_height_source'], 'assumed')
            self.assertEqual(existing_skin_map_data['patient_mass_source'], 'assumed')
            self.assertEqual(existing_skin_map_data['patient_orientation_source'], 'extracted')
            self.assertEqual(existing_skin_map_data['skin_map_version'], '0.7')
            self.assertEqual(existing_skin_map_data['skin_map'], SKIN_MAP_HFS)

        os.remove(skin_map_path)

    def test_skin_map_hfp(self):
        """
        Test known Siemens Zee RDSR.
        Patient orientation Head First Prone. Events were:

        Ev  Tube    Tube    Table H     Tble Z      Tble x  Fld sz
        1   -180    0       15          90          0       22      pt neck
        2   -180    0       15          70          0       22      pt abdomen
        3   -135    0       15          70          0       22      tube is pt left
        4   -135    0       15          70          10      22      tube moved to pt left

        """
        study = GeneralStudyModuleAttr.objects.order_by('id')[1]
        make_skin_map(study.pk)
        study_date = study.study_date
        skin_map_path = os.path.join(MEDIA_ROOT, 'skin_maps', "{0:0>4}".format(study_date.year),
                                     "{0:0>2}".format(study_date.month), "{0:0>2}".format(study_date.day),
                                     'skin_map_' + str(study.pk) + '.p')
        with gzip.open(skin_map_path, 'rb') as f:
            existing_skin_map_data = pickle.load(f)

            self.assertAlmostEqual(existing_skin_map_data['width'], 90)
            self.assertAlmostEqual(existing_skin_map_data['height'], 70)
            self.assertAlmostEqual(existing_skin_map_data['phantom_width'], 34)
            self.assertAlmostEqual(existing_skin_map_data['phantom_height'], 70)
            self.assertAlmostEqual(existing_skin_map_data['phantom_depth'], 20)
            self.assertAlmostEqual(existing_skin_map_data['phantom_flat_dist'], 14)
            self.assertAlmostEqual(existing_skin_map_data['phantom_curved_dist'], 31)
            self.assertAlmostEqual(existing_skin_map_data['patient_height'], 178.6)
            self.assertAlmostEqual(existing_skin_map_data['patient_mass'], 73.2)
            self.assertEqual(existing_skin_map_data['patient_orientation'], 'HFP')
            self.assertEqual(existing_skin_map_data['patient_height_source'], 'assumed')
            self.assertEqual(existing_skin_map_data['patient_mass_source'], 'assumed')
            self.assertEqual(existing_skin_map_data['patient_orientation_source'], 'extracted')
            self.assertEqual(existing_skin_map_data['skin_map_version'], '0.7')
            self.assertEqual(existing_skin_map_data['skin_map'], SKIN_MAP_HFP)

        os.remove(skin_map_path)

    def test_skin_map_ffs(self):
        """
        Test known Siemens Zee RDSR. Map is currently all zeros.
        Patient orientation Feet First Supine. Events were:

        Ev  Tube    Tube    Table H     Tble Z      Tble x  Fld sz
        1   0       0       -15         -75         0       22      pt neck
        2   0       0       -15         -90         0       22      pt abdomen
        3   -45     0       -15         -90         0       22      tube is pt left
        4   0       0       -15         -90         15      22      tube moved to pt left

        """
        study = GeneralStudyModuleAttr.objects.order_by('id')[2]
        make_skin_map(study.pk)
        study_date = study.study_date
        skin_map_path = os.path.join(MEDIA_ROOT, 'skin_maps', "{0:0>4}".format(study_date.year),
                                     "{0:0>2}".format(study_date.month), "{0:0>2}".format(study_date.day),
                                     'skin_map_' + str(study.pk) + '.p')
        with gzip.open(skin_map_path, 'rb') as f:
            existing_skin_map_data = pickle.load(f)

            self.assertAlmostEqual(existing_skin_map_data['width'], 90)
            self.assertAlmostEqual(existing_skin_map_data['height'], 70)
            self.assertAlmostEqual(existing_skin_map_data['phantom_width'], 34)
            self.assertAlmostEqual(existing_skin_map_data['phantom_height'], 70)
            self.assertAlmostEqual(existing_skin_map_data['phantom_depth'], 20)
            self.assertAlmostEqual(existing_skin_map_data['phantom_flat_dist'], 14)
            self.assertAlmostEqual(existing_skin_map_data['phantom_curved_dist'], 31)
            self.assertAlmostEqual(existing_skin_map_data['patient_height'], 178.6)
            self.assertAlmostEqual(existing_skin_map_data['patient_mass'], 73.2)
            self.assertEqual(existing_skin_map_data['patient_orientation'], 'FFS')
            self.assertEqual(existing_skin_map_data['patient_height_source'], 'assumed')
            self.assertEqual(existing_skin_map_data['patient_mass_source'], 'assumed')
            self.assertEqual(existing_skin_map_data['patient_orientation_source'], 'extracted')
            self.assertEqual(existing_skin_map_data['skin_map_version'], '0.7')
            self.assertEqual(existing_skin_map_data['skin_map'], SKIN_MAP_FFS)

        os.remove(skin_map_path)
