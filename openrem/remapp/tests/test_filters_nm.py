# This Python file uses the following encoding: utf-8
# test_filters_nm.py

import os
from django.contrib.auth.models import User, Group
from django.test import TestCase
from django.urls import reverse
from remapp.extractors import rdsr, nm_image
from remapp.models import PatientIDSettings, FilterLibrary
from .test_filters_data import get_simple_multiple_query

SHARE_BUTTON_TEXT = "Make available for all users"
REMOVE_SHARED_BUTTON_TEXT = "Remove from all users"

TEMPUSER_FILTER_1 = "tempuser_nm_1"
TEMPUSER_FILTER_2 = "tempuser_nm_2"
TEMPUSER_WRONG_MOD = "tempuser_wrong_mod"

SUPERUSER_FILTER = "superuser_nm"

SUPERUSER_SHARED_FILTER_1 = "superuser_shared_nm_1"
SUPERUSER_SHARED_FILTER_2 = "superuser_shared_nm_2"
SUPERUSER_SHARED_WRONG_MOD = "superuser_shared_wrong_mod"


class FilterViewTests(TestCase):
    """
    Class to test the filter views for fluoroscopy
    """

    def setUp(self):
        """
        Load in all the nm objects so that there is something to filter!
        And add filters to library
        """
        PatientIDSettings.objects.create()
        tempuser = User.objects.create_user(
            "temporary", "temporary@gmail.com", "temporary"
        )
        superuser = User.objects.create_superuser(
            "superuser", "super@fantastic.cloud", "superpassword"
        )

        admingroup = Group.objects.create(name="admingroup")
        superuser.groups.add(admingroup)

        nm1 = "test_files/NM-NmIm-Siemens.dcm"
        nm2 = "test_files/NM-PetIm-GE.dcm"
        nm3 = "test_files/NM-PetIm-Siemens.dcm"
        nm_rdsr = "test_files/NM-RRDSR-Siemens-Extended.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        path_nm1 = os.path.join(root_tests, nm1)
        path_nm2 = os.path.join(root_tests, nm2)
        path_nm3 = os.path.join(root_tests, nm3)
        path_nm_rdsr = os.path.join(root_tests, nm_rdsr)

        nm_image.nm_image(path_nm1)
        nm_image.nm_image(path_nm2)
        nm_image.nm_image(path_nm3)
        rdsr.rdsr(path_nm_rdsr)

        basic_pattern = get_simple_multiple_query({})

        FilterLibrary.objects.create(
            pattern=basic_pattern,
            name=TEMPUSER_FILTER_1,
            modality_type="NM",
            user=tempuser,
            shared=False,
        )
        FilterLibrary.objects.create(
            pattern=basic_pattern,
            name=TEMPUSER_FILTER_2,
            modality_type="NM",
            user=tempuser,
            shared=False,
        )
        FilterLibrary.objects.create(
            pattern=basic_pattern,
            name=TEMPUSER_WRONG_MOD,
            modality_type="RF",
            user=tempuser,
            shared=False,
        )

        FilterLibrary.objects.create(
            pattern=basic_pattern,
            name=SUPERUSER_FILTER,
            modality_type="NM",
            user=superuser,
            shared=False,
        )

        FilterLibrary.objects.create(
            pattern=basic_pattern,
            name=SUPERUSER_SHARED_FILTER_1,
            modality_type="NM",
            user=superuser,
            shared=True,
        )
        FilterLibrary.objects.create(
            pattern=basic_pattern,
            name=SUPERUSER_SHARED_FILTER_2,
            modality_type="NM",
            user=superuser,
            shared=True,
        )
        FilterLibrary.objects.create(
            pattern=basic_pattern,
            name=SUPERUSER_SHARED_WRONG_MOD,
            modality_type="RF",
            user=superuser,
            shared=True,
        )

    def test_list_all_nm(self):
        """
        Initial test to ensure four (4) studies are listed with no filter
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(reverse("nm_summary_list_filter"), follow=True)
        self.assertEqual(response.status_code, 200)
        responses_text = "There are 4 studies in this list."
        self.assertContains(response, responses_text)

    def test_show_tempuser_filter(self):
        """
        Tests whether user sees his own filters and the system-wide (shared) filters but not the filters from other modalities and users
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(reverse("nm_summary_list_filter"), follow=True)

        self.assertEqual(response.status_code, 200)

        self.assertContains(response, TEMPUSER_FILTER_1)
        self.assertContains(response, TEMPUSER_FILTER_2)
        self.assertContains(response, SUPERUSER_SHARED_FILTER_1)
        self.assertContains(response, SUPERUSER_SHARED_FILTER_2)

        self.assertNotContains(response, TEMPUSER_WRONG_MOD)
        self.assertNotContains(response, SUPERUSER_FILTER)
        self.assertNotContains(response, SUPERUSER_SHARED_WRONG_MOD)

        self.assertNotContains(response, SHARE_BUTTON_TEXT)
        self.assertNotContains(response, REMOVE_SHARED_BUTTON_TEXT)

    def test_show_superuser_filter(self):
        """
        Tests whether superuser sees his own filters and the system-wide (shared) filters but not the filters from other modalities and users
        """
        self.client.login(username="superuser", password="superpassword")
        response = self.client.get(reverse("nm_summary_list_filter"), follow=True)

        self.assertEqual(response.status_code, 200)

        self.assertContains(response, SUPERUSER_FILTER)
        self.assertContains(response, SUPERUSER_SHARED_FILTER_1)
        self.assertContains(response, SUPERUSER_SHARED_FILTER_2)

        self.assertNotContains(response, TEMPUSER_FILTER_1)
        self.assertNotContains(response, TEMPUSER_FILTER_2)
        self.assertNotContains(response, TEMPUSER_WRONG_MOD)
        self.assertNotContains(response, SUPERUSER_SHARED_WRONG_MOD)

        self.assertContains(response, SHARE_BUTTON_TEXT)
        self.assertContains(response, REMOVE_SHARED_BUTTON_TEXT)
