# This Python file uses the following encoding: utf-8
# test_filters_rf.py

import os
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse_lazy, reverse
from remapp.extractors import rdsr
from remapp.models import PatientIDSettings


class FilterViewTests(TestCase):
    """
    Class to test the filter views for fluoroscopy
    """

    def setUp(self):
        """
        Load in all the rf objects so that there is something to filter!
        """
        PatientIDSettings.objects.create()
        User.objects.create_user("temporary", "temporary@gmail.com", "temporary")

        rf1 = "test_files/RF-RDSR-Philips_Allura.dcm"
        rf2 = "test_files/RF-RDSR-Siemens-Zee.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        path_rf1 = os.path.join(root_tests, rf1)
        path_rf2 = os.path.join(root_tests, rf2)

        rdsr.rdsr(path_rf1)
        rdsr.rdsr(path_rf2)

    def test_list_all_fluoro(self):
        """
        Initial test to ensure two studies are listed with no filter
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(reverse("rf_summary_list_filter"), follow=True)
        self.assertEqual(response.status_code, 200)
        three_responses_text = "There are 2 studies in this list."
        self.assertContains(response, three_responses_text)

    def test_filter_study_desc(self):
        """
        Apply study description filter
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(
            reverse_lazy("rf_summary_list_filter")
            + "?study_description=liuotushoidon+raajojen",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number = (
            "01234.1234"  # Accession number of study with matching study description
        )
        self.assertContains(response, accession_number)

    def test_filter_acquisition_protocol(self):
        """
        Apply acquisition protocol filter
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(
            reverse_lazy("rf_summary_list_filter")
            + "?projectionxrayradiationdose__irradeventxraydata__acquisition_protocol=2fps",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number = (
            "01234.1234"  # Accession number of study with matching acquisition protocol
        )
        self.assertContains(response, accession_number)

    def test_filter_patient_weight(self):
        """
        Apply patient weight filter
        """
        self.client.login(username="temporary", password="temporary")

        # The Philips test study has patient weight of 86.2 kg
        # The Siemens test study does not have patient weight data

        # Test filtering using min weight of 80 kg and max weight of 90 kg - this should exclude the Siemens study
        response = self.client.get(
            reverse_lazy("rf_summary_list_filter")
            + "?patientstudymoduleattr__patient_weight__gte=80&patientstudymoduleattr__patient_weight__lte=90",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number = "01234.1234"  # Accession number of the Allura study which has patient weight of 86.2 kg
        self.assertContains(response, accession_number)

        # Test filtering using min weight of 90 kg - this should exclude both studies
        response = self.client.get(
            reverse_lazy("rf_summary_list_filter")
            + "?patientstudymoduleattr__patient_weight__gte=90",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        zero_responses_text = "There are 0 studies in this list."
        self.assertContains(response, zero_responses_text)

        # Test filtering using max weight of 80 kg - this should exclude both studies
        response = self.client.get(
            reverse_lazy("rf_summary_list_filter")
            + "?patientstudymoduleattr__patient_weight__lte=80",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        zero_responses_text = "There are 0 studies in this list."
        self.assertContains(response, zero_responses_text)
