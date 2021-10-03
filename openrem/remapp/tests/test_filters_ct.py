# This Python file uses the following encoding: utf-8
# test_filters_ct.py

import os
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse_lazy, reverse
from ..extractors import rdsr, ct_philips
from ..models import PatientIDSettings


class FilterViewTests(TestCase):
    """
    Class to test the filter views for CT
    """

    def setUp(self):
        """
        Load in all the CT objects so that there is something to filter!
        """
        PatientIDSettings.objects.create()
        User.objects.create_user("temporary", "temporary@gmail.com", "temporary")

        ct1 = "test_files/CT-ESR-GE_Optima.dcm"
        ct2 = "test_files/CT-ESR-GE_VCT.dcm"
        ct3 = "test_files/CT-RDSR-GEPixelMed.dcm"
        ct4 = "test_files/CT-RDSR-Siemens_Flash-QA-DS.dcm"
        ct5 = "test_files/CT-RDSR-Siemens_Flash-TAP-SS.dcm"
        ct6 = "test_files/CT-RDSR-ToshibaPixelMed.dcm"
        ct7 = "test_files/CT-SC-Philips_Brilliance16P.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        path_ct1 = os.path.join(root_tests, ct1)
        path_ct2 = os.path.join(root_tests, ct2)
        path_ct3 = os.path.join(root_tests, ct3)
        path_ct4 = os.path.join(root_tests, ct4)
        path_ct5 = os.path.join(root_tests, ct5)
        path_ct6 = os.path.join(root_tests, ct6)
        path_ct7 = os.path.join(root_tests, ct7)

        rdsr.rdsr(path_ct1)
        rdsr.rdsr(path_ct2)
        rdsr.rdsr(path_ct3)
        rdsr.rdsr(path_ct4)
        rdsr.rdsr(path_ct5)
        rdsr.rdsr(path_ct6)
        ct_philips.ct_philips(path_ct7)

    def test_list_all_ct(self):
        """
        Initial test to ensure seven studies are listed with no filter
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(reverse("ct_summary_list_filter"), follow=True)
        self.assertEqual(response.status_code, 200)
        responses_text = "There are 7 studies in this list."
        self.assertContains(response, responses_text)

    def test_filter_study_desc(self):
        """
        Apply study description filter
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(
            "http://test/openrem/ct/?study_description=abdomen", follow=True
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        display_name = (
            "PHILIPS-E71E3F0"  # Display name of study with matching study description
        )
        self.assertContains(response, display_name)

    def test_filter_procedure(self):
        """
        Apply procedure filter
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(
            "http://test/openrem/ct/?procedure_code_meaning=abdomen", follow=True
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        display_name = (
            "ACC12345601"  # Display name of study with matching study description
        )
        self.assertContains(response, display_name)

    def test_filter_requested_procedure(self):
        """
        Apply procedure filter
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(
            "http://test/openrem/ct/?requested_procedure_code_meaning=bones",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        display_name = (
            "001234512345678"  # Display name of study with matching study description
        )
        self.assertContains(response, display_name)

    def test_filter_acquisition_protocol(self):
        """
        Apply acquisition protocol filter
        """
        self.client.login(username="temporary", password="temporary")
        response = self.client.get(
            "http://test/openrem/ct/?ctradiationdose__ctirradiationeventdata__acquisition_protocol=monitoring",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number = "ACC12345601"  # Accession number of study with matching acquisition protocol
        self.assertContains(response, accession_number)

    def test_specify_event_numbers(self):
        """
        Apply specific event number filters
        """
        self.client.login(username="temporary", password="temporary")

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=2&num_axial_events=&num_spr_events=&num_stationary_events=",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        four_responses_text = "There are 4 studies in this list."
        self.assertContains(response, four_responses_text)
        accession_number_1 = "4935683"
        accession_number_2 = "74624646290"
        accession_number_3 = "001234512345678"
        accession_number_4 = "0012345.12345678"
        self.assertContains(response, accession_number_1)
        self.assertContains(response, accession_number_2)
        self.assertContains(response, accession_number_3)
        self.assertContains(response, accession_number_4)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=2&num_axial_events=0&num_spr_events=&num_stationary_events=",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        three_responses_text = "There are 3 studies in this list."
        self.assertContains(response, three_responses_text)
        accession_number_1 = "4935683"
        accession_number_2 = "74624646290"
        accession_number_3 = "0012345.12345678"
        self.assertContains(response, accession_number_1)
        self.assertContains(response, accession_number_2)
        self.assertContains(response, accession_number_3)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=2&num_axial_events=5&num_spr_events=&num_stationary_events=4",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number_1 = "001234512345678"
        self.assertContains(response, accession_number_1)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=1&num_axial_events=0&num_spr_events=1&num_stationary_events=2",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number_1 = "ACC12345601"
        self.assertContains(response, accession_number_1)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=&num_axial_events=5&num_spr_events=&num_stationary_events=",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number_1 = "001234512345678"
        self.assertContains(response, accession_number_1)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=&num_axial_events=&num_spr_events=1&num_stationary_events=",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        three_responses_text = "There are 3 studies in this list."
        self.assertContains(response, three_responses_text)
        accession_number_1 = "IH_KUB_2mm"
        accession_number_2 = "4935683"
        accession_number_3 = "ACC12345601"
        self.assertContains(response, accession_number_1)
        self.assertContains(response, accession_number_2)
        self.assertContains(response, accession_number_3)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=&num_axial_events=&num_spr_events=&num_stationary_events=2",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number_1 = "ACC12345601"
        self.assertContains(response, accession_number_1)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=&num_axial_events=5&num_spr_events=&num_stationary_events=4",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number_1 = "001234512345678"
        self.assertContains(response, accession_number_1)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=1&num_axial_events=&num_spr_events=1&num_stationary_events=",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number_1 = "ACC12345601"
        self.assertContains(response, accession_number_1)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=2&num_axial_events=&num_spr_events=&num_stationary_events=7",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number_1 = "74624646290"
        self.assertContains(response, accession_number_1)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=1&num_axial_events=&num_spr_events=1&num_stationary_events=2",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        one_responses_text = "There are 1 studies in this list."
        self.assertContains(response, one_responses_text)
        accession_number_1 = "ACC12345601"
        self.assertContains(response, accession_number_1)

        response = self.client.get(
            "http://test/openrem/ct/?num_spiral_events=&num_axial_events=&num_spr_events=some&num_stationary_events=",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        five_responses_text = "There are 5 studies in this list."
        self.assertContains(response, five_responses_text)

    def test_filter_patient_weight(self):
        """
        Apply patient weight filter
        """
        self.client.login(username="temporary", password="temporary")

        # 78.2 kg: CT-ESR-GE_Optima.dcm"
        # None:    CT-ESR-GE_VCT.dcm"
        # None:    CT-RDSR-GEPixelMed.dcm"
        # 75.0 kg: CT-RDSR-Siemens_Flash-QA-DS.dcm"
        # 87.0 kg: CT-RDSR-Siemens_Flash-TAP-SS.dcm"
        # 75.0 kg: CT-RDSR-ToshibaPixelMed.dcm"
        # None:    CT-SC-Philips_Brilliance16P.dcm"

        # Filter min weight 70 kg, max weight 90 kg
        # This should leave the four studies that have weight data
        response = self.client.get(
            reverse_lazy("ct_summary_list_filter")
            + "?patientstudymoduleattr__patient_weight__gte=70&patientstudymoduleattr__patient_weight__lte=90",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        four_responses_text = "There are 4 studies in this list."
        self.assertContains(response, four_responses_text)
        accession_number_optima = "0012345.12345678"
        self.assertContains(response, accession_number_optima)
        accession_number_flash_qa = "74624646290"
        self.assertContains(response, accession_number_flash_qa)
        accession_number_flash_tap = "ACC12345601"
        self.assertContains(response, accession_number_flash_tap)
        accession_number_toshiba = "4935683"
        self.assertContains(response, accession_number_toshiba)

        # Filter min weight 70 kg, max weight 76 kg
        # This should leave two studies
        response = self.client.get(
            reverse_lazy("ct_summary_list_filter")
            + "?patientstudymoduleattr__patient_weight__gte=70&patientstudymoduleattr__patient_weight__lte=76",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        two_responses_text = "There are 2 studies in this list."
        self.assertContains(response, two_responses_text)
        accession_number_flash_qa = "74624646290"
        self.assertContains(response, accession_number_flash_qa)
        accession_number_toshiba = "4935683"
        self.assertContains(response, accession_number_toshiba)

        # Filter min weight 76 kg, max weight 90 kg
        # This should leave two studies
        response = self.client.get(
            reverse_lazy("ct_summary_list_filter")
            + "?patientstudymoduleattr__patient_weight__gte=76&patientstudymoduleattr__patient_weight__lte=90",
            follow=True,
        )
        self.assertEqual(response.status_code, 200)
        two_responses_text = "There are 2 studies in this list."
        self.assertContains(response, two_responses_text)
        accession_number_optima = "0012345.12345678"
        self.assertContains(response, accession_number_optima)
        accession_number_flash_tap = "ACC12345601"
        self.assertContains(response, accession_number_flash_tap)
