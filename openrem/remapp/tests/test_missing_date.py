import os
from django.conf import settings
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from remapp.extractors import rdsr, dx
from remapp.models import PatientIDSettings, HomePageAdminSettings


class FilterViewTests(TestCase):
    """
    Class to test the filter views for radiography
    """

    def setUp(self):
        """
        Load in study with missing date attributes
        """
        PatientIDSettings.objects.create()
        User.objects.create_user("temporary", "temporary@gmail.com", "temporary")
        HomePageAdminSettings.objects.create()

        dx1 = "test_files/DX-Im-Date-Missing.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        path_dx1 = os.path.join(root_tests, dx1)

        dx.dx(path_dx1)

    def test_list_all_dx(self):
        """
        Initial test to ensure five studies are listed with no filter
        """
        self.client.login(username="temporary", password="temporary")
        self.client.cookies.load({settings.LANGUAGE_COOKIE_NAME: "en"})
        response = self.client.post(
            reverse("update_latest_studies"),
            data={"modality": "DX"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        self.assertEqual(response.status_code, 200)
        responses_text = "DX Date Removed KODAK7500"
        self.assertContains(response, responses_text)
