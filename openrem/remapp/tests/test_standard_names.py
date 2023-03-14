# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2012,2013  The Royal Marsden NHS Foundation Trust
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    Additional permission under section 7 of GPLv3:
#    You shall not make any use of the name of The Royal Marsden NHS
#    Foundation trust in connection with this Program in any press or
#    other public announcement without the prior written consent of
#    The Royal Marsden NHS Foundation Trust.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from datetime import date, time, datetime
import os
from decimal import Decimal
from typing import Dict
from django.test import TestCase
from django.urls import reverse_lazy, reverse
from remapp.extractors import rdsr, ct_philips

from django.contrib.auth.models import User, Group
from remapp.models import PatientIDSettings, StandardNames, GeneralStudyModuleAttr


class StandardNamesTest(TestCase):
    def setUp(self) -> None:
        PatientIDSettings.objects.create()
        user = User.objects.create_user("temporary", "temporary@gmail.com", "temporary")
        admingroup = Group.objects.create(name="admingroup")
        user.groups.add(admingroup)

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

    def test_create_ct_standard_name(self):
        self.client.login(username="temporary", password="temporary")
        response = self.client.post(
            reverse("add_name_ct"),
            {
                "standard_name": "ct",
                "study_description": ["FACIAL BONES"],
                "diagnostic_reference_level_criteria": "bmi",
                "drl_alert_factor": "15.0",
                "k_factor_criteria": "age",
                "modality": "CT",
                "drl_formset-TOTAL_FORMS": " 1",
                "drl_formset-INITIAL_FORMS": "0",
                "drl_formset-MIN_NUM_FORMS": "0",
                "drl_formset-MAX_NUM_FORMS": "1000",
                "drl_formset-0-id": "",
                "drl_formset-0-lower_bound": "",
                "drl_formset-0-upper_bound": "",
                "drl_formset-0-diagnostic_reference_level": "",
                "kfactor_formset-TOTAL_FORMS": " 1",
                "kfactor_formset-INITIAL_FORMS": "0",
                "kfactor_formset-MIN_NUM_FORMS": "0",
                "kfactor_formset-MAX_NUM_FORMS": "1000",
                "kfactor_formset-0-id": "",
                "kfactor_formset-0-lower_bound": "",
                "kfactor_formset-0-upper_bound": "",
                "kfactor_formset-0-k_factor": "",
            },
        )
        self.assertEqual(response.status_code, 302)

        std_name = StandardNames.objects.filter(standard_name="ct")

        self.assertEqual(std_name.count(), 1)

        std_name = std_name[0]

        self.assertEqual(std_name.standard_name, "ct")
        self.assertEqual(std_name.diagnostic_reference_level_criteria, "bmi")
        self.assertAlmostEqual(std_name.drl_alert_factor, Decimal(15.0))
        self.assertEqual(std_name.k_factor_criteria, "age")
