# This Python file uses the following encoding: utf-8
# test_charts_dx.py

import os
from django.contrib.auth.models import User, Group
from django.test import TestCase, RequestFactory
from remapp.extractors import dx
from remapp.models import GeneralStudyModuleAttr, PatientIDSettings
from django.db.models import Q
from remapp.interface.mod_filters import DXSummaryListFilter
import numpy as np
import math


class ChartsDX(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username="jacob", email="jacob@…", password="top_secret"
        )
        eg = Group(name="viewgroup")
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

        dx_ge_xr220_1 = os.path.join("test_files", "DX-Im-GE_XR220-1.dcm")
        dx_ge_xr220_2 = os.path.join("test_files", "DX-Im-GE_XR220-2.dcm")
        dx_ge_xr220_3 = os.path.join("test_files", "DX-Im-GE_XR220-3.dcm")
        dx_carestream_dr7500_1 = os.path.join(
            "test_files", "DX-Im-Carestream_DR7500-1.dcm"
        )
        dx_carestream_dr7500_2 = os.path.join(
            "test_files", "DX-Im-Carestream_DR7500-2.dcm"
        )
        root_tests = os.path.dirname(os.path.abspath(__file__))

        dx.dx(os.path.join(root_tests, dx_ge_xr220_1))
        dx.dx(os.path.join(root_tests, dx_ge_xr220_2))
        dx.dx(os.path.join(root_tests, dx_ge_xr220_3))
        dx.dx(os.path.join(root_tests, dx_carestream_dr7500_1))
        dx.dx(os.path.join(root_tests, dx_carestream_dr7500_2))

    def user_profile_reset(self):
        self.user.userprofile.plotCharts = True

        self.user.userprofile.plotGroupingChoice = "system"
        self.user.userprofile.plotSeriesPerSystem = False
        self.user.userprofile.plotCaseInsensitiveCategories = False

        self.user.userprofile.plotInitialSortingDirection = 0
        self.user.userprofile.plotDXInitialSortingChoice = "frequency"

        self.user.userprofile.plotAverageChoice = "mean"
        self.user.userprofile.plotMean = False
        self.user.userprofile.plotMedian = False
        self.user.userprofile.plotBoxplots = False

        self.user.userprofile.plotHistograms = False
        self.user.userprofile.plotHistogramBins = 10

        self.user.userprofile.plotDXAcquisitionMeanDAP = False
        self.user.userprofile.plotDXAcquisitionMeankVp = False
        self.user.userprofile.plotDXAcquisitionMeanmAs = False
        self.user.userprofile.plotDXAcquisitionFreq = False
        self.user.userprofile.plotDXAcquisitionDAPvsMass = False
        self.user.userprofile.plotDXAcquisitionMeankVpOverTime = False
        self.user.userprofile.plotDXAcquisitionMeanmAsOverTime = False
        self.user.userprofile.plotDXAcquisitionMeanDAPOverTime = False

        self.user.userprofile.plotDXStudyMeanDAP = False
        self.user.userprofile.plotDXStudyFreq = False
        self.user.userprofile.plotDXStudyDAPvsMass = False
        self.user.userprofile.plotDXStudyPerDayAndHour = False

        self.user.userprofile.plotDXRequestMeanDAP = False
        self.user.userprofile.plotDXRequestFreq = False
        self.user.userprofile.plotDXRequestDAPvsMass = False

        self.user.userprofile.save()

    def obtain_chart_data(self, f):
        from remapp.views_charts_dx import dx_plot_calculations

        self.chart_data = dx_plot_calculations(
            f, self.user.userprofile, return_as_dict=True
        )

    def test_dap_per_acq(self):
        # Test of mean and median DAP, count, system and acquisition protocol names
        # Also tests raw data going into the box plots
        self.client.login(username="jacob", password="top_secret")

        # I can add to the filter_set to control what type of chart data is calculated
        filter_set = ""

        f = DXSummaryListFilter(
            filter_set,
            queryset=GeneralStudyModuleAttr.objects.filter(
                Q(modality_type__exact="DX") | Q(modality_type__exact="CR")
            )
            .order_by()
            .distinct(),
        )

        # Set user profile options
        self.user_profile_reset()
        self.user.userprofile.plotDXAcquisitionMeanDAP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotDXAcquisitionFreq = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the chart data
        # Acquisition name test
        acq_names = ["ABD_1_VIEW", "AEC"]
        self.assertListEqual(
            list(self.chart_data["acquisitionMeanDAPData"]["data"][0]["x"]), acq_names
        )

        # Acquisition system name test
        acq_system_names = "All systems"
        self.assertEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][0]["name"],
            acq_system_names,
        )

        # Check on mean data
        acq_data = [[0.0, 10.93333333, 3.0], [0.0, 105.85, 2.0]]

        # Check on mean DAP values
        self.assertAlmostEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"][0][1],
            acq_data[0][1],
        )
        self.assertAlmostEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"][1][1],
            acq_data[1][1],
        )
        # Check on counts in the mean data
        self.assertEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"][0][2],
            acq_data[0][2],
        )
        self.assertEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"][1][2],
            acq_data[1][2],
        )

        # Check on median values
        acq_data = [[0.0, 8.2, 3.0], [0.0, 105.85, 2.0]]

        # Check on median DAP values
        self.assertAlmostEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][0]["customdata"][0][1],
            acq_data[0][1],
        )
        self.assertAlmostEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][0]["customdata"][1][1],
            acq_data[1][1],
        )
        # Check on counts in the median data
        self.assertEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][0]["customdata"][0][2],
            acq_data[0][2],
        )
        self.assertEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][0]["customdata"][1][2],
            acq_data[1][2],
        )

        # Check on the boxplot data x-axis values
        acq_data = "All systems"
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["name"], acq_data
        )

        acq_data = ["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW", "AEC", "AEC"]
        self.assertListEqual(
            list(self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["x"]), acq_data
        )

        # Check the boxplot data y-axis values
        acq_data = [4.1, 8.2, 20.5, 101.57, 110.13]
        for idx, value in enumerate(acq_data):
            self.assertAlmostEqual(
                np.sort(self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["y"])[
                    idx
                ],
                value,
            )

        # The frequency chart - system names
        acq_data = "All systems"
        self.assertEqual(
            self.chart_data["acquisitionFrequencyData"]["data"][0]["x"], acq_data
        )
        self.assertEqual(
            self.chart_data["acquisitionFrequencyData"]["data"][1]["x"], acq_data
        )

        # The frequency chart - acquisition protocol names
        acq_data = ["ABD_1_VIEW", "AEC"]
        self.assertEqual(
            self.chart_data["acquisitionFrequencyData"]["data"][0]["name"], acq_data[0]
        )
        self.assertEqual(
            self.chart_data["acquisitionFrequencyData"]["data"][1]["name"], acq_data[1]
        )

        # The frequency chart - frequencies
        acq_data = [3, 2]
        self.assertEqual(
            self.chart_data["acquisitionFrequencyData"]["data"][0]["y"], acq_data[0]
        )
        self.assertEqual(
            self.chart_data["acquisitionFrequencyData"]["data"][1]["y"], acq_data[1]
        )

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Acquisition name test
        acq_names = ["ABD_1_VIEW", "AEC"]
        self.assertListEqual(
            list(self.chart_data["acquisitionMeanDAPData"]["data"][0]["x"]), acq_names
        )

        # Acquisition system name test
        acq_system_names = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        self.assertEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][0]["name"],
            acq_system_names[0],
        )
        self.assertEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][1]["name"],
            acq_system_names[1],
        )

        # Check on mean data of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 105.85, 2.0]]
        # Check on mean DAP values
        self.assertTrue(
            math.isnan(
                self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"][0][1]
            )
        )
        self.assertAlmostEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"][1][1],
            acq_data[1][1],
        )
        # Check on counts in the mean data
        self.assertEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"][0][2],
            acq_data[0][2],
        )
        self.assertEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"][1][2],
            acq_data[1][2],
        )

        # Check on mean data of series 1
        acq_data = [[1.0, 10.93333333, 3.0], [1.0, np.nan, 0.0]]
        # Check on mean DAP values
        self.assertAlmostEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][1]["customdata"][0][1],
            acq_data[0][1],
        )
        self.assertTrue(
            math.isnan(
                self.chart_data["acquisitionMeanDAPData"]["data"][1]["customdata"][1][1]
            )
        )
        # Check on counts in the mean data
        self.assertEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][1]["customdata"][0][2],
            acq_data[0][2],
        )
        self.assertEqual(
            self.chart_data["acquisitionMeanDAPData"]["data"][1]["customdata"][1][2],
            acq_data[1][2],
        )

        # Check on median values of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 105.85, 2.0]]
        # Check on mean DAP values
        self.assertTrue(
            math.isnan(
                self.chart_data["acquisitionMedianDAPData"]["data"][0]["customdata"][0][
                    1
                ]
            )
        )
        self.assertAlmostEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][0]["customdata"][1][1],
            acq_data[1][1],
        )
        # Check on counts in the mean data
        self.assertEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][0]["customdata"][0][2],
            acq_data[0][2],
        )
        self.assertEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][0]["customdata"][1][2],
            acq_data[1][2],
        )

        # Check on median values of series 1
        acq_data = [[1.0, 8.2, 3.0], [1.0, np.nan, 0.0]]
        # Check on mean DAP values
        self.assertAlmostEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][1]["customdata"][0][1],
            acq_data[0][1],
        )
        self.assertTrue(
            math.isnan(
                self.chart_data["acquisitionMedianDAPData"]["data"][1]["customdata"][1][
                    1
                ]
            )
        )
        # Check on counts in the mean data
        self.assertEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][1]["customdata"][0][2],
            acq_data[0][2],
        )
        self.assertEqual(
            self.chart_data["acquisitionMedianDAPData"]["data"][1]["customdata"][1][2],
            acq_data[1][2],
        )

        # Check on the boxplot data system names
        acq_data = ["Carestream Clinic KODAK7500", "Digital Mobile Hospital 01234MOB54"]
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["name"], acq_data[0]
        )
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][1]["name"], acq_data[1]
        )

        # Check on the boxplot data x-axis values
        acq_data = ["AEC", "AEC"]
        self.assertListEqual(
            list(self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["x"]), acq_data
        )
        acq_data = ["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW"]
        self.assertListEqual(
            list(self.chart_data["acquisitionBoxplotDAPData"]["data"][1]["x"]), acq_data
        )

        # Check on the boxplot data y-axis values
        acq_data = [101.57, 110.13]
        for idx, value in enumerate(acq_data):
            self.assertAlmostEqual(
                np.sort(self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["y"])[
                    idx
                ],
                value,
            )

        acq_data = [4.1, 8.2, 20.5]
        for idx, value in enumerate(acq_data):
            self.assertAlmostEqual(
                np.sort(self.chart_data["acquisitionBoxplotDAPData"]["data"][1]["y"])[
                    idx
                ],
                value,
            )

        # The frequency chart - system names
        acq_data = ["Carestream Clinic KODAK7500", "Digital Mobile Hospital 01234MOB54"]
        self.assertListEqual(
            list(self.chart_data["acquisitionFrequencyData"]["data"][0]["x"]), acq_data
        )
        self.assertListEqual(
            list(self.chart_data["acquisitionFrequencyData"]["data"][1]["x"]), acq_data
        )

        # The frequency chart - acquisition protocol names
        acq_data = ["ABD_1_VIEW", "AEC"]
        self.assertEqual(
            self.chart_data["acquisitionFrequencyData"]["data"][0]["name"], acq_data[0]
        )
        self.assertEqual(
            self.chart_data["acquisitionFrequencyData"]["data"][1]["name"], acq_data[1]
        )

        # The frequency chart - frequencies
        acq_data = [0, 3]
        self.assertListEqual(
            list(self.chart_data["acquisitionFrequencyData"]["data"][0]["y"]), acq_data
        )
        acq_data = [2, 0]
        self.assertListEqual(
            list(self.chart_data["acquisitionFrequencyData"]["data"][1]["y"]), acq_data
        )
