# This Python file uses the following encoding: utf-8
# test_charts_dx.py

import os
from django.contrib.auth.models import User, Group
from django.test import TestCase, RequestFactory
from remapp.extractors import mam, rdsr
from remapp.models import GeneralStudyModuleAttr, PatientIDSettings
from remapp.interface.mod_filters import MGSummaryListFilter
import numpy as np
import math


class ChartsMG(TestCase):
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

        mam1 = os.path.join("test_files", "MG-Im-Hologic-PropProj.dcm")
        mam2 = os.path.join("test_files", "MG-Im-GE-SenDS-scaled.dcm")
        mam3 = os.path.join("test_files", "MG-RDSR-Hologic_2D.dcm")
        root_tests = os.path.dirname(os.path.abspath(__file__))

        mam.mam(os.path.join(root_tests, mam1))
        mam.mam(os.path.join(root_tests, mam2))
        rdsr.rdsr(os.path.join(root_tests, mam3))

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

        self.user.userprofile.plotMGaverageAGD = False
        self.user.userprofile.plotMGacquisitionFreq = False
        self.user.userprofile.plotMGAGDvsThickness = False
        self.user.userprofile.plotMGmAsvsThickness = False
        self.user.userprofile.plotMGkVpvsThickness = False
        self.user.userprofile.plotMGaverageAGDvsThickness = False

        self.user.userprofile.plotMGStudyPerDayAndHour = False

        self.user.userprofile.save()

    def login_get_filterset(self):
        self.client.login(username="jacob", password="top_secret")
        # I can add to the filter_set to control what type of chart data is calculated
        filter_set = ""
        f = MGSummaryListFilter(
            filter_set,
            queryset=GeneralStudyModuleAttr.objects.filter(
                modality_type__exact="MG"
            )
            .order_by()
            .distinct(),
        )
        # Reset the user profile
        self.user_profile_reset()
        return f

    def obtain_chart_data(self, f):
        from remapp.views_charts_mg import mg_plot_calculations

        self.chart_data = mg_plot_calculations(
            f, self.user.userprofile, return_as_dict=True
        )

    def check_series_and_category_names(self, category_names, series_names, chart_data):
        for idx, series_name in enumerate(series_names):
            self.assertEqual(
                chart_data[idx]["name"],
                series_name,
            )
            self.assertListEqual(list(chart_data[idx]["x"]), category_names)

    def check_avg_and_counts(self, comparison_data, chart_data):
        for idx in range(len(comparison_data)):
            # If the comparison value is a nan then check that the chart value is too
            if math.isnan(comparison_data[idx][1]):
                self.assertTrue(math.isnan(chart_data[idx][1]))
            # Otherwise compare the values
            else:
                self.assertAlmostEqual(
                    chart_data[idx][1],
                    comparison_data[idx][1],
                )
            self.assertEqual(
                chart_data[idx][2],
                comparison_data[idx][2],
            )

    def check_frequencies(self, comparison_data, chart_data):
        for idx, values in enumerate(comparison_data):
            self.assertListEqual(list(chart_data[idx]["y"]), comparison_data[idx])

    def check_boxplot_xy(self, x_data, y_data, chart_data):
        for i in range(len(x_data)):
            std_x_data = x_data[i]
            std_y_data = y_data[i]
            std_y_data = [y for y, _ in sorted(zip(std_y_data, std_x_data))]
            std_x_data = sorted(std_x_data)
            std_x_data = [x for _, x in sorted(zip(std_y_data, std_x_data))]
            std_y_data = sorted(std_y_data)

            chart_y_data = chart_data[i]["y"]
            chart_x_data = chart_data[i]["x"]
            chart_y_data = [y for y, _ in sorted(zip(chart_y_data, chart_x_data))]
            chart_x_data = sorted(chart_x_data)
            chart_x_data = [x for _, x in sorted(zip(chart_y_data, chart_x_data))]
            chart_y_data = sorted(chart_y_data)

            np.testing.assert_equal(chart_x_data, std_x_data)
            np.testing.assert_almost_equal(chart_y_data, std_y_data)

    def test_required_charts(self):
        from remapp.views_charts_mg import generate_required_mg_charts_list

        f = self.login_get_filterset()

        # Set user profile options to use all charts
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotHistograms = True
        self.user.plotMGStudyPerDayAndHour = True
        self.user.plotMGAGDvsThickness = True
        self.user.plotMGkVpvsThickness = True
        self.user.plotMGmAsvsThickness = True
        self.user.plotMGaverageAGDvsThickness = True
        self.user.plotMGaverageAGD = True
        self.user.plotMGacquisitionFreq = True
        self.user.userprofile.save()

        required_charts_list = generate_required_mg_charts_list(self.user.userprofile)

        chart_var_names = []
        for item in required_charts_list:
            chart_var_names.append(item["var_name"])

        # Just check the variable names - I don't mind if the titles change
        reference_var_names = [
            "acquisitionScatterAGDvsThick",
            "acquisitionFrequency",
            "acquisitionMeanAGD",
            "acquisitionMedianAGD",
            "acquisitionBoxplotAGD",
            "acquisitionHistogramAGD",
            "acquisitionMeanAGDvsThick",
            "acquisitionMedianAGDvsThick",
            "acquisitionScatterkVpvsThick",
            "acquisitionScattermAsvsThick",
            "studyWorkload",
        ]

        for ref_var_name in reference_var_names:
            self.assertTrue(ref_var_name in chart_var_names)

    def test_acq_agd(self):
        # Test of mean and median AGD, count, system and acquisition protocol names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotMGaverageAGD = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the mean data
        standard_data = [
            {
                'customdata': np.array([[0., 1.29, 2.],
                                        [0., 0.26, 1.],
                                        [0., 1.373, 1.]]),
                'name': 'All systems',
                'x': np.array(['Blank', 'Flat Field Tomo', 'ROUTINE'], dtype=object),
                'y': np.array([1.29, 0.26, 1.373]), }]

        chart_data = self.chart_data["acquisitionMeanAGDData"]["data"]

        self.assertEqual(standard_data[0]["name"], chart_data[0]["name"])
        np.testing.assert_array_equal(standard_data[0]["x"], chart_data[0]["x"])
        np.testing.assert_array_almost_equal(standard_data[0]["customdata"], chart_data[0]["customdata"])

        # Test the median data
        standard_data = [
            {
                'customdata': np.array([[0., 1.29, 2.],
                                        [0., 0.26, 1.],
                                        [0., 1.373, 1.]]),
                'name': 'All systems',
                'x': np.array(['Blank', 'Flat Field Tomo', 'ROUTINE'], dtype=object),
                'y': np.array([1.29, 0.26, 1.373]), }]

        chart_data = self.chart_data["acquisitionMedianAGDData"]["data"]

        self.assertEqual(standard_data[0]["name"], chart_data[0]["name"])
        np.testing.assert_array_equal(standard_data[0]["x"], chart_data[0]["x"])
        np.testing.assert_array_almost_equal(standard_data[0]["customdata"], chart_data[0]["customdata"])

        # Check the boxplot data
        standard_data = [
            {
                'name': 'All systems',
                'x': np.array(['Flat Field Tomo', 'Blank', 'Blank', 'ROUTINE'], dtype=object),
                'y': np.array([0.26, 1.28, 1.3, 1.373]),
            }
        ]

        chart_data = self.chart_data["acquisitionBoxplotAGDData"]["data"]

        self.assertEqual(standard_data[0]["name"], chart_data[0]["name"])
        np.testing.assert_array_equal(standard_data[0]["x"], chart_data[0]["x"])
        np.testing.assert_array_almost_equal(standard_data[0]["y"], chart_data[0]["y"])


        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)
