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

    def login_get_filterset(self):
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
        # Reset the user profile
        self.user_profile_reset()
        return f

    def obtain_chart_data(self, f):
        from remapp.views_charts_dx import dx_plot_calculations

        self.chart_data = dx_plot_calculations(
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
        for idx in range(0, 1):
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
            chart_y_data = chart_data[i]["y"]
            chart_x_data = chart_data[i]["x"]

            chart_y_data, chart_x_data = np.sort(np.array([chart_y_data, chart_x_data]))

            for j in range(len(chart_x_data)):
                self.assertEqual(chart_x_data[j], x_data[i][j])

                self.assertAlmostEqual(
                    chart_y_data[j],
                    y_data[i][j],
                )

    def test_acq_dap(self):
        # Test of mean and median DAP, count, system and acquisition protocol names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionMeanDAP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Acquisition name and system name test
        acq_system_names = ["All systems"]
        acq_names = ["ABD_1_VIEW", "AEC"]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"]
        self.check_series_and_category_names(acq_names, acq_system_names, chart_data)

        # Check on mean DAP values and counts
        acq_data = [[0.0, 10.93333333, 3.0], [0.0, 105.85, 2.0]]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median DAP values and counts
        acq_data = [[0.0, 8.2, 3.0], [0.0, 105.85, 2.0]]
        chart_data = self.chart_data["acquisitionMedianDAPData"]["data"][0][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on the boxplot data system names
        acq_data = "All systems"
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["name"], acq_data
        )

        # Check the boxplot x and y data values
        acq_x_data = [["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW", "AEC", "AEC"]]
        acq_y_data = [[4.1, 8.2, 20.5, 101.57, 110.13]]
        chart_data = self.chart_data["acquisitionBoxplotDAPData"]["data"]
        self.check_boxplot_xy(acq_x_data, acq_y_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Acquisition name and system name test
        acq_system_names = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        acq_names = ["ABD_1_VIEW", "AEC"]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"]
        self.check_series_and_category_names(acq_names, acq_system_names, chart_data)

        # Check on mean data of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 105.85, 2.0]]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on mean data of series 1
        acq_data = [[1.0, 10.93333333, 3.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 105.85, 2.0]]
        chart_data = self.chart_data["acquisitionMedianDAPData"]["data"][0][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values of series 1
        acq_data = [[1.0, 8.2, 3.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMedianDAPData"]["data"][1][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on the boxplot data system names
        acq_data = ["Carestream Clinic KODAK7500", "Digital Mobile Hospital 01234MOB54"]
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["name"], acq_data[0]
        )
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][1]["name"], acq_data[1]
        )

        # Check the boxplot x and y data values
        acq_x_data = [["AEC", "AEC"], ["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW"]]
        acq_y_data = [[101.57, 110.13], [4.1, 8.2, 20.5]]
        chart_data = self.chart_data["acquisitionBoxplotDAPData"]["data"]
        self.check_boxplot_xy(acq_x_data, acq_y_data, chart_data)

    def test_acq_mas(self):
        # Test of mean and median mas, count, system and acquisition protocol names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionMeanmAs = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Acquisition name and system name test
        acq_system_names = ["All systems"]
        acq_names = ["ABD_1_VIEW", "AEC"]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"]
        self.check_series_and_category_names(acq_names, acq_system_names, chart_data)

        # Check on mean data
        acq_data = [[0.0, 2.70666667, 3.0], [0.0, 9.5, 2.0]]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"][0]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values
        acq_data = [[0.0, 2.04, 3.0], [0.0, 9.5, 2.0]]
        chart_data = self.chart_data["acquisitionMedianmAsData"]["data"][0][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on the boxplot system names
        acq_data = "All systems"
        self.assertEqual(
            self.chart_data["acquisitionBoxplotmAsData"]["data"][0]["name"], acq_data
        )

        # Check the boxplot x and y data values
        acq_x_data = [["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW", "AEC", "AEC"]]
        acq_y_data = [[1.04, 2.04, 5.04, 9.0, 10.0]]
        chart_data = self.chart_data["acquisitionBoxplotmAsData"]["data"]
        self.check_boxplot_xy(acq_x_data, acq_y_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Acquisition name and system name test
        acq_system_names = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        acq_names = ["ABD_1_VIEW", "AEC"]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"]
        self.check_series_and_category_names(acq_names, acq_system_names, chart_data)

        # Check on mean data of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 9.5, 2.0]]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"][0]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on mean data of series 1
        acq_data = [[1.0, 2.70666667, 3.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"][1]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 9.5, 2.0]]
        chart_data = self.chart_data["acquisitionMedianmAsData"]["data"][0][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values of series 1
        acq_data = [[1.0, 2.04, 3.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMedianmAsData"]["data"][1][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on the boxplot data system names
        acq_data = ["Carestream Clinic KODAK7500", "Digital Mobile Hospital 01234MOB54"]
        self.assertEqual(
            self.chart_data["acquisitionBoxplotmAsData"]["data"][0]["name"], acq_data[0]
        )
        self.assertEqual(
            self.chart_data["acquisitionBoxplotmAsData"]["data"][1]["name"], acq_data[1]
        )

        # Check the boxplot x and y data values
        acq_x_data = [["AEC", "AEC"], ["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW"]]
        acq_y_data = [[9.0, 10.0], [1.04, 2.04, 5.04]]
        chart_data = self.chart_data["acquisitionBoxplotmAsData"]["data"]
        self.check_boxplot_xy(acq_x_data, acq_y_data, chart_data)

    def test_acq_freq(self):
        # Test of mean and median DAP, count, system and acquisition protocol names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionFreq = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Acquisition name and system name test
        acq_system_names = ["All systems"]
        acq_names = ["ABD_1_VIEW", "AEC"]
        chart_data = self.chart_data["acquisitionFrequencyData"]["data"]
        self.check_series_and_category_names(acq_system_names, acq_names, chart_data)

        # The frequency chart - frequencies
        acq_data = [[3], [2]]
        chart_data = self.chart_data["acquisitionFrequencyData"]["data"]
        self.check_frequencies(acq_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Acquisition name and system name test
        acq_system_names = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        acq_names = ["ABD_1_VIEW", "AEC"]
        chart_data = self.chart_data["acquisitionFrequencyData"]["data"]
        self.check_series_and_category_names(acq_system_names, acq_names, chart_data)

        # The frequency chart - frequencies
        acq_data = [[0, 3], [2, 0]]
        chart_data = self.chart_data["acquisitionFrequencyData"]["data"]
        self.check_frequencies(acq_data, chart_data)

    def test_request_dap(self):
        # Test of mean and median DAP, count, system and requested procedure protocol names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXRequestMeanDAP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # requested procedure name and system name test
        request_system_names = ["All systems"]
        request_names = ["Blank"]
        chart_data = self.chart_data["requestMeanDAPData"]["data"]
        self.check_series_and_category_names(
            request_names, request_system_names, chart_data
        )

        # Check on mean DAP values and counts
        request_data = [[0.0, 122.25, 2.0]]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on median DAP values and counts
        request_data = [[0.0, 122.25, 2.0]]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on the boxplot data system names
        request_data = "All systems"
        self.assertEqual(
            self.chart_data["requestBoxplotDAPData"]["data"][0]["name"], request_data
        )

        # Check the boxplot x and y data values
        request_x_data = [["Blank", "Blank"]]
        request_y_data = [[32.8, 211.7]]
        chart_data = self.chart_data["requestBoxplotDAPData"]["data"]
        self.check_boxplot_xy(request_x_data, request_y_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # request name and system name test
        request_system_names = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        request_names = ["Blank"]
        chart_data = self.chart_data["requestMeanDAPData"]["data"]
        self.check_series_and_category_names(
            request_names, request_system_names, chart_data
        )

        # Check on mean data of series 0
        request_data = [[0.0, 211.7, 1.0]]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on mean data of series 1
        request_data = [[1.0, 32.8, 1.0]]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on median values of series 0
        request_data = [[0.0, 211.7, 1.0]]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on median values of series 1
        request_data = [[1.0, 32.8, 1.0]]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on the boxplot data system names
        request_data = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        self.assertEqual(
            self.chart_data["requestBoxplotDAPData"]["data"][0]["name"], request_data[0]
        )
        self.assertEqual(
            self.chart_data["requestBoxplotDAPData"]["data"][1]["name"], request_data[1]
        )

        # Check the boxplot x and y data values
        request_x_data = [["Blank"], ["Blank"]]
        request_y_data = [[211.7], [32.8]]
        chart_data = self.chart_data["requestBoxplotDAPData"]["data"]
        self.check_boxplot_xy(request_x_data, request_y_data, chart_data)

    def test_request_freq(self):
        # Test of mean and median DAP, count, system and requested procedure names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXRequestFreq = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Request name and system name test
        request_system_names = ["All systems"]
        request_names = ["Blank"]
        chart_data = self.chart_data["requestFrequencyData"]["data"]
        self.check_series_and_category_names(
            request_system_names, request_names, chart_data
        )

        # The frequency chart - frequencies
        request_data = [[2]]
        chart_data = self.chart_data["requestFrequencyData"]["data"]
        self.check_frequencies(request_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # request name and system name test
        request_system_names = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        request_names = ["Blank"]
        chart_data = self.chart_data["requestFrequencyData"]["data"]
        self.check_series_and_category_names(
            request_system_names, request_names, chart_data
        )

        # The frequency chart - frequencies
        request_data = [[1, 1]]
        chart_data = self.chart_data["requestFrequencyData"]["data"]
        self.check_frequencies(request_data, chart_data)

    def test_study_dap(self):
        # Test of mean and median DAP, count, system and study description protocol names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXStudyMeanDAP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # study description name and system name test
        study_system_names = ["All systems"]
        study_names = ["AEC", "Abdomen"]
        chart_data = self.chart_data["studyMeanDAPData"]["data"]
        self.check_series_and_category_names(
            study_names, study_system_names, chart_data
        )

        # Check on mean DAP values and counts
        study_data = [[0.0, 211.7, 1.0], [0.0, 32.8, 1.0]]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on median DAP values and counts
        study_data = [[0.0, 211.7, 1.0], [0.0, 32.8, 1.0]]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on the boxplot data system names
        study_data = "All systems"
        self.assertEqual(
            self.chart_data["studyBoxplotDAPData"]["data"][0]["name"], study_data
        )

        # Check the boxplot x and y data values
        study_x_data = [["AEC", "Abdomen"]]
        study_y_data = [[32.8, 211.7]]
        chart_data = self.chart_data["studyBoxplotDAPData"]["data"]
        self.check_boxplot_xy(study_x_data, study_y_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # study name and system name test
        study_system_names = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        study_names = ["AEC", "Abdomen"]
        chart_data = self.chart_data["studyMeanDAPData"]["data"]
        self.check_series_and_category_names(
            study_names, study_system_names, chart_data
        )

        # Check on mean data of series 0
        study_data = [[0.0, 211.7, 1.0]]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on mean data of series 1
        study_data = [[0.0, np.nan, 0.0]]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on median values of series 0
        study_data = [[0.0, 211.7, 1.0]]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on median values of series 1
        study_data = [[0.0, np.nan, 0.0]]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on the boxplot data system names
        study_data = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        self.assertEqual(
            self.chart_data["studyBoxplotDAPData"]["data"][0]["name"], study_data[0]
        )
        self.assertEqual(
            self.chart_data["studyBoxplotDAPData"]["data"][1]["name"], study_data[1]
        )

        # Check the boxplot x and y data values
        study_x_data = [["AEC"], ["Abdomen"]]
        study_y_data = [[211.7], [32.8]]
        chart_data = self.chart_data["studyBoxplotDAPData"]["data"]
        self.check_boxplot_xy(study_x_data, study_y_data, chart_data)

    def test_study_freq(self):
        # Test of mean and median DAP, count, system and study description names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXStudyFreq = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # study name and system name test
        study_system_names = ["All systems"]
        study_names = ["AEC", "Abdomen"]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        self.check_series_and_category_names(
            study_system_names, study_names, chart_data
        )

        # The frequency chart - frequencies
        study_data = [[1], [1]]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        self.check_frequencies(study_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # study name and system name test
        study_system_names = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
        ]
        study_names = ["AEC", "Abdomen"]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        self.check_series_and_category_names(
            study_system_names, study_names, chart_data
        )

        # The frequency chart - frequencies
        study_data = [[1, 0], [0, 1]]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        self.check_frequencies(study_data, chart_data)
