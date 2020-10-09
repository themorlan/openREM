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
        dx_carestream_dxr_1 = os.path.join("test_files", "DX-Im-Carestream_DRX.dcm")
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
        dx.dx(os.path.join(root_tests, dx_carestream_dxr_1))
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
            chart_y_data = chart_data[i]["y"]
            chart_x_data = chart_data[i]["x"]

            chart_x_data = [x for _, x in sorted(zip(chart_y_data, chart_x_data))]
            chart_y_data = sorted(chart_y_data)

            for j in range(len(chart_x_data)):
                self.assertEqual(chart_x_data[j], x_data[i][j])

                self.assertAlmostEqual(
                    chart_y_data[j],
                    y_data[i][j],
                )

    def test_required_charts(self):
        from remapp.views_charts_dx import generate_required_dx_charts_list

        f = self.login_get_filterset()

        # Set user profile options to use all charts
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.plotDXAcquisitionMeanDAP = True
        self.user.userprofile.plotDXAcquisitionFreq = True
        self.user.userprofile.plotDXAcquisitionDAPvsMass = True
        self.user.userprofile.plotDXStudyMeanDAP = True
        self.user.userprofile.plotDXStudyFreq = True
        self.user.userprofile.plotDXStudyDAPvsMass = True
        self.user.userprofile.plotDXRequestMeanDAP = True
        self.user.userprofile.plotDXRequestFreq = True
        self.user.userprofile.plotDXRequestDAPvsMass = True
        self.user.userprofile.plotDXAcquisitionMeankVp = True
        self.user.userprofile.plotDXAcquisitionMeanmAs = True
        self.user.userprofile.plotDXStudyPerDayAndHour = True
        self.user.userprofile.plotDXAcquisitionMeankVpOverTime = True
        self.user.userprofile.plotDXAcquisitionMeanmAsOverTime = True
        self.user.userprofile.plotDXAcquisitionMeanDAPOverTime = True
        self.user.userprofile.save()

        required_charts_list = generate_required_dx_charts_list(self.user.userprofile)

        chart_var_names = []
        for item in required_charts_list:
            chart_var_names.append(item["var_name"])

        # Just check the variable names - I don't mind if the titles change
        reference_var_names = [
            "acquisitionMeanDAP",
            "acquisitionMedianDAP",
            "acquisitionBoxplotDAP",
            "acquisitionHistogramDAP",
            "acquisitionFrequency",
            "studyMeanDAP",
            "studyMedianDAP",
            "studyBoxplotDAP",
            "studyHistogramDAP",
            "studyFrequency",
            "requestMeanDAP",
            "requestMedianDAP",
            "requestBoxplotDAP",
            "requestHistogramDAP",
            "requestFrequency",
            "acquisitionMeankVp",
            "acquisitionMediankVp",
            "acquisitionBoxplotkVp",
            "acquisitionHistogramkVp",
            "acquisitionMeanmAs",
            "acquisitionMedianmAs",
            "acquisitionBoxplotmAs",
            "acquisitionHistogrammAs",
            "studyWorkload",
            "acquisitionMeankVpOverTime",
            "acquisitionMediankVpOverTime",
            "acquisitionMeanmAsOverTime",
            "acquisitionMedianmAsOverTime",
            "acquisitionMeanDAPOverTime",
            "acquisitionMedianDAPOverTime",
            "acquisitionDAPvsMass",
            "studyDAPvsMass",
            "requestDAPvsMass",
        ]

        for ref_var_name in reference_var_names:
            self.assertTrue(ref_var_name in chart_var_names)

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
        acq_names = ["ABD_1_VIEW", "AEC", "AP"]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"]
        self.check_series_and_category_names(acq_names, acq_system_names, chart_data)

        # Check on mean DAP values and counts
        acq_data = [[0.0, 10.93333333, 3.0], [0.0, 105.85, 2.0], [0.0, 6.33, 1.0]]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median DAP values and counts
        acq_data = [[0.0, 8.2, 3.0], [0.0, 105.85, 2.0], [0.0, 6.33, 1.0]]
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
        acq_x_data = [["ABD_1_VIEW", "AP", "ABD_1_VIEW", "ABD_1_VIEW", "AEC", "AEC"]]
        acq_y_data = [[4.1, 6.33, 8.2, 20.5, 101.57, 110.13]]
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
            "LICARDR0004",
        ]
        acq_names = ["ABD_1_VIEW", "AEC", "AP"]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"]
        self.check_series_and_category_names(acq_names, acq_system_names, chart_data)

        # Check on mean data of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 105.85, 2.0], [0.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on mean data of series 1
        acq_data = [[1.0, 10.93333333, 3.0], [1.0, np.nan, 0.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on mean data of series 2
        acq_data = [[2.0, np.nan, 0.0], [2.0, np.nan, 0.0], [2.0, 6.33, 1.0]]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"][2]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 105.85, 2.0], [0.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMedianDAPData"]["data"][0][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values of series 1
        acq_data = [[1.0, 8.2, 3.0], [1.0, np.nan, 0.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMedianDAPData"]["data"][1][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values of series 2
        acq_data = [[2.0, np.nan, 0.0], [2.0, np.nan, 0.0], [2.0, 6.33, 1.0]]
        chart_data = self.chart_data["acquisitionMedianDAPData"]["data"][2][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on the boxplot data system names
        acq_data = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
            "LICARDR0004",
        ]
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["name"], acq_data[0]
        )
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][1]["name"], acq_data[1]
        )
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][2]["name"], acq_data[2]
        )

        # Check the boxplot x and y data values
        acq_x_data = [
            ["AEC", "AEC"],
            ["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW"],
            ["AP"],
        ]
        acq_y_data = [[101.57, 110.13], [4.1, 8.2, 20.5], [6.33]]
        chart_data = self.chart_data["acquisitionBoxplotDAPData"]["data"]
        self.check_boxplot_xy(acq_x_data, acq_y_data, chart_data)

    def test_acq_dap_histogram(self):
        # Test of DAP histogram
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionMeanDAP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["acquisitionHistogramDAPData"]["data"]

        standard_data = [
            {
                "name": "ABD_1_VIEW",
                "text": np.array(
                    [
                        "4.10 to 14.70",
                        "14.70 to 25.31",
                        "25.31 to 35.91",
                        "35.91 to 46.51",
                        "46.51 to 57.12",
                        "57.12 to 67.72",
                        "67.72 to 78.32",
                        "78.32 to 88.92",
                        "88.92 to 99.53",
                        "99.53 to 110.13",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        9.4015,
                        20.0045,
                        30.6075,
                        41.2105,
                        51.8135,
                        62.4165,
                        73.0195,
                        83.6225,
                        94.2255,
                        104.8285,
                    ]
                ),
                "y": np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            },
            {
                "name": "AEC",
                "text": np.array(
                    [
                        "4.10 to 14.70",
                        "14.70 to 25.31",
                        "25.31 to 35.91",
                        "35.91 to 46.51",
                        "46.51 to 57.12",
                        "57.12 to 67.72",
                        "67.72 to 78.32",
                        "78.32 to 88.92",
                        "88.92 to 99.53",
                        "99.53 to 110.13",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        9.4015,
                        20.0045,
                        30.6075,
                        41.2105,
                        51.8135,
                        62.4165,
                        73.0195,
                        83.6225,
                        94.2255,
                        104.8285,
                    ]
                ),
                "y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2]),
            },
            {
                "name": "AP",
                "text": np.array(
                    [
                        "4.10 to 14.70",
                        "14.70 to 25.31",
                        "25.31 to 35.91",
                        "35.91 to 46.51",
                        "46.51 to 57.12",
                        "57.12 to 67.72",
                        "67.72 to 78.32",
                        "78.32 to 88.92",
                        "88.92 to 99.53",
                        "99.53 to 110.13",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        9.4015,
                        20.0045,
                        30.6075,
                        41.2105,
                        51.8135,
                        62.4165,
                        73.0195,
                        83.6225,
                        94.2255,
                        104.8285,
                    ]
                ),
                "y": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            },
        ]

        for idx, dataset in enumerate(standard_data):
            self.assertEqual(chart_data[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(chart_data[idx]["x"], dataset["x"])
            np.testing.assert_equal(chart_data[idx]["y"], dataset["y"])

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
        acq_names = ["ABD_1_VIEW", "AEC", "AP"]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"]
        self.check_series_and_category_names(acq_names, acq_system_names, chart_data)

        # Check on mean data
        acq_data = [[0.0, 2.70666667, 3.0], [0.0, 9.5, 2.0], [0.0, 1.0, 1.0]]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"][0]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values
        acq_data = [[0.0, 2.04, 3.0], [0.0, 9.5, 2.0], [0.0, 1.0, 1.0]]
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
        acq_x_data = [["AP", "ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW", "AEC", "AEC"]]
        acq_y_data = [[1.00, 1.04, 2.04, 5.04, 9.0, 10.0]]
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
            "LICARDR0004",
        ]
        acq_names = ["ABD_1_VIEW", "AEC", "AP"]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"]
        self.check_series_and_category_names(acq_names, acq_system_names, chart_data)

        # Check on mean data of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 9.5, 2.0], [0.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"][0]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on mean data of series 1
        acq_data = [[1.0, 2.70666667, 3.0], [1.0, np.nan, 0.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"][1]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on mean data of series 2
        acq_data = [[2.0, np.nan, 0.0], [2.0, np.nan, 0.0], [2.0, 1.00, 1.0]]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"][2]["customdata"]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values of series 0
        acq_data = [[0.0, np.nan, 0.0], [0.0, 9.5, 2.0], [0.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMedianmAsData"]["data"][0][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median values of series 1
        acq_data = [[1.0, 2.04, 3.0], [1.0, np.nan, 0.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["acquisitionMedianmAsData"]["data"][1][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on median data of series 2
        acq_data = [[2.0, np.nan, 0.0], [2.0, np.nan, 0.0], [2.0, 1.00, 1.0]]
        chart_data = self.chart_data["acquisitionMedianmAsData"]["data"][2][
            "customdata"
        ]
        self.check_avg_and_counts(acq_data, chart_data)

        # Check on the boxplot data system names
        acq_data = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
            "LICARDR0004",
        ]
        self.assertEqual(
            self.chart_data["acquisitionBoxplotmAsData"]["data"][0]["name"], acq_data[0]
        )
        self.assertEqual(
            self.chart_data["acquisitionBoxplotmAsData"]["data"][1]["name"], acq_data[1]
        )
        self.assertEqual(
            self.chart_data["acquisitionBoxplotmAsData"]["data"][2]["name"], acq_data[2]
        )

        # Check the boxplot x and y data values
        acq_x_data = [
            ["AEC", "AEC"],
            ["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW"],
            ["AP"],
        ]
        acq_y_data = [[9.0, 10.0], [1.04, 2.04, 5.04], [1.0]]
        chart_data = self.chart_data["acquisitionBoxplotmAsData"]["data"]
        self.check_boxplot_xy(acq_x_data, acq_y_data, chart_data)

    def test_acq_mas_histogram(self):
        # Test of mAs histogram
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionMeanmAs = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["acquisitionHistogrammAsData"]["data"]

        standard_data = [
            {
                "name": "ABD_1_VIEW",
                "text": np.array(
                    [
                        "1.00 to 1.90",
                        "1.90 to 2.80",
                        "2.80 to 3.70",
                        "3.70 to 4.60",
                        "4.60 to 5.50",
                        "5.50 to 6.40",
                        "6.40 to 7.30",
                        "7.30 to 8.20",
                        "8.20 to 9.10",
                        "9.10 to 10.00",
                    ],
                    dtype="<U13",
                ),
                "x": np.array(
                    [1.45, 2.35, 3.25, 4.15, 5.05, 5.95, 6.85, 7.75, 8.65, 9.55]
                ),
                "y": np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0]),
            },
            {
                "name": "AEC",
                "text": np.array(
                    [
                        "1.00 to 1.90",
                        "1.90 to 2.80",
                        "2.80 to 3.70",
                        "3.70 to 4.60",
                        "4.60 to 5.50",
                        "5.50 to 6.40",
                        "6.40 to 7.30",
                        "7.30 to 8.20",
                        "8.20 to 9.10",
                        "9.10 to 10.00",
                    ],
                    dtype="<U13",
                ),
                "x": np.array(
                    [1.45, 2.35, 3.25, 4.15, 5.05, 5.95, 6.85, 7.75, 8.65, 9.55]
                ),
                "y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            },
            {
                "name": "AP",
                "text": np.array(
                    [
                        "1.00 to 1.90",
                        "1.90 to 2.80",
                        "2.80 to 3.70",
                        "3.70 to 4.60",
                        "4.60 to 5.50",
                        "5.50 to 6.40",
                        "6.40 to 7.30",
                        "7.30 to 8.20",
                        "8.20 to 9.10",
                        "9.10 to 10.00",
                    ],
                    dtype="<U13",
                ),
                "x": np.array(
                    [1.45, 2.35, 3.25, 4.15, 5.05, 5.95, 6.85, 7.75, 8.65, 9.55]
                ),
                "y": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            },
        ]

        for idx, dataset in enumerate(standard_data):
            self.assertEqual(chart_data[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(chart_data[idx]["x"], dataset["x"])
            np.testing.assert_equal(chart_data[idx]["y"], dataset["y"])

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
        acq_names = ["ABD_1_VIEW", "AEC", "AP"]
        chart_data = self.chart_data["acquisitionFrequencyData"]["data"]
        self.check_series_and_category_names(acq_system_names, acq_names, chart_data)

        # The frequency chart - frequencies
        acq_data = [[3], [2], [1]]
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
            "LICARDR0004",
        ]
        acq_names = ["ABD_1_VIEW", "AEC", "AP"]
        chart_data = self.chart_data["acquisitionFrequencyData"]["data"]
        self.check_series_and_category_names(acq_system_names, acq_names, chart_data)

        # The frequency chart - frequencies
        acq_data = [[0, 3, 0], [2, 0, 0], [0, 0, 1]]
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
        request_names = ["Blank", "XR CHEST"]
        chart_data = self.chart_data["requestMeanDAPData"]["data"]
        self.check_series_and_category_names(
            request_names, request_system_names, chart_data
        )

        # Check on mean DAP values and counts
        request_data = [[0.0, 122.25, 2.0], [0.0, 6.33, 1.0]]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on median DAP values and counts - series 0
        request_data = [[0.0, 122.25, 2.0], [0.0, 6.33, 1.0]]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on the boxplot data system names
        request_data = "All systems"
        self.assertEqual(
            self.chart_data["requestBoxplotDAPData"]["data"][0]["name"], request_data
        )

        # Check the boxplot x and y data values
        request_x_data = [["XR CHEST", "Blank", "Blank"]]
        request_y_data = [[6.33, 32.8, 211.7]]
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
            "LICARDR0004",
        ]
        request_names = ["Blank", "XR CHEST"]
        chart_data = self.chart_data["requestMeanDAPData"]["data"]
        self.check_series_and_category_names(
            request_names, request_system_names, chart_data
        )

        # Check on mean data of series 0
        request_data = [[0.0, 211.7, 1.0], [0.0, np.nan, 0.0]]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on mean data of series 1
        request_data = [[1.0, 32.8, 1.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on mean data of series 2
        request_data = [[2.0, np.nan, 0.0], [2.0, 6.33, 1.0]]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][2]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on median values of series 0
        request_data = [[0.0, 211.7, 1.0], [0.0, np.nan, 0.0]]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on median values of series 1
        request_data = [[1.0, 32.8, 1.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on median data of series 2
        request_data = [[2.0, np.nan, 0.0], [2.0, 6.33, 1.0]]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][2]["customdata"]
        self.check_avg_and_counts(request_data, chart_data)

        # Check on the boxplot data system names
        request_data = [
            "Carestream Clinic KODAK7500",
            "Digital Mobile Hospital 01234MOB54",
            "LICARDR0004",
        ]
        self.assertEqual(
            self.chart_data["requestBoxplotDAPData"]["data"][0]["name"], request_data[0]
        )
        self.assertEqual(
            self.chart_data["requestBoxplotDAPData"]["data"][1]["name"], request_data[1]
        )
        self.assertEqual(
            self.chart_data["requestBoxplotDAPData"]["data"][2]["name"], request_data[2]
        )

        # Check the boxplot x and y data values
        request_x_data = [["Blank"], ["Blank"], ["XR CHEST"]]
        request_y_data = [[211.7], [32.8], [6.33]]
        chart_data = self.chart_data["requestBoxplotDAPData"]["data"]
        self.check_boxplot_xy(request_x_data, request_y_data, chart_data)

    def test_request_dap_histogram(self):
        # Test of DAP histogram
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXRequestMeanDAP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["requestHistogramDAPData"]["data"]

        standard_data = [
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "6.33 to 26.87",
                        "26.87 to 47.40",
                        "47.40 to 67.94",
                        "67.94 to 88.48",
                        "88.48 to 109.02",
                        "109.02 to 129.55",
                        "129.55 to 150.09",
                        "150.09 to 170.63",
                        "170.63 to 191.16",
                        "191.16 to 211.70",
                    ],
                    dtype="<U16",
                ),
                "x": np.array(
                    [
                        16.5985,
                        37.1355,
                        57.6725,
                        78.2095,
                        98.7465,
                        119.2835,
                        139.8205,
                        160.3575,
                        180.8945,
                        201.4315,
                    ]
                ),
                "y": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1]),
            },
            {
                "name": "XR CHEST",
                "text": np.array(
                    [
                        "6.33 to 26.87",
                        "26.87 to 47.40",
                        "47.40 to 67.94",
                        "67.94 to 88.48",
                        "88.48 to 109.02",
                        "109.02 to 129.55",
                        "129.55 to 150.09",
                        "150.09 to 170.63",
                        "170.63 to 191.16",
                        "191.16 to 211.70",
                    ],
                    dtype="<U16",
                ),
                "x": np.array(
                    [
                        16.5985,
                        37.1355,
                        57.6725,
                        78.2095,
                        98.7465,
                        119.2835,
                        139.8205,
                        160.3575,
                        180.8945,
                        201.4315,
                    ]
                ),
                "y": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            },
        ]

        for idx, dataset in enumerate(standard_data):
            self.assertEqual(chart_data[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(chart_data[idx]["x"], dataset["x"])
            np.testing.assert_equal(chart_data[idx]["y"], dataset["y"])

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
        request_names = ["Blank", "XR CHEST"]
        chart_data = self.chart_data["requestFrequencyData"]["data"]
        self.check_series_and_category_names(
            request_system_names, request_names, chart_data
        )

        # The frequency chart - frequencies
        request_data = [[2], [1]]
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
            "LICARDR0004",
        ]
        request_names = ["Blank", "XR CHEST"]
        chart_data = self.chart_data["requestFrequencyData"]["data"]
        self.check_series_and_category_names(
            request_system_names, request_names, chart_data
        )

        # The frequency chart - frequencies
        request_data = [[1, 1, 0], [0, 0, 1]]
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
        study_names = ["AEC", "Abdomen", "XR CHEST"]
        chart_data = self.chart_data["studyMeanDAPData"]["data"]
        self.check_series_and_category_names(
            study_names, study_system_names, chart_data
        )

        # Check on mean DAP values and counts
        study_data = [[0.0, 211.7, 1.0], [0.0, 32.8, 1.0], [0.0, 6.33, 1.0]]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on median DAP values and counts
        study_data = [[0.0, 211.7, 1.0], [0.0, 32.8, 1.0], [0.0, 6.33, 1.0]]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on the boxplot data system names
        study_data = "All systems"
        self.assertEqual(
            self.chart_data["studyBoxplotDAPData"]["data"][0]["name"], study_data
        )

        # Check the boxplot x and y data values
        study_x_data = [["XR CHEST", "Abdomen", "AEC"]]
        study_y_data = [[6.33, 32.8, 211.7]]
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
            "LICARDR0004",
        ]
        study_names = ["AEC", "Abdomen", "XR CHEST"]
        chart_data = self.chart_data["studyMeanDAPData"]["data"]
        self.check_series_and_category_names(
            study_names, study_system_names, chart_data
        )

        # Check on mean data of series 0
        study_data = [[0.0, 211.7, 1.0], [0.0, np.nan, 0.0], [0.0, np.nan, 0.0]]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on mean data of series 1
        study_data = [[0.0, np.nan, 0.0], [0.0, 32.8, 1.0], [0.0, np.nan, 0.0]]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on mean data of series 2
        study_data = [[2.0, np.nan, 0.0], [2.0, np.nan, 0.0], [2.0, 6.33, 1.0]]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][2]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on median values of series 0
        study_data = [[0.0, 211.7, 1.0], [0.0, np.nan, 0.0], [0.0, np.nan, 0.0]]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][0]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on median values of series 1
        study_data = [[1.0, np.nan, 0.0], [1.0, 32.8, 1.0], [1.0, np.nan, 0.0]]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][1]["customdata"]
        self.check_avg_and_counts(study_data, chart_data)

        # Check on median data of series 2
        study_data = [[2.0, np.nan, 0.0], [2.0, np.nan, 0.0], [2.0, 6.33, 1.0]]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][2]["customdata"]
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

    def test_study_dap_histogram(self):
        # Test of DAP histogram
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXStudyMeanDAP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["studyHistogramDAPData"]["data"]

        standard_data = [
            {
                "name": "AEC",
                "text": np.array(
                    [
                        "6.33 to 26.87",
                        "26.87 to 47.40",
                        "47.40 to 67.94",
                        "67.94 to 88.48",
                        "88.48 to 109.02",
                        "109.02 to 129.55",
                        "129.55 to 150.09",
                        "150.09 to 170.63",
                        "170.63 to 191.16",
                        "191.16 to 211.70",
                    ],
                    dtype="<U16",
                ),
                "x": np.array(
                    [
                        16.5985,
                        37.1355,
                        57.6725,
                        78.2095,
                        98.7465,
                        119.2835,
                        139.8205,
                        160.3575,
                        180.8945,
                        201.4315,
                    ]
                ),
                "y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            },
            {
                "name": "Abdomen",
                "text": np.array(
                    [
                        "6.33 to 26.87",
                        "26.87 to 47.40",
                        "47.40 to 67.94",
                        "67.94 to 88.48",
                        "88.48 to 109.02",
                        "109.02 to 129.55",
                        "129.55 to 150.09",
                        "150.09 to 170.63",
                        "170.63 to 191.16",
                        "191.16 to 211.70",
                    ],
                    dtype="<U16",
                ),
                "x": np.array(
                    [
                        16.5985,
                        37.1355,
                        57.6725,
                        78.2095,
                        98.7465,
                        119.2835,
                        139.8205,
                        160.3575,
                        180.8945,
                        201.4315,
                    ]
                ),
                "y": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            },
            {
                "name": "XR CHEST",
                "text": np.array(
                    [
                        "6.33 to 26.87",
                        "26.87 to 47.40",
                        "47.40 to 67.94",
                        "67.94 to 88.48",
                        "88.48 to 109.02",
                        "109.02 to 129.55",
                        "129.55 to 150.09",
                        "150.09 to 170.63",
                        "170.63 to 191.16",
                        "191.16 to 211.70",
                    ],
                    dtype="<U16",
                ),
                "x": np.array(
                    [
                        16.5985,
                        37.1355,
                        57.6725,
                        78.2095,
                        98.7465,
                        119.2835,
                        139.8205,
                        160.3575,
                        180.8945,
                        201.4315,
                    ]
                ),
                "y": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            },
        ]

        for idx, dataset in enumerate(standard_data):
            self.assertEqual(chart_data[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(chart_data[idx]["x"], dataset["x"])
            np.testing.assert_equal(chart_data[idx]["y"], dataset["y"])

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
        study_names = ["AEC", "Abdomen", "XR CHEST"]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        self.check_series_and_category_names(
            study_system_names, study_names, chart_data
        )

        # The frequency chart - frequencies
        study_data = [[1], [1], [1]]
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
            "LICARDR0004",
        ]
        study_names = ["AEC", "Abdomen", "XR CHEST"]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        self.check_series_and_category_names(
            study_system_names, study_names, chart_data
        )

        # The frequency chart - frequencies
        study_data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        self.check_frequencies(study_data, chart_data)
