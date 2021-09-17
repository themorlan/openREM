# This Python file uses the following encoding: utf-8
# test_charts_dx.py

import os
from django.contrib.auth.models import User, Group
from django.test import TestCase, RequestFactory
from remapp.extractors import dx
from remapp.models import GeneralStudyModuleAttr, PatientIDSettings
from django.db.models import Q
from remapp.interface.mod_filters import DXSummaryListFilter
from remapp.tests.test_charts_common import (
    check_series_and_category_names,
    check_frequencies,
    check_boxplot_xy,
    check_boxplot_data,
    check_average_data,
    check_avg_and_counts,
    user_profile_reset,
)
import numpy as np


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
        user_profile_reset(self)
        return f

    def obtain_chart_data(self, f):
        from remapp.views_charts_dx import dx_plot_calculations

        self.chart_data = dx_plot_calculations(
            f, self.user.userprofile, return_as_dict=True
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
        check_series_and_category_names(self, acq_names, acq_system_names, chart_data)

        # Check on mean DAP values and counts
        acq_data = [[0.0, 10.93, 3], [0.0, 105.85, 2], [0.0, 6.33, 1]]
        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"][0]
        check_avg_and_counts(self, acq_data, chart_data)

        # Check on median DAP values and counts
        acq_data = [[0.0, 8.2, 3], [0.0, 105.85, 2], [0.0, 6.33, 1]]
        chart_data = self.chart_data["acquisitionMedianDAPData"]["data"][0]
        check_avg_and_counts(self, acq_data, chart_data)

        # Check on the boxplot data system names
        acq_data = "All systems"
        self.assertEqual(
            self.chart_data["acquisitionBoxplotDAPData"]["data"][0]["name"], acq_data
        )

        # Check the boxplot x and y data values
        acq_x_data = [["ABD_1_VIEW", "AP", "ABD_1_VIEW", "ABD_1_VIEW", "AEC", "AEC"]]
        acq_y_data = [[4.1, 6.33, 8.2, 20.5, 101.57, 110.13]]
        chart_data = self.chart_data["acquisitionBoxplotDAPData"]["data"]
        check_boxplot_xy(self, acq_x_data, acq_y_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Check the mean
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["Digital Mobile Hospital 01234MOB54", 3],
                        ["Digital Mobile Hospital 01234MOB54", 0],
                        ["Digital Mobile Hospital 01234MOB54", 0],
                    ],
                    dtype=object,
                ),
                "name": "Digital Mobile Hospital 01234MOB54",
                "x": np.array(
                    ["ABD_1_VIEW", "AEC", "AP"],
                    dtype=object,
                ),
                "y": np.array([10.933333, np.nan, np.nan]),
            },
            {
                "customdata": np.array(
                    [
                        ["Carestream Clinic KODAK7500", 0],
                        ["Carestream Clinic KODAK7500", 2],
                        ["Carestream Clinic KODAK7500", 0],
                    ],
                    dtype=object,
                ),
                "name": "Carestream Clinic KODAK7500",
                "x": np.array(
                    ["ABD_1_VIEW", "AEC", "AP"],
                    dtype=object,
                ),
                "y": np.array([np.nan, 105.85, np.nan]),
            },
            {
                "customdata": np.array(
                    [["LICARDR0004", 0], ["LICARDR0004", 0], ["LICARDR0004", 1]],
                    dtype=object,
                ),
                "name": "LICARDR0004",
                "x": np.array(
                    ["ABD_1_VIEW", "AEC", "AP"],
                    dtype=object,
                ),
                "y": np.array([np.nan, np.nan, 6.33]),
            },
        ]

        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"]
        check_average_data(self, chart_data, standard_data)

        # Check the median
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["Digital Mobile Hospital 01234MOB54", 3],
                        ["Digital Mobile Hospital 01234MOB54", 0],
                        ["Digital Mobile Hospital 01234MOB54", 0],
                    ],
                    dtype=object,
                ),
                "name": "Digital Mobile Hospital 01234MOB54",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"]),
                "y": np.array([8.2, np.nan, np.nan]),
            },
            {
                "customdata": np.array(
                    [
                        ["Carestream Clinic KODAK7500", 0],
                        ["Carestream Clinic KODAK7500", 2],
                        ["Carestream Clinic KODAK7500", 0],
                    ],
                    dtype=object,
                ),
                "name": "Carestream Clinic KODAK7500",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"]),
                "y": np.array(
                    [np.nan, 105.85, np.nan],
                ),
            },
            {
                "customdata": np.array(
                    [["LICARDR0004", 0], ["LICARDR0004", 0], ["LICARDR0004", 1]],
                    dtype=object,
                ),
                "name": "LICARDR0004",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"]),
                "y": np.array([np.nan, np.nan, 6.33]),
            },
        ]

        chart_data = self.chart_data["acquisitionMedianDAPData"]["data"]
        check_average_data(self, chart_data, standard_data)

        # Check the box plot
        standard_data = [
            {
                "name": "Digital Mobile Hospital 01234MOB54",
                "x": np.array(
                    [
                        "ABD_1_VIEW",
                        "AEC",
                        "AP",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [10.933333, np.nan, np.nan],
                ),
            },
            {
                "name": "Carestream Clinic KODAK7500",
                "x": np.array(
                    ["ABD_1_VIEW", "AEC", "AP"],
                    dtype=object,
                ),
                "y": np.array([np.nan, 105.85, np.nan]),
            },
            {
                "name": "LICARDR0004",
                "x": np.array(
                    ["ABD_1_VIEW", "AEC", "AP"],
                    dtype=object,
                ),
                "y": np.array([np.nan, np.nan, 6.33]),
            },
        ]

        chart_data = self.chart_data["acquisitionMeanDAPData"]["data"]

        check_boxplot_data(self, chart_data, standard_data)

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
            np.testing.assert_almost_equal(
                chart_data[idx]["x"], dataset["x"], decimal=4
            )
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
        check_series_and_category_names(self, acq_names, acq_system_names, chart_data)

        # Check on mean data
        acq_data = [[0.0, 2.71, 3], [0.0, 9.5, 2], [0.0, 1.0, 1]]
        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"][0]
        check_avg_and_counts(self, acq_data, chart_data)

        # Check on median values
        acq_data = [[0.0, 2.04, 3], [0.0, 9.5, 2], [0.0, 1.0, 1]]
        chart_data = self.chart_data["acquisitionMedianmAsData"]["data"][0]
        check_avg_and_counts(self, acq_data, chart_data)

        # Check on the boxplot system names
        acq_data = "All systems"
        self.assertEqual(
            self.chart_data["acquisitionBoxplotmAsData"]["data"][0]["name"], acq_data
        )

        # Check the boxplot x and y data values
        acq_x_data = [["AP", "ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW", "AEC", "AEC"]]
        acq_y_data = [[1.00, 1.04, 2.04, 5.04, 9.0, 10.0]]
        chart_data = self.chart_data["acquisitionBoxplotmAsData"]["data"]
        check_boxplot_xy(self, acq_x_data, acq_y_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Check the mean, frequency and names
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["Digital Mobile Hospital 01234MOB54", 3],
                        ["Digital Mobile Hospital 01234MOB54", 0],
                        ["Digital Mobile Hospital 01234MOB54", 0],
                    ],
                    dtype=object,
                ),
                "name": "Digital Mobile Hospital 01234MOB54",
                "x": np.array(
                    [
                        "ABD_1_VIEW",
                        "AEC",
                        "AP",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        2.7066667,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "customdata": np.array(
                    [
                        ["Carestream Clinic KODAK7500", 0],
                        ["Carestream Clinic KODAK7500", 2],
                        ["Carestream Clinic KODAK7500", 0],
                    ],
                    dtype=object,
                ),
                "name": "Carestream Clinic KODAK7500",
                "x": np.array(
                    ["ABD_1_VIEW", "AEC", "AP"],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        9.5,
                        np.nan,
                    ]
                ),
            },
            {
                "customdata": np.array(
                    [["LICARDR0004", 0], ["LICARDR0004", 0], ["LICARDR0004", 1]],
                    dtype=object,
                ),
                "name": "LICARDR0004",
                "offsetgroup": "LICARDR0004",
                "x": np.array(
                    ["ABD_1_VIEW", "AEC", "AP"],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        1.0,
                    ]
                ),
            },
        ]

        chart_data = self.chart_data["acquisitionMeanmAsData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Check the median, frequency and names
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["Digital Mobile Hospital 01234MOB54", 3],
                        ["Digital Mobile Hospital 01234MOB54", 0],
                        ["Digital Mobile Hospital 01234MOB54", 0],
                    ],
                    dtype=object,
                ),
                "name": "Digital Mobile Hospital 01234MOB54",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"]),
                "y": np.array([2.04, np.nan, np.nan]),
            },
            {
                "customdata": np.array(
                    [
                        ["Carestream Clinic KODAK7500", 0],
                        ["Carestream Clinic KODAK7500", 2],
                        ["Carestream Clinic KODAK7500", 0],
                    ],
                    dtype=object,
                ),
                "name": "Carestream Clinic KODAK7500",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"]),
                "y": np.array([np.nan, 9.5, np.nan]),
            },
            {
                "customdata": np.array(
                    [["LICARDR0004", 0], ["LICARDR0004", 0], ["LICARDR0004", 1]],
                    dtype=object,
                ),
                "name": "LICARDR0004",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"]),
                "y": np.array([np.nan, np.nan, 1.0]),
            },
        ]

        chart_data = self.chart_data["acquisitionMedianmAsData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Check on the boxplot data system names
        acq_data = [
            "Digital Mobile Hospital 01234MOB54",
            "Carestream Clinic KODAK7500",
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

        # Check the boxplot data
        standard_data = [
            {
                "name": "Digital Mobile Hospital 01234MOB54",
                "x": np.array(["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW"], dtype=object),
                "y": np.array([2.04, 1.04, 5.04]),
            },
            {
                "name": "Carestream Clinic KODAK7500",
                "x": np.array(["AEC", "AEC"], dtype=object),
                "y": np.array([9.0, 10.0]),
            },
            {
                "name": "LICARDR0004",
                "x": np.array(["AP"], dtype=object),
                "y": np.array([1.0]),
            },
        ]

        chart_data = self.chart_data["acquisitionBoxplotmAsData"]["data"]

        check_boxplot_data(self, chart_data, standard_data)

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
            np.testing.assert_almost_equal(
                chart_data[idx]["x"], dataset["x"], decimal=2
            )
            np.testing.assert_equal(chart_data[idx]["y"], dataset["y"])

    def test_acq_kvp(self):
        # Test of mean and median mas, count, system and acquisition protocol names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionMeankVp = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Acquisition average data tests
        chart_data = self.chart_data["acquisitionMeankVpData"]["data"]

        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 69.82, 3],
                        ["All systems", 80.0, 2],
                        ["All systems", 100.0, 1],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"], dtype=object),
                "y": np.array([69.82, 80.0, 100]),
            }
        ]

        check_average_data(self, chart_data, standard_data)

        # Check on the boxplot data
        chart_data = self.chart_data["acquisitionBoxplotkVpData"]["data"]

        standard_data = [
            {
                "name": "All systems",
                "x": np.array(
                    ["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW", "AEC", "AEC", "AP"],
                    dtype=object,
                ),
                "y": np.array([69.64, 69.86, 69.96, 80.0, 80.0, 100.0]),
            }
        ]

        np.testing.assert_equal(chart_data[0]["name"], standard_data[0]["name"])
        check_boxplot_xy(
            self, [standard_data[0]["x"]], [standard_data[0]["y"]], chart_data
        )

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["acquisitionMeankVpData"]["data"]

        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["Digital Mobile Hospital 01234MOB54", 69.82, 3],
                        ["Digital Mobile Hospital 01234MOB54", np.nan, 0],
                        ["Digital Mobile Hospital 01234MOB54", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "Digital Mobile Hospital 01234MOB54",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"], dtype=object),
                "y": np.array([69.82, np.nan, np.nan]),
            },
            {
                "customdata": np.array(
                    [
                        ["Carestream Clinic KODAK7500", np.nan, 0],
                        ["Carestream Clinic KODAK7500", 80.0, 2],
                        ["Carestream Clinic KODAK7500", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "Carestream Clinic KODAK7500",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"], dtype=object),
                "y": np.array([np.nan, 80.0, np.nan]),
            },
            {
                "customdata": np.array(
                    [
                        ["LICARDR0004", np.nan, 0],
                        ["LICARDR0004", np.nan, 0],
                        ["LICARDR0004", 100.0, 1],
                    ],
                    dtype=object,
                ),
                "name": "LICARDR0004",
                "x": np.array(["ABD_1_VIEW", "AEC", "AP"], dtype=object),
                "y": np.array([np.nan, np.nan, 100.0]),
            },
        ]

        check_average_data(self, chart_data, standard_data)

        # Check on the boxplot data
        chart_data = self.chart_data["acquisitionBoxplotkVpData"]["data"]

        standard_data = [
            {
                "name": "Digital Mobile Hospital 01234MOB54",
                "x": np.array(["ABD_1_VIEW", "ABD_1_VIEW", "ABD_1_VIEW"], dtype=object),
                "y": np.array([69.64, 69.86, 69.96]),
            },
            {
                "name": "Carestream Clinic KODAK7500",
                "x": np.array(["AEC", "AEC"], dtype=object),
                "y": np.array([80.0, 80.0]),
            },
            {
                "name": "LICARDR0004",
                "x": np.array(["AP"], dtype=object),
                "y": np.array([100.0]),
            },
        ]

        for idx, dataset in enumerate(standard_data):
            np.testing.assert_equal(chart_data[idx]["name"], dataset["name"])
            check_boxplot_xy(
                self, [list(dataset["x"])], [list(dataset["y"])], [chart_data[idx]]
            )

    def test_acq_kvp_histogram(self):
        # Test of kVp histogram
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionMeankVp = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["acquisitionHistogramkVpData"]["data"]

        standard_data = [
            {
                "name": "ABD_1_VIEW",
                "text": np.array(
                    [
                        "69.64 to 72.68",
                        "72.68 to 75.71",
                        "75.71 to 78.75",
                        "78.75 to 81.78",
                        "81.78 to 84.82",
                        "84.82 to 87.86",
                        "87.86 to 90.89",
                        "90.89 to 93.93",
                        "93.93 to 96.96",
                        "96.96 to 100.00",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        71.15799905,
                        74.19399915,
                        77.22999925,
                        80.26599935,
                        83.30199945,
                        86.33799955,
                        89.37399965,
                        92.40999975,
                        95.44599985,
                        98.48199995,
                    ]
                ),
                "y": np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            },
            {
                "name": "AEC",
                "text": np.array(
                    [
                        "69.64 to 72.68",
                        "72.68 to 75.71",
                        "75.71 to 78.75",
                        "78.75 to 81.78",
                        "81.78 to 84.82",
                        "84.82 to 87.86",
                        "87.86 to 90.89",
                        "90.89 to 93.93",
                        "93.93 to 96.96",
                        "96.96 to 100.00",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        71.15799905,
                        74.19399915,
                        77.22999925,
                        80.26599935,
                        83.30199945,
                        86.33799955,
                        89.37399965,
                        92.40999975,
                        95.44599985,
                        98.48199995,
                    ]
                ),
                "y": np.array([0, 0, 0, 2, 0, 0, 0, 0, 0, 0]),
            },
            {
                "name": "AP",
                "text": np.array(
                    [
                        "69.64 to 72.68",
                        "72.68 to 75.71",
                        "75.71 to 78.75",
                        "78.75 to 81.78",
                        "81.78 to 84.82",
                        "84.82 to 87.86",
                        "87.86 to 90.89",
                        "90.89 to 93.93",
                        "93.93 to 96.96",
                        "96.96 to 100.00",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        71.15799905,
                        74.19399915,
                        77.22999925,
                        80.26599935,
                        83.30199945,
                        86.33799955,
                        89.37399965,
                        92.40999975,
                        95.44599985,
                        98.48199995,
                    ]
                ),
                "y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            },
        ]

        for idx, dataset in enumerate(standard_data):
            self.assertEqual(chart_data[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(
                chart_data[idx]["x"], dataset["x"], decimal=4
            )
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
        check_series_and_category_names(self, acq_system_names, acq_names, chart_data)

        # The frequency chart - frequencies
        acq_data = [[3], [2], [1]]
        chart_data = self.chart_data["acquisitionFrequencyData"]["data"]
        check_frequencies(self, acq_data, chart_data)

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
        check_series_and_category_names(self, acq_system_names, acq_names, chart_data)

        # The frequency chart - frequencies
        acq_data = [[0, 3, 0], [2, 0, 0], [0, 0, 1]]
        chart_data = self.chart_data["acquisitionFrequencyData"]["data"]
        check_frequencies(self, acq_data, chart_data)

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
        check_series_and_category_names(
            self, request_names, request_system_names, chart_data
        )

        # Check on mean DAP values and counts
        request_data = [["All systems", 122.25, 2], ["All systems", 6.33, 1]]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][0]
        check_avg_and_counts(self, request_data, chart_data)

        # Check on median DAP values and counts - series 0
        request_data = [["All systems", 122.25, 2], ["All systems", 6.33, 1]]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][0]
        check_avg_and_counts(self, request_data, chart_data)

        # Check on the boxplot data system names
        request_data = "All systems"
        self.assertEqual(
            self.chart_data["requestBoxplotDAPData"]["data"][0]["name"], request_data
        )

        # Check the boxplot x and y data values
        request_x_data = [["XR CHEST", "Blank", "Blank"]]
        request_y_data = [[6.33, 32.8, 211.7]]
        chart_data = self.chart_data["requestBoxplotDAPData"]["data"]
        check_boxplot_xy(self, request_x_data, request_y_data, chart_data)

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
        check_series_and_category_names(
            self, request_names, request_system_names, chart_data
        )

        # Check on mean data of series 0
        request_data = [
            ["Carestream Clinic KODAK7500", 211.7, 1],
            ["Carestream Clinic KODAK7500", np.nan, 0],
        ]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][0]
        check_avg_and_counts(self, request_data, chart_data)

        # Check on mean data of series 1
        request_data = [
            ["Digital Mobile Hospital 01234MOB54", 32.8, 1],
            ["Digital Mobile Hospital 01234MOB54", np.nan, 0],
        ]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][1]
        check_avg_and_counts(self, request_data, chart_data)

        # Check on mean data of series 2
        request_data = [["LICARDR0004", np.nan, 0], ["LICARDR0004", 6.33, 1]]
        chart_data = self.chart_data["requestMeanDAPData"]["data"][2]
        check_avg_and_counts(self, request_data, chart_data)

        # Check on median values of series 0
        request_data = [
            ["Carestream Clinic KODAK7500", 211.7, 1],
            ["Carestream Clinic KODAK7500", np.nan, 0],
        ]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][0]
        check_avg_and_counts(self, request_data, chart_data)

        # Check on median values of series 1
        request_data = [
            ["Digital Mobile Hospital 01234MOB54", 32.8, 1],
            ["Digital Mobile Hospital 01234MOB54", np.nan, 0],
        ]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][1]
        check_avg_and_counts(self, request_data, chart_data)

        # Check on median data of series 2
        request_data = [["LICARDR0004", np.nan, 0], ["LICARDR0004", 6.33, 1]]
        chart_data = self.chart_data["requestMedianDAPData"]["data"][2]
        check_avg_and_counts(self, request_data, chart_data)

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
        check_boxplot_xy(self, request_x_data, request_y_data, chart_data)

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
            np.testing.assert_almost_equal(
                chart_data[idx]["x"], dataset["x"], decimal=4
            )
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
        check_series_and_category_names(
            self, request_system_names, request_names, chart_data
        )

        # The frequency chart - frequencies
        request_data = [[2], [1]]
        chart_data = self.chart_data["requestFrequencyData"]["data"]
        check_frequencies(self, request_data, chart_data)

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
        check_series_and_category_names(
            self, request_system_names, request_names, chart_data
        )

        # The frequency chart - frequencies
        request_data = [[1, 1, 0], [0, 0, 1]]
        chart_data = self.chart_data["requestFrequencyData"]["data"]
        check_frequencies(self, request_data, chart_data)

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
        check_series_and_category_names(
            self, study_names, study_system_names, chart_data
        )

        # Check on mean DAP values and counts
        study_data = [
            ["All systems", 211.7, 1],
            ["All systems", 32.8, 1],
            ["All systems", 6.33, 1],
        ]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][0]
        check_avg_and_counts(self, study_data, chart_data)

        # Check on median DAP values and counts
        study_data = [
            ["All systems", 211.7, 1],
            ["All systems", 32.8, 1],
            ["All systems", 6.33, 1],
        ]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][0]
        check_avg_and_counts(self, study_data, chart_data)

        # Check on the boxplot data system names
        study_data = "All systems"
        self.assertEqual(
            self.chart_data["studyBoxplotDAPData"]["data"][0]["name"], study_data
        )

        # Check the boxplot x and y data values
        study_x_data = [["XR CHEST", "Abdomen", "AEC"]]
        study_y_data = [[6.33, 32.8, 211.7]]
        chart_data = self.chart_data["studyBoxplotDAPData"]["data"]
        check_boxplot_xy(self, study_x_data, study_y_data, chart_data)

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
        check_series_and_category_names(
            self, study_names, study_system_names, chart_data
        )

        # Check on mean data of series 0
        study_data = [
            ["Carestream Clinic KODAK7500", 211.7, 1],
            ["Carestream Clinic KODAK7500", np.nan, 0],
            ["Carestream Clinic KODAK7500", np.nan, 0],
        ]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][0]
        check_avg_and_counts(self, study_data, chart_data)

        # Check on mean data of series 1
        study_data = [
            ["Digital Mobile Hospital 01234MOB54", np.nan, 0],
            ["Digital Mobile Hospital 01234MOB54", 32.8, 1],
            ["Digital Mobile Hospital 01234MOB54", np.nan, 0],
        ]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][1]
        check_avg_and_counts(self, study_data, chart_data)

        # Check on mean data of series 2
        study_data = [
            ["LICARDR0004", np.nan, 0],
            ["LICARDR0004", np.nan, 0],
            ["LICARDR0004", 6.33, 1],
        ]
        chart_data = self.chart_data["studyMeanDAPData"]["data"][2]
        check_avg_and_counts(self, study_data, chart_data)

        # Check on median values of series 0
        study_data = [
            ["Carestream Clinic KODAK7500", 211.7, 1],
            ["Carestream Clinic KODAK7500", np.nan, 0],
            ["Carestream Clinic KODAK7500", np.nan, 0],
        ]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][0]
        check_avg_and_counts(self, study_data, chart_data)

        # Check on median values of series 1
        study_data = [
            ["Digital Mobile Hospital 01234MOB54", np.nan, 0],
            ["Digital Mobile Hospital 01234MOB54", 32.8, 1],
            ["Digital Mobile Hospital 01234MOB54", np.nan, 0],
        ]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][1]
        check_avg_and_counts(self, study_data, chart_data)

        # Check on median data of series 2
        study_data = [
            ["LICARDR0004", np.nan, 0],
            ["LICARDR0004", np.nan, 0],
            ["LICARDR0004", 6.33, 1],
        ]
        chart_data = self.chart_data["studyMedianDAPData"]["data"][2]
        check_avg_and_counts(self, study_data, chart_data)

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
        check_boxplot_xy(self, study_x_data, study_y_data, chart_data)

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
            np.testing.assert_almost_equal(
                chart_data[idx]["x"], dataset["x"], decimal=4
            )
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
        check_series_and_category_names(
            self, study_system_names, study_names, chart_data
        )

        # The frequency chart - frequencies
        study_data = [[1], [1], [1]]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        check_frequencies(self, study_data, chart_data)

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
        check_series_and_category_names(
            self, study_system_names, study_names, chart_data
        )

        # The frequency chart - frequencies
        study_data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        check_frequencies(self, study_data, chart_data)

    def test_study_workload(self):
        # Test of study workload
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXStudyPerDayAndHour = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 10],
                        ["All systems", 11],
                        ["All systems", 14],
                        ["All systems", 10],
                        ["All systems", 11],
                        ["All systems", 14],
                        ["All systems", 10],
                        ["All systems", 11],
                        ["All systems", 14],
                    ],
                    dtype=object,
                ),
                "x": np.array(
                    [
                        "Friday",
                        "Friday",
                        "Friday",
                        "Monday",
                        "Monday",
                        "Monday",
                        "Tuesday",
                        "Tuesday",
                        "Tuesday",
                    ],
                    dtype=object,
                ),
                "y": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
            }
        ]

        chart_data = self.chart_data["studyWorkloadData"]["data"]

        for idx, dataset in enumerate(standard_data):
            for i, entry in enumerate(dataset["customdata"]):
                np.testing.assert_equal(entry, chart_data[idx]["customdata"][i])

            np.testing.assert_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_equal(dataset["y"], chart_data[idx]["y"])

        # Repeat with series per system enabled
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        self.obtain_chart_data(f)

        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["Carestream Clinic KODAK7500", 10],
                        ["Carestream Clinic KODAK7500", 11],
                        ["Carestream Clinic KODAK7500", 14],
                        ["Carestream Clinic KODAK7500", 10],
                        ["Carestream Clinic KODAK7500", 11],
                        ["Carestream Clinic KODAK7500", 14],
                        ["Carestream Clinic KODAK7500", 10],
                        ["Carestream Clinic KODAK7500", 11],
                        ["Carestream Clinic KODAK7500", 14],
                    ],
                    dtype=object,
                ),
                "x": np.array(
                    [
                        "Friday",
                        "Friday",
                        "Friday",
                        "Monday",
                        "Monday",
                        "Monday",
                        "Tuesday",
                        "Tuesday",
                        "Tuesday",
                    ],
                    dtype=object,
                ),
                "y": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),
            },
            {
                "customdata": np.array(
                    [
                        ["Digital Mobile Hospital 01234MOB54", 10],
                        ["Digital Mobile Hospital 01234MOB54", 11],
                        ["Digital Mobile Hospital 01234MOB54", 14],
                        ["Digital Mobile Hospital 01234MOB54", 10],
                        ["Digital Mobile Hospital 01234MOB54", 11],
                        ["Digital Mobile Hospital 01234MOB54", 14],
                        ["Digital Mobile Hospital 01234MOB54", 10],
                        ["Digital Mobile Hospital 01234MOB54", 11],
                        ["Digital Mobile Hospital 01234MOB54", 14],
                    ],
                    dtype=object,
                ),
                "x": np.array(
                    [
                        "Friday",
                        "Friday",
                        "Friday",
                        "Monday",
                        "Monday",
                        "Monday",
                        "Tuesday",
                        "Tuesday",
                        "Tuesday",
                    ],
                    dtype=object,
                ),
                "y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
            },
            {
                "customdata": np.array(
                    [
                        ["LICARDR0004", 10],
                        ["LICARDR0004", 11],
                        ["LICARDR0004", 14],
                        ["LICARDR0004", 10],
                        ["LICARDR0004", 11],
                        ["LICARDR0004", 14],
                        ["LICARDR0004", 10],
                        ["LICARDR0004", 11],
                        ["LICARDR0004", 14],
                    ],
                    dtype=object,
                ),
                "x": np.array(
                    [
                        "Friday",
                        "Friday",
                        "Friday",
                        "Monday",
                        "Monday",
                        "Monday",
                        "Tuesday",
                        "Tuesday",
                        "Tuesday",
                    ],
                    dtype=object,
                ),
                "y": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),
            },
        ]

        chart_data = self.chart_data["studyWorkloadData"]["data"]

        for idx, dataset in enumerate(standard_data):
            for i, entry in enumerate(dataset["customdata"]):
                np.testing.assert_equal(entry, chart_data[idx]["customdata"][i])

            np.testing.assert_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_equal(dataset["y"], chart_data[idx]["y"])

    def test_acquisition_dap_over_time(self):
        from datetime import datetime

        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionMeanDAPOverTime = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        standard_data = [
            {
                "name": "ABD_1_VIEW",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        10.93333333,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "name": "AEC",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        105.85,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "name": "AP",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        6.33,
                    ]
                ),
            },
        ]

        chart_data = self.chart_data["acquisitionMeanDAPOverTime"]["data"]

        for idx, dataset in enumerate(standard_data):
            np.testing.assert_array_equal(dataset["name"], chart_data[idx]["name"])
            np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_array_almost_equal(
                dataset["y"], chart_data[idx]["y"], decimal=2
            )

        # Now test the median data
        standard_data = [
            {
                "name": "ABD_1_VIEW",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        8.2,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "name": "AEC",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        105.85,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "name": "AP",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        6.33,
                    ]
                ),
            },
        ]

        chart_data = self.chart_data["acquisitionMedianDAPOverTime"]["data"]

        for idx, dataset in enumerate(standard_data):
            np.testing.assert_array_equal(dataset["name"], chart_data[idx]["name"])
            np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_array_almost_equal(
                dataset["y"], chart_data[idx]["y"], decimal=2
            )

    def test_acquisition_kvp_over_time(self):
        from datetime import datetime

        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionMeankVpOverTime = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        standard_data = [
            {
                "name": "ABD_1_VIEW",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        69.81999967,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "name": "AEC",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        80.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "name": "AP",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        100.0,
                    ]
                ),
            },
        ]

        chart_data = self.chart_data["acquisitionMeankVpOverTime"]["data"]

        for idx, dataset in enumerate(standard_data):
            np.testing.assert_array_equal(dataset["name"], chart_data[idx]["name"])
            np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_array_almost_equal(
                dataset["y"], chart_data[idx]["y"], decimal=1
            )

        # Now test the median data
        standard_data = [
            {
                "name": "ABD_1_VIEW",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        69.860001,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "name": "AEC",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        80.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "name": "AP",
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        100.0,
                    ]
                ),
            },
        ]

        chart_data = self.chart_data["acquisitionMediankVpOverTime"]["data"]

        for idx, dataset in enumerate(standard_data):
            np.testing.assert_array_equal(dataset["name"], chart_data[idx]["name"])
            np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_array_almost_equal(
                dataset["y"], chart_data[idx]["y"], decimal=1
            )

    def test_acquisition_mas_over_time(self):
        from datetime import datetime

        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionMeanmAsOverTime = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        standard_data = [
            {
                "hovertemplate": "Acquisition protocol=ABD_1_VIEW<br>System=All systems<br>Study date=%{x}<br>mAs=%{y}<extra></extra>",
                "legendgroup": "ABD_1_VIEW",
                "marker": {"color": "#a50026", "symbol": "circle"},
                "mode": "markers+lines",
                "name": "ABD_1_VIEW",
                "orientation": "v",
                "showlegend": True,
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "xaxis": "x",
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        2.70666667,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
                "yaxis": "y",
                "type": "scatter",
            },
            {
                "hovertemplate": "Acquisition protocol=AEC<br>System=All systems<br>Study date=%{x}<br>mAs=%{y}<extra></extra>",
                "legendgroup": "AEC",
                "marker": {"color": "#feffc0", "symbol": "circle"},
                "mode": "markers+lines",
                "name": "AEC",
                "orientation": "v",
                "showlegend": True,
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "xaxis": "x",
                "y": np.array(
                    [
                        9.5,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
                "yaxis": "y",
                "type": "scatter",
            },
            {
                "hovertemplate": "Acquisition protocol=AP<br>System=All systems<br>Study date=%{x}<br>mAs=%{y}<extra></extra>",
                "legendgroup": "AP",
                "marker": {"color": "#313695", "symbol": "circle"},
                "mode": "markers+lines",
                "name": "AP",
                "orientation": "v",
                "showlegend": True,
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "xaxis": "x",
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                    ]
                ),
                "yaxis": "y",
                "type": "scatter",
            },
        ]

        chart_data = self.chart_data["acquisitionMeanmAsOverTime"]["data"]

        for idx, dataset in enumerate(standard_data):
            np.testing.assert_array_equal(dataset["name"], chart_data[idx]["name"])
            np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_array_almost_equal(
                dataset["y"], chart_data[idx]["y"], decimal=1
            )

        # Now test the median data
        standard_data = [
            {
                "hovertemplate": "Acquisition protocol=ABD_1_VIEW<br>System=All systems<br>Study date=%{x}<br>mAs=%{y}<extra></extra>",
                "legendgroup": "ABD_1_VIEW",
                "marker": {"color": "#a50026", "symbol": "circle"},
                "mode": "markers+lines",
                "name": "ABD_1_VIEW",
                "orientation": "v",
                "showlegend": True,
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "xaxis": "x",
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        2.04,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
                "yaxis": "y",
                "type": "scatter",
            },
            {
                "hovertemplate": "Acquisition protocol=AEC<br>System=All systems<br>Study date=%{x}<br>mAs=%{y}<extra></extra>",
                "legendgroup": "AEC",
                "marker": {"color": "#feffc0", "symbol": "circle"},
                "mode": "markers+lines",
                "name": "AEC",
                "orientation": "v",
                "showlegend": True,
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "xaxis": "x",
                "y": np.array(
                    [
                        9.5,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
                "yaxis": "y",
                "type": "scatter",
            },
            {
                "hovertemplate": "Acquisition protocol=AP<br>System=All systems<br>Study date=%{x}<br>mAs=%{y}<extra></extra>",
                "legendgroup": "AP",
                "marker": {"color": "#313695", "symbol": "circle"},
                "mode": "markers+lines",
                "name": "AP",
                "orientation": "v",
                "showlegend": True,
                "x": np.array(
                    [
                        datetime(2014, 6, 30, 0, 0),
                        datetime(2014, 7, 31, 0, 0),
                        datetime(2014, 8, 31, 0, 0),
                        datetime(2014, 9, 30, 0, 0),
                        datetime(2014, 10, 31, 0, 0),
                        datetime(2014, 11, 30, 0, 0),
                        datetime(2014, 12, 31, 0, 0),
                        datetime(2015, 1, 31, 0, 0),
                        datetime(2015, 2, 28, 0, 0),
                        datetime(2015, 3, 31, 0, 0),
                        datetime(2015, 4, 30, 0, 0),
                        datetime(2015, 5, 31, 0, 0),
                        datetime(2015, 6, 30, 0, 0),
                        datetime(2015, 7, 31, 0, 0),
                        datetime(2015, 8, 31, 0, 0),
                        datetime(2015, 9, 30, 0, 0),
                        datetime(2015, 10, 31, 0, 0),
                        datetime(2015, 11, 30, 0, 0),
                        datetime(2015, 12, 31, 0, 0),
                        datetime(2016, 1, 31, 0, 0),
                        datetime(2016, 2, 29, 0, 0),
                        datetime(2016, 3, 31, 0, 0),
                        datetime(2016, 4, 30, 0, 0),
                        datetime(2016, 5, 31, 0, 0),
                        datetime(2016, 6, 30, 0, 0),
                        datetime(2016, 7, 31, 0, 0),
                        datetime(2016, 8, 31, 0, 0),
                        datetime(2016, 9, 30, 0, 0),
                        datetime(2016, 10, 31, 0, 0),
                        datetime(2016, 11, 30, 0, 0),
                        datetime(2016, 12, 31, 0, 0),
                        datetime(2017, 1, 31, 0, 0),
                        datetime(2017, 2, 28, 0, 0),
                        datetime(2017, 3, 31, 0, 0),
                        datetime(2017, 4, 30, 0, 0),
                        datetime(2017, 5, 31, 0, 0),
                    ],
                    dtype=object,
                ),
                "xaxis": "x",
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                    ]
                ),
                "yaxis": "y",
                "type": "scatter",
            },
        ]

        chart_data = self.chart_data["acquisitionMedianmAsOverTime"]["data"]

        for idx, dataset in enumerate(standard_data):
            np.testing.assert_array_equal(dataset["name"], chart_data[idx]["name"])
            np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_array_almost_equal(
                dataset["y"], chart_data[idx]["y"], decimal=1
            )

    def test_empty_acquisition_dap_vs_mass(self):
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotDXAcquisitionDAPvsMass = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        standard_data = "<div class='alert alert-warning' role='alert'>No data left after excluding missing values.</div>"

        chart_data = self.chart_data["acquisitionDAPvsMass"]

        self.assertEqual(standard_data, chart_data)
