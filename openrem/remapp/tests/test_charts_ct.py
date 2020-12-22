# This Python file uses the following encoding: utf-8
# test_charts_ct.py

import os
from django.contrib.auth.models import User, Group
from django.test import TestCase, RequestFactory
from remapp.extractors import rdsr
from remapp.models import GeneralStudyModuleAttr, PatientIDSettings
from remapp.interface.mod_filters import CTSummaryListFilter
import numpy as np
import math


class ChartsCT(TestCase):
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

        ct1 = os.path.join("test_files", "CT-ESR-GE_Optima.dcm")
        ct2 = os.path.join("test_files", "CT-ESR-GE_VCT.dcm")
        ct3 = os.path.join("test_files", "CT-RDSR-GEPixelMed.dcm")
        ct4 = os.path.join("test_files", "CT-RDSR-Siemens_Flash-QA-DS.dcm")
        ct5 = os.path.join("test_files", "CT-RDSR-Siemens_Flash-TAP-SS.dcm")
        ct6 = os.path.join("test_files", "CT-RDSR-ToshibaPixelMed.dcm")
        ct7 = os.path.join("test_files", "CT-SC-Philips_Brilliance16P.dcm")
        root_tests = os.path.dirname(os.path.abspath(__file__))

        rdsr.rdsr(os.path.join(root_tests, ct1))
        rdsr.rdsr(os.path.join(root_tests, ct2))
        rdsr.rdsr(os.path.join(root_tests, ct3))
        rdsr.rdsr(os.path.join(root_tests, ct4))
        rdsr.rdsr(os.path.join(root_tests, ct5))
        rdsr.rdsr(os.path.join(root_tests, ct6))
        rdsr.rdsr(os.path.join(root_tests, ct7))

    def user_profile_reset(self):
        self.user.userprofile.plotCharts = True

        self.user.userprofile.plotGroupingChoice = "system"
        self.user.userprofile.plotSeriesPerSystem = False
        self.user.userprofile.plotCaseInsensitiveCategories = False

        self.user.userprofile.plotInitialSortingDirection = 0
        self.user.userprofile.plotCTInitialSortingChoice = "frequency"

        self.user.userprofile.plotAverageChoice = "mean"
        self.user.userprofile.plotMean = False
        self.user.userprofile.plotMedian = False
        self.user.userprofile.plotBoxplots = False

        self.user.userprofile.plotHistograms = False
        self.user.userprofile.plotHistogramBins = 10

        self.user.userprofile.plotCTAcquisitionMeanDLP = False
        self.user.userprofile.plotCTAcquisitionMeanCTDI = False
        self.user.userprofile.plotCTAcquisitionFreq = False
        self.user.userprofile.plotCTAcquisitionCTDIvsMass = False
        self.user.userprofile.plotCTAcquisitionDLPvsMass = False
        self.user.userprofile.plotCTAcquisitionCTDIOverTime = False
        self.user.userprofile.plotCTAcquisitionDLPOverTime = False

        self.user.userprofile.plotCTStudyMeanDLP = False
        self.user.userprofile.plotCTStudyMeanCTDI = False
        self.user.userprofile.plotCTStudyFreq = False
        self.user.userprofile.plotCTStudyNumEvents = False
        self.user.userprofile.plotCTStudyPerDayAndHour = False
        self.user.userprofile.plotCTStudyMeanDLPOverTime = False

        self.user.userprofile.plotCTRequestMeanDLP = False
        self.user.userprofile.plotCTRequestFreq = False
        self.user.userprofile.plotCTRequestNumEvents = False
        self.user.userprofile.plotCTRequestDLPOverTime = False

        self.user.userprofile.save()

    def login_get_filterset(self):
        self.client.login(username="jacob", password="top_secret")
        # I can add to the filter_set to control what type of chart data is calculated
        filter_set = ""
        f = CTSummaryListFilter(
            filter_set,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT")
            .order_by()
            .distinct(),
        )
        # Reset the user profile
        self.user_profile_reset()
        return f

    def obtain_chart_data(self, f):
        from remapp.views_charts_ct import ct_plot_calculations

        self.chart_data = ct_plot_calculations(
            f, self.user.userprofile, return_as_dict=True
        )

    def check_series_and_category_names(self, category_names, series_names, chart_data):
        for idx, series_name in enumerate(series_names):
            self.assertEqual(chart_data[idx]["name"], series_name)
            self.assertListEqual(list(chart_data[idx]["x"]), category_names)

    def check_average_data(self, chart_data, standard_data):
        for idx, dataset in enumerate(standard_data):
            self.assertEqual(dataset["name"], chart_data[idx]["name"])
            np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_array_almost_equal(dataset["y"], chart_data[idx]["y"])

            # Check the system names
            np.testing.assert_array_equal([i[0] for i in dataset["customdata"]], [i[0] for i in chart_data[idx]["customdata"]])

            # Check the average values
            np.testing.assert_array_almost_equal([i[1] for i in dataset["customdata"]], [i[1] for i in chart_data[idx]["customdata"]])

            # Check the frequency values
            np.testing.assert_array_almost_equal([i[2] for i in dataset["customdata"]], [i[2] for i in chart_data[idx]["customdata"]])

    def check_frequency_data(self, chart_data, standard_data):
        for idx, dataset in enumerate(standard_data["data"]):
            np.testing.assert_equal(dataset["name"], chart_data["data"][idx]["name"])
            np.testing.assert_equal(dataset["x"], chart_data["data"][idx]["x"])
            np.testing.assert_equal(dataset["y"], chart_data["data"][idx]["y"])

    def check_workload_data(self, chart_data, standard_data):
        for idx, dataset in enumerate(standard_data):
            np.testing.assert_array_equal(
                dataset["customdata"], chart_data[idx]["customdata"]
            )
            np.testing.assert_array_equal(
                dataset["hovertext"], chart_data[idx]["hovertext"]
            )
            np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_array_equal(dataset["y"], chart_data[idx]["y"])

    def check_sys_name_x_y_data(self, chart_data, standard_data):
        for idx, dataset in enumerate(standard_data["data"]):
            self.assertTrue(dataset["system"] in chart_data[idx]["hovertemplate"])
            np.testing.assert_array_equal(dataset["name"], chart_data[idx]["name"])

            std_x_data = dataset["x"]
            std_y_data = dataset["y"]
            std_y_data = [y for y, _ in sorted(zip(std_y_data, std_x_data))]
            std_x_data = sorted(std_x_data)
            std_x_data = [x for _, x in sorted(zip(std_y_data, std_x_data))]
            std_y_data = sorted(std_y_data)

            chart_y_data = chart_data[idx]["y"]
            chart_x_data = chart_data[idx]["x"]
            chart_y_data = [y for y, _ in sorted(zip(chart_y_data, chart_x_data))]
            chart_x_data = sorted(chart_x_data)
            chart_x_data = [x for _, x in sorted(zip(chart_y_data, chart_x_data))]
            chart_y_data = sorted(chart_y_data)

            np.testing.assert_array_equal(std_x_data, chart_x_data)
            np.testing.assert_array_almost_equal(std_y_data, chart_y_data)

    def check_avg_and_counts(self, comparison_data, chart_data):
        for idx in range(len(comparison_data)):
            # If the comparison value is a nan then check that the chart value is too
            if math.isnan(comparison_data[idx][1]):
                self.assertTrue(math.isnan(chart_data[idx][1]))
            # Otherwise compare the values
            else:
                self.assertAlmostEqual(chart_data[idx][1], comparison_data[idx][1])
            self.assertEqual(chart_data[idx][2], comparison_data[idx][2])

    def check_frequencies(self, comparison_data, chart_data):
        for idx, values in enumerate(comparison_data):
            self.assertListEqual(list(chart_data[idx]["y"]), comparison_data[idx])

    def check_boxplot_data(self, chart_data, standard_data):
        for idx, dataset in enumerate(standard_data):
            self.assertEqual(dataset["name"], chart_data[idx]["name"])
            self.check_boxplot_xy([dataset["x"]], [dataset["y"]], [chart_data[idx]])

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
        from remapp.views_charts_ct import generate_required_ct_charts_list

        f = self.login_get_filterset()

        # Set user profile options to use all charts
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.plotCTAcquisitionMeanDLP = True
        self.user.userprofile.plotCTAcquisitionMeanCTDI = True
        self.user.userprofile.plotCTAcquisitionFreq = True
        self.user.userprofile.plotCTAcquisitionCTDIvsMass = True
        self.user.userprofile.plotCTAcquisitionDLPvsMass = True
        self.user.userprofile.plotCTAcquisitionCTDIOverTime = True
        self.user.userprofile.plotCTAcquisitionDLPOverTime = True
        self.user.userprofile.plotCTStudyMeanDLP = True
        self.user.userprofile.plotCTStudyMeanCTDI = True
        self.user.userprofile.plotCTStudyFreq = True
        self.user.userprofile.plotCTStudyNumEvents = True
        self.user.userprofile.plotCTStudyPerDayAndHour = True
        self.user.userprofile.plotCTStudyMeanDLPOverTime = True
        self.user.userprofile.plotCTRequestMeanDLP = True
        self.user.userprofile.plotCTRequestFreq = True
        self.user.userprofile.plotCTRequestNumEvents = True
        self.user.userprofile.plotCTRequestDLPOverTime = True
        self.user.userprofile.save()

        required_charts_list = generate_required_ct_charts_list(self.user.userprofile)

        chart_var_names = []
        for item in required_charts_list:
            chart_var_names.append(item["var_name"])

        # Just check the variable names - I don't mind if the titles change
        reference_var_names = [
            "acquisitionMeanDLP",
            "acquisitionMedianDLP",
            "acquisitionBoxplotDLP",
            "acquisitionHistogramDLP",
            "acquisitionMeanCTDI",
            "acquisitionMedianCTDI",
            "acquisitionBoxplotCTDI",
            "acquisitionHistogramCTDI",
            "acquisitionFrequency",
            "acquisitionScatterCTDIvsMass",
            "acquisitionScatterDLPvsMass",
            "acquisitionMeanCTDIOverTime",
            "acquisitionMedianCTDIOverTime",
            "acquisitionMeanDLPOverTime",
            "acquisitionMedianDLPOverTime",
            "studyMeanDLP",
            "studyMedianDLP",
            "studyBoxplotDLP",
            "studyHistogramDLP",
            "studyMeanCTDI",
            "studyMedianCTDI",
            "studyBoxplotCTDI",
            "studyHistogramCTDI",
            "studyFrequency",
            "studyMeanNumEvents",
            "studyMedianNumEvents",
            "studyBoxplotNumEvents",
            "studyHistogramNumEvents",
            "requestMeanDLP",
            "requestMedianDLP",
            "requestBoxplotDLP",
            "requestHistogramDLP",
            "requestFrequency",
            "requestMeanNumEvents",
            "requestMedianNumEvents",
            "requestBoxplotNumEvents",
            "requestHistogramNumEvents",
            "requestMeanDLPOverTime",
            "requestMedianDLPOverTime",
            "studyWorkload",
            "studyMeanDLPOverTime",
            "studyMedianDLPOverTime",
        ]

        for ref_var_name in reference_var_names:
            self.assertTrue(ref_var_name in chart_var_names)

    def test_acq_dlp(self):
        # Test of mean and median acquisition DLP, count, system and acquisition protocol names
        # Also tests raw data going into the box plots
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotCTAcquisitionMeanDLP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the mean data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 111.3, 1.0],
                        ["All systems", 202.684375, 16.0],
                        ["All systems", 29.67, 1.0],
                        ["All systems", 50.58, 1.0],
                        ["All systems", 129.89, 1.0],
                        ["All systems", 21.18, 1.0],
                        ["All systems", 24.05, 1.0],
                        ["All systems", 74.98, 2.0],
                        ["All systems", 369.34, 1.0],
                        ["All systems", 815.33, 1.0],
                        ["All systems", 3.61, 1.0],
                        ["All systems", 1.2, 1.0],
                        ["All systems", 708.2, 1.0],
                        ["All systems", 11.51, 1.0],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        111.3,
                        202.684375,
                        29.67,
                        50.58,
                        129.89,
                        21.18,
                        24.05,
                        74.98,
                        369.34,
                        815.33,
                        3.61,
                        1.2,
                        708.2,
                        11.51,
                    ]
                ),
            }
        ]

        chart_data = self.chart_data["acquisitionMeanDLPData"]["data"]

        self.check_average_data(chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 111.3, 1.0],
                        ["All systems", 148.585, 16.0],
                        ["All systems", 29.67, 1.0],
                        ["All systems", 50.58, 1.0],
                        ["All systems", 129.89, 1.0],
                        ["All systems", 21.18, 1.0],
                        ["All systems", 24.05, 1.0],
                        ["All systems", 74.98, 2.0],
                        ["All systems", 369.34, 1.0],
                        ["All systems", 815.33, 1.0],
                        ["All systems", 3.61, 1.0],
                        ["All systems", 1.2, 1.0],
                        ["All systems", 708.2, 1.0],
                        ["All systems", 11.51, 1.0],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        111.3,
                        148.585,
                        29.67,
                        50.58,
                        129.89,
                        21.18,
                        24.05,
                        74.98,
                        369.34,
                        815.33,
                        3.61,
                        1.2,
                        708.2,
                        11.51,
                    ]
                ),
            }
        ]

        chart_data = self.chart_data["acquisitionMedianDLPData"]["data"]

        self.check_average_data(chart_data, standard_data)

        # Check the boxplot data
        standard_data = [
            {
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "10.13 RADIOTHERAPY QA",
                        "DE_laser align",
                        "DS axial std",
                        "DS 50mAs",
                        "DS 140kV",
                        "DS 100kV",
                        "DS 80kV",
                        "DS axial std",
                        "DS_helical",
                        "DS_hel p 0.23",
                        "testÃ¦Ã¸Ã¥",
                        "PreMonitoring",
                        "Monitoring",
                        "TAP",
                        "Blank",
                        "Blank",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        155.97,
                        259.85,
                        16.41,
                        429.19,
                        246.69,
                        3.12,
                        890.26,
                        2.92,
                        352.24,
                        14.66,
                        14.66,
                        15.83,
                        16.41,
                        475.04,
                        111.3,
                        29.67,
                        84.28,
                        21.18,
                        129.89,
                        50.58,
                        24.05,
                        65.68,
                        815.33,
                        369.34,
                        11.51,
                        1.2,
                        3.61,
                        708.2,
                        208.5,
                        141.2,
                    ]
                ),
            }
        ]

        chart_data = self.chart_data["acquisitionBoxplotDLPData"]["data"]

        self.check_boxplot_data(chart_data, standard_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the mean data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", 182.03545455, 11.0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "A VCT Hospital VCTScanner",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        182.03545455,
                        np.nan,
                        np.nan,
                        np.nan,
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
                "customdata": np.array(
                    [
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", 207.91, 2.0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "An Optima Hospital geoptima",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        207.91,
                        np.nan,
                        np.nan,
                        np.nan,
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
                "customdata": np.array(
                    [
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", 29.67, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 50.58, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 129.89, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 21.18, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 24.05, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 74.98, 2.0],
                        ["Gnats Bottom Hospital CTAWP91919", 369.34, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 815.33, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "Gnats Bottom Hospital CTAWP91919",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        29.67,
                        50.58,
                        129.89,
                        21.18,
                        24.05,
                        74.98,
                        369.34,
                        815.33,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "customdata": np.array(
                    [
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", 3.61, 1.0],
                        ["Hospital Number One Trust CTAWP00001", 1.2, 1.0],
                        ["Hospital Number One Trust CTAWP00001", 708.2, 1.0],
                        ["Hospital Number One Trust CTAWP00001", 11.51, 1.0],
                    ],
                    dtype=object,
                ),
                "name": "Hospital Number One Trust CTAWP00001",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
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
                        3.61,
                        1.2,
                        708.2,
                        11.51,
                    ]
                ),
            },
            {
                "customdata": np.array(
                    [
                        ["OpenREM centre médical rt16", 111.3, 1.0],
                        ["OpenREM centre médical rt16", 475.04, 1.0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "OpenREM centre médical rt16",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        111.3,
                        475.04,
                        np.nan,
                        np.nan,
                        np.nan,
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
                "customdata": np.array(
                    [
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", 174.85, 2.0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "Oxbridge County Hospital CTTOSHIBA1",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        174.85,
                        np.nan,
                        np.nan,
                        np.nan,
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
        ]

        chart_data = self.chart_data["acquisitionMeanDLPData"]["data"]

        self.check_average_data(chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", 16.41, 11.0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "A VCT Hospital VCTScanner",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        16.41,
                        np.nan,
                        np.nan,
                        np.nan,
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
                "customdata": np.array(
                    [
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", 207.91, 2.0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "An Optima Hospital geoptima",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        207.91,
                        np.nan,
                        np.nan,
                        np.nan,
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
                "customdata": np.array(
                    [
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", 29.67, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 50.58, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 129.89, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 21.18, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 24.05, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 74.98, 2.0],
                        ["Gnats Bottom Hospital CTAWP91919", 369.34, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", 815.33, 1.0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "Gnats Bottom Hospital CTAWP91919",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        np.nan,
                        29.67,
                        50.58,
                        129.89,
                        21.18,
                        24.05,
                        74.98,
                        369.34,
                        815.33,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                ),
            },
            {
                "customdata": np.array(
                    [
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ["Hospital Number One Trust CTAWP00001", 3.61, 1.0],
                        ["Hospital Number One Trust CTAWP00001", 1.2, 1.0],
                        ["Hospital Number One Trust CTAWP00001", 708.2, 1.0],
                        ["Hospital Number One Trust CTAWP00001", 11.51, 1.0],
                    ],
                    dtype=object,
                ),
                "name": "Hospital Number One Trust CTAWP00001",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
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
                        3.61,
                        1.2,
                        708.2,
                        11.51,
                    ]
                ),
            },
            {
                "customdata": np.array(
                    [
                        ["OpenREM centre médical rt16", 111.3, 1.0],
                        ["OpenREM centre médical rt16", 475.04, 1.0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                        ["OpenREM centre médical rt16", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "OpenREM centre médical rt16",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        111.3,
                        475.04,
                        np.nan,
                        np.nan,
                        np.nan,
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
                "customdata": np.array(
                    [
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", 174.85, 2.0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                    ],
                    dtype=object,
                ),
                "name": "Oxbridge County Hospital CTTOSHIBA1",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "Blank",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        np.nan,
                        174.85,
                        np.nan,
                        np.nan,
                        np.nan,
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
        ]

        chart_data = self.chart_data["acquisitionMedianDLPData"]["data"]

        self.check_average_data(chart_data, standard_data)

        # Check the boxplot data
        standard_data = [
            {
                "name": "A VCT Hospital VCTScanner",
                "x": np.array(
                    [
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        890.26,
                        3.12,
                        2.92,
                        429.19,
                        16.41,
                        352.24,
                        14.66,
                        14.66,
                        15.83,
                        16.41,
                        246.69,
                    ]
                ),
            },
            {
                "name": "An Optima Hospital geoptima",
                "x": np.array(["Blank", "Blank"], dtype=object),
                "y": np.array([259.85, 155.97]),
            },
            {
                "name": "Gnats Bottom Hospital CTAWP91919",
                "x": np.array(
                    [
                        "DS_hel p 0.23",
                        "DS_helical",
                        "DS axial std",
                        "DS 80kV",
                        "DS 140kV",
                        "DS 100kV",
                        "DS axial std",
                        "DE_laser align",
                        "DS 50mAs",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [369.34, 815.33, 65.68, 24.05, 129.89, 50.58, 84.28, 29.67, 21.18]
                ),
            },
            {
                "name": "Hospital Number One Trust CTAWP00001",
                "x": np.array(
                    ["testÃ¦Ã¸Ã¥", "PreMonitoring", "Monitoring", "TAP"], dtype=object
                ),
                "y": np.array([11.51, 1.2, 3.61, 708.2]),
            },
            {
                "name": "OpenREM centre médical rt16",
                "x": np.array(["10.13 RADIOTHERAPY QA", "Blank"], dtype=object),
                "y": np.array([111.3, 475.04]),
            },
            {
                "name": "Oxbridge County Hospital CTTOSHIBA1",
                "x": np.array(["Blank", "Blank"], dtype=object),
                "y": np.array([208.5, 141.2]),
            },
        ]

        chart_data = self.chart_data["acquisitionBoxplotDLPData"]["data"]

        self.check_boxplot_data(chart_data, standard_data)
