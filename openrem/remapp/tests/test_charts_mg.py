# This Python file uses the following encoding: utf-8
# test_charts_mg.py

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
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="MG")
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
            self.assertEqual(chart_data[idx]["name"], series_name)
            self.assertListEqual(list(chart_data[idx]["x"]), category_names)

    def check_average_data(self, chart_data, standard_data):
        for idx, dataset in enumerate(standard_data):
            self.assertEqual(dataset["name"], chart_data[idx]["name"])
            np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
            np.testing.assert_array_almost_equal(dataset["y"], chart_data[idx]["y"])
            np.testing.assert_array_almost_equal(
                dataset["customdata"], chart_data[idx]["customdata"]
            )

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
        from remapp.views_charts_mg import generate_required_mg_charts_list

        f = self.login_get_filterset()

        # Set user profile options to use all charts
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.plotMGStudyPerDayAndHour = True
        self.user.userprofile.plotMGAGDvsThickness = True
        self.user.userprofile.plotMGkVpvsThickness = True
        self.user.userprofile.plotMGmAsvsThickness = True
        self.user.userprofile.plotMGaverageAGDvsThickness = True
        self.user.userprofile.plotMGaverageAGD = True
        self.user.userprofile.plotMGacquisitionFreq = True
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
                "customdata": np.array(
                    [[0.0, 1.29, 2.0], [0.0, 0.26, 1.0], [0.0, 1.373, 1.0]]
                ),
                "name": "All systems",
                "x": np.array(["Blank", "Flat Field Tomo", "ROUTINE"], dtype=object),
                "y": np.array([1.29, 0.26, 1.373]),
            }
        ]

        chart_data = self.chart_data["acquisitionMeanAGDData"]["data"]

        self.check_average_data(chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [[0.0, 1.29, 2.0], [0.0, 0.26, 1.0], [0.0, 1.373, 1.0]]
                ),
                "name": "All systems",
                "x": np.array(["Blank", "Flat Field Tomo", "ROUTINE"], dtype=object),
                "y": np.array([1.29, 0.26, 1.373]),
            }
        ]

        chart_data = self.chart_data["acquisitionMedianAGDData"]["data"]

        self.check_average_data(chart_data, standard_data)

        # Check the boxplot data
        standard_data = [
            {
                "name": "All systems",
                "x": np.array(
                    ["Flat Field Tomo", "Blank", "Blank", "ROUTINE"], dtype=object
                ),
                "y": np.array([0.26, 1.28, 1.3, 1.373]),
            }
        ]

        chart_data = self.chart_data["acquisitionBoxplotAGDData"]["data"]

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
                    [[0.0, np.nan, 0.0], [0.0, 0.26, 1.0], [0.0, np.nan, 0.0]]
                ),
                "name": "Breast Imaging Clinic PQW_HOL_SELENIA",
                "x": np.array(["Blank", "Flat Field Tomo", "ROUTINE"], dtype=object),
                "y": np.array([np.nan, 0.26, np.nan]),
            },
            {
                "customdata": np.array(
                    [[1.0, 1.29, 2.0], [1.0, np.nan, 0.0], [1.0, np.nan, 0.0]]
                ),
                "name": "OpenREM Dimensions",
                "x": np.array(["Blank", "Flat Field Tomo", "ROUTINE"], dtype=object),
                "y": np.array([1.29, np.nan, np.nan]),
            },
            {
                "customdata": np.array(
                    [[2.0, np.nan, 0.0], [2.0, np.nan, 0.0], [2.0, 1.373, 1.0]]
                ),
                "name": "中心医院 SENODS01",
                "x": np.array(["Blank", "Flat Field Tomo", "ROUTINE"], dtype=object),
                "y": np.array([np.nan, np.nan, 1.373]),
            },
        ]

        chart_data = self.chart_data["acquisitionMeanAGDData"]["data"]

        self.check_average_data(chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [[0.0, np.nan, 0.0], [0.0, 0.26, 1.0], [0.0, np.nan, 0.0]]
                ),
                "name": "Breast Imaging Clinic PQW_HOL_SELENIA",
                "x": np.array(["Blank", "Flat Field Tomo", "ROUTINE"], dtype=object),
                "y": np.array([np.nan, 0.26, np.nan]),
            },
            {
                "customdata": np.array(
                    [[1.0, 1.29, 2.0], [1.0, np.nan, 0.0], [1.0, np.nan, 0.0]]
                ),
                "name": "OpenREM Dimensions",
                "x": np.array(["Blank", "Flat Field Tomo", "ROUTINE"], dtype=object),
                "y": np.array([1.29, np.nan, np.nan]),
            },
            {
                "customdata": np.array(
                    [[2.0, np.nan, 0.0], [2.0, np.nan, 0.0], [2.0, 1.373, 1.0]]
                ),
                "name": "中心医院 SENODS01",
                "x": np.array(["Blank", "Flat Field Tomo", "ROUTINE"], dtype=object),
                "y": np.array([np.nan, np.nan, 1.373]),
            },
        ]

        chart_data = self.chart_data["acquisitionMedianAGDData"]["data"]

        self.check_average_data(chart_data, standard_data)

        # Check the boxplot data
        standard_data = [
            {
                "name": "Breast Imaging Clinic PQW_HOL_SELENIA",
                "notched": False,
                "x": np.array(["Flat Field Tomo"], dtype=object),
                "y": np.array([0.26]),
            },
            {
                "name": "OpenREM Dimensions",
                "x": np.array(["Blank", "Blank"], dtype=object),
                "y": np.array([1.28, 1.3]),
            },
            {
                "name": "中心医院 SENODS01",
                "x": np.array(["ROUTINE"], dtype=object),
                "y": np.array([1.373]),
            },
        ]

        chart_data = self.chart_data["acquisitionBoxplotAGDData"]["data"]

        self.check_boxplot_data(chart_data, standard_data)

    def test_acq_agd_histogram(self):
        # Test of AGD histogram
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotMGaverageAGD = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["acquisitionHistogramAGDData"]["data"]

        standard_data = [
            {
                "name": "Blank",
                "x": np.array(
                    [
                        0.31565,
                        0.42695,
                        0.53825,
                        0.64955,
                        0.76085,
                        0.87215,
                        0.98345,
                        1.09475,
                        1.20605,
                        1.31735,
                    ]
                ),
                "y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2]),
            },
            {
                "name": "Flat Field Tomo",
                "x": np.array(
                    [
                        0.31565,
                        0.42695,
                        0.53825,
                        0.64955,
                        0.76085,
                        0.87215,
                        0.98345,
                        1.09475,
                        1.20605,
                        1.31735,
                    ]
                ),
                "y": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            },
            {
                "name": "ROUTINE",
                "x": np.array(
                    [
                        0.31565,
                        0.42695,
                        0.53825,
                        0.64955,
                        0.76085,
                        0.87215,
                        0.98345,
                        1.09475,
                        1.20605,
                        1.31735,
                    ]
                ),
                "y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
            },
        ]

        for idx, dataset in enumerate(standard_data):
            self.assertEqual(chart_data[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(chart_data[idx]["x"], dataset["x"])
            np.testing.assert_equal(chart_data[idx]["y"], dataset["y"])

    def test_acq_freq(self):
        # Test of acquisition protocol frequency
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotMGacquisitionFreq = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["acquisitionFrequencyData"]

        standard_data = {
            "data": [
                {
                    "name": "Blank",
                    "x": np.array(["All systems"], dtype=object),
                    "y": np.array([2]),
                },
                {
                    "name": "Flat Field Tomo",
                    "x": np.array(["All systems"], dtype=object),
                    "y": np.array([1]),
                },
                {
                    "name": "ROUTINE",
                    "x": np.array(["All systems"], dtype=object),
                    "y": np.array([1]),
                },
            ]
        }

        self.check_frequency_data(chart_data, standard_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["acquisitionFrequencyData"]

        standard_data = {
            "data": [
                {
                    "name": "Blank",
                    "x": np.array(
                        [
                            "Breast Imaging Clinic PQW_HOL_SELENIA",
                            "OpenREM Dimensions",
                            "中心医院 SENODS01",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([0, 2, 0]),
                },
                {
                    "name": "Flat Field Tomo",
                    "x": np.array(
                        [
                            "Breast Imaging Clinic PQW_HOL_SELENIA",
                            "OpenREM Dimensions",
                            "中心医院 SENODS01",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([1, 0, 0]),
                },
                {
                    "name": "ROUTINE",
                    "x": np.array(
                        [
                            "Breast Imaging Clinic PQW_HOL_SELENIA",
                            "OpenREM Dimensions",
                            "中心医院 SENODS01",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([0, 0, 1]),
                },
            ]
        }

        self.check_frequency_data(chart_data, standard_data)

    def test_study_workload(self):
        # Test of study workload
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotMGStudyPerDayAndHour = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        standard_data = [
            {
                "customdata": np.array(
                    [
                        [0, "Friday", 12, 1],
                        [0, "Sunday", 12, 1],
                        [0, "Thursday", 12, 1],
                    ],
                    dtype=object,
                ),
                "name": "",
                "hovertext": np.array(
                    ["All systems", "All systems", "All systems"], dtype=object
                ),
                "x": np.array(["Friday", "Sunday", "Thursday"], dtype=object),
                "y": np.array([1, 1, 1]),
            }
        ]

        chart_data = self.chart_data["studyWorkloadData"]["data"]

        self.check_workload_data(chart_data, standard_data)

        # Repeat with series per system enabled
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        self.obtain_chart_data(f)

        standard_data = [
            {
                "customdata": np.array(
                    [
                        [0, "Friday", 12, 0],
                        [0, "Sunday", 12, 0],
                        [0, "Thursday", 12, 1],
                    ],
                    dtype=object,
                ),
                "name": "",
                "hovertext": np.array(
                    [
                        "Breast Imaging Clinic PQW_HOL_SELENIA",
                        "Breast Imaging Clinic PQW_HOL_SELENIA",
                        "Breast Imaging Clinic PQW_HOL_SELENIA",
                    ],
                    dtype=object,
                ),
                "x": np.array(["Friday", "Sunday", "Thursday"], dtype=object),
                "y": np.array([0, 0, 1]),
            },
            {
                "customdata": np.array(
                    [
                        [1, "Friday", 12, 0],
                        [1, "Sunday", 12, 1],
                        [1, "Thursday", 12, 0],
                    ],
                    dtype=object,
                ),
                "name": "",
                "hovertext": np.array(
                    ["OpenREM Dimensions", "OpenREM Dimensions", "OpenREM Dimensions"],
                    dtype=object,
                ),
                "x": np.array(["Friday", "Sunday", "Thursday"], dtype=object),
                "y": np.array([0, 1, 0]),
            },
            {
                "customdata": np.array(
                    [
                        [2, "Friday", 12, 1],
                        [2, "Sunday", 12, 0],
                        [2, "Thursday", 12, 0],
                    ],
                    dtype=object,
                ),
                "name": "",
                "hovertext": np.array(
                    ["中心医院 SENODS01", "中心医院 SENODS01", "中心医院 SENODS01"], dtype=object
                ),
                "x": np.array(["Friday", "Sunday", "Thursday"], dtype=object),
                "y": np.array([1, 0, 0]),
            },
        ]

        chart_data = self.chart_data["studyWorkloadData"]["data"]

        self.check_workload_data(chart_data, standard_data)

    def test_avg_agd_vs_cbt(self):
        # Test of AGD histogram
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotMGaverageAGDvsThickness = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["meanAGDvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "All systems",
                    "name": "Blank",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [np.nan, np.nan, np.nan, 1.29, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
                {
                    "system": "All systems",
                    "name": "Flat Field Tomo",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [0.26, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
                {
                    "system": "All systems",
                    "name": "ROUTINE",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [np.nan, np.nan, np.nan, np.nan, 1.373, np.nan, np.nan, np.nan]
                    ),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)

        chart_data = self.chart_data["medianAGDvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "All systems",
                    "name": "Blank",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [np.nan, np.nan, np.nan, 1.29, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
                {
                    "system": "All systems",
                    "name": "Flat Field Tomo",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [0.26, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
                {
                    "system": "All systems",
                    "name": "ROUTINE",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [np.nan, np.nan, np.nan, np.nan, 1.373, np.nan, np.nan, np.nan]
                    ),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)

        # Repeat with series per system enabled
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        self.obtain_chart_data(f)

        chart_data = self.chart_data["meanAGDvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "Breast Imaging Clinic PQW_HOL_SELENIA",
                    "name": "Flat Field Tomo",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [0.26, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
                {
                    "system": "OpenREM Dimensions",
                    "name": "Blank",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [np.nan, np.nan, np.nan, 1.29, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
                {
                    "system": "中心医院 SENODS01",
                    "name": "ROUTINE",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [np.nan, np.nan, np.nan, np.nan, 1.373, np.nan, np.nan, np.nan]
                    ),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)

        chart_data = self.chart_data["medianAGDvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "Breast Imaging Clinic PQW_HOL_SELENIA",
                    "name": "Flat Field Tomo",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [0.26, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
                {
                    "system": "OpenREM Dimensions",
                    "name": "Blank",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [np.nan, np.nan, np.nan, 1.29, np.nan, np.nan, np.nan, np.nan]
                    ),
                },
                {
                    "system": "中心医院 SENODS01",
                    "name": "ROUTINE",
                    "x": np.array(
                        [
                            "18≤x<20",
                            "20≤x<30",
                            "30≤x<40",
                            "40≤x<50",
                            "50≤x<60",
                            "60≤x<70",
                            "70≤x<80",
                            "80≤x<90",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [np.nan, np.nan, np.nan, np.nan, 1.373, np.nan, np.nan, np.nan]
                    ),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)

    def test_agd_vs_cbt(self):
        # Test of AGD vs CBT scatter chart
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotMGAGDvsThickness = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["AGDvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "All systems",
                    "name": "Blank",
                    "x": np.array([43.0, 43.0]),
                    "y": np.array([1.28, 1.3]),
                },
                {
                    "system": "All systems",
                    "name": "Flat Field Tomo",
                    "x": np.array([18.0]),
                    "y": np.array([0.26]),
                },
                {
                    "system": "All systems",
                    "name": "ROUTINE",
                    "x": np.array([53.0]),
                    "y": np.array([1.373]),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)

        # Repeat with series per system enabled
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        self.obtain_chart_data(f)

        chart_data = self.chart_data["AGDvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "OpenREM Dimensions",
                    "name": "Blank",
                    "x": np.array([43.0, 43.0]),
                    "y": np.array([1.28, 1.3]),
                },
                {
                    "system": "Breast Imaging Clinic PQW_HOL_SELENIA",
                    "name": "Flat Field Tomo",
                    "x": np.array([18.0]),
                    "y": np.array([0.26]),
                },
                {
                    "system": "中心医院 SENODS01",
                    "name": "ROUTINE",
                    "x": np.array([53.0]),
                    "y": np.array([1.373]),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)

    def test_kvp_vs_cbt(self):
        # Test of kVp vs CBT scatter chart
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotMGkVpvsThickness = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["kVpvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "All systems",
                    "name": "Blank",
                    "x": np.array([43.0, 43.0]),
                    "y": np.array([28.0, 28.0]),
                },
                {
                    "system": "All systems",
                    "name": "Flat Field Tomo",
                    "x": np.array([18.0]),
                    "y": np.array([28.0]),
                },
                {
                    "system": "All systems",
                    "name": "ROUTINE",
                    "x": np.array([53.0]),
                    "y": np.array([29.0]),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)

        # Repeat with series per system enabled
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        self.obtain_chart_data(f)

        chart_data = self.chart_data["kVpvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "OpenREM Dimensions",
                    "name": "Blank",
                    "x": np.array([43.0, 43.0]),
                    "y": np.array([28.0, 28.0]),
                },
                {
                    "system": "Breast Imaging Clinic PQW_HOL_SELENIA",
                    "name": "Flat Field Tomo",
                    "x": np.array([18.0]),
                    "y": np.array([28.0]),
                },
                {
                    "system": "中心医院 SENODS01",
                    "name": "ROUTINE",
                    "x": np.array([53.0]),
                    "y": np.array([29.0]),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)

    def test_mas_vs_cbt(self):
        # Test of mAs vs CBT scatter chart
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotMGmAsvsThickness = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        chart_data = self.chart_data["mAsvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "All systems",
                    "name": "Blank",
                    "x": np.array([43.0, 43.0]),
                    "y": np.array([88800.0, 90200.0]),
                },
                {
                    "system": "All systems",
                    "name": "Flat Field Tomo",
                    "x": np.array([18.0]),
                    "y": np.array([6000.0]),
                },
                {
                    "system": "All systems",
                    "name": "ROUTINE",
                    "x": np.array([53.0]),
                    "y": np.array([51800.0]),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)

        # Repeat with series per system enabled
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        self.obtain_chart_data(f)

        chart_data = self.chart_data["mAsvsThickness"]["data"]

        standard_data = {
            "data": [
                {
                    "system": "OpenREM Dimensions",
                    "name": "Blank",
                    "x": np.array([43.0, 43.0]),
                    "y": np.array([88800.0, 90200.0]),
                },
                {
                    "system": "Breast Imaging Clinic PQW_HOL_SELENIA",
                    "name": "Flat Field Tomo",
                    "x": np.array([18.0]),
                    "y": np.array([6000.0]),
                },
                {
                    "system": "中心医院 SENODS01",
                    "name": "ROUTINE",
                    "x": np.array([53.0]),
                    "y": np.array([51800.0]),
                },
            ]
        }

        self.check_sys_name_x_y_data(chart_data, standard_data)
