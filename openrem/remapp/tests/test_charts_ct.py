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
            np.testing.assert_array_equal(dataset["y"], chart_data[idx]["y"])
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
