# This Python file uses the following encoding: utf-8
# test_charts_common.py

import numpy as np
import math


def check_series_and_category_names(
    chartTests, category_names, series_names, chart_data
):
    for idx, series_name in enumerate(series_names):
        chartTests.assertEqual(chart_data[idx]["name"], series_name)
        chartTests.assertListEqual(list(chart_data[idx]["x"]), category_names)


def check_frequency_data(chartTests, chart_data, standard_data):
    for idx, dataset in enumerate(standard_data["data"]):
        np.testing.assert_equal(dataset["name"], chart_data["data"][idx]["name"])
        np.testing.assert_equal(dataset["x"], chart_data["data"][idx]["x"])
        np.testing.assert_equal(dataset["y"], chart_data["data"][idx]["y"])


def check_frequencies(chartTests, comparison_data, chart_data):
    for idx, values in enumerate(comparison_data):
        chartTests.assertListEqual(list(chart_data[idx]["y"]), comparison_data[idx])


def check_boxplot_data(chartTests, chart_data, standard_data):
    for idx, dataset in enumerate(standard_data):
        chartTests.assertEqual(dataset["name"], chart_data[idx]["name"])
        check_boxplot_xy(chartTests, [dataset["x"]], [dataset["y"]], [chart_data[idx]])


def check_boxplot_xy(chartTests, x_data, y_data, chart_data):
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
        np.testing.assert_almost_equal(chart_y_data, std_y_data, decimal=3)


def check_average_data(chartTests, chart_data, standard_data):
    for idx, dataset in enumerate(standard_data):
        # Check the name
        chartTests.assertEqual(dataset["name"], chart_data[idx]["name"])

        # Check the x data labels
        np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])

        # Check the y average values
        np.testing.assert_array_almost_equal(
            dataset["y"], chart_data[idx]["y"], decimal=3
        )

        # Check the system names and frequencies
        for i, entry in enumerate(dataset["customdata"]):
            if entry.size == 2:
                np.testing.assert_array_equal(entry, chart_data[idx]["customdata"][i])
            else:
                np.testing.assert_array_equal(
                    np.take(entry, [0, 2]), chart_data[idx]["customdata"][i]
                )


def check_workload_data(chartTests, chart_data, standard_data):
    for idx, dataset in enumerate(standard_data):
        for i, entry in enumerate(dataset["customdata"]):
            np.testing.assert_array_equal(entry, chart_data[idx]["customdata"][i])

        np.testing.assert_array_equal(
            dataset["hovertext"], chart_data[idx]["hovertext"]
        )
        np.testing.assert_array_equal(dataset["x"], chart_data[idx]["x"])
        np.testing.assert_array_equal(dataset["y"], chart_data[idx]["y"])


def check_sys_name_x_y_data(chartTests, chart_data, standard_data):
    for idx, dataset in enumerate(standard_data["data"]):
        chartTests.assertTrue(dataset["system"] in chart_data[idx]["hovertemplate"])
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
        np.testing.assert_array_almost_equal(std_y_data, chart_y_data, decimal=4)


def check_avg_and_counts(chartTests, comparison_data, chart_data):
    for idx in range(len(comparison_data)):

        # If the comparison average value is a nan then check that the chart value is too
        if math.isnan(comparison_data[idx][1]):
            chartTests.assertTrue(math.isnan(chart_data["y"][idx]))

        # Otherwise compare the average values
        else:
            chartTests.assertAlmostEqual(
                chart_data["y"][idx], comparison_data[idx][1], places=2
            )

        # Compare the frequency (count) data
        chartTests.assertEqual(
            chart_data["customdata"][idx][1], comparison_data[idx][2]
        )


def user_profile_reset(chartTests):
    chartTests.user.userprofile.plotCharts = True

    chartTests.user.userprofile.plotGroupingChoice = "system"
    chartTests.user.userprofile.plotSeriesPerSystem = False
    chartTests.user.userprofile.plotCaseInsensitiveCategories = False

    chartTests.user.userprofile.plotInitialSortingDirection = 0
    chartTests.user.userprofile.plotMGInitialSortingChoice = "frequency"

    chartTests.user.userprofile.plotAverageChoice = "mean"
    chartTests.user.userprofile.plotMean = False
    chartTests.user.userprofile.plotMedian = False
    chartTests.user.userprofile.plotBoxplots = False

    chartTests.user.userprofile.plotHistograms = False
    chartTests.user.userprofile.plotHistogramBins = 10

    chartTests.user.userprofile.plotMGaverageAGD = False
    chartTests.user.userprofile.plotMGacquisitionFreq = False
    chartTests.user.userprofile.plotMGAGDvsThickness = False
    chartTests.user.userprofile.plotMGmAsvsThickness = False
    chartTests.user.userprofile.plotMGkVpvsThickness = False
    chartTests.user.userprofile.plotMGaverageAGDvsThickness = False

    chartTests.user.userprofile.plotMGStudyPerDayAndHour = False

    chartTests.user.userprofile.plotCTAcquisitionMeanDLP = False
    chartTests.user.userprofile.plotCTAcquisitionMeanCTDI = False
    chartTests.user.userprofile.plotCTAcquisitionFreq = False
    chartTests.user.userprofile.plotCTAcquisitionCTDIvsMass = False
    chartTests.user.userprofile.plotCTAcquisitionDLPvsMass = False
    chartTests.user.userprofile.plotCTAcquisitionCTDIOverTime = False
    chartTests.user.userprofile.plotCTAcquisitionDLPOverTime = False

    chartTests.user.userprofile.plotCTSequencedAcquisition = True
    chartTests.user.userprofile.plotCTSpiralAcquisition = True
    chartTests.user.userprofile.plotCTConstantAngleAcquisition = True
    chartTests.user.userprofile.plotCTStationaryAcquisition = True
    chartTests.user.userprofile.plotCTFreeAcquisition = True

    chartTests.user.userprofile.plotCTStudyMeanDLP = False
    chartTests.user.userprofile.plotCTStudyMeanCTDI = False
    chartTests.user.userprofile.plotCTStudyFreq = False
    chartTests.user.userprofile.plotCTStudyNumEvents = False
    chartTests.user.userprofile.plotCTStudyPerDayAndHour = False
    chartTests.user.userprofile.plotCTStudyMeanDLPOverTime = False

    chartTests.user.userprofile.plotCTRequestMeanDLP = False
    chartTests.user.userprofile.plotCTRequestFreq = False
    chartTests.user.userprofile.plotCTRequestNumEvents = False
    chartTests.user.userprofile.plotCTRequestDLPOverTime = False

    chartTests.user.userprofile.plotDXAcquisitionMeanDAP = False
    chartTests.user.userprofile.plotDXAcquisitionMeankVp = False
    chartTests.user.userprofile.plotDXAcquisitionMeanmAs = False
    chartTests.user.userprofile.plotDXAcquisitionFreq = False
    chartTests.user.userprofile.plotDXAcquisitionDAPvsMass = False
    chartTests.user.userprofile.plotDXAcquisitionMeankVpOverTime = False
    chartTests.user.userprofile.plotDXAcquisitionMeanmAsOverTime = False
    chartTests.user.userprofile.plotDXAcquisitionMeanDAPOverTime = False

    chartTests.user.userprofile.plotDXStudyMeanDAP = False
    chartTests.user.userprofile.plotDXStudyFreq = False
    chartTests.user.userprofile.plotDXStudyDAPvsMass = False
    chartTests.user.userprofile.plotDXStudyPerDayAndHour = False

    chartTests.user.userprofile.plotDXRequestMeanDAP = False
    chartTests.user.userprofile.plotDXRequestFreq = False
    chartTests.user.userprofile.plotDXRequestDAPvsMass = False

    chartTests.user.userprofile.save()
