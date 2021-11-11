# This Python file uses the following encoding: utf-8
# test_charts_ct.py

import os
from django.contrib.auth.models import User, Group
from django.test import TestCase, RequestFactory
from remapp.extractors import rdsr
from remapp.models import (
    GeneralStudyModuleAttr,
    PatientIDSettings,
    StandardNameSettings,
)
from django.core.exceptions import ObjectDoesNotExist
from remapp.interface.mod_filters import CTSummaryListFilter
from remapp.tests.test_charts_common import (
    check_series_and_category_names,
    check_frequencies,
    check_boxplot_data,
    check_boxplot_xy,
    check_average_data,
    user_profile_reset,
)
import numpy as np


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

        try:
            StandardNameSettings.objects.get()
        except ObjectDoesNotExist:
            StandardNameSettings.objects.create()
        standard_name_settings = StandardNameSettings.objects.get()
        standard_name_settings.enable_standard_names = True
        standard_name_settings.save()

        ct1 = os.path.join("test_files", "CT-ESR-GE_Optima.dcm")
        ct2 = os.path.join("test_files", "CT-ESR-GE_VCT.dcm")
        ct3 = os.path.join("test_files", "CT-RDSR-GEPixelMed.dcm")
        ct4 = os.path.join("test_files", "CT-RDSR-Siemens_Flash-QA-DS.dcm")
        ct5 = os.path.join("test_files", "CT-RDSR-Siemens_Flash-TAP-SS.dcm")
        ct6 = os.path.join("test_files", "CT-RDSR-ToshibaPixelMed.dcm")
        root_tests = os.path.dirname(os.path.abspath(__file__))

        rdsr.rdsr(os.path.join(root_tests, ct1))
        rdsr.rdsr(os.path.join(root_tests, ct2))
        rdsr.rdsr(os.path.join(root_tests, ct3))
        rdsr.rdsr(os.path.join(root_tests, ct4))
        rdsr.rdsr(os.path.join(root_tests, ct5))
        rdsr.rdsr(os.path.join(root_tests, ct6))

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
        user_profile_reset(self)
        return f

    def obtain_chart_data(self, f):
        from remapp.views_charts_ct import ct_plot_calculations

        self.chart_data = ct_plot_calculations(
            f, self.user.userprofile, return_as_dict=True
        )

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
        self.user.userprofile.plotCTStandardAcquisitionMeanDLP = True
        self.user.userprofile.plotCTStandardAcquisitionMeanCTDI = True
        self.user.userprofile.plotCTStandardAcquisitionDLPOverTime = True
        self.user.userprofile.plotCTStandardAcquisitionCTDIOverTime = True
        self.user.userprofile.plotCTStandardAcquisitionDLPvsMass = True
        self.user.userprofile.plotCTStandardAcquisitionCTDIvsMass = True
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
        self.user.userprofile.plotCTStandardStudyMeanDLP = True
        self.user.userprofile.plotCTStandardStudyNumEvents = True
        self.user.userprofile.plotCTStandardStudyFreq = True
        self.user.userprofile.plotCTStandardStudyPerDayAndHour = True
        self.user.userprofile.plotCTStandardStudyMeanDLPOverTime = True
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
            "standardAcquisitionMeanDLP",
            "standardAcquisitionMedianDLP",
            "standardAcquisitionBoxplotDLP",
            "standardAcquisitionHistogramDLP",
            "standardAcquisitionMeanCTDI",
            "standardAcquisitionMedianCTDI",
            "standardAcquisitionBoxplotCTDI",
            "standardAcquisitionHistogramCTDI",
            "standardAcquisitionMeanDLPOverTime",
            "standardAcquisitionMedianDLPOverTime",
            "standardAcquisitionMeanCTDIOverTime",
            "standardAcquisitionMedianCTDIOverTime",
            "standardAcquisitionScatterCTDIvsMass",
            "standardAcquisitionScatterDLPvsMass",
            "studyMeanDLP",
            "studyMedianDLP",
            "studyBoxplotDLP",
            "studyHistogramDLP",
            "standardStudyMeanDLP",
            "standardStudyMedianDLP",
            "standardStudyBoxplotDLP",
            "standardStudyHistogramDLP",
            "standardStudyMeanNumEvents",
            "standardStudyMedianNumEvents",
            "standardStudyBoxplotNumEvents",
            "standardStudyHistogramNumEvents",
            "standardStudyFrequency",
            "standardStudyWorkload",
            "standardStudyMeanDLPOverTime",
            "standardStudyMedianDLPOverTime",
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
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the mean data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 111.3, 1],
                        ["All systems", 202.684, 16],
                        ["All systems", 29.67, 1],
                        ["All systems", 50.58, 1],
                        ["All systems", 129.89, 1],
                        ["All systems", 21.18, 1],
                        ["All systems", 24.05, 1],
                        ["All systems", 74.98, 2],
                        ["All systems", 369.34, 1],
                        ["All systems", 815.33, 1],
                        ["All systems", 3.61, 1],
                        ["All systems", 1.2, 1],
                        ["All systems", 708.2, 1],
                        ["All systems", 11.51, 1],
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
                        202.684,
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

        check_average_data(self, chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 111.3, 1],
                        ["All systems", 148.585, 16],
                        ["All systems", 29.67, 1],
                        ["All systems", 50.58, 1],
                        ["All systems", 129.89, 1],
                        ["All systems", 21.18, 1],
                        ["All systems", 24.05, 1],
                        ["All systems", 74.98, 2],
                        ["All systems", 369.34, 1],
                        ["All systems", 815.33, 1],
                        ["All systems", 3.61, 1],
                        ["All systems", 1.2, 1],
                        ["All systems", 708.2, 1],
                        ["All systems", 11.51, 1],
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

        check_average_data(self, chart_data, standard_data)

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

        check_boxplot_data(self, chart_data, standard_data)

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
                        ["A VCT Hospital VCTScanner", 182.035, 11],
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
                        182.035,
                        np.nan,
                        np.nan,
                        np.nan,
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
                        ["Gnats Bottom Hospital CTAWP91919", 29.67, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 50.58, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 129.89, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 21.18, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 24.05, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 74.98, 2],
                        ["Gnats Bottom Hospital CTAWP91919", 369.34, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 815.33, 1],
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
                        ["Hospital Number One Trust CTAWP00001", 3.61, 1],
                        ["Hospital Number One Trust CTAWP00001", 1.2, 1],
                        ["Hospital Number One Trust CTAWP00001", 708.2, 1],
                        ["Hospital Number One Trust CTAWP00001", 11.51, 1],
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
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", 207.91, 2],
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
                        ["OpenREM centre médical rt16", 111.3, 1],
                        ["OpenREM centre médical rt16", 475.04, 1],
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
                        ["Oxbridge County Hospital CTTOSHIBA1", 174.85, 2],
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

        check_average_data(self, chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["A VCT Hospital VCTScanner", np.nan, 0],
                        ["A VCT Hospital VCTScanner", 16.41, 11],
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
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ["Gnats Bottom Hospital CTAWP91919", 29.67, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 50.58, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 129.89, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 21.18, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 24.05, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 74.98, 2],
                        ["Gnats Bottom Hospital CTAWP91919", 369.34, 1],
                        ["Gnats Bottom Hospital CTAWP91919", 815.33, 1],
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
                        ["Hospital Number One Trust CTAWP00001", 3.61, 1],
                        ["Hospital Number One Trust CTAWP00001", 1.2, 1],
                        ["Hospital Number One Trust CTAWP00001", 708.2, 1],
                        ["Hospital Number One Trust CTAWP00001", 11.51, 1],
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
                        ["An Optima Hospital geoptima", np.nan, 0],
                        ["An Optima Hospital geoptima", 207.91, 2],
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
                        ["OpenREM centre médical rt16", 111.3, 1],
                        ["OpenREM centre médical rt16", 475.04, 1],
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
                        ["Oxbridge County Hospital CTTOSHIBA1", 174.85, 2],
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

        check_average_data(self, chart_data, standard_data)

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
                "name": "An Optima Hospital geoptima",
                "x": np.array(["Blank", "Blank"], dtype=object),
                "y": np.array([259.85, 155.97]),
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

        check_boxplot_data(self, chart_data, standard_data)

        # Check the histogram data
        standard_data = [
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "2.92≤x<91.65",
                        "91.65≤x<180.39",
                        "180.39≤x<269.12",
                        "269.12≤x<357.86",
                        "357.86≤x<446.59",
                        "446.59≤x<535.32",
                        "535.32≤x<624.06",
                        "624.06≤x<712.79",
                        "712.79≤x<801.53",
                        "801.53≤x<890.26",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [7, 0, 1, 1, 1, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        47.287,
                        136.021,
                        224.755,
                        313.489,
                        402.223,
                        490.957,
                        579.691,
                        668.425,
                        757.159,
                        845.893,
                    ],
                ),
            },
            {
                "name": "DS axial std",
                "text": np.array(
                    [
                        "21.18≤x<100.59",
                        "100.59≤x<180.01",
                        "180.01≤x<259.43",
                        "259.43≤x<338.84",
                        "338.84≤x<418.26",
                        "418.26≤x<497.67",
                        "497.67≤x<577.09",
                        "577.09≤x<656.50",
                        "656.50≤x<735.91",
                        "735.91≤x<815.33",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        60.8875,
                        140.3025,
                        219.7175,
                        299.1325,
                        378.5475,
                        457.9625,
                        537.3775,
                        616.7925,
                        696.2075,
                        775.6225,
                    ],
                ),
            },
            {
                "name": "DE_laser align",
                "text": np.array(
                    [
                        "21.18≤x<100.59",
                        "100.59≤x<180.01",
                        "180.01≤x<259.43",
                        "259.43≤x<338.84",
                        "338.84≤x<418.26",
                        "418.26≤x<497.67",
                        "497.67≤x<577.09",
                        "577.09≤x<656.50",
                        "656.50≤x<735.91",
                        "735.91≤x<815.33",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        60.8875,
                        140.3025,
                        219.7175,
                        299.1325,
                        378.5475,
                        457.9625,
                        537.3775,
                        616.7925,
                        696.2075,
                        775.6225,
                    ],
                ),
            },
            {
                "name": "DS 100kV",
                "text": np.array(
                    [
                        "21.18≤x<100.59",
                        "100.59≤x<180.01",
                        "180.01≤x<259.43",
                        "259.43≤x<338.84",
                        "338.84≤x<418.26",
                        "418.26≤x<497.67",
                        "497.67≤x<577.09",
                        "577.09≤x<656.50",
                        "656.50≤x<735.91",
                        "735.91≤x<815.33",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        60.8875,
                        140.3025,
                        219.7175,
                        299.1325,
                        378.5475,
                        457.9625,
                        537.3775,
                        616.7925,
                        696.2075,
                        775.6225,
                    ],
                ),
            },
            {
                "name": "DS 140kV",
                "text": np.array(
                    [
                        "21.18≤x<100.59",
                        "100.59≤x<180.01",
                        "180.01≤x<259.43",
                        "259.43≤x<338.84",
                        "338.84≤x<418.26",
                        "418.26≤x<497.67",
                        "497.67≤x<577.09",
                        "577.09≤x<656.50",
                        "656.50≤x<735.91",
                        "735.91≤x<815.33",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        60.8875,
                        140.3025,
                        219.7175,
                        299.1325,
                        378.5475,
                        457.9625,
                        537.3775,
                        616.7925,
                        696.2075,
                        775.6225,
                    ],
                ),
            },
            {
                "name": "DS 50mAs",
                "text": np.array(
                    [
                        "21.18≤x<100.59",
                        "100.59≤x<180.01",
                        "180.01≤x<259.43",
                        "259.43≤x<338.84",
                        "338.84≤x<418.26",
                        "418.26≤x<497.67",
                        "497.67≤x<577.09",
                        "577.09≤x<656.50",
                        "656.50≤x<735.91",
                        "735.91≤x<815.33",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        60.8875,
                        140.3025,
                        219.7175,
                        299.1325,
                        378.5475,
                        457.9625,
                        537.3775,
                        616.7925,
                        696.2075,
                        775.6225,
                    ],
                ),
            },
            {
                "name": "DS 80kV",
                "text": np.array(
                    [
                        "21.18≤x<100.59",
                        "100.59≤x<180.01",
                        "180.01≤x<259.43",
                        "259.43≤x<338.84",
                        "338.84≤x<418.26",
                        "418.26≤x<497.67",
                        "497.67≤x<577.09",
                        "577.09≤x<656.50",
                        "656.50≤x<735.91",
                        "735.91≤x<815.33",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        60.8875,
                        140.3025,
                        219.7175,
                        299.1325,
                        378.5475,
                        457.9625,
                        537.3775,
                        616.7925,
                        696.2075,
                        775.6225,
                    ],
                ),
            },
            {
                "name": "DS_hel p 0.23",
                "text": np.array(
                    [
                        "21.18≤x<100.59",
                        "100.59≤x<180.01",
                        "180.01≤x<259.43",
                        "259.43≤x<338.84",
                        "338.84≤x<418.26",
                        "418.26≤x<497.67",
                        "497.67≤x<577.09",
                        "577.09≤x<656.50",
                        "656.50≤x<735.91",
                        "735.91≤x<815.33",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        60.8875,
                        140.3025,
                        219.7175,
                        299.1325,
                        378.5475,
                        457.9625,
                        537.3775,
                        616.7925,
                        696.2075,
                        775.6225,
                    ],
                ),
            },
            {
                "name": "DS_helical",
                "text": np.array(
                    [
                        "21.18≤x<100.59",
                        "100.59≤x<180.01",
                        "180.01≤x<259.43",
                        "259.43≤x<338.84",
                        "338.84≤x<418.26",
                        "418.26≤x<497.67",
                        "497.67≤x<577.09",
                        "577.09≤x<656.50",
                        "656.50≤x<735.91",
                        "735.91≤x<815.33",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        60.8875,
                        140.3025,
                        219.7175,
                        299.1325,
                        378.5475,
                        457.9625,
                        537.3775,
                        616.7925,
                        696.2075,
                        775.6225,
                    ],
                ),
            },
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "155.97≤x<166.36",
                        "166.36≤x<176.75",
                        "176.75≤x<187.13",
                        "187.13≤x<197.52",
                        "197.52≤x<207.91",
                        "207.91≤x<218.30",
                        "218.30≤x<228.69",
                        "228.69≤x<239.07",
                        "239.07≤x<249.46",
                        "249.46≤x<259.85",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        161.164,
                        171.552,
                        181.94,
                        192.328,
                        202.716,
                        213.104,
                        223.492,
                        233.88,
                        244.268,
                        254.656,
                    ],
                ),
            },
            {
                "name": "Monitoring",
                "text": np.array(
                    [
                        "1.20≤x<71.90",
                        "71.90≤x<142.60",
                        "142.60≤x<213.30",
                        "213.30≤x<284.00",
                        "284.00≤x<354.70",
                        "354.70≤x<425.40",
                        "425.40≤x<496.10",
                        "496.10≤x<566.80",
                        "566.80≤x<637.50",
                        "637.50≤x<708.20",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        36.55,
                        107.25,
                        177.95,
                        248.65,
                        319.35,
                        390.05,
                        460.75,
                        531.45,
                        602.15,
                        672.85,
                    ],
                ),
            },
            {
                "name": "PreMonitoring",
                "text": np.array(
                    [
                        "1.20≤x<71.90",
                        "71.90≤x<142.60",
                        "142.60≤x<213.30",
                        "213.30≤x<284.00",
                        "284.00≤x<354.70",
                        "354.70≤x<425.40",
                        "425.40≤x<496.10",
                        "496.10≤x<566.80",
                        "566.80≤x<637.50",
                        "637.50≤x<708.20",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        36.55,
                        107.25,
                        177.95,
                        248.65,
                        319.35,
                        390.05,
                        460.75,
                        531.45,
                        602.15,
                        672.85,
                    ],
                ),
            },
            {
                "name": "TAP",
                "text": np.array(
                    [
                        "1.20≤x<71.90",
                        "71.90≤x<142.60",
                        "142.60≤x<213.30",
                        "213.30≤x<284.00",
                        "284.00≤x<354.70",
                        "354.70≤x<425.40",
                        "425.40≤x<496.10",
                        "496.10≤x<566.80",
                        "566.80≤x<637.50",
                        "637.50≤x<708.20",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        36.55,
                        107.25,
                        177.95,
                        248.65,
                        319.35,
                        390.05,
                        460.75,
                        531.45,
                        602.15,
                        672.85,
                    ],
                ),
            },
            {
                "name": "testÃ¦Ã¸Ã¥",
                "text": np.array(
                    [
                        "1.20≤x<71.90",
                        "71.90≤x<142.60",
                        "142.60≤x<213.30",
                        "213.30≤x<284.00",
                        "284.00≤x<354.70",
                        "354.70≤x<425.40",
                        "425.40≤x<496.10",
                        "496.10≤x<566.80",
                        "566.80≤x<637.50",
                        "637.50≤x<708.20",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        36.55,
                        107.25,
                        177.95,
                        248.65,
                        319.35,
                        390.05,
                        460.75,
                        531.45,
                        602.15,
                        672.85,
                    ],
                ),
            },
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "141.20≤x<147.93",
                        "147.93≤x<154.66",
                        "154.66≤x<161.39",
                        "161.39≤x<168.12",
                        "168.12≤x<174.85",
                        "174.85≤x<181.58",
                        "181.58≤x<188.31",
                        "188.31≤x<195.04",
                        "195.04≤x<201.77",
                        "201.77≤x<208.50",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        144.565,
                        151.295,
                        158.025,
                        164.755,
                        171.485,
                        178.215,
                        184.945,
                        191.675,
                        198.405,
                        205.135,
                    ],
                ),
            },
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "111.30≤x<147.67",
                        "147.67≤x<184.05",
                        "184.05≤x<220.42",
                        "220.42≤x<256.80",
                        "256.80≤x<293.17",
                        "293.17≤x<329.54",
                        "329.54≤x<365.92",
                        "365.92≤x<402.29",
                        "402.29≤x<438.67",
                        "438.67≤x<475.04",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        129.487,
                        165.861,
                        202.235,
                        238.609,
                        274.983,
                        311.357,
                        347.731,
                        384.105,
                        420.479,
                        456.853,
                    ],
                ),
            },
            {
                "name": "10.13 RADIOTHERAPY QA",
                "text": np.array(
                    [
                        "111.30≤x<147.67",
                        "147.67≤x<184.05",
                        "184.05≤x<220.42",
                        "220.42≤x<256.80",
                        "256.80≤x<293.17",
                        "293.17≤x<329.54",
                        "329.54≤x<365.92",
                        "365.92≤x<402.29",
                        "402.29≤x<438.67",
                        "438.67≤x<475.04",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        129.487,
                        165.861,
                        202.235,
                        238.609,
                        274.983,
                        311.357,
                        347.731,
                        384.105,
                        420.479,
                        456.853,
                    ],
                ),
            },
        ]
        chart_data = self.chart_data["acquisitionHistogramDLPData"]["data"]
        for idx, dataset in enumerate(standard_data):
            self.assertEqual(chart_data[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(
                chart_data[idx]["x"], dataset["x"], decimal=6
            )
            np.testing.assert_equal(chart_data[idx]["y"], dataset["y"])

    def test_study_dlp(self):
        # Test of mean and median study DLP
        # Also tests raw data going into the box plots
        # Also test histogram data. Then repeat above
        # with plotseriespersystem selected
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotCTStudyMeanDLP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the mean data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 468.02, 2],
                        ["All systems", 415.82, 1],
                        ["All systems", 2002.39, 1],
                        ["All systems", 1590.0, 1],
                        ["All systems", 724.52, 1],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "Colonography",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "Thorax^TAP (Adult)",
                    ],
                    dtype=object,
                ),
                "y": np.array([468.02, 415.82, 2002.39, 1590.00, 724.52]),
            }
        ]

        chart_data = self.chart_data["studyMeanDLPData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 468.02, 2],
                        ["All systems", 415.82, 1],
                        ["All systems", 2002.39, 1],
                        ["All systems", 1590.0, 1],
                        ["All systems", 724.52, 1],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "Colonography",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "Thorax^TAP (Adult)",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        468.02,
                        415.82,
                        2002.39,
                        1590.0,
                        724.52,
                    ]
                ),
            }
        ]

        chart_data = self.chart_data["studyMedianDLPData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Check the boxplot data
        standard_data = [
            {
                "name": "All systems",
                "x": np.array(
                    [
                        "Colonography",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "Thorax^TAP (Adult)",
                        "Blank",
                        "Blank",
                    ],
                    dtype=object,
                ),
                "y": np.array([415.82, 2002.39, 1590.0, 724.52, 586.34, 349.7]),
            }
        ]

        chart_data = self.chart_data["studyBoxplotDLPData"]["data"]
        check_boxplot_data(self, chart_data, standard_data)

        # Check the histogram data
        standard_data = [
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "349.70≤x<514.97",
                        "514.97≤x<680.24",
                        "680.24≤x<845.51",
                        "845.51≤x<1010.78",
                        "1010.78≤x<1176.05",
                        "1176.05≤x<1341.31",
                        "1341.31≤x<1506.58",
                        "1506.58≤x<1671.85",
                        "1671.85≤x<1837.12",
                        "1837.12≤x<2002.39",
                    ],
                    dtype="<U17",
                ),
                "y": np.array(
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        432.3345,
                        597.6035,
                        762.8725,
                        928.1415,
                        1093.4105,
                        1258.6795,
                        1423.9485,
                        1589.2175,
                        1754.4865,
                        1919.7555,
                    ],
                ),
            },
            {
                "name": "Colonography",
                "text": np.array(
                    [
                        "349.70≤x<514.97",
                        "514.97≤x<680.24",
                        "680.24≤x<845.51",
                        "845.51≤x<1010.78",
                        "1010.78≤x<1176.05",
                        "1176.05≤x<1341.31",
                        "1341.31≤x<1506.58",
                        "1506.58≤x<1671.85",
                        "1671.85≤x<1837.12",
                        "1837.12≤x<2002.39",
                    ],
                    dtype="<U17",
                ),
                "x": np.array(
                    [
                        432.3345,
                        597.6035,
                        762.8725,
                        928.1415,
                        1093.4105,
                        1258.6795,
                        1423.9485,
                        1589.2175,
                        1754.4865,
                        1919.7555,
                    ],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "FACIAL BONES",
                "text": np.array(
                    [
                        "349.70≤x<514.97",
                        "514.97≤x<680.24",
                        "680.24≤x<845.51",
                        "845.51≤x<1010.78",
                        "1010.78≤x<1176.05",
                        "1176.05≤x<1341.31",
                        "1341.31≤x<1506.58",
                        "1506.58≤x<1671.85",
                        "1671.85≤x<1837.12",
                        "1837.12≤x<2002.39",
                    ],
                    dtype="<U17",
                ),
                "x": np.array(
                    [
                        432.3345,
                        597.6035,
                        762.8725,
                        928.1415,
                        1093.4105,
                        1258.6795,
                        1423.9485,
                        1589.2175,
                        1754.4865,
                        1919.7555,
                    ],
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
            },
            {
                "name": "Specials^PhysicsTesting (Adult)",
                "text": np.array(
                    [
                        "349.70≤x<514.97",
                        "514.97≤x<680.24",
                        "680.24≤x<845.51",
                        "845.51≤x<1010.78",
                        "1010.78≤x<1176.05",
                        "1176.05≤x<1341.31",
                        "1341.31≤x<1506.58",
                        "1506.58≤x<1671.85",
                        "1671.85≤x<1837.12",
                        "1837.12≤x<2002.39",
                    ],
                    dtype="<U17",
                ),
                "x": np.array(
                    [
                        432.3345,
                        597.6035,
                        762.8725,
                        928.1415,
                        1093.4105,
                        1258.6795,
                        1423.9485,
                        1589.2175,
                        1754.4865,
                        1919.7555,
                    ],
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                ),
            },
            {
                "name": "Thorax^TAP (Adult)",
                "text": np.array(
                    [
                        "349.70≤x<514.97",
                        "514.97≤x<680.24",
                        "680.24≤x<845.51",
                        "845.51≤x<1010.78",
                        "1010.78≤x<1176.05",
                        "1176.05≤x<1341.31",
                        "1341.31≤x<1506.58",
                        "1506.58≤x<1671.85",
                        "1671.85≤x<1837.12",
                        "1837.12≤x<2002.39",
                    ],
                    dtype="<U17",
                ),
                "x": np.array(
                    [
                        432.3345,
                        597.6035,
                        762.8725,
                        928.1415,
                        1093.4105,
                        1258.6795,
                        1423.9485,
                        1589.2175,
                        1754.4865,
                        1919.7555,
                    ],
                ),
                "y": np.array(
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
        ]

        chart_data1 = self.chart_data["studyHistogramDLPData"]["data"]
        for idx, dataset in enumerate(standard_data):
            self.assertEqual(chart_data1[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(
                chart_data1[idx]["x"], dataset["x"], decimal=6
            )
            np.testing.assert_equal(chart_data1[idx]["y"], dataset["y"])

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
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", 2002.39, 1],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "A VCT Hospital VCTScanner",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            np.nan,
                            np.nan,
                            2002.39,
                            np.nan,
                            np.nan,
                        ]
                    ),
                },
                {
                    "customdata": np.array(
                        [
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", 415.82, 1],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            np.nan,
                            415.82,
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
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", 1590.0, 1],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            1590.0,
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
                            ["Hospital Number One Trust CTAWP00001", 724.52, 1],
                        ],
                        dtype=object,
                    ),
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            724.52,
                        ]
                    ),
                },
                {
                    "customdata": np.array(
                        [
                            ["OpenREM centre médical rt16", 586.34, 1],
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
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([586.34, np.nan, np.nan, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Oxbridge County Hospital CTTOSHIBA1", 349.7, 1],
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
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            349.7,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ]
                    ),
                },
            ]
            chart_data = self.chart_data["studyMeanDLPData"]["data"]
            check_average_data(self, chart_data, standard_data)

            # Test the median data
            standard_data = [
                {
                    "customdata": np.array(
                        [
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", 2002.39, 1],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "A VCT Hospital VCTScanner",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            np.nan,
                            np.nan,
                            2002.39,
                            np.nan,
                            np.nan,
                        ]
                    ),
                },
                {
                    "customdata": np.array(
                        [
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", 415.82, 1],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            np.nan,
                            415.82,
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
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", 1590.0, 1],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            1590.0,
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
                            ["Hospital Number One Trust CTAWP00001", 724.52, 1],
                        ],
                        dtype=object,
                    ),
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            724.52,
                        ]
                    ),
                },
                {
                    "customdata": np.array(
                        [
                            ["OpenREM centre médical rt16", 586.34, 1],
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
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([586.34, np.nan, np.nan, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Oxbridge County Hospital CTTOSHIBA1", 349.7, 1],
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
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [
                            349.7,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                        ]
                    ),
                },
            ]
            chart_data = self.chart_data["studyMedianDLPData"]["data"]

            check_average_data(self, chart_data, standard_data)

            # Check the boxplot data
            standard_data = [
                {
                    "name": "A VCT Hospital VCTScanner",
                    "x": np.array(
                        [
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([2002.39]),
                },
                {
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        [
                            "Colonography",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([415.82]),
                },
                {
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "Specials^PhysicsTesting (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([1590.0]),
                },
                {
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        [
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([724.52]),
                },
                {
                    "name": "OpenREM centre médical rt16",
                    "x": np.array(
                        [
                            "Blank",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([586.34]),
                },
                {
                    "name": "Oxbridge County Hospital CTTOSHIBA1",
                    "x": np.array(
                        [
                            "Blank",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([349.7]),
                },
            ]
        chart_data = self.chart_data["studyBoxplotDLPData"]["data"]

        check_boxplot_data(self, chart_data, standard_data)

        # Check the histogram data
        standard_data = [
            {
                "name": "FACIAL BONES",
                "text": np.array(
                    [
                        "2002.39≤x<2002.39",
                        "2002.39≤x<2002.39",
                        "2002.39≤x<2002.39",
                        "2002.39≤x<2002.39",
                        "2002.39≤x<2002.39",
                        "2002.39≤x<2002.39",
                        "2002.39≤x<2002.39",
                        "2002.39≤x<2002.39",
                        "2002.39≤x<2002.39",
                        "2002.39≤x<2002.39",
                    ],
                    dtype="<U17",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        2002.39,
                        2002.39,
                        2002.39,
                        2002.39,
                        2002.39,
                        2002.39,
                        2002.39,
                        2002.39,
                        2002.39,
                        2002.39,
                    ],
                ),
            },
            {
                "name": "Colonography",
                "text": np.array(
                    [
                        "415.82≤x<415.82",
                        "415.82≤x<415.82",
                        "415.82≤x<415.82",
                        "415.82≤x<415.82",
                        "415.82≤x<415.82",
                        "415.82≤x<415.82",
                        "415.82≤x<415.82",
                        "415.82≤x<415.82",
                        "415.82≤x<415.82",
                        "415.82≤x<415.82",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        415.82,
                        415.82,
                        415.82,
                        415.82,
                        415.82,
                        415.82,
                        415.82,
                        415.82,
                        415.82,
                        415.82,
                    ],
                ),
            },
            {
                "name": "Specials^PhysicsTesting (Adult)",
                "text": np.array(
                    [
                        "1590.00≤x<1590.00",
                        "1590.00≤x<1590.00",
                        "1590.00≤x<1590.00",
                        "1590.00≤x<1590.00",
                        "1590.00≤x<1590.00",
                        "1590.00≤x<1590.00",
                        "1590.00≤x<1590.00",
                        "1590.00≤x<1590.00",
                        "1590.00≤x<1590.00",
                        "1590.00≤x<1590.00",
                    ],
                    dtype="<U17",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        1590.0,
                        1590.0,
                        1590.0,
                        1590.0,
                        1590.0,
                        1590.0,
                        1590.0,
                        1590.0,
                        1590.0,
                        1590.0,
                    ],
                ),
            },
            {
                "name": "Thorax^TAP (Adult)",
                "text": np.array(
                    [
                        "724.52≤x<724.52",
                        "724.52≤x<724.52",
                        "724.52≤x<724.52",
                        "724.52≤x<724.52",
                        "724.52≤x<724.52",
                        "724.52≤x<724.52",
                        "724.52≤x<724.52",
                        "724.52≤x<724.52",
                        "724.52≤x<724.52",
                        "724.52≤x<724.52",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        724.52,
                        724.52,
                        724.52,
                        724.52,
                        724.52,
                        724.52,
                        724.52,
                        724.52,
                        724.52,
                        724.52,
                    ],
                ),
            },
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "586.34≤x<586.34",
                        "586.34≤x<586.34",
                        "586.34≤x<586.34",
                        "586.34≤x<586.34",
                        "586.34≤x<586.34",
                        "586.34≤x<586.34",
                        "586.34≤x<586.34",
                        "586.34≤x<586.34",
                        "586.34≤x<586.34",
                        "586.34≤x<586.34",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        586.34,
                        586.34,
                        586.34,
                        586.34,
                        586.34,
                        586.34,
                        586.34,
                        586.34,
                        586.34,
                        586.34,
                    ],
                ),
            },
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "349.70≤x<349.70",
                        "349.70≤x<349.70",
                        "349.70≤x<349.70",
                        "349.70≤x<349.70",
                        "349.70≤x<349.70",
                        "349.70≤x<349.70",
                        "349.70≤x<349.70",
                        "349.70≤x<349.70",
                        "349.70≤x<349.70",
                        "349.70≤x<349.70",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        349.7,
                        349.7,
                        349.7,
                        349.7,
                        349.7,
                        349.7,
                        349.7,
                        349.7,
                        349.7,
                        349.7,
                    ],
                ),
            },
        ]
        chart_data2 = self.chart_data["studyHistogramDLPData"]["data"]
        for idx, dataset in enumerate(standard_data):
            self.assertEqual(chart_data2[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(
                chart_data2[idx]["x"], dataset["x"], decimal=6
            )
            np.testing.assert_equal(chart_data2[idx]["y"], dataset["y"])

    def test_request_dlp(self):
        # Test of mean and median requested procedure DLP,
        # Also tests raw data going into the box plots
        # Also test histogram data. Then repeat above
        # with plotseriespersystem selected
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotCTRequestMeanDLP = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the mean data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 735.465, 4],
                        ["All systems", 724.52, 1],
                        ["All systems", 2002.39, 1],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "CT Thorax abdomen and pelvis with contrast",
                        "FACIAL BONES",
                    ],
                    dtype=object,
                ),
                "y": np.array([735.465, 724.52, 2002.39]),
            }
        ]

        chart_data = self.chart_data["requestMeanDLPData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 501.08, 4],
                        ["All systems", 724.52, 1],
                        ["All systems", 2002.39, 1],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "CT Thorax abdomen and pelvis with contrast",
                        "FACIAL BONES",
                    ],
                    dtype=object,
                ),
                "y": np.array([501.08, 724.52, 2002.39]),
            }
        ]

        chart_data = self.chart_data["requestMedianDLPData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Check the boxplot data
        standard_data = [
            {
                "name": "All systems",
                "x": np.array(
                    [
                        "CT Thorax abdomen and pelvis with contrast",
                        "FACIAL BONES",
                        "Blank",
                        "Blank",
                        "Blank",
                        "Blank",
                    ],
                    dtype=object,
                ),
                "y": np.array([724.52, 2002.39, 349.7, 415.82, 586.34, 1590.0]),
            }
        ]

        chart_data = self.chart_data["requestBoxplotDLPData"]["data"]
        check_boxplot_data(self, chart_data, standard_data)

        # Check the histogram data
        standard_data1 = [
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "349.70≤x<514.97",
                        "514.97≤x<680.24",
                        "680.24≤x<845.51",
                        "845.51≤x<1010.78",
                        "1010.78≤x<1176.05",
                        "1176.05≤x<1341.31",
                        "1341.31≤x<1506.58",
                        "1506.58≤x<1671.85",
                        "1671.85≤x<1837.12",
                        "1837.12≤x<2002.39",
                    ],
                    dtype="<U17",
                ),
                "y": np.array(
                    [2, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        432.3345,
                        597.6035,
                        762.8725,
                        928.1415,
                        1093.4105,
                        1258.6795,
                        1423.9485,
                        1589.2175,
                        1754.4865,
                        1919.7555,
                    ],
                ),
            },
            {
                "name": "CT Thorax abdomen and pelvis with contrast",
                "text": np.array(
                    [
                        "349.70≤x<514.97",
                        "514.97≤x<680.24",
                        "680.24≤x<845.51",
                        "845.51≤x<1010.78",
                        "1010.78≤x<1176.05",
                        "1176.05≤x<1341.31",
                        "1341.31≤x<1506.58",
                        "1506.58≤x<1671.85",
                        "1671.85≤x<1837.12",
                        "1837.12≤x<2002.39",
                    ],
                    dtype="<U17",
                ),
                "x": np.array(
                    [
                        432.3345,
                        597.6035,
                        762.8725,
                        928.1415,
                        1093.4105,
                        1258.6795,
                        1423.9485,
                        1589.2175,
                        1754.4865,
                        1919.7555,
                    ],
                ),
                "y": np.array(
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "FACIAL BONES",
                "text": np.array(
                    [
                        "349.70≤x<514.97",
                        "514.97≤x<680.24",
                        "680.24≤x<845.51",
                        "845.51≤x<1010.78",
                        "1010.78≤x<1176.05",
                        "1176.05≤x<1341.31",
                        "1341.31≤x<1506.58",
                        "1506.58≤x<1671.85",
                        "1671.85≤x<1837.12",
                        "1837.12≤x<2002.39",
                    ],
                    dtype="<U17",
                ),
                "x": np.array(
                    [
                        432.3345,
                        597.6035,
                        762.8725,
                        928.1415,
                        1093.4105,
                        1258.6795,
                        1423.9485,
                        1589.2175,
                        1754.4865,
                        1919.7555,
                    ],
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
            },
        ]

        chart_data1 = self.chart_data["requestHistogramDLPData"]["data"]

        for idx, dataset in enumerate(standard_data1):
            self.assertEqual(chart_data1[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(
                chart_data1[idx]["x"], dataset["x"], decimal=6
            )
            np.testing.assert_equal(chart_data1[idx]["y"], dataset["y"])

            # Almost equal used for equivalence because the chart data isn't equal to the standard data
            # at a high number of decimal places
            # chart_data renamed chart_data1 in this instance so that the for loop above related to the histograms
            # does not apply to all other tests

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
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", 2002.39, 1],
                        ],
                        dtype=object,
                    ),
                    "name": "A VCT Hospital VCTScanner",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, np.nan, 2002.39]),
                },
                {
                    "customdata": np.array(
                        [
                            ["An Optima Hospital geoptima", 415.82, 1],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([415.82, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Gnats Bottom Hospital CTAWP91919", 1590.0, 1],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([1590.0, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", 724.52, 1],
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, 724.52, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["OpenREM centre médical rt16", 586.34, 1],
                            ["OpenREM centre médical rt16", np.nan, 0],
                            ["OpenREM centre médical rt16", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "OpenREM centre médical rt16",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([586.34, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Oxbridge County Hospital CTTOSHIBA1", 349.7, 1],
                            ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                            ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Oxbridge County Hospital CTTOSHIBA1",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([349.7, np.nan, np.nan]),
                },
            ]
            chart_data = self.chart_data["requestMeanDLPData"]["data"]

            check_average_data(self, chart_data, standard_data)

            # Test the median data
            standard_data = [
                {
                    "customdata": np.array(
                        [
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", 2002.39, 1],
                        ],
                        dtype=object,
                    ),
                    "name": "A VCT Hospital VCTScanner",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, np.nan, 2002.39]),
                },
                {
                    "customdata": np.array(
                        [
                            ["An Optima Hospital geoptima", 415.82, 1],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([415.82, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Gnats Bottom Hospital CTAWP91919", 1590.0, 1],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([1590.0, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", 724.52, 1],
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, 724.52, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["OpenREM centre médical rt16", 586.34, 1],
                            ["OpenREM centre médical rt16", np.nan, 0],
                            ["OpenREM centre médical rt16", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "OpenREM centre médical rt16",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([586.34, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Oxbridge County Hospital CTTOSHIBA1", 349.7, 1],
                            ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                            ["Oxbridge County Hospital CTTOSHIBA1", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Oxbridge County Hospital CTTOSHIBA1",
                    "x": np.array(
                        [
                            "Blank",
                            "CT Thorax abdomen and pelvis with contrast",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([349.7, np.nan, np.nan]),
                },
            ]
            chart_data = self.chart_data["requestMedianDLPData"]["data"]

            check_average_data(self, chart_data, standard_data)

            # Check the boxplot data
            standard_data = [
                {
                    "name": "A VCT Hospital VCTScanner",
                    "x": np.array(
                        [
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([2002.39]),
                },
                {
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        [
                            "Blank",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([415.82]),
                },
                {
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "Blank",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([1590.0]),
                },
                {
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        [
                            "CT Thorax abdomen and pelvis with contrast",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([724.52]),
                },
                {
                    "name": "OpenREM centre médical rt16",
                    "x": np.array(
                        [
                            "Blank",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([586.34]),
                },
                {
                    "name": "Oxbridge County Hospital CTTOSHIBA1",
                    "x": np.array(
                        [
                            "Blank",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([349.7]),
                },
            ]
            chart_data = self.chart_data["requestBoxplotDLPData"]["data"]

            check_boxplot_data(self, chart_data, standard_data)

            # Check the histogram data
            standard_data = [
                {
                    "name": "FACIAL BONES",
                    "text": np.array(
                        [
                            "2002.39≤x<2002.39",
                            "2002.39≤x<2002.39",
                            "2002.39≤x<2002.39",
                            "2002.39≤x<2002.39",
                            "2002.39≤x<2002.39",
                            "2002.39≤x<2002.39",
                            "2002.39≤x<2002.39",
                            "2002.39≤x<2002.39",
                            "2002.39≤x<2002.39",
                            "2002.39≤x<2002.39",
                        ],
                        dtype="<U17",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            2002.39,
                            2002.39,
                            2002.39,
                            2002.39,
                            2002.39,
                            2002.39,
                            2002.39,
                            2002.39,
                            2002.39,
                            2002.39,
                        ],
                    ),
                },
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "415.82≤x<415.82",
                            "415.82≤x<415.82",
                            "415.82≤x<415.82",
                            "415.82≤x<415.82",
                            "415.82≤x<415.82",
                            "415.82≤x<415.82",
                            "415.82≤x<415.82",
                            "415.82≤x<415.82",
                            "415.82≤x<415.82",
                            "415.82≤x<415.82",
                        ],
                        dtype="<U15",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            415.82,
                            415.82,
                            415.82,
                            415.82,
                            415.82,
                            415.82,
                            415.82,
                            415.82,
                            415.82,
                            415.82,
                        ],
                    ),
                },
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "1590.00≤x<1590.00",
                            "1590.00≤x<1590.00",
                            "1590.00≤x<1590.00",
                            "1590.00≤x<1590.00",
                            "1590.00≤x<1590.00",
                            "1590.00≤x<1590.00",
                            "1590.00≤x<1590.00",
                            "1590.00≤x<1590.00",
                            "1590.00≤x<1590.00",
                            "1590.00≤x<1590.00",
                        ],
                        dtype="<U17",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            1590.0,
                            1590.0,
                            1590.0,
                            1590.0,
                            1590.0,
                            1590.0,
                            1590.0,
                            1590.0,
                            1590.0,
                            1590.0,
                        ],
                    ),
                },
                {
                    "name": "CT Thorax abdomen and pelvis with contrast",
                    "text": np.array(
                        [
                            "724.52≤x<724.52",
                            "724.52≤x<724.52",
                            "724.52≤x<724.52",
                            "724.52≤x<724.52",
                            "724.52≤x<724.52",
                            "724.52≤x<724.52",
                            "724.52≤x<724.52",
                            "724.52≤x<724.52",
                            "724.52≤x<724.52",
                            "724.52≤x<724.52",
                        ],
                        dtype="<U15",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            724.52,
                            724.52,
                            724.52,
                            724.52,
                            724.52,
                            724.52,
                            724.52,
                            724.52,
                            724.52,
                            724.52,
                        ],
                    ),
                },
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "586.34≤x<586.34",
                            "586.34≤x<586.34",
                            "586.34≤x<586.34",
                            "586.34≤x<586.34",
                            "586.34≤x<586.34",
                            "586.34≤x<586.34",
                            "586.34≤x<586.34",
                            "586.34≤x<586.34",
                            "586.34≤x<586.34",
                            "586.34≤x<586.34",
                        ],
                        dtype="<U15",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            586.34,
                            586.34,
                            586.34,
                            586.34,
                            586.34,
                            586.34,
                            586.34,
                            586.34,
                            586.34,
                            586.34,
                        ],
                    ),
                },
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "349.70≤x<349.70",
                            "349.70≤x<349.70",
                            "349.70≤x<349.70",
                            "349.70≤x<349.70",
                            "349.70≤x<349.70",
                            "349.70≤x<349.70",
                            "349.70≤x<349.70",
                            "349.70≤x<349.70",
                            "349.70≤x<349.70",
                            "349.70≤x<349.70",
                        ],
                        dtype="<U15",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            349.7,
                            349.7,
                            349.7,
                            349.7,
                            349.7,
                            349.7,
                            349.7,
                            349.7,
                            349.7,
                            349.7,
                        ],
                    ),
                },
            ]
            chart_data = self.chart_data["requestHistogramDLPData"]["data"]
            for idx, dataset in enumerate(standard_data):
                self.assertEqual(chart_data[idx]["name"], dataset["name"])
                np.testing.assert_almost_equal(
                    chart_data[idx]["x"], dataset["x"], decimal=6
                )
                np.testing.assert_equal(chart_data[idx]["y"], dataset["y"])

    def test_acq_ctdi(self):
        # Test of mean and median acquisition CTDI,
        # Also tests raw data going into the box plots
        # Also test histogram data. Then repeat above
        # with plotseriespersystem selected
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotCTAcquisitionMeanCTDI = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the mean data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 222.59, 1],
                        ["All systems", 31.194, 16],
                        ["All systems", 15.45, 1],
                        ["All systems", 13.17, 1],
                        ["All systems", 33.83, 1],
                        ["All systems", 5.52, 1],
                        ["All systems", 6.26, 1],
                        ["All systems", 19.525, 2],
                        ["All systems", 29.67, 1],
                        ["All systems", 65.47, 1],
                        ["All systems", 3.61, 1],
                        ["All systems", 1.2, 1],
                        ["All systems", 9.91, 1],
                        ["All systems", 0.14, 1],
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
                        2.225900e02,
                        3.119375e01,
                        1.545000e01,
                        1.317000e01,
                        3.383000e01,
                        5.520000e00,
                        6.260000e00,
                        1.952500e01,
                        2.967000e01,
                        6.547000e01,
                        3.610000e00,
                        1.200000e00,
                        9.910000e00,
                        1.400000e-01,
                    ]
                ),
            }
        ]

        chart_data = self.chart_data["acquisitionMeanCTDIData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 222.59, 1],
                        ["All systems", 25.05, 16],
                        ["All systems", 15.45, 1],
                        ["All systems", 13.17, 1],
                        ["All systems", 33.83, 1],
                        ["All systems", 5.52, 1],
                        ["All systems", 6.26, 1],
                        ["All systems", 19.525, 2],
                        ["All systems", 29.67, 1],
                        ["All systems", 65.47, 1],
                        ["All systems", 3.61, 1],
                        ["All systems", 1.2, 1],
                        ["All systems", 9.91, 1],
                        ["All systems", 0.14, 1],
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
                        2.2259e02,
                        2.5050e01,
                        1.5450e01,
                        1.3170e01,
                        3.3830e01,
                        5.5200e00,
                        6.2600e00,
                        1.9525e01,
                        2.9670e01,
                        6.5470e01,
                        3.6100e00,
                        1.2000e00,
                        9.9100e00,
                        1.4000e-01,
                    ]
                ),
            }
        ]

        chart_data = self.chart_data["acquisitionMedianCTDIData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Check the boxplot data
        standard_data = [
            {
                "name": "All systems",
                "x": np.array(
                    [
                        "10.13 RADIOTHERAPY QA",
                        "DE_laser align",
                        "DS 100kV",
                        "DS 140kV",
                        "DS 50mAs",
                        "DS 80kV",
                        "DS axial std",
                        "DS axial std",
                        "DS_hel p 0.23",
                        "DS_helical",
                        "Monitoring",
                        "PreMonitoring",
                        "TAP",
                        "testÃ¦Ã¸Ã¥",
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
                        "Blank",
                        "Blank",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        2.2259e02,
                        1.5450e01,
                        1.3170e01,
                        3.3830e01,
                        5.5200e00,
                        6.2600e00,
                        2.1950e01,
                        1.7100e01,
                        2.9670e01,
                        6.5470e01,
                        3.6100e00,
                        1.2000e00,
                        9.9100e00,
                        1.4000e-01,
                        3.2300e00,
                        5.3000e00,
                        3.2830e01,
                        8.7400e00,
                        4.9300e00,
                        6.2300e00,
                        2.2260e01,
                        5.8400e00,
                        1.7612e02,
                        2.9310e01,
                        2.9310e01,
                        3.1660e01,
                        3.2830e01,
                        6.0410e01,
                        2.5400e01,
                        2.4700e01,
                    ]
                ),
            }
        ]

        chart_data = self.chart_data["acquisitionBoxplotCTDIData"]["data"]
        check_boxplot_data(self, chart_data, standard_data)

        # Check the histogram data
        standard_data1 = [
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [7, 7, 1, 0, 0, 0, 0, 1, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
            },
            {
                "name": "DS axial std",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "10.13 RADIOTHERAPY QA",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
            },
            {
                "name": "DE_laser align",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "DS 100kV",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "DS 140kV",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "DS 50mAs",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "DS 80kV",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "DS_hel p 0.23",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "DS_helical",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "Monitoring",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "PreMonitoring",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "TAP",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "testÃ¦Ã¸Ã¥",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
        ]

        chart_data1 = self.chart_data["acquisitionHistogramCTDIData"]["data"]

        for idx, dataset in enumerate(standard_data1):
            self.assertEqual(chart_data1[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(
                chart_data1[idx]["x"], dataset["x"], decimal=6
            )
            np.testing.assert_equal(chart_data1[idx]["y"], dataset["y"])

            # Almost equal used for equivalence because the chart data isn't equal to the standard data
            # at a high number of decimal places
            # chart_data renamed chart_data1 in this instance so that the for loop above related to the histograms
            # does not apply to all other tests

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
                            ["A VCT Hospital VCTScanner", 34.55, 11],
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
                            34.55,
                            np.nan,
                            np.nan,
                            np.nan,
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
                            ["Gnats Bottom Hospital CTAWP91919", 15.45, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 13.17, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 33.83, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 5.52, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 6.26, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 19.525, 2],
                            ["Gnats Bottom Hospital CTAWP91919", 29.67, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 65.47, 1],
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
                            15.45,
                            13.17,
                            33.83,
                            5.52,
                            6.26,
                            19.525,
                            29.67,
                            65.47,
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
                            ["Hospital Number One Trust CTAWP00001", 3.61, 1],
                            ["Hospital Number One Trust CTAWP00001", 1.2, 1],
                            ["Hospital Number One Trust CTAWP00001", 9.91, 1],
                            ["Hospital Number One Trust CTAWP00001", 0.14, 1],
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
                            9.91,
                            0.14,
                        ]
                    ),
                },
                {
                    "customdata": np.array(
                        [
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", 4.265, 2],
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
                            4.265,
                            np.nan,
                            np.nan,
                            np.nan,
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
                            ["OpenREM centre médical rt16", 222.59, 1],
                            ["OpenREM centre médical rt16", 60.41, 1],
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
                            222.59,
                            60.41,
                            np.nan,
                            np.nan,
                            np.nan,
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
                            [
                                "Oxbridge County Hospital CTTOSHIBA1",
                                25.05,
                                2,
                            ],
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
                            25.05,
                            np.nan,
                            np.nan,
                            np.nan,
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

            chart_data = self.chart_data["acquisitionMeanCTDIData"]["data"]

            check_average_data(self, chart_data, standard_data)

            # Test the median data
            standard_data = [
                {
                    "customdata": np.array(
                        [
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", 29.31, 11],
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
                            29.31,
                            np.nan,
                            np.nan,
                            np.nan,
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
                            ["Gnats Bottom Hospital CTAWP91919", 15.45, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 13.17, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 33.83, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 5.52, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 6.26, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 19.525, 2],
                            ["Gnats Bottom Hospital CTAWP91919", 29.67, 1],
                            ["Gnats Bottom Hospital CTAWP91919", 65.47, 1],
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
                            15.45,
                            13.17,
                            33.83,
                            5.52,
                            6.26,
                            19.525,
                            29.67,
                            65.47,
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
                            ["Hospital Number One Trust CTAWP00001", 3.61, 1],
                            ["Hospital Number One Trust CTAWP00001", 1.2, 1],
                            ["Hospital Number One Trust CTAWP00001", 9.91, 1],
                            ["Hospital Number One Trust CTAWP00001", 0.14, 1],
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
                            9.91,
                            0.14,
                        ]
                    ),
                },
                {
                    "customdata": np.array(
                        [
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", 4.265, 2],
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
                            4.265,
                            np.nan,
                            np.nan,
                            np.nan,
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
                            ["OpenREM centre médical rt16", 222.59, 1],
                            ["OpenREM centre médical rt16", 60.41, 1],
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
                            222.59,
                            60.41,
                            np.nan,
                            np.nan,
                            np.nan,
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
                            [
                                "Oxbridge County Hospital CTTOSHIBA1",
                                25.05,
                                2,
                            ],
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
                            25.05,
                            np.nan,
                            np.nan,
                            np.nan,
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

            chart_data = self.chart_data["acquisitionMedianCTDIData"]["data"]

            check_average_data(self, chart_data, standard_data)

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
                            8.74,
                            4.93,
                            6.23,
                            22.26,
                            5.84,
                            176.12,
                            29.31,
                            29.31,
                            32.83,
                            31.66,
                            32.83,
                        ]
                    ),
                },
                {
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "DE_laser align",
                            "DS 100kV",
                            "DS 140kV",
                            "DS 50mAs",
                            "DS 80kV",
                            "DS axial std",
                            "DS_helical",
                            "DS_hel p 0.23",
                            "DS axial std",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [15.45, 13.17, 33.83, 5.52, 6.26, 21.95, 65.47, 29.67, 17.1]
                    ),
                },
                {
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        ["Monitoring", "TAP", "testÃ¦Ã¸Ã¥", "PreMonitoring"],
                        dtype=object,
                    ),
                    "y": np.array([3.61, 9.91, 0.14, 1.2]),
                },
                {
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        ["Blank", "Blank"],
                        dtype=object,
                    ),
                    "y": np.array([5.3, 3.23]),
                },
                {
                    "name": "OpenREM centre médical rt16",
                    "x": np.array(
                        ["Blank", "10.13 RADIOTHERAPY QA"],
                        dtype=object,
                    ),
                    "y": np.array([60.41, 222.59]),
                },
                {
                    "name": "Oxbridge County Hospital CTTOSHIBA1",
                    "x": np.array(
                        ["Blank", "Blank"],
                        dtype=object,
                    ),
                    "y": np.array([25.4, 24.7]),
                },
            ]

            chart_data = self.chart_data["acquisitionBoxplotCTDIData"]["data"]
            check_boxplot_data(self, chart_data, standard_data)

            # Check the histogram data
            standard_data1 = [
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "4.93≤x<22.05",
                            "22.05≤x<39.17",
                            "39.17≤x<56.29",
                            "56.29≤x<73.41",
                            "73.41≤x<90.53",
                            "90.53≤x<107.64",
                            "107.64≤x<124.76",
                            "124.76≤x<141.88",
                            "141.88≤x<159.00",
                            "159.00≤x<176.12",
                        ],
                        dtype="<U15",
                    ),
                    "y": np.array(
                        [4, 6, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            13.4895,
                            30.6085,
                            47.7275,
                            64.8465,
                            81.9655,
                            99.0845,
                            116.2035,
                            133.3225,
                            150.4415,
                            167.5605,
                        ],
                    ),
                },
                {
                    "name": "DS axial std",
                    "text": np.array(
                        [
                            "5.52≤x<11.52",
                            "11.52≤x<17.51",
                            "17.51≤x<23.50",
                            "23.50≤x<29.50",
                            "29.50≤x<35.50",
                            "35.50≤x<41.49",
                            "41.49≤x<47.48",
                            "47.48≤x<53.48",
                            "53.48≤x<59.47",
                            "59.47≤x<65.47",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            8.5175,
                            14.5125,
                            20.5075,
                            26.5025,
                            32.4975,
                            38.4925,
                            44.4875,
                            50.4825,
                            56.4775,
                            62.4725,
                        ],
                    ),
                },
                {
                    "name": "DE_laser align",
                    "text": np.array(
                        [
                            "5.52≤x<11.52",
                            "11.52≤x<17.51",
                            "17.51≤x<23.50",
                            "23.50≤x<29.50",
                            "29.50≤x<35.50",
                            "35.50≤x<41.49",
                            "41.49≤x<47.48",
                            "47.48≤x<53.48",
                            "53.48≤x<59.47",
                            "59.47≤x<65.47",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            8.5175,
                            14.5125,
                            20.5075,
                            26.5025,
                            32.4975,
                            38.4925,
                            44.4875,
                            50.4825,
                            56.4775,
                            62.4725,
                        ],
                    ),
                },
                {
                    "name": "DS 100kV",
                    "text": np.array(
                        [
                            "5.52≤x<11.52",
                            "11.52≤x<17.51",
                            "17.51≤x<23.50",
                            "23.50≤x<29.50",
                            "29.50≤x<35.50",
                            "35.50≤x<41.49",
                            "41.49≤x<47.48",
                            "47.48≤x<53.48",
                            "53.48≤x<59.47",
                            "59.47≤x<65.47",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            8.5175,
                            14.5125,
                            20.5075,
                            26.5025,
                            32.4975,
                            38.4925,
                            44.4875,
                            50.4825,
                            56.4775,
                            62.4725,
                        ],
                    ),
                },
                {
                    "name": "DS 140kV",
                    "text": np.array(
                        [
                            "5.52≤x<11.52",
                            "11.52≤x<17.51",
                            "17.51≤x<23.50",
                            "23.50≤x<29.50",
                            "29.50≤x<35.50",
                            "35.50≤x<41.49",
                            "41.49≤x<47.48",
                            "47.48≤x<53.48",
                            "53.48≤x<59.47",
                            "59.47≤x<65.47",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            8.5175,
                            14.5125,
                            20.5075,
                            26.5025,
                            32.4975,
                            38.4925,
                            44.4875,
                            50.4825,
                            56.4775,
                            62.4725,
                        ],
                    ),
                },
                {
                    "name": "DS 50mAs",
                    "text": np.array(
                        [
                            "5.52≤x<11.52",
                            "11.52≤x<17.51",
                            "17.51≤x<23.50",
                            "23.50≤x<29.50",
                            "29.50≤x<35.50",
                            "35.50≤x<41.49",
                            "41.49≤x<47.48",
                            "47.48≤x<53.48",
                            "53.48≤x<59.47",
                            "59.47≤x<65.47",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            8.5175,
                            14.5125,
                            20.5075,
                            26.5025,
                            32.4975,
                            38.4925,
                            44.4875,
                            50.4825,
                            56.4775,
                            62.4725,
                        ],
                    ),
                },
                {
                    "name": "DS 80kV",
                    "text": np.array(
                        [
                            "5.52≤x<11.52",
                            "11.52≤x<17.51",
                            "17.51≤x<23.50",
                            "23.50≤x<29.50",
                            "29.50≤x<35.50",
                            "35.50≤x<41.49",
                            "41.49≤x<47.48",
                            "47.48≤x<53.48",
                            "53.48≤x<59.47",
                            "59.47≤x<65.47",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            8.5175,
                            14.5125,
                            20.5075,
                            26.5025,
                            32.4975,
                            38.4925,
                            44.4875,
                            50.4825,
                            56.4775,
                            62.4725,
                        ],
                    ),
                },
                {
                    "name": "DS_hel p 0.23",
                    "text": np.array(
                        [
                            "5.52≤x<11.52",
                            "11.52≤x<17.51",
                            "17.51≤x<23.50",
                            "23.50≤x<29.50",
                            "29.50≤x<35.50",
                            "35.50≤x<41.49",
                            "41.49≤x<47.48",
                            "47.48≤x<53.48",
                            "53.48≤x<59.47",
                            "59.47≤x<65.47",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            8.5175,
                            14.5125,
                            20.5075,
                            26.5025,
                            32.4975,
                            38.4925,
                            44.4875,
                            50.4825,
                            56.4775,
                            62.4725,
                        ],
                    ),
                },
                {
                    "name": "DS_helical",
                    "text": np.array(
                        [
                            "5.52≤x<11.52",
                            "11.52≤x<17.51",
                            "17.51≤x<23.50",
                            "23.50≤x<29.50",
                            "29.50≤x<35.50",
                            "35.50≤x<41.49",
                            "41.49≤x<47.48",
                            "47.48≤x<53.48",
                            "53.48≤x<59.47",
                            "59.47≤x<65.47",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            8.5175,
                            14.5125,
                            20.5075,
                            26.5025,
                            32.4975,
                            38.4925,
                            44.4875,
                            50.4825,
                            56.4775,
                            62.4725,
                        ],
                    ),
                },
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "3.23≤x<3.44",
                            "3.44≤x<3.64",
                            "3.64≤x<3.85",
                            "3.85≤x<4.06",
                            "4.06≤x<4.26",
                            "4.26≤x<4.47",
                            "4.47≤x<4.68",
                            "4.68≤x<4.89",
                            "4.89≤x<5.09",
                            "5.09≤x<5.30",
                        ],
                        dtype="<U11",
                    ),
                    "y": np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            3.3335,
                            3.5405,
                            3.7475,
                            3.9545,
                            4.1615,
                            4.3685,
                            4.5755,
                            4.7825,
                            4.9895,
                            5.1965,
                        ],
                    ),
                },
                {
                    "name": "Monitoring",
                    "text": np.array(
                        [
                            "0.14≤x<1.12",
                            "1.12≤x<2.09",
                            "2.09≤x<3.07",
                            "3.07≤x<4.05",
                            "4.05≤x<5.02",
                            "5.02≤x<6.00",
                            "6.00≤x<6.98",
                            "6.98≤x<7.96",
                            "7.96≤x<8.93",
                            "8.93≤x<9.91",
                        ],
                        dtype="<U11",
                    ),
                    "y": np.array(
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            0.6285,
                            1.6055,
                            2.5825,
                            3.5595,
                            4.5365,
                            5.5135,
                            6.4905,
                            7.4675,
                            8.4445,
                            9.4215,
                        ],
                    ),
                },
                {
                    "name": "PreMonitoring",
                    "text": np.array(
                        [
                            "0.14≤x<1.12",
                            "1.12≤x<2.09",
                            "2.09≤x<3.07",
                            "3.07≤x<4.05",
                            "4.05≤x<5.02",
                            "5.02≤x<6.00",
                            "6.00≤x<6.98",
                            "6.98≤x<7.96",
                            "7.96≤x<8.93",
                            "8.93≤x<9.91",
                        ],
                        dtype="<U11",
                    ),
                    "y": np.array(
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            0.6285,
                            1.6055,
                            2.5825,
                            3.5595,
                            4.5365,
                            5.5135,
                            6.4905,
                            7.4675,
                            8.4445,
                            9.4215,
                        ],
                    ),
                },
                {
                    "name": "TAP",
                    "text": np.array(
                        [
                            "0.14≤x<1.12",
                            "1.12≤x<2.09",
                            "2.09≤x<3.07",
                            "3.07≤x<4.05",
                            "4.05≤x<5.02",
                            "5.02≤x<6.00",
                            "6.00≤x<6.98",
                            "6.98≤x<7.96",
                            "7.96≤x<8.93",
                            "8.93≤x<9.91",
                        ],
                        dtype="<U11",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            0.6285,
                            1.6055,
                            2.5825,
                            3.5595,
                            4.5365,
                            5.5135,
                            6.4905,
                            7.4675,
                            8.4445,
                            9.4215,
                        ],
                    ),
                },
                {
                    "name": "testÃ¦Ã¸Ã¥",
                    "text": np.array(
                        [
                            "0.14≤x<1.12",
                            "1.12≤x<2.09",
                            "2.09≤x<3.07",
                            "3.07≤x<4.05",
                            "4.05≤x<5.02",
                            "5.02≤x<6.00",
                            "6.00≤x<6.98",
                            "6.98≤x<7.96",
                            "7.96≤x<8.93",
                            "8.93≤x<9.91",
                        ],
                        dtype="<U11",
                    ),
                    "y": np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            0.6285,
                            1.6055,
                            2.5825,
                            3.5595,
                            4.5365,
                            5.5135,
                            6.4905,
                            7.4675,
                            8.4445,
                            9.4215,
                        ],
                    ),
                },
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "24.70≤x<24.77",
                            "24.77≤x<24.84",
                            "24.84≤x<24.91",
                            "24.91≤x<24.98",
                            "24.98≤x<25.05",
                            "25.05≤x<25.12",
                            "25.12≤x<25.19",
                            "25.19≤x<25.26",
                            "25.26≤x<25.33",
                            "25.33≤x<25.40",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            24.735,
                            24.805,
                            24.875,
                            24.945,
                            25.015,
                            25.085,
                            25.155,
                            25.225,
                            25.295,
                            25.365,
                        ],
                    ),
                },
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "60.41≤x<76.63",
                            "76.63≤x<92.85",
                            "92.85≤x<109.06",
                            "109.06≤x<125.28",
                            "125.28≤x<141.50",
                            "141.50≤x<157.72",
                            "157.72≤x<173.94",
                            "173.94≤x<190.15",
                            "190.15≤x<206.37",
                            "206.37≤x<222.59",
                        ],
                        dtype="<U15",
                    ),
                    "y": np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            68.519,
                            84.737,
                            100.955,
                            117.173,
                            133.391,
                            149.609,
                            165.827,
                            182.045,
                            198.263,
                            214.481,
                        ],
                    ),
                },
                {
                    "name": "10.13 RADIOTHERAPY QA",
                    "text": np.array(
                        [
                            "60.41≤x<76.63",
                            "76.63≤x<92.85",
                            "92.85≤x<109.06",
                            "109.06≤x<125.28",
                            "125.28≤x<141.50",
                            "141.50≤x<157.72",
                            "157.72≤x<173.94",
                            "173.94≤x<190.15",
                            "190.15≤x<206.37",
                            "206.37≤x<222.59",
                        ],
                        dtype="<U15",
                    ),
                    "y": np.array(
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            68.519,
                            84.737,
                            100.955,
                            117.173,
                            133.391,
                            149.609,
                            165.827,
                            182.045,
                            198.263,
                            214.481,
                        ],
                    ),
                },
            ]

            chart_data2 = self.chart_data["acquisitionHistogramCTDIData"]["data"]

            for idx, dataset in enumerate(standard_data1):
                self.assertEqual(chart_data2[idx]["name"], dataset["name"])
                np.testing.assert_almost_equal(
                    chart_data2[idx]["x"], dataset["x"], decimal=6
                )
                np.testing.assert_equal(chart_data2[idx]["y"], dataset["y"])

    def test_study_ctdi(self):
        # Test of mean and median study CTDI,
        # Also tests raw data going into the box plots
        # Also test histogram data. Then repeat above
        # with plotseriespersystem selected
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotCTStudyMeanCTDI = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the mean data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 83.275, 4],
                        ["All systems", 4.265, 2],
                        ["All systems", 35.324, 9],
                        ["All systems", 23.158, 9],
                        ["All systems", 3.715, 4],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "Colonography",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "Thorax^TAP (Adult)",
                    ],
                    dtype=object,
                ),
                "y": np.array([83.275, 4.265, 35.324, 23.158, 3.715]),
            }
        ]

        chart_data = self.chart_data["studyMeanCTDIData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 42.905, 4],
                        ["All systems", 4.265, 2],
                        ["All systems", 22.26, 9],
                        ["All systems", 17.1, 9],
                        ["All systems", 2.405, 4],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "Colonography",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "Thorax^TAP (Adult)",
                    ],
                    dtype=object,
                ),
                "y": np.array([42.905, 4.265, 22.26, 17.1, 2.405]),
            }
        ]

        chart_data = self.chart_data["studyMedianCTDIData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Check the boxplot data
        standard_data = [
            {
                "name": "All systems",
                "x": np.array(
                    [
                        "Thorax^TAP (Adult)",
                        "Thorax^TAP (Adult)",
                        "Colonography",
                        "Thorax^TAP (Adult)",
                        "FACIAL BONES",
                        "Colonography",
                        "Specials^PhysicsTesting (Adult)",
                        "FACIAL BONES",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "FACIAL BONES",
                        "Thorax^TAP (Adult)",
                        "Specials^PhysicsTesting (Adult)",
                        "Specials^PhysicsTesting (Adult)",
                        "Specials^PhysicsTesting (Adult)",
                        "Specials^PhysicsTesting (Adult)",
                        "FACIAL BONES",
                        "Blank",
                        "Blank",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "FACIAL BONES",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "Blank",
                        "Specials^PhysicsTesting (Adult)",
                        "FACIAL BONES",
                        "Blank",
                    ],
                    dtype=object,
                ),
                "y": np.array(
                    [
                        1.4000e-01,
                        1.2000e00,
                        3.2300e00,
                        3.6100e00,
                        4.9300e00,
                        5.3000e00,
                        5.5200e00,
                        5.8400e00,
                        6.2300e00,
                        6.2600e00,
                        8.7400e00,
                        9.9100e00,
                        1.3170e01,
                        1.5450e01,
                        1.7100e01,
                        2.1950e01,
                        2.2260e01,
                        2.4700e01,
                        2.5400e01,
                        2.9310e01,
                        2.9670e01,
                        3.1660e01,
                        3.2830e01,
                        3.3830e01,
                        6.0410e01,
                        6.5470e01,
                        1.7612e02,
                        2.2259e02,
                    ]
                ),
            }
        ]

        chart_data = self.chart_data["studyBoxplotCTDIData"]["data"]
        check_boxplot_data(self, chart_data, standard_data)

        # Check the histogram data
        standard_data1 = [
            {
                "name": "FACIAL BONES",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "y": np.array(
                    [5, 3, 0, 0, 0, 0, 0, 1, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
            },
            {
                "name": "Specials^PhysicsTesting (Adult)",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [6, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [0, 2, 1, 0, 0, 0, 0, 0, 0, 1],
                ),
            },
            {
                "name": "Thorax^TAP (Adult)",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "Colonography",
                "text": np.array(
                    [
                        "0.14≤x<22.39",
                        "22.39≤x<44.63",
                        "44.63≤x<66.88",
                        "66.88≤x<89.12",
                        "89.12≤x<111.37",
                        "111.37≤x<133.61",
                        "133.61≤x<155.85",
                        "155.85≤x<178.10",
                        "178.10≤x<200.34",
                        "200.34≤x<222.59",
                    ],
                    dtype="<U15",
                ),
                "x": np.array(
                    [
                        11.2625,
                        33.5075,
                        55.7525,
                        77.9975,
                        100.2425,
                        122.4875,
                        144.7325,
                        166.9775,
                        189.2225,
                        211.4675,
                    ],
                ),
                "y": np.array(
                    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
        ]

        chart_data1 = self.chart_data["studyHistogramCTDIData"]["data"]

        for idx, dataset in enumerate(standard_data1):
            self.assertEqual(chart_data1[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(
                chart_data1[idx]["x"], dataset["x"], decimal=6
            )
            np.testing.assert_equal(chart_data1[idx]["y"], dataset["y"])

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
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", 35.324, 9],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "A VCT Hospital VCTScanner",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, np.nan, 35.324, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", 23.158, 9],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, np.nan, np.nan, 23.158, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", 3.715, 4],
                        ],
                        dtype=object,
                    ),
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, np.nan, np.nan, np.nan, 3.715]),
                },
                {
                    "customdata": np.array(
                        [
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", 4.265, 2],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, 4.265, np.nan, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["OpenREM centre médical rt16", 141.5, 2],
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
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([141.5, np.nan, np.nan, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            [
                                "Oxbridge County Hospital CTTOSHIBA1",
                                25.05,
                                2,
                            ],
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
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([25.05, np.nan, np.nan, np.nan, np.nan]),
                },
            ]

            chart_data = self.chart_data["studyMeanCTDIData"]["data"]

            check_average_data(self, chart_data, standard_data)

            # Test the median data
            standard_data = [
                {
                    "customdata": np.array(
                        [
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", 22.26, 9],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                            ["A VCT Hospital VCTScanner", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "A VCT Hospital VCTScanner",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, np.nan, 22.26, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                            ["Gnats Bottom Hospital CTAWP91919", 17.1, 9],
                            ["Gnats Bottom Hospital CTAWP91919", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, np.nan, np.nan, 17.1, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", np.nan, 0],
                            ["Hospital Number One Trust CTAWP00001", 2.405, 4],
                        ],
                        dtype=object,
                    ),
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, np.nan, np.nan, np.nan, 2.405]),
                },
                {
                    "customdata": np.array(
                        [
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", 4.265, 2],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                            ["An Optima Hospital geoptima", np.nan, 0],
                        ],
                        dtype=object,
                    ),
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        [
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([np.nan, 4.265, np.nan, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            ["OpenREM centre médical rt16", 141.5, 2],
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
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([141.5, np.nan, np.nan, np.nan, np.nan]),
                },
                {
                    "customdata": np.array(
                        [
                            [
                                "Oxbridge County Hospital CTTOSHIBA1",
                                25.05,
                                2,
                            ],
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
                            "Blank",
                            "Colonography",
                            "FACIAL BONES",
                            "Specials^PhysicsTesting (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([25.05, np.nan, np.nan, np.nan, np.nan]),
                },
            ]

            chart_data = self.chart_data["studyMedianCTDIData"]["data"]

            check_average_data(self, chart_data, standard_data)

            # Check the boxplot data
            standard_data = [
                {
                    "name": "A VCT Hospital VCTScanner",
                    "x": np.array(
                        [
                            "FACIAL BONES",
                            "FACIAL BONES",
                            "FACIAL BONES",
                            "FACIAL BONES",
                            "FACIAL BONES",
                            "FACIAL BONES",
                            "FACIAL BONES",
                            "FACIAL BONES",
                            "FACIAL BONES",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [29.31, 176.12, 4.93, 5.84, 6.23, 8.74, 22.26, 32.83, 31.66]
                    ),
                },
                {
                    "name": "Gnats Bottom Hospital CTAWP91919",
                    "x": np.array(
                        [
                            "Specials^PhysicsTesting (Adult)",
                            "Specials^PhysicsTesting (Adult)",
                            "Specials^PhysicsTesting (Adult)",
                            "Specials^PhysicsTesting (Adult)",
                            "Specials^PhysicsTesting (Adult)",
                            "Specials^PhysicsTesting (Adult)",
                            "Specials^PhysicsTesting (Adult)",
                            "Specials^PhysicsTesting (Adult)",
                            "Specials^PhysicsTesting (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array(
                        [65.47, 33.83, 29.67, 21.95, 13.17, 15.45, 6.26, 5.52, 17.1]
                    ),
                },
                {
                    "name": "Hospital Number One Trust CTAWP00001",
                    "x": np.array(
                        [
                            "Thorax^TAP (Adult)",
                            "Thorax^TAP (Adult)",
                            "Thorax^TAP (Adult)",
                            "Thorax^TAP (Adult)",
                        ],
                        dtype=object,
                    ),
                    "y": np.array([0.14, 1.2, 3.61, 9.91]),
                },
                {
                    "name": "An Optima Hospital geoptima",
                    "x": np.array(
                        ["Colonography", "Colonography"],
                        dtype=object,
                    ),
                    "y": np.array([3.23, 5.3]),
                },
                {
                    "name": "OpenREM centre médical rt16",
                    "x": np.array(
                        ["Blank", "Blank"],
                        dtype=object,
                    ),
                    "y": np.array([222.59, 60.41]),
                },
                {
                    "name": "Oxbridge County Hospital CTTOSHIBA1",
                    "x": np.array(
                        ["Blank", "Blank"],
                        dtype=object,
                    ),
                    "y": np.array([24.7, 25.4]),
                },
            ]

            chart_data = self.chart_data["studyBoxplotCTDIData"]["data"]
            check_boxplot_data(self, chart_data, standard_data)

            # Check the histogram data
            standard_data1 = [
                {
                    "name": "FACIAL BONES",
                    "text": np.array(
                        [
                            "4.93≤x<22.05",
                            "22.05≤x<39.17",
                            "39.17≤x<56.29",
                            "56.29≤x<73.41",
                            "73.41≤x<90.53",
                            "90.53≤x<107.64",
                            "107.64≤x<124.76",
                            "124.76≤x<141.88",
                            "141.88≤x<159.00",
                            "159.00≤x<176.12",
                        ],
                        dtype="<U15",
                    ),
                    "y": np.array(
                        [4, 4, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            13.4895,
                            30.6085,
                            47.7275,
                            64.8465,
                            81.9655,
                            99.0845,
                            116.2035,
                            133.3225,
                            150.4415,
                            167.5605,
                        ],
                    ),
                },
                {
                    "name": "Specials^PhysicsTesting (Adult)",
                    "text": np.array(
                        [
                            "29.50≤x<35.50",
                            "35.50≤x<41.49",
                            "41.49≤x<47.48",
                            "47.48≤x<53.48",
                            "53.48≤x<59.47",
                            "59.47≤x<65.47",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [2, 3, 1, 0, 2, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            8.5175,
                            14.5125,
                            20.5075,
                            26.5025,
                            32.4975,
                            38.4925,
                            44.4875,
                            50.4825,
                            56.4775,
                            62.4725,
                        ],
                    ),
                },
                {
                    "name": "Thorax^TAP (Adult)",
                    "text": np.array(
                        [
                            "0.14≤x<1.12",
                            "1.12≤x<2.09",
                            "2.09≤x<3.07",
                            "3.07≤x<4.05",
                            "4.05≤x<5.02",
                            "5.02≤x<6.00",
                            "6.00≤x<6.98",
                            "6.98≤x<7.96",
                            "7.96≤x<8.93",
                            "8.93≤x<9.91",
                        ],
                        dtype="<U11",
                    ),
                    "y": np.array(
                        [1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            0.6285,
                            1.6055,
                            2.5825,
                            3.5595,
                            4.5365,
                            5.5135,
                            6.4905,
                            7.4675,
                            8.4445,
                            9.4215,
                        ],
                    ),
                },
                {
                    "name": "Colonography",
                    "text": np.array(
                        [
                            "3.23≤x<3.44",
                            "3.44≤x<3.64",
                            "3.64≤x<3.85",
                            "3.85≤x<4.06",
                            "4.06≤x<4.26",
                            "4.26≤x<4.47",
                            "4.47≤x<4.68",
                            "4.68≤x<4.89",
                            "4.89≤x<5.09",
                            "5.09≤x<5.30",
                        ],
                        dtype="<U11",
                    ),
                    "y": np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            3.3335,
                            3.5405,
                            3.7475,
                            3.9545,
                            4.1615,
                            4.3685,
                            4.5755,
                            4.7825,
                            4.9895,
                            5.1965,
                        ],
                    ),
                },
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "24.70≤x<24.77",
                            "24.77≤x<24.84",
                            "24.84≤x<24.91",
                            "24.91≤x<24.98",
                            "24.98≤x<25.05",
                            "25.05≤x<25.12",
                            "25.12≤x<25.19",
                            "25.19≤x<25.26",
                            "25.26≤x<25.33",
                            "25.33≤x<25.40",
                        ],
                        dtype="<U13",
                    ),
                    "y": np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            24.735,
                            24.805,
                            24.875,
                            24.945,
                            25.015,
                            25.085,
                            25.155,
                            25.225,
                            25.295,
                            25.365,
                        ],
                    ),
                },
                {
                    "name": "Blank",
                    "text": np.array(
                        [
                            "60.41≤x<76.63",
                            "76.63≤x<92.85",
                            "92.85≤x<109.06",
                            "109.06≤x<125.28",
                            "125.28≤x<141.50",
                            "141.50≤x<157.72",
                            "157.72≤x<173.94",
                            "173.94≤x<190.15",
                            "190.15≤x<206.37",
                            "206.37≤x<222.59",
                        ],
                        dtype="<U15",
                    ),
                    "y": np.array(
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ),
                    # frequency data corresponds to the 'Blank' bars on the histogram
                    "x": np.array(
                        [
                            68.519,
                            84.737,
                            100.955,
                            117.173,
                            133.391,
                            149.609,
                            165.827,
                            182.045,
                            198.263,
                            214.481,
                        ],
                    ),
                },
            ]

        chart_data2 = self.chart_data["studyHistogramCTDIData"]["data"]
        for idx, dataset in enumerate(standard_data1):
            self.assertEqual(chart_data2[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(
                chart_data2[idx]["x"], dataset["x"], decimal=6
            )
            np.testing.assert_equal(chart_data2[idx]["y"], dataset["y"])

    def test_study_freq(self):

        # test study names and associated frequencies
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotCTStudyFreq = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Study name and system name test
        study_system_names = ["All systems"]
        study_names = [
            "Blank",
            "Colonography",
            "FACIAL BONES",
            "Specials^PhysicsTesting (Adult)",
            "Thorax^TAP (Adult)",
        ]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        check_series_and_category_names(
            self, study_system_names, study_names, chart_data
        )

        # The frequency chart - frequencies
        study_data = [[2], [1], [1], [1], [1]]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        check_frequencies(self, study_data, chart_data)

        # Repeat the above, but plot a series per system
        self.user.userprofile.plotSeriesPerSystem = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # study name and system name test
        study_system_names = [
            "A VCT Hospital VCTScanner",
            "An Optima Hospital geoptima",
            "Gnats Bottom Hospital CTAWP91919",
            "Hospital Number One Trust CTAWP00001",
            "OpenREM centre médical rt16",
            "Oxbridge County Hospital CTTOSHIBA1",
        ]
        study_names = [
            "Blank",
            "Colonography",
            "FACIAL BONES",
            "Specials^PhysicsTesting (Adult)",
            "Thorax^TAP (Adult)",
        ]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        check_series_and_category_names(
            self, study_system_names, study_names, chart_data
        )

        # The frequency chart - frequencies
        study_data = [
            [0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ]
        chart_data = self.chart_data["studyFrequencyData"]["data"]
        check_frequencies(self, study_data, chart_data)

    def test_study_numevents(self):
        # Test of mean and median study number of events,
        # Also tests raw data going into the box plots
        # Also test histogram data. Then repeat above
        # with plotseriespersystem selected
        f = self.login_get_filterset()

        # Set user profile options
        self.user.userprofile.plotCTStudyNumEvents = True
        self.user.userprofile.plotMean = True
        self.user.userprofile.plotMedian = True
        self.user.userprofile.plotBoxplots = True
        self.user.userprofile.plotHistograms = True
        self.user.userprofile.save()

        # Obtain chart data
        self.obtain_chart_data(f)

        # Test the mean data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 2.5, 2],
                        ["All systems", 6.0, 1],
                        ["All systems", 27.0, 1],
                        ["All systems", 9.0, 1],
                        ["All systems", 4.0, 1],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "Colonography",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "Thorax^TAP (Adult)",
                    ],
                    dtype=object,
                ),
                "y": np.array([2.5, 6.0, 27.0, 9.0, 4.0]),
            }
        ]

        chart_data = self.chart_data["studyMeanNumEventsData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Test the median data
        standard_data = [
            {
                "customdata": np.array(
                    [
                        ["All systems", 2.5, 2],
                        ["All systems", 6.0, 1],
                        ["All systems", 27.0, 1],
                        ["All systems", 9.0, 1],
                        ["All systems", 4.0, 1],
                    ],
                    dtype=object,
                ),
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "Colonography",
                        "FACIAL BONES",
                        "Specials^PhysicsTesting (Adult)",
                        "Thorax^TAP (Adult)",
                    ],
                    dtype=object,
                ),
                "y": np.array([2.5, 6.0, 27.0, 9.0, 4.0]),
            }
        ]

        chart_data = self.chart_data["studyMedianNumEventsData"]["data"]

        check_average_data(self, chart_data, standard_data)

        # Check the boxplot data
        standard_data = [
            {
                "name": "All systems",
                "x": np.array(
                    [
                        "Blank",
                        "Blank",
                        "Thorax^TAP (Adult)",
                        "Colonography",
                        "Specials^PhysicsTesting (Adult)",
                        "FACIAL BONES",
                    ],
                    dtype=object,
                ),
                "y": np.array([2, 3, 4, 6, 9, 27]),
            }
        ]

        chart_data = self.chart_data["studyBoxplotNumEventsData"]["data"]
        check_boxplot_data(self, chart_data, standard_data)

        # Check the histogram data
        standard_data1 = [
            {
                "name": "Blank",
                "text": np.array(
                    [
                        "2.00≤x<4.50",
                        "4.50≤x<7.00",
                        "7.00≤x<9.50",
                        "9.50≤x<12.00",
                        "12.00≤x<14.50",
                        "14.50≤x<17.00",
                        "17.00≤x<19.50",
                        "19.50≤x<22.00",
                        "22.00≤x<24.50",
                        "24.50≤x<27.00",
                    ],
                    dtype="<U13",
                ),
                "y": np.array(
                    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
                # frequency data corresponds to the 'Blank' bars on the histogram
                "x": np.array(
                    [3.25, 5.75, 8.25, 10.75, 13.25, 15.75, 18.25, 20.75, 23.25, 25.75],
                ),
            },
            {
                "name": "Colonography",
                "text": np.array(
                    [
                        "2.00≤x<4.50",
                        "4.50≤x<7.00",
                        "7.00≤x<9.50",
                        "9.50≤x<12.00",
                        "12.00≤x<14.50",
                        "14.50≤x<17.00",
                        "17.00≤x<19.50",
                        "19.50≤x<22.00",
                        "22.00≤x<24.50",
                        "24.50≤x<27.00",
                    ],
                    dtype="<U13",
                ),
                "x": np.array(
                    [3.25, 5.75, 8.25, 10.75, 13.25, 15.75, 18.25, 20.75, 23.25, 25.75],
                ),
                "y": np.array(
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "FACIAL BONES",
                "text": np.array(
                    [
                        "2.00≤x<4.50",
                        "4.50≤x<7.00",
                        "7.00≤x<9.50",
                        "9.50≤x<12.00",
                        "12.00≤x<14.50",
                        "14.50≤x<17.00",
                        "17.00≤x<19.50",
                        "19.50≤x<22.00",
                        "22.00≤x<24.50",
                        "24.50≤x<27.00",
                    ],
                    dtype="<U13",
                ),
                "x": np.array(
                    [3.25, 5.75, 8.25, 10.75, 13.25, 15.75, 18.25, 20.75, 23.25, 25.75],
                ),
                "y": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                ),
            },
            {
                "name": "Specials^PhysicsTesting (Adult)",
                "text": np.array(
                    [
                        "2.00≤x<4.50",
                        "4.50≤x<7.00",
                        "7.00≤x<9.50",
                        "9.50≤x<12.00",
                        "12.00≤x<14.50",
                        "14.50≤x<17.00",
                        "17.00≤x<19.50",
                        "19.50≤x<22.00",
                        "22.00≤x<24.50",
                        "24.50≤x<27.00",
                    ],
                    dtype="<U13",
                ),
                "x": np.array(
                    [3.25, 5.75, 8.25, 10.75, 13.25, 15.75, 18.25, 20.75, 23.25, 25.75],
                ),
                "y": np.array(
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
            {
                "name": "Thorax^TAP (Adult)",
                "text": np.array(
                    [
                        "2.00≤x<4.50",
                        "4.50≤x<7.00",
                        "7.00≤x<9.50",
                        "9.50≤x<12.00",
                        "12.00≤x<14.50",
                        "14.50≤x<17.00",
                        "17.00≤x<19.50",
                        "19.50≤x<22.00",
                        "22.00≤x<24.50",
                        "24.50≤x<27.00",
                    ],
                    dtype="<U13",
                ),
                "x": np.array(
                    [3.25, 5.75, 8.25, 10.75, 13.25, 15.75, 18.25, 20.75, 23.25, 25.75],
                ),
                "y": np.array(
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ),
            },
        ]

        chart_data1 = self.chart_data["studyHistogramNumEventsData"]["data"]

        for idx, dataset in enumerate(standard_data1):
            self.assertEqual(chart_data1[idx]["name"], dataset["name"])
            np.testing.assert_almost_equal(
                chart_data1[idx]["x"], dataset["x"], decimal=6
            )
            np.testing.assert_equal(chart_data1[idx]["y"], dataset["y"])
