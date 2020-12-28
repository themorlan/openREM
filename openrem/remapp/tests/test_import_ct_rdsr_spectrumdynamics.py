# This Python file uses the following encoding: utf-8
# test_import_ct_rdsr_siemens.py

"""
..  module:: test_import_rdsr_toshiba_dosecheck
    :synopsis: Test module focusing on proper extraction of dose check data

..  moduleauthor:: Ed McDonagh
"""

import os
from decimal import Decimal
from django.test import TestCase
from ..extractors import rdsr
from ..models import GeneralStudyModuleAttr, PatientIDSettings


class ImportVeritonWithDoseCheck(TestCase):
    """Test module for Spectrum Dynamics Veriton SPECT-CT RDSR with invalid Person Participant in Dose Check section"""

    def test_spectrum_dynamics_import(self):
        """Imports a known RDSR and checks for a correct import"""
        PatientIDSettings.objects.create()

        dicom_file = "test_files/CT-RDSR-SpectrumDynamics.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        self.assertEqual(study.modality_type, "CT")
        self.assertAlmostEqual(
            study.ctradiationdose_set.get()
            .ctaccumulateddosedata_set.get()
            .ct_dose_length_product_total,
            Decimal(187.339),
        )

        series2_dose_check = (
            study.ctradiationdose_set.get()
            .ctirradiationeventdata_set.order_by("id")[1]
            .ctdosecheckdetails_set.get()
        )
        series3_dose_check = (
            study.ctradiationdose_set.get()
            .ctirradiationeventdata_set.order_by("id")[2]
            .ctdosecheckdetails_set.get()
        )
        series4_dose_check = (
            study.ctradiationdose_set.get()
            .ctirradiationeventdata_set.order_by("id")[3]
            .ctdosecheckdetails_set.get()
        )
        series5_dose_check = (
            study.ctradiationdose_set.get()
            .ctirradiationeventdata_set.order_by("id")[4]
            .ctdosecheckdetails_set.get()
        )

        self.assertFalse(series2_dose_check.dlp_alert_value_configured)
        self.assertFalse(series2_dose_check.ctdivol_alert_value_configured)
        self.assertAlmostEqual(
            series2_dose_check.accumulated_dlp_forward_estimate, Decimal(21.64)
        )
        self.assertFalse(series2_dose_check.dlp_notification_value_configured)
        self.assertFalse(series2_dose_check.ctdivol_notification_value_configured)

        self.assertFalse(series3_dose_check.dlp_alert_value_configured)
        self.assertFalse(series3_dose_check.ctdivol_alert_value_configured)
        self.assertAlmostEqual(
            series3_dose_check.accumulated_dlp_forward_estimate, Decimal(47.0858)
        )
        self.assertFalse(series3_dose_check.dlp_notification_value_configured)
        self.assertFalse(series3_dose_check.ctdivol_notification_value_configured)

        self.assertFalse(series4_dose_check.dlp_alert_value_configured)
        self.assertFalse(series4_dose_check.ctdivol_alert_value_configured)
        self.assertAlmostEqual(
            series4_dose_check.accumulated_dlp_forward_estimate, Decimal(172.421)
        )
        self.assertFalse(series4_dose_check.dlp_notification_value_configured)
        self.assertFalse(series4_dose_check.ctdivol_notification_value_configured)

        self.assertFalse(series5_dose_check.dlp_alert_value_configured)
        self.assertFalse(series5_dose_check.ctdivol_alert_value_configured)
        self.assertAlmostEqual(
            series5_dose_check.accumulated_dlp_forward_estimate, Decimal(243.091)
        )
        self.assertFalse(series5_dose_check.dlp_notification_value_configured)
        self.assertFalse(series5_dose_check.ctdivol_notification_value_configured)
