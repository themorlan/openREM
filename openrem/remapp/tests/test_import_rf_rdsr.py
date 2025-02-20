# This Python file uses the following encoding: utf-8
# test_import_rf_rdsr_philips.py

import os
from decimal import Decimal
import datetime

from django.test import TestCase

from remapp.extractors import rdsr
from remapp.models import GeneralStudyModuleAttr, PatientIDSettings


class ImportRFRDSRPhilips(TestCase):
    """Tests for importing the Philips Allura RDSR"""

    def test_private_collimation_data(self):
        """Tests that the collimated field information has been successfully obtained

        :return: None
        """

        PatientIDSettings.objects.create()

        dicom_file = "test_files/RF-RDSR-Philips_Allura.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        projection_dose = study.projectionxrayradiationdose_set.get()
        first_source_data = projection_dose.irradeventxraydata_set.order_by("pk")[
            0
        ].irradeventxraysourcedata_set.get()

        first_field_height = Decimal((164.5 + 164.5) * 1.194)
        first_field_width = Decimal((131.0 + 131.0) * 1.194)
        first_field_area = (first_field_height * first_field_width) / 1000000

        self.assertAlmostEqual(
            first_source_data.collimated_field_height, first_field_height
        )
        self.assertAlmostEqual(
            first_source_data.collimated_field_width, first_field_width
        )
        self.assertAlmostEqual(
            first_source_data.collimated_field_area, first_field_area
        )

        performing_physician_name = study.performing_physician_name
        self.assertEqual(
            performing_physician_name, "Yamada^Tarou=山田^太郎=やまだ^たろう"
        )


class ImportRFRDSRPhilipsAzurion(TestCase):
    """
    Test importing Azurion RDSR with empty calibration data and incorrect Acquisition Device Type

    *** Currently disabled as RDSR too big - import adds 15 seconds to tests! ***
    To enable, remove '_disable_' from the function name
    """

    def _disable_test_azurion_import(self):
        """
        Tests that the file was imported without error (empty calibrations)
        Tests that the accumulated fluoroscopy data is captured (incorrect Acquisition Device Type)
        :return: None
        """

        PatientIDSettings.objects.create()

        dicom_file = "test_files/RF-RDSR-Philips_Azurion.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        fluoro_totals = (
            study.projectionxrayradiationdose_set.get()
            .accumxraydose_set.get()
            .accumprojxraydose_set.get()
        )

        self.assertAlmostEqual(
            fluoro_totals.fluoro_dose_area_product_total, Decimal(0.00101567)
        )


class DAPUnitsTest(TestCase):
    """
    Test handling of incorrect DAP units found in Toshiba/Canon RF Ultimax
    """

    def test_dgycm2(self):
        """
        Initial test of sequence as presented in Ultimax RDSR
        :return: None
        """
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence
        from remapp.extractors.rdsr_methods import _check_dap_units

        units_sequence = Dataset()
        units_sequence.CodeValue = "dGy.cm2"
        units_sequence.CodingSchemeDesignator = "UCUM"
        units_sequence.CodeMeaning = "dGy.cm2"
        measured_values_sequence = Dataset()
        measured_values_sequence.NumericValue = 1.034
        measured_values_sequence.MeasurementUnitsCodeSequence = Sequence(
            [units_sequence]
        )

        dap = _check_dap_units(measured_values_sequence)
        self.assertAlmostEqual(dap, 0.00001034)

    def test_gym2(self):
        """
        Test case of correct sequence as presented in conformant RDSR
        :return: None
        """
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence
        from remapp.extractors.rdsr_methods import _check_dap_units

        units_sequence = Dataset()
        units_sequence.CodeValue = "Gym2"
        units_sequence.CodingSchemeDesignator = "UCUM"
        units_sequence.CodeMeaning = "Gym2"
        measured_values_sequence = Dataset()
        measured_values_sequence.NumericValue = 1.6e-005
        measured_values_sequence.MeasurementUnitsCodeSequence = Sequence(
            [units_sequence]
        )

        dap = _check_dap_units(measured_values_sequence)
        self.assertAlmostEqual(dap, 0.000016)

    def test_no_units(self):
        """
        Test case of missing units sequence - not seen by the auther in the wild
        :return: None
        """
        from pydicom.dataset import Dataset
        from remapp.extractors.rdsr_methods import _check_dap_units

        measured_values_sequence = Dataset()
        measured_values_sequence.NumericValue = 1.6e-005

        dap = _check_dap_units(measured_values_sequence)
        self.assertAlmostEqual(dap, 0.000016)


class RPDoseUnitsTest(TestCase):
    """
    Test handling of incorrect dose at reference point units found in a Canon RF Ultimax-i
    """

    def test_mgy(self):
        """
        Initial test of sequence as presented in Ultimax-i RDSR
        :return: None
        """
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence
        from remapp.extractors.rdsr_methods import _check_rp_dose_units

        units_sequence = Dataset()
        units_sequence.CodeValue = "mGy"
        units_sequence.CodingSchemeDesignator = "UCUM"
        units_sequence.CodeMeaning = "mGy"
        measured_values_sequence = Dataset()
        measured_values_sequence.NumericValue = 1.034
        measured_values_sequence.MeasurementUnitsCodeSequence = Sequence(
            [units_sequence]
        )

        dap = _check_rp_dose_units(measured_values_sequence)
        self.assertAlmostEqual(dap, 0.001034)

    def test_gy(self):
        """
        Test case of correct sequence as presented in conformant RDSR
        :return: None
        """
        from pydicom.dataset import Dataset
        from pydicom.sequence import Sequence
        from remapp.extractors.rdsr_methods import _check_rp_dose_units

        units_sequence = Dataset()
        units_sequence.CodeValue = "Gy"
        units_sequence.CodingSchemeDesignator = "UCUM"
        units_sequence.CodeMeaning = "Gy"
        measured_values_sequence = Dataset()
        measured_values_sequence.NumericValue = 1.6e-003
        measured_values_sequence.MeasurementUnitsCodeSequence = Sequence(
            [units_sequence]
        )

        dap = _check_rp_dose_units(measured_values_sequence)
        self.assertAlmostEqual(dap, 0.0016)

    def test_no_units(self):
        """
        Test case of missing units sequence
        :return: None
        """
        from pydicom.dataset import Dataset
        from remapp.extractors.rdsr_methods import _check_rp_dose_units

        measured_values_sequence = Dataset()
        measured_values_sequence.NumericValue = 1.6e-003

        dap = _check_rp_dose_units(measured_values_sequence)
        self.assertAlmostEqual(dap, 0.0016)


class ImportRFRDSRSiemens(TestCase):
    """Tests for importing the Siemens Zee RDSR"""

    def test_comment_xml_extraction(self):
        """Tests that the patient orientation and II size information has been successfully obtained

        :return: None
        """

        PatientIDSettings.objects.create()

        dicom_file = "test_files/RF-RDSR-Siemens-Zee.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        event_data = (
            study.projectionxrayradiationdose_set.get().irradeventxraydata_set.order_by(
                "id"
            )[0]
        )
        self.assertEqual(
            event_data.patient_table_relationship_cid.code_value, "F-10470"
        )
        self.assertEqual(event_data.patient_orientation_cid.code_value, "F-10450")
        self.assertEqual(
            event_data.patient_orientation_modifier_cid.code_meaning, "supine"
        )
        source_data = event_data.irradeventxraysourcedata_set.get()
        self.assertEqual(source_data.ii_field_size, 220)

        # Test summary fields
        self.assertAlmostEqual(study.total_dap_a, Decimal(0.000016))
        self.assertAlmostEqual(study.total_dap, Decimal(0.000016))
        self.assertAlmostEqual(study.total_rp_dose_a, Decimal(0.00252))
        self.assertEqual(study.number_of_events, 8)
        self.assertEqual(study.number_of_planes, 1)


class ImportRFRDSRGESurgical(TestCase):
    """
    Tests for importing an RDSR from a GE Surgical C-Arm FPD system
    """

    def test_ge_c_arm_rdsr(self):
        """Tests for extracting from GE RDSRs, particularly the fields that have the wrong type
        or typographical errors.

        :return: None
        """

        PatientIDSettings.objects.create()

        dicom_file = "test_files/RF-RDSR-GE.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        device_observer_uid = (
            study.generalequipmentmoduleattr_set.get().unique_equipment_name.device_observer_uid
        )
        self.assertEqual(device_observer_uid, "1.3.6.1.4.1.45593.912345678.9876543123")

        accum_proj = (
            study.projectionxrayradiationdose_set.get()
            .accumxraydose_set.get()
            .accumprojxraydose_set.get()
        )
        total_fluoro_dap = accum_proj.fluoro_dose_area_product_total
        total_fluoro_rp_dose = accum_proj.fluoro_dose_rp_total
        total_acq_dap = accum_proj.acquisition_dose_area_product_total
        total_acq_rp_dose = accum_proj.acquisition_dose_rp_total
        self.assertAlmostEqual(total_fluoro_dap, Decimal(0.00024126))
        self.assertAlmostEqual(total_fluoro_rp_dose, Decimal(0.01173170))
        self.assertEqual(total_acq_dap, Decimal(0))
        self.assertEqual(total_acq_rp_dose, Decimal(0))

        events = (
            study.projectionxrayradiationdose_set.get().irradeventxraydata_set.order_by(
                "pk"
            )
        )
        event_4 = events[3]
        self.assertEqual(
            event_4.date_time_started, datetime.datetime(2019, 3, 16, 13, 27, 25)
        )
        self.assertEqual(event_4.reference_point_definition.code_value, "113861")
        self.assertEqual(
            event_4.irradiation_event_uid,
            "1.3.6.1.4.1.5962.99.1.3577657414.286912992.1554060884038.8.0",
        )
        self.assertAlmostEqual(event_4.dose_area_product, Decimal(0.00004334))
        event_4_source = event_4.irradeventxraysourcedata_set.get()
        self.assertAlmostEqual(event_4_source.dose_rp, Decimal(0.00210763))
        self.assertAlmostEqual(
            event_4_source.collimated_field_area, Decimal(0.04196800)
        )
        self.assertAlmostEqual(
            event_4_source.average_xray_tube_current, Decimal(18.90340042)
        )
        self.assertEqual(
            event_4_source.xraygrid_set.get().xray_grid.code_meaning, "Fixed Grid"
        )
        self.assertEqual(
            event_4_source.xraygrid_set.get().xray_grid.code_value, "111641"
        )

        # Test summary fields
        self.assertAlmostEqual(study.total_dap_a, Decimal(0.00024126))
        self.assertAlmostEqual(study.total_dap, Decimal(0.00024126))
        self.assertAlmostEqual(study.total_rp_dose_a, Decimal(0.01173170))
        self.assertEqual(study.number_of_events, 8)
        self.assertEqual(study.number_of_planes, 1)


class ImportRFRDSRGEOECMiniView(TestCase):
    """
    Tests for importing an RDSR from a GE ELite Mini View C-Arm that doesn't declare a template type
    """

    def test_ge_mini_view_rdsr(self):
        """Tests that GE OEC Elite Mini View RDSR imports correctly as no template is declared

        :return: None
        """

        PatientIDSettings.objects.create()

        dicom_file = "test_files/RF-RDSR-GE-OECEliteMiniView.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        accum_proj = (
            study.projectionxrayradiationdose_set.get()
            .accumxraydose_set.get()
            .accumprojxraydose_set.get()
        )
        total_fluoro_dap = accum_proj.fluoro_dose_area_product_total
        total_fluoro_rp_dose = accum_proj.fluoro_dose_rp_total
        total_acq_dap = accum_proj.acquisition_dose_area_product_total
        total_acq_rp_dose = accum_proj.acquisition_dose_rp_total
        self.assertAlmostEqual(total_fluoro_dap, Decimal(0.0000013316568))
        self.assertAlmostEqual(total_fluoro_rp_dose, Decimal(0.00022034578))
        self.assertEqual(total_acq_dap, Decimal(0))
        self.assertEqual(total_acq_rp_dose, Decimal(0))


class ImportRFRDSRCanonUltimaxi(TestCase):
    """
    Tests for importing an RDSR from a Canon Ultimax-i where the dose at reference point is stored as mGy
    """

    def test_canon_ultimax_i_rdsr(self):
        """Tests that Canon Ultimax-i dose at reference point imports correctly as the values are stored in mGy rather
        than Gy

        :return: None
        """

        PatientIDSettings.objects.create()

        dicom_file = "test_files/RF-RDSR-Canon-Ultimaxi-mGyDoseAtRP.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.order_by("id")[0]

        # Test the total reference point doses
        # The total fluoro RP dose is stored in the RDSR as 25.664 with units of mGy
        # The total acquisition RP dose is stored in the RDSR as 4.909 with units of mGy
        # The total RP dose is stored in the RDSR as 30.573 with units of mGy
        accum_proj = (
            study.projectionxrayradiationdose_set.get()
            .accumxraydose_set.get()
            .accumprojxraydose_set.get()
        )
        total_fluoro_rp_dose = accum_proj.fluoro_dose_rp_total
        total_acq_rp_dose = accum_proj.acquisition_dose_rp_total
        total_rp_dose = accum_proj.dose_rp_total

        self.assertAlmostEqual(total_fluoro_rp_dose, Decimal(0.025664000000))
        self.assertAlmostEqual(total_acq_rp_dose, Decimal(0.004909000000))
        self.assertAlmostEqual(total_rp_dose, Decimal(0.030573000000))

        # Test a reference point dose from an individual exposure
        # The first exposure RP dose is is stored in the RDSR as 0.384 with units of mGy
        events = (
            study.projectionxrayradiationdose_set.get().irradeventxraydata_set.order_by(
                "pk"
            )
        )
        event_1_source = events[0].irradeventxraysourcedata_set.get()
        self.assertAlmostEqual(event_1_source.dose_rp, Decimal(0.000384000000))
