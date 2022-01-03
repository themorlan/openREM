# This Python file uses the following encoding: utf-8
# test_export_ct_xlsx.py

import os
from django.contrib.auth.models import User, Group
from django.test import TestCase, RequestFactory
import xlrd
from remapp.extractors import rdsr
from remapp.exports.ct_export import ctxlsx, ct_csv, ct_phe_2019
from remapp.models import PatientIDSettings, Exports

xlrd.xlsx.ensure_elementtree_imported(False, None)
xlrd.xlsx.Element_has_iter = True

class ExportCTxlsx(TestCase):
    """Test class for CT exports to XLSX"""

    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username="jacob", email="jacob@…", password="top_secret"
        )
        eg = Group(name="exportgroup")
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

        ct_ge_ct660 = os.path.join("test_files", "CT-ESR-GE_Optima.dcm")
        ct_ge_vct = os.path.join("test_files", "CT-ESR-GE_VCT.dcm")
        ct_siemens_flash_ss = os.path.join(
            "test_files", "CT-RDSR-Siemens_Flash-TAP-SS.dcm"
        )
        ct_toshiba_dosecheck = os.path.join(
            "test_files", "CT-RDSR-Toshiba_DoseCheck.dcm"
        )
        root_tests = os.path.dirname(os.path.abspath(__file__))

        rdsr.rdsr(os.path.join(root_tests, ct_ge_ct660))
        rdsr.rdsr(os.path.join(root_tests, ct_ge_vct))
        rdsr.rdsr(os.path.join(root_tests, ct_siemens_flash_ss))
        rdsr.rdsr(os.path.join(root_tests, ct_toshiba_dosecheck))

    def test_id_as_text(self):  # See https://bitbucket.org/openrem/openrem/issues/443
        filter_set = {"o": "-study_date"}
        pid = True
        name = False
        patient_id = True

        ctxlsx(filter_set, pid=pid, name=name, patid=patient_id, user=self.user)

        task = Exports.objects.all()[0]

        book = xlrd.open_workbook(task.filename.path)
        all_data_sheet = book.sheet_by_name("All data")
        headers = all_data_sheet.row(0)

        patient_id_col = [i for i, x in enumerate(headers) if x.value == "Patient ID"][0]
        accession_number_col = [i for i, x in enumerate(headers) if x.value == "Accession number"][0]
        dlp_total_col = [i for i, x in enumerate(headers) if x.value == "Total DLP (mGy.cm)"][0]
        e1_dose_check_col = [i for i, x in enumerate(headers) if x.value == "E1 Dose check alerts"][0]
        e2_dose_check_col = [i for i, x in enumerate(headers) if x.value == "E2 Dose check alerts"][0]

        e1_dose_check_string = (
            u"Dose check alerts:\n"
            u"DLP alert is configured at 100.0 mGy.cm\n"
            u"with an accumulated forward estimate of 251.2 mGy.cm\n"
            u"CTDIvol alert is configured at 10.0 mGy\n"
            u"Person authorizing irradiation: Luuk"
        )
        e2_dose_check_string = (
            u"Dose check alerts:\n"
            u"DLP alert is configured at 100.0 mGy.cm\n"
            u"with an accumulated forward estimate of 502.4 mGy.cm\n"
            u"CTDIvol alert is configured at 10.0 mGy\n"
            u"with an accumulated forward estimate of 10.6 mGy\n"
            u"Person authorizing irradiation: Luuk"
        )

        self.assertEqual(all_data_sheet.cell_type(1, patient_id_col), xlrd.XL_CELL_TEXT)
        self.assertEqual(all_data_sheet.cell_type(1, accession_number_col), xlrd.XL_CELL_TEXT)
        self.assertEqual(all_data_sheet.cell_type(1, dlp_total_col), xlrd.XL_CELL_NUMBER)
        self.assertEqual(all_data_sheet.cell_value(1, patient_id_col), "00001234")
        self.assertEqual(all_data_sheet.cell_value(1, accession_number_col), "0012345.12345678")
        self.assertAlmostEqual(all_data_sheet.cell_value(1, dlp_total_col), 415.82, 2)
        self.assertEqual(all_data_sheet.cell_value(1, e1_dose_check_col), "")
        self.assertEqual(all_data_sheet.cell_value(1, e2_dose_check_col), "")

        self.assertEqual(all_data_sheet.cell_type(2, patient_id_col), xlrd.XL_CELL_TEXT)
        self.assertEqual(all_data_sheet.cell_type(2, accession_number_col), xlrd.XL_CELL_TEXT)
        self.assertEqual(all_data_sheet.cell_type(2, dlp_total_col), xlrd.XL_CELL_NUMBER)
        self.assertEqual(all_data_sheet.cell_value(2, patient_id_col), "008F/g234")
        self.assertEqual(all_data_sheet.cell_value(2, accession_number_col), "001234512345678")
        self.assertAlmostEqual(all_data_sheet.cell_value(2, dlp_total_col), 2002.39, 2)
        self.assertEqual(all_data_sheet.cell_value(2, e1_dose_check_col), "")
        self.assertEqual(all_data_sheet.cell_value(2, e2_dose_check_col), "")

        self.assertEqual(all_data_sheet.cell_type(3, patient_id_col), xlrd.XL_CELL_TEXT)
        self.assertEqual(all_data_sheet.cell_type(3, accession_number_col), xlrd.XL_CELL_TEXT)
        self.assertEqual(all_data_sheet.cell_type(3, dlp_total_col), xlrd.XL_CELL_NUMBER)
        self.assertEqual(all_data_sheet.cell_value(3, patient_id_col), "4018119567876617")
        self.assertEqual(all_data_sheet.cell_value(3, accession_number_col), "3599305798462538")
        self.assertAlmostEqual(all_data_sheet.cell_value(3, dlp_total_col), 502.40, 2)
        self.assertEqual(all_data_sheet.cell_value(3, e1_dose_check_col), e1_dose_check_string)
        self.assertEqual(all_data_sheet.cell_value(3, e2_dose_check_col), e2_dose_check_string)

        self.assertEqual(all_data_sheet.cell_type(4, patient_id_col), xlrd.XL_CELL_TEXT)
        self.assertEqual(all_data_sheet.cell_type(4, accession_number_col), xlrd.XL_CELL_TEXT)
        self.assertEqual(all_data_sheet.cell_type(4, dlp_total_col), xlrd.XL_CELL_NUMBER)
        self.assertEqual(all_data_sheet.cell_value(4, patient_id_col), "123456")
        self.assertEqual(all_data_sheet.cell_value(4, accession_number_col), "ACC12345601")
        self.assertAlmostEqual(all_data_sheet.cell_value(4, dlp_total_col), 724.52, 2)
        self.assertEqual(all_data_sheet.cell_value(4, e1_dose_check_col), "")
        self.assertEqual(all_data_sheet.cell_value(4, e2_dose_check_col), "")

        # cleanup
        task.filename.delete()  # delete file so local testing doesn't get too messy!

    def test_zero_filter(self):
        """Test error handled correctly when empty filter."""
        filter_set = {"study_description": "asd"}
        pid = True
        name = False
        patient_id = True

        ctxlsx(filter_set, pid=pid, name=name, patid=patient_id, user=self.user)

        task = Exports.objects.all()[0]
        self.assertEqual(u"ERROR", task.status)

    def test_acq_type_filter_spiral(self):
        """Test to check that filtering CT by acquisition type works
        as expected.

        """
        filter_set = {
            "num_spiral_events": "some",
            "o": "-study_date",
        }
        pid = True
        name = False
        patient_id = True

        ctxlsx(filter_set, pid=pid, name=name, patid=patient_id, user=self.user)

        task = Exports.objects.all()[0]
        self.assertEqual(4, task.num_records)

        # cleanup
        task.filename.delete()  # delete file so local testing doesn't get too messy!

    def test_acq_type_filter_sequenced(self):
        """Test to check that filtering CT by acquisition type works
        as expected.

        """
        filter_set = {
            "num_axial_events": "some",
            "o": "-study_date",
        }
        pid = True
        name = False
        patient_id = True

        ctxlsx(filter_set, pid=pid, name=name, patid=patient_id, user=self.user)

        task = Exports.objects.all()[0]
        self.assertEqual(1, task.num_records)

        # cleanup
        task.filename.delete()  # delete file so local testing doesn't get too messy!

    def test_acq_type_filter_spiral_and_sequenced(self):
        """Test to check that filtering CT by acquisition type works
        as expected.

        """
        filter_set = {
            "num_spiral_events": "some",
            "num_axial_events": "some",
            "o": "-study_date",
        }
        pid = True
        name = False
        patient_id = True

        ctxlsx(filter_set, pid=pid, name=name, patid=patient_id, user=self.user)

        task = Exports.objects.all()[0]
        self.assertEqual(1, task.num_records)

        # cleanup
        task.filename.delete()  # delete file so local testing doesn't get too messy!

    def test_export_phe(self):
        filter_set = {"num_spiral_events": "2", "o": "-study_date"}

        ct_phe_2019(filter_set, user=self.user)

        task = Exports.objects.order_by("pk")[0]
        self.assertEqual(3, task.num_records)

        book = xlrd.open_workbook(task.filename.path)
        phe_sheet = book.sheet_by_name("PHE CT 2019")

        self.assertEqual(phe_sheet.cell_value(1, 4), 487)  # first series imaged length
        self.assertEqual(phe_sheet.cell_value(2, 4), 5)
        self.assertEqual(phe_sheet.cell_value(3, 4), 418.75)
        self.assertEqual(phe_sheet.cell_value(1, 10), 5.3)  # first series CTDIvol
        self.assertEqual(phe_sheet.cell_value(2, 10), 32.83)
        self.assertEqual(phe_sheet.cell_value(3, 10), 3.23)
        self.assertEqual(phe_sheet.cell_value(1, 19), 251.2)  # second series DLP
        self.assertEqual(phe_sheet.cell_value(2, 19), 429.19)
        self.assertEqual(phe_sheet.cell_value(3, 19), 259.85)
        self.assertEqual(phe_sheet.cell_value(1, 36), 502.4)  # total DLP
        self.assertEqual(phe_sheet.cell_value(2, 36), 2002.39)
        self.assertEqual(phe_sheet.cell_value(3, 36), 415.82)
        self.assertEqual(phe_sheet.cell_value(3, 2), 78.2)
        self.assertEqual(phe_sheet.cell_value(3, 3), 164)

        # cleanup
        task.filename.delete()  # delete file so local testing doesn't get too messy!
