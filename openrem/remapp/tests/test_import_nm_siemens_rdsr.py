# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2012,2013  The Royal Marsden NHS Foundation Trust
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    Additional permission under section 7 of GPLv3:
#    You shall not make any use of the name of The Royal Marsden NHS
#    Foundation trust in connection with this Program in any press or
#    other public announcement without the prior written consent of
#    The Royal Marsden NHS Foundation Trust.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from datetime import date, time, datetime
import os
from decimal import Decimal
from typing import Dict
from django.test import TestCase

from remapp.extractors import rdsr
from remapp.models import GeneralStudyModuleAttr, PatientIDSettings


class ImportTest(TestCase):
    def _get_dcm_file(self, dicom_file):
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path = os.path.join(root_tests, dicom_file)
        return dicom_path

    def setUp(self) -> None:
        u: PatientIDSettings = PatientIDSettings.objects.create()
        u.name_stored = True
        u.name_hashed = False
        u.save()

    def _verify_equality(self, value, expect_value, location):
        msg = f"At {location}"
        check_type = type(expect_value)
        self.assertEqual(check_type, type(value), msg)

        if (
            check_type == date
            or check_type == str
            or check_type == time
            or check_type == datetime
            or check_type == bool
            or check_type == int
        ):
            self.assertEqual(value, expect_value, msg)
        elif check_type == Decimal:
            self.assertAlmostEqual(value, expect_value, msg=msg)
        elif check_type == type(None):
            self.assertIsNone(value, msg)
        else:
            raise NotImplementedError(
                f"Type {check_type} has no associated check at {location}"
            )

    def _check_values(self, level, data, trace=""):
        for name, subdata in data.items():
            try:
                if callable(name):
                    location = f"{trace}.func"
                    current = name(level)
                else:
                    location = f"{trace}.{name}"
                    current = getattr(level, name)
            except AttributeError:
                raise ValueError(f"{name} was expected, but not found. Trace: {trace}")
            try:
                if (
                    callable(current) and not "_set" in name
                ):  # Shortcut so one can use strings for zero parameter functions
                    current = current()
            except Exception as e:
                raise ValueError(
                    f"{name} is a function. Only functions without parameters usable at {location}"
                )
            if not isinstance(subdata, Dict):
                self._verify_equality(current, subdata, location)
            else:
                self._check_values(current, subdata, location)


class ImportNMRDSR(ImportTest):
    def test_import_siemens_rrdsr(self):
        """Loads a Siemens RRDSR-File and checks all values against the expected ones"""

        dicom_path = self._get_dcm_file("test_files/NM-RRDSR-Siemens.dcm")
        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.filter(modality_type__exact="NM").get()

        # We verify only the first and the last organ dose
        expected_data = {
            "patientmoduleattr_set": {
                "get": {
                    "not_patient_indicator": None,
                    "patient_name": "REMOVED1",
                }
            },
            "study_date": date(2022, 2, 24),
            "study_time": time(11, 50, 25, 472000),
            "accession_number": "TEST123456",
            "study_description": "PET/CT Ganzkoerper FDG",
            "modality_type": "NM",
            "referring_physician_name": "Kim",
            "generalequipmentmoduleattr_set": {
                "get": {
                    "institution_name": "REMOVED",
                    "manufacturer": "SIEMENS",
                    "manufacturer_model_name": "Biograph64_Vision 600_Vision 600-1208",
                    "station_name": "CTAWP00001",
                }
            },
            "patientstudymoduleattr_set": {
                "get": {
                    "patient_age": "063Y",
                }
            },
            "radiopharmaceuticalradiationdose_set": {
                "get": {
                    "associated_procedure": {
                        "code_meaning": "PET study for localization of tumor"
                    },
                    "has_intent": {"code_meaning": "Diagnostic Intent"},
                    "radiopharmaceuticaladministrationeventdata_set": {
                        "get": {
                            "radiopharmaceutical_agent": {
                                "code_meaning": "Fluorodeoxyglucose F^18^"
                            },
                            "radionuclide": {"code_meaning": "^18^Fluorine"},
                            "radionuclide_half_life": Decimal(6586.2),
                            "radiopharmaceutical_administration_event_uid": "1.3.12.2.1107.5.1.4.11090.20220224104830.0",
                            "radiopharmaceutical_start_datetime": datetime(
                                2022, 2, 24, 10, 40, 30, 0
                            ),
                            "radiopharmaceutical_stop_datetime": datetime(
                                2022, 2, 24, 10, 40, 30, 0
                            ),
                            "administered_activity": Decimal(394.0),
                            "effective_dose": Decimal(7.486),
                            "organdose_set": {
                                "first": {
                                    "finding_site": {"code_meaning": "Adrenal gland"},
                                    "laterality": {"code_meaning": "Right and left"},
                                    "organ_dose": Decimal(4.73),
                                    "reference_authority_text": "ICRP Publication 128",
                                },
                                "last": {
                                    "finding_site": {"code_meaning": "Bladder"},
                                    "laterality": None,
                                    "organ_dose": Decimal(51.22),
                                    "reference_authority_text": "ICRP Publication 128",
                                },
                            },
                            "route_of_administration": {
                                "code_meaning": "Intravenous route"
                            },
                            "site_of": {"code_meaning": "Via vein"},
                            "personparticipant_set": {
                                "get": {
                                    "person_name": "Unknown",
                                    "person_role_in_procedure_cid": {
                                        "code_meaning": "Irradiation Administering"
                                    },
                                }
                            },
                        }
                    },
                    "radiopharmaceuticaladministrationpatientcharacteristics_set": {
                        "get": {
                            "subject_age": Decimal(63),
                            "subject_sex": {"code_value": "M"},
                            "patient_height": Decimal(1.78),
                            "patient_weight": Decimal(110),
                        }
                    },
                }
            },
        }

        self._check_values(study, expected_data)

    def test_import_generated_full_nm(self):

        dicom_path = self._get_dcm_file("test_files/NM-RRDSR-Siemens-Extended.dcm")
        rdsr.rdsr(dicom_path)
        study = GeneralStudyModuleAttr.objects.filter(modality_type__exact="NM").get()

        expected_data = {
            "patientmoduleattr_set": {
                "get": {
                    "not_patient_indicator": None,
                    "patient_name": "REMOVED2",
                }
            },
            "study_date": date(2022, 2, 23),
            "study_time": time(9, 32, 12, 576000),
            "accession_number": "XYZ123",
            "study_description": "PET/CT Herz FDG",
            "modality_type": "NM",
            "referring_physician_name": "REMOVED",
            "generalequipmentmoduleattr_set": {
                "get": {
                    "institution_name": "REMOVED",
                    "manufacturer": "SIEMENS",
                    "manufacturer_model_name": "Biograph64_Vision 600_Vision 600-1208",
                    "station_name": "REMOVED",
                }
            },
            "patientstudymoduleattr_set": {
                "get": {
                    "patient_age": "047Y",
                }
            },
            "radiopharmaceuticalradiationdose_set": {
                "get": {
                    "languageofcontentitemanddescendants_set": {
                        "get": {
                            "language_of_contentitem_and_descendants": {
                                "code_meaning": "German"
                            },
                            "country_of_language": {"code_meaning": "Bahrain"},
                        }
                    },
                    "associated_procedure": {
                        "code_meaning": "PET study for localization of tumor"
                    },
                    "has_intent": {"code_meaning": "Diagnostic Intent"},
                    "radiopharmaceuticaladministrationeventdata_set": {
                        "get": {
                            "radiopharmaceutical_agent": {
                                "code_meaning": "Fluorodeoxyglucose F^18^"
                            },
                            "radionuclide": {"code_meaning": "^18^Fluorine"},
                            "radionuclide_half_life": Decimal(6586.2),
                            "radiopharmaceutical_specific_activity": Decimal(10.1),
                            "radiopharmaceutical_administration_event_uid": "1.3.12.2.1107.5.1.4.11090.20220223082918.0",
                            "intravenousextravasationsymptoms_set": {
                                "first": {
                                    "intravenous_extravasation_symptoms": {
                                        "code_meaning": "Injection site abscess"
                                    }
                                },
                                "last": {
                                    "intravenous_extravasation_symptoms": {
                                        "code_meaning": "Injection site anesthesia"
                                    }
                                },
                            },
                            "estimated_extravasation_activity": Decimal(10.0),
                            "radiopharmaceutical_volume": Decimal(100),
                            "radiopharmaceutical_start_datetime": datetime(
                                2022, 2, 23, 8, 29, 18, 0
                            ),
                            "radiopharmaceutical_stop_datetime": datetime(
                                2022, 2, 23, 8, 29, 18, 0
                            ),
                            "administered_activity": Decimal(250.0),
                            "effective_dose": Decimal(4.75),
                            "pre_administration_measured_activity": Decimal(11.0),
                            "pre_activity_measurement_device": {
                                "code_meaning": "Dose Calibrator"
                            },
                            "observercontext_set": {
                                "first": {
                                    "radiopharmaceutical_administration_is_pre_observer": True,
                                    "observer_type": {"code_meaning": "Person"},
                                }
                            },
                            "post_administration_measured_activity": Decimal(12.0),
                            "post_activity_measurement_device": {
                                "code_meaning": "Dose Calibrator"
                            },
                            "route_of_administration": {
                                "code_meaning": "Intravenous route"
                            },
                            "site_of": {"code_meaning": "Via vein"},
                            "personparticipant_set": {
                                "get": {
                                    "person_name": "Unknown",
                                    "person_role_in_procedure_cid": {
                                        "code_meaning": "Irradiation Administering"
                                    },
                                }
                            },
                            "billingcode_set": {
                                "get": {
                                    "billing_code": {
                                        "code_meaning": "Nuclear Medicine Procedure and Services"
                                    }
                                }
                            },
                            "drugproductidentifier_set": {
                                "get": {
                                    "drug_product_identifier": {
                                        "code_meaning": "Aconitum radix"
                                    }
                                }
                            },
                            "brand_name": "Some Brand",
                            "radiopharmaceutical_dispense_unit_identifier": "Dispenser",
                            "radiopharmaceuticallotidentifier_set": {
                                "get": {"radiopharmaceutical_lot_identifier": "lot id"}
                            },
                            "reagentvialidentifier_set": {
                                "get": {"reagent_vial_identifier": "vial id"}
                            },
                            "radionuclideidentifier_set": {
                                "get": {"radionuclide_identifier": "radio id"}
                            },
                            "prescription_identifier": "pres id",
                            "comment": "any comment",
                        }
                    },
                    "radiopharmaceuticaladministrationpatientcharacteristics_set": {
                        "get": {
                            "patientstate_set": {
                                "get": {
                                    "patient_state": {
                                        "code_meaning": "Acute unilateral renal blockage"
                                    }
                                }
                            },
                            "subject_age": Decimal(47),
                            "subject_sex": {"code_value": "F"},
                            "patient_height": Decimal(1.68),
                            "patient_weight": Decimal(68),
                            "body_surface_area": Decimal(1.5),
                            "body_surface_area_formula": {"code_value": "122240"},
                            "body_mass_index": Decimal(23.0),
                            "equation": {"code_value": "122265"},
                            "glucose": Decimal(0.87),
                            "fasting_duration": Decimal(4.0),
                            "hydration_volume": Decimal(310),
                            "recent_physical_activity": "None",
                            "serum_creatinine": Decimal(4.3),
                            "glomerularfiltrationrate_set": {
                                "get": {
                                    "glomerular_filtration_rate": Decimal(12.21),
                                    "equivalent_meaning_of_concept_name": {
                                        "code_meaning": "Glomerular Filtration Rate Cystatin-based formula"
                                    },
                                    "measurement_method": {
                                        "code_meaning": "Glomerular Filtration Rate black (MDRD)"
                                    },
                                }
                            },
                        }
                    },
                }
            },
        }

        self._check_values(study, expected_data)

    def _2_associated_studies(self):
        self.assertEqual(GeneralStudyModuleAttr.objects.count(), 2)
        study1 = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").get()
        study2 = GeneralStudyModuleAttr.objects.filter(modality_type__exact="NM").get()
        self.assertEqual(study1.study_id, study2.study_id)

    def test_load_associated_ct_before(self):
        """Check if loading the associated RDSR before the RRDSR works correctly"""
        rdsr_path = self._get_dcm_file("test_files/NM-CT-RDSR-Siemens.dcm")
        rdsr.rdsr(rdsr_path)
        self.test_import_siemens_rrdsr()

        self._2_associated_studies()

    def test_load_associated_ct_after(self):
        """Check if loading the associated RDSR after the RRDSR works correctly"""
        self.test_import_siemens_rrdsr()
        rdsr_path = self._get_dcm_file("test_files/NM-CT-RDSR-Siemens.dcm")
        rdsr.rdsr(rdsr_path)

        self._2_associated_studies()

    def test_no_reimport(self):
        self.test_import_siemens_rrdsr()
        self.test_import_siemens_rrdsr()
        rdsr_path = self._get_dcm_file("test_files/NM-CT-RDSR-Siemens.dcm")
        rdsr.rdsr(rdsr_path)
        rdsr.rdsr(rdsr_path)

        self._2_associated_studies()
