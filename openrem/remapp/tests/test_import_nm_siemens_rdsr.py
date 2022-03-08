from datetime import date, time, datetime
import os
from decimal import Decimal
from typing import Callable, Dict, Tuple
from django.test import TestCase

from remapp.extractors import rdsr
from remapp.models import GeneralStudyModuleAttr, PatientIDSettings

class ImportNMRDSR(TestCase):
    def _verify_equality(self, value, expect_value, location):
        msg = f"At {location}"
        check_type = type(expect_value)
        self.assertEqual(check_type, type(value), msg)

        if (check_type == date or check_type == str or
            check_type == time or check_type == datetime):
            self.assertEqual(value, expect_value, msg)
        elif check_type == Decimal:
            self.assertAlmostEqual(value, expect_value, msg=msg)
        elif check_type == type(None):
            self.assertIsNone(value, msg)
        else:
            raise NotImplementedError(f"Type {check_type} has no associated check at {location}")

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
                if callable(current) and not "_set" in name: # Shortcut so one can use strings for zero parameter functions
                    current = current()
            except Exception as e:
                raise ValueError(f"{name} is a function. Only functions without parameters usable at {location}")
            if not isinstance(subdata, Dict):
                self._verify_equality(current, subdata, location)
            else:
                self._check_values(current, subdata, location)
            

    def test_import_siemens_rrdsr(self):
        """Loads a Siemens RRDSR-File and checks all values against the expected ones"""

        u: PatientIDSettings = PatientIDSettings.objects.create()
        u.name_stored = True
        u.name_hashed = False
        u.save()

        dicom_file = "PET_1_STUDY_0_RRDSR"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        root_tests = "/home/medphys/Schreibtisch/jannis_local/DICOM-Daten"
        dicom_path = os.path.join(root_tests, dicom_file)

        rdsr.rdsr(dicom_path)

        study = GeneralStudyModuleAttr.objects.first()

        # We verify only the first and the last organ dose
        expected_data = {
            "patientmoduleattr_set": {
                "get": {
                    "not_patient_indicator": None,
                    "patient_name": "REMOVED1",
                }
            },
            "study_date": date(2000, 3, 24),
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
                    "has_intent": {
                        "code_meaning": "Diagnostic Intent"
                    },
                    "radiopharmaceuticaladministrationeventdata_set": {
                        "get": {
                            "radiopharmaceutical_agent": {
                                "code_meaning": "Fluorodeoxyglucose F^18^"
                            },
                            "radionuclide": {
                                "code_meaning": "^18^Fluorine"
                            },
                            "radionuclide_half_life": Decimal(6586.2),
                            "radiopharmaceutical_administration_event_uid": "1.3.12.2.1107.5.1.4.11090.20220224104830.000000",
                            "radiopharmaceutical_start_datetime": datetime(2000, 3, 24, 10, 40, 30, 0),
                            "radiopharmaceutical_stop_datetime": datetime(2000, 3, 24, 10, 40, 40, 0),
                            "administered_activity": Decimal(394.0),
                            "organdose_set": {
                                "first": {
                                    "finding_site": {
                                        "code_meaning": "Adrenal gland"
                                    },
                                    "laterality": {
                                        "code_meaning": "Right and left"
                                    },
                                    "organ_dose": Decimal(4.73),
                                    "reference_authority_text": "ICRP Publication 128"
                                },
                                "last": {
                                    "finding_site": {
                                        "code_meaning": "Bladder"
                                    },
                                    "laterality": None,
                                    "organ_dose": Decimal(51.22),
                                    "reference_authority_text": "ICRP Publication 128"
                                }
                            },
                            "route_of_administration": {
                                "code_meaning": "Intravenous route"
                            },
                            "site_of": {
                                "code_meaning": "Via vein"
                            },
                            "personparticipant_set": {
                                "get": {
                                    "person_name": "Unknown",
                                    "person_role_in_procedure_cid": {
                                        "code_meaning": "Irradiation Administering"
                                    }
                                }
                            }
                        }
                    },
                    "radiopharmaceuticaladministrationpatientcharacteristics_set": {
                        "get": {
                            "subject_age": Decimal(63),
                            "subject_sex": {
                                "code_value": "M"
                            },
                            "patient_height": Decimal(1.78),
                            "patient_weight": Decimal(110),
                        }
                    }
                }
            }
        }

        self._check_values(study, expected_data)