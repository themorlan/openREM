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


from remapp.models import GeneralStudyModuleAttr
from .test_import_nm_siemens_rdsr import ImportTest
from remapp.extractors.nm_image import nm_image
from remapp.extractors.rdsr import rdsr
from mock import patch

from datetime import datetime
from decimal import Decimal

class ImportNMImage(ImportTest):
    def test_pet_image_alone(self):
        nm_image("/home/medphys/Schreibtisch/jannis_local/DICOM-Daten/PET/PET_1_SIEMENS/DICOM/ST000000/SE000002/PT000000")

        expected = {
            "modality_type": "NM",
            "patientmoduleattr_set": {
                "get": {
                    "not_patient_indicator": None,
                    "patient_name": "REMOVED1",
                }
            },
            "patientstudymoduleattr_set": {
                "get": {
                    "patient_age": "063Y",
                }
            },
            "radiopharmaceuticalradiationdose_set": {
                "get": {
                    "radiopharmaceuticaladministrationeventdata_set": {
                        "get": {
                            "radiopharmaceutical_agent": {
                                "code_meaning": "Fluorodeoxyglucose F^18^"
                            },
                            "radionuclide": {"code_meaning": "^18^Fluorine"},
                            "radionuclide_half_life": Decimal(6586.2),
                            "administered_activity": Decimal(394.0),
                            "radiopharmaceutical_start_datetime": datetime(
                                2022, 2, 24, 10, 48, 30, 0
                            ),
                            "radiopharmaceutical_stop_datetime": datetime(
                                2022, 2, 24, 10, 48, 30, 0
                            ),
                        }
                    },
                }
            }
        }

        study = GeneralStudyModuleAttr.objects.get()
        self._check_values(study, expected)


    @patch('remapp.extractors.nm_image.logger')
    def test_pet_image_rrdsr(self, logger_mock):
        rrdsr_file = self._get_dcm_file("test_files/NM-RRDSR-Siemens.dcm")
        rdsr(rrdsr_file)
        nm_image("/home/medphys/Schreibtisch/jannis_local/DICOM-Daten/PET/PET_1_SIEMENS/DICOM/ST000000/SE000002/PT000000")

        logger_mock.warn.assert_called() # Dates are set differently, therefore logger warns

        expected = {
            "radiopharmaceuticalradiationdose_set": {
                "get": {
                    "radiopharmaceuticaladministrationeventdata_set": {
                        "get": {
                            "radiopharmaceutical_agent": {
                                "code_meaning": "Fluorodeoxyglucose F^18^"
                            },
                            "radionuclide": {"code_meaning": "^18^Fluorine"},
                            "radionuclide_half_life": Decimal(6586.2),
                            "radiopharmaceutical_administration_event_uid": "1.3.12.2.1107.5.1.4.11090.20220224104830.000000",
                            "administered_activity": Decimal(394.0),
                            "radiopharmaceutical_start_datetime": datetime( # first loaded takes precedence
                                2000, 3, 24, 10, 40, 30
                            ),
                            "radiopharmaceutical_stop_datetime": datetime(
                                2000, 3, 24, 10, 40, 40
                            ),
                        }
                    },
                }
            }
        }

        study = GeneralStudyModuleAttr.objects.get()
        self._check_values(study, expected)

