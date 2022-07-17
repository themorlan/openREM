# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2022  The Royal Marsden NHS Foundation Trust
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
    def _get_siemens_image_expected_radioadmin(self):
        return {
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
        }

    def test_pet_image_siemens_alone(self):
        """
        Loads a single PET image
        """
        img_loc = self._get_dcm_file("test_files/NM-PetIm-Siemens.dcm")
        nm_image(img_loc)

        expected = {
            "modality_type": "NM",
            "patientmoduleattr_set": {
                "get": {
                    "not_patient_indicator": None,
                    "patient_name": "REMOVED",
                }
            },
            "patientstudymoduleattr_set": {
                "get": {
                    "patient_age": "063Y",
                }
            },
            "radiopharmaceuticalradiationdose_set": {
                "get": {
                    "radiopharmaceuticaladministrationeventdata_set": self._get_siemens_image_expected_radioadmin(),
                    "petseries_set": {
                        lambda x: x.get(): {
                            "series_datetime": datetime(2022, 2, 24, 11, 56),
                            "number_of_rr_intervals": None,
                            "number_of_time_slots": None,
                            "number_of_time_slices": None,
                            "number_of_slices": Decimal(476),
                            "reconstruction_method": "PSF+TOF 4i5s",
                            "coincidence_window_width": None,
                            "energy_window_lower_limit": Decimal(435),
                            "energy_window_upper_limit": Decimal(585),
                            "scan_progression_direction": None,
                            "petseriescorrection_set": {
                                "all": {
                                    lambda x: x[0]: {"corrected_image": "NORM"},
                                    lambda x: x[1]: {"corrected_image": "DTIM"},
                                    lambda x: x[2]: {"corrected_image": "ATTN"},
                                    lambda x: x[3]: {"corrected_image": "SCAT"},
                                    lambda x: x[4]: {"corrected_image": "DECY"},
                                    lambda x: x[5]: {"corrected_image": "RAN"},
                                }
                            },
                            "petseriestype_set": {
                                "all": {
                                    lambda x: x[0]: {"series_type": "WHOLE BODY"},
                                }
                            },
                        }
                    },
                }
            },
        }

        study = GeneralStudyModuleAttr.objects.get()
        self._check_values(study, expected)

    def test_pet_image_ge_alone(self):
        """
        Loads a single PET Image
        """
        img_loc = self._get_dcm_file("test_files/NM-PetIm-GE.dcm")
        nm_image(img_loc)

        expected = {
            "modality_type": "NM",
            "patientmoduleattr_set": {
                "get": {
                    "not_patient_indicator": None,
                    "patient_name": "REMOVED",
                }
            },
            "patientstudymoduleattr_set": {
                "get": {
                    "patient_age": "051Y",
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
                            "radiopharmaceutical_agent_string": "FDG -- fluorodeoxyglucose",
                            "radionuclide_half_life": Decimal(6586.2001953125),
                            "administered_activity": Decimal(221.596288),
                            "radiopharmaceutical_start_datetime": datetime(
                                2022, 3, 10, 12, 41, 0
                            ),
                            "radiopharmaceutical_stop_datetime": datetime(
                                2022, 3, 10, 12, 42, 0
                            ),
                        }
                    },
                    "petseries_set": {
                        "get": {
                            "series_datetime": datetime(2022, 3, 10, 13, 31, 55),
                            "number_of_rr_intervals": None,
                            "number_of_time_slots": None,
                            "number_of_time_slices": None,
                            "number_of_slices": Decimal(313),
                            "reconstruction_method": "VPFXS",
                            "coincidence_window_width": Decimal(0.0),
                            "energy_window_lower_limit": Decimal(425),
                            "energy_window_upper_limit": Decimal(650),
                            "scan_progression_direction": None,
                            "petseriescorrection_set": {
                                "all": {
                                    lambda x: x[0]: {"corrected_image": "DECY"},
                                    "last": {"corrected_image": "NORM"},
                                }
                            },
                            "petseriestype_set": {
                                "first": {"series_type": "STATIC"},
                                "last": {"series_type": "IMAGE"},
                            },
                        }
                    },
                }
            },
        }

        study = GeneralStudyModuleAttr.objects.get()
        self._check_values(study, expected)

    def _get_pet_simens_rrdsr_one_img_combined(self):
        return {
            "radiopharmaceuticalradiationdose_set": {
                "get": {
                    "radiopharmaceuticaladministrationeventdata_set": {
                        "get": {
                            "radiopharmaceutical_agent": {
                                "code_meaning": "Fluorodeoxyglucose F^18^"
                            },
                            "radionuclide": {"code_meaning": "^18^Fluorine"},
                            "radionuclide_half_life": Decimal(6586.2),
                            "radiopharmaceutical_administration_event_uid": "1.3.12.2.1107.5.1.4.11090.20220224104830.0",
                            "administered_activity": Decimal(394.0),
                            "radiopharmaceutical_start_datetime": datetime(  # first loaded takes precedence
                                2022, 2, 24, 10, 40, 30
                            ),
                            "radiopharmaceutical_stop_datetime": datetime(
                                2022, 2, 24, 10, 40, 30
                            ),
                        }
                    },
                    "petseries_set": {
                        lambda x: x.get(): {
                            "series_datetime": datetime(2022, 2, 24, 11, 56),
                            "number_of_rr_intervals": None,
                            "number_of_time_slots": None,
                            "number_of_time_slices": None,
                            "number_of_slices": Decimal(476),
                            "reconstruction_method": "PSF+TOF 4i5s",
                            "coincidence_window_width": None,
                            "energy_window_lower_limit": Decimal(435),
                            "energy_window_upper_limit": Decimal(585),
                            "scan_progression_direction": None,
                            "petseriescorrection_set": {
                                "first": {"corrected_image": "NORM"},
                            },
                            "petseriestype_set": {
                                "all": {
                                    lambda x: x[0]: {"series_type": "WHOLE BODY"},
                                }
                            },
                        }
                    },
                }
            }
        }

    @patch("remapp.extractors.nm_image.logger")
    def test_rrdsr_pet_image(self, logger_mock):
        """
        Loads the rrdsr, followed by an associated PET-Image
        """
        rrdsr_file = self._get_dcm_file("test_files/NM-RRDSR-Siemens.dcm")
        rdsr(rrdsr_file)
        img_loc = self._get_dcm_file("test_files/NM-PetIm-Siemens.dcm")
        nm_image(img_loc)

        logger_mock.warning.assert_called()  # Dates are set differently, therefore logger warns

        study = GeneralStudyModuleAttr.objects.get()
        self._check_values(study, self._get_pet_simens_rrdsr_one_img_combined())

    def test_pet_image_rrdsr(self):
        """
        Loads the PET-Image, followed by an associated rrdsr
        """
        img_loc = self._get_dcm_file("test_files/NM-PetIm-Siemens.dcm")
        nm_image(img_loc)
        rrdsr_file = self._get_dcm_file("test_files/NM-RRDSR-Siemens.dcm")
        rdsr(rrdsr_file)

        study = GeneralStudyModuleAttr.objects.get()
        self._check_values(study, self._get_pet_simens_rrdsr_one_img_combined())

    @patch("remapp.extractors.nm_image.logger")
    def test_no_reloads_1(self, logger_mock):
        """
        Loads the image twice to verify that it is imported only once
        """
        img_loc = self._get_dcm_file("test_files/NM-PetIm-Siemens.dcm")
        nm_image(img_loc)
        logger_mock.info.assert_not_called()
        nm_image(img_loc)
        logger_mock.info.assert_called_once()

    @patch("remapp.extractors.nm_image.logger")
    def test_no_reloads_2(self, logger_mock):
        """
        Loads the image twice to verify that it is imported only once. Same as above, but now the rrdsr has been imported previously.
        """
        rrdsr_file = self._get_dcm_file("test_files/NM-RRDSR-Siemens.dcm")
        rdsr(rrdsr_file)
        img_loc = self._get_dcm_file("test_files/NM-PetIm-Siemens.dcm")
        nm_image(img_loc)
        logger_mock.info.assert_not_called()
        nm_image(img_loc)
        logger_mock.info.assert_called_once()

    def test_load_spect_image_alone(self):
        img_loc = self._get_dcm_file("test_files/NM-NmIm-Siemens.dcm")
        nm_image(img_loc)

        expected = {
            "modality_type": "NM",
            "study_date": datetime(2022, 2, 24).date(),
            "patientmoduleattr_set": {
                "get": {
                    "not_patient_indicator": None,
                    "patient_name": "REMOVED",
                    "patient_sex": "F",
                }
            },
            "patientstudymoduleattr_set": {
                "get": {
                    "patient_age": "050Y",
                }
            },
            "generalequipmentmoduleattr_set": {
                "get": {
                    "manufacturer": "SIEMENS NM",
                }
            },
            "radiopharmaceuticalradiationdose_set": {
                "get": {
                    "radiopharmaceuticaladministrationeventdata_set": {
                        "get": {
                            "radiopharmaceutical_agent_string": "DPD",
                            "radionuclide": {"code_meaning": "99m Technetium"},
                            "administered_activity": Decimal(764.0),
                        }
                    },
                }
            },
        }
        study = GeneralStudyModuleAttr.objects.get()
        self._check_values(study, expected)

    @patch("remapp.extractors.nm_image.logger")
    def test_pet_image_siemens_multiple(self, logger_mock):
        nm_image(self._get_dcm_file("test_files/NM-PetIm-Siemens.dcm"))
        nm_image(self._get_dcm_file("test_files/NM-PetIm-Siemens-3-sameseries.dcm"))
        nm_image(self._get_dcm_file("test_files/NM-PetIm-Siemens-2.dcm"))
        nm_image(self._get_dcm_file("test_files/NM-PetIm-Siemens-3.dcm"))

        logger_mock.info.assert_called()  # 3 will not be loaded the second time, because it's already imported

        study = GeneralStudyModuleAttr.objects.get()
        expected = {
            "modality_type": "NM",
            "patientmoduleattr_set": {
                "get": {
                    "not_patient_indicator": None,
                    "patient_name": "REMOVED",
                }
            },
            "patientstudymoduleattr_set": {
                "get": {
                    "patient_age": "063Y",
                }
            },
            "radiopharmaceuticalradiationdose_set": {
                "get": {
                    "radiopharmaceuticaladministrationeventdata_set": self._get_siemens_image_expected_radioadmin(),
                    "petseries_set": {
                        "count": 3,
                        "all": {
                            lambda x: x[0]: {
                                "series_datetime": datetime(2022, 2, 24, 11, 56),
                                "number_of_rr_intervals": None,
                                "number_of_time_slots": None,
                                "number_of_time_slices": None,
                                "number_of_slices": Decimal(476),
                                "reconstruction_method": "PSF+TOF 4i5s",
                                "energy_window_lower_limit": Decimal(435),
                                "energy_window_upper_limit": Decimal(585),
                                "scan_progression_direction": None,
                                "petseriescorrection_set": {"count": 6},
                                "petseriestype_set": {"count": 2},
                            },
                            lambda x: x[1]: {
                                "series_datetime": datetime(2022, 2, 24, 11, 56),
                                "number_of_slices": Decimal(476),
                                "reconstruction_method": "PSF+TOF 7i5s",
                                "energy_window_lower_limit": Decimal(435),
                                "energy_window_upper_limit": Decimal(585),
                                "scan_progression_direction": None,
                                "petseriescorrection_set": {"count": 6},
                                "petseriestype_set": {"count": 2},
                            },
                            lambda x: x[2]: {
                                "petseriescorrection_set": {"count": 4},
                                "petseriestype_set": {"count": 2},
                            },
                        },
                    },
                }
            },
        }
        self._check_values(study, expected)

    def test_nm_image_multiple(self):
        nm_image(self._get_dcm_file("test_files/NM-NmIm-Siemens-s1-1.dcm"))
        nm_image(self._get_dcm_file("test_files/NM-NmIm-Siemens-s1-2.dcm"))

        study = GeneralStudyModuleAttr.objects.get()
        expected = {
            "modality_type": "NM",
            "study_date": datetime(2022, 2, 18).date(),
            "patientmoduleattr_set": {
                "get": {
                    "not_patient_indicator": None,
                    "patient_name": "REMOVED",
                    "patient_sex": "F",
                }
            },
            "patientstudymoduleattr_set": {
                "get": {
                    "patient_age": "071Y",
                }
            },
            "generalequipmentmoduleattr_set": {
                "get": {
                    "manufacturer": "SIEMENS NM",
                }
            },
            "radiopharmaceuticalradiationdose_set": {
                "get": {
                    "radiopharmaceuticaladministrationeventdata_set": {
                        "get": {
                            "radiopharmaceutical_agent_string": "Sestamibi",
                            "radionuclide": {"code_meaning": "99m Technetium"},
                            "administered_activity": Decimal(320.0),
                        }
                    },
                }
            },
        }
        self._check_values(study, expected)
