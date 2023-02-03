# This Python file uses the following encoding: utf-8
# test_mod_filters.py

from decimal import Decimal
import os
from django.test import TestCase

from django.db.models import QuerySet
from remapp.interface.mod_filters import json_to_query
from remapp.models import GeneralStudyModuleAttr, User, GeneralEquipmentModuleAttr
from ..extractors import rdsr, ct_philips
from .test_filters_data import (
    get_simple_query,
    get_simple_multiple_query,
    DummyModClass,
    get_or_query,
)
from ..models import CtIrradiationEventData, PatientIDSettings

class ModFiltersTest(TestCase):
    def setUp(self):
        """
        Load in all the CT objects so that there is something to filter!
        """
        PatientIDSettings.objects.create()
        User.objects.create_user("temporary", "temporary@gmail.com", "temporary")

        ct1 = "test_files/CT-ESR-GE_Optima.dcm"
        ct2 = "test_files/CT-ESR-GE_VCT.dcm"
        ct3 = "test_files/CT-RDSR-GEPixelMed.dcm"
        ct4 = "test_files/CT-RDSR-Siemens_Flash-QA-DS.dcm"
        ct5 = "test_files/CT-RDSR-Siemens_Flash-TAP-SS.dcm"
        ct6 = "test_files/CT-RDSR-ToshibaPixelMed.dcm"
        ct7 = "test_files/CT-SC-Philips_Brilliance16P.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        path_ct1 = os.path.join(root_tests, ct1)
        path_ct2 = os.path.join(root_tests, ct2)
        path_ct3 = os.path.join(root_tests, ct3)
        path_ct4 = os.path.join(root_tests, ct4)
        path_ct5 = os.path.join(root_tests, ct5)
        path_ct6 = os.path.join(root_tests, ct6)
        path_ct7 = os.path.join(root_tests, ct7)

        rdsr.rdsr(path_ct1)
        rdsr.rdsr(path_ct2)
        rdsr.rdsr(path_ct3)
        rdsr.rdsr(path_ct4)
        rdsr.rdsr(path_ct5)
        rdsr.rdsr(path_ct6)
        ct_philips.ct_philips(path_ct7)

    def test_unfiltered(self):
        """
        Initial test to ensure seven studies are listed with no filter
        """
        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result: QuerySet[GeneralStudyModuleAttr] = json_to_query(get_simple_multiple_query({}), qs, DummyModClass)
        self.assertEqual(result.count(), 7)


    def test_filter_simple_study_description(self):
        """
        Apply study description filter
        """
        filter_pattern = get_simple_query("study_description", "abdomen")
        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 1)

        study: GeneralStudyModuleAttr = result[0]

        display_name = GeneralEquipmentModuleAttr.objects.get(
            general_study_module_attributes=study.pk
        ).unique_equipment_name.display_name
        self.assertEqual(display_name, "PHILIPS-E71E3F0")

    def test_filter_simple_inverted_study_description(self):
        """
        Apply study description filter
        """
        filter_pattern = get_simple_query(
            "study_description", ["abdomen", "icontains", True]
        )
        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 6)

    def test_filter_simple_model(self):
        """
        Apply model filter
        """
        filter_pattern = get_simple_query(
            "generalequipmentmoduleattr__manufacturer_model_name",
            ["SOMATOM Definition Flash", "iexact", False],
        )
        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 2)

    def test_filter_simple_description_and_model(self):
        """
        Apply study description and model filter
        """
        filter_pattern = get_simple_multiple_query(
            {
                "study_description": ["Thorax", "icontains", False],
                "generalequipmentmoduleattr__manufacturer_model_name": [
                    "SOMATOM Definition Flash",
                    "iexact",
                    False,
                ],
            }
        )
        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 1)

        study: GeneralStudyModuleAttr = result[0]

        self.assertEqual(study.accession_number, "ACC12345601")

    def test_filter_or(self):
        """
        Apply study description or (study description and model) filter
        """
        filter_pattern = get_or_query(
            {
                "study_description": ["abdomen", "icontains", False],
            },
            {
                "study_description": ["Thorax", "icontains", False],
                "generalequipmentmoduleattr__manufacturer_model_name": [
                    "SOMATOM Definition Flash",
                    "iexact",
                    False,
                ],
            },
        )
        qs = (
            GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT")
            .distinct()
            .order_by("study_description")
        )

        result: QuerySet[GeneralStudyModuleAttr] = json_to_query(
            filter_pattern, qs, DummyModClass
        )
        self.assertEqual(result.count(), 2)

        self.assertEqual(result[0].study_description, "Abdomen/IH_KUB_2mm")
        self.assertEqual(result[1].study_description, "Thorax^TAP (Adult)")


    def test_filter_simple_total_dlp(self):
        """
        Apply study description filter
        """
        filter_pattern = get_simple_multiple_query({
            "total_dlp__range_min": "1",
            "total_dlp__range_max": "2"
        })
        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 0)


        filter_pattern = get_simple_multiple_query({
            "total_dlp__range_min": "400",
            "total_dlp__range_max": "500"
        })
        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result: QuerySet[GeneralStudyModuleAttr] = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 1)
        self.assertEqual(result[0].accession_number, "0012345.12345678")


        filter_pattern = get_simple_multiple_query({
            "total_dlp__range_min": "1103",
        })
        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result: QuerySet[GeneralStudyModuleAttr] = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 2)


    def test_filter_simple_event_dlp(self):
        """
        Apply study description filter
        """
        filter_pattern = get_simple_multiple_query({
            "event_dlp__range_min": "655",
            "event_dlp__range_max": "657.8"
        })

        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 0)


        filter_pattern = get_simple_multiple_query({
            "event_dlp__range_min": "657.8",
            "event_dlp__range_max": "658"
        })

        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result: QuerySet[GeneralStudyModuleAttr] = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 1)
        study = result[0]
        study_events: QuerySet[CtIrradiationEventData] = study.ctradiationdose_set.get().ctirradiationeventdata_set.select_related(
                    "ct_acquisition_type", "ctdiw_phantom_type"
                ).order_by("-pk")
        study_event = study_events.get(acquisition_protocol__exact="100S C/A/P")
        self.assertAlmostEqual(study_event.dlp, Decimal(657.85012000))

        # Matches three (3) studies, while one study has two matching events
        filter_pattern = get_simple_multiple_query({
            "event_dlp__range_min": "195",
            "event_dlp__range_max": "250"
        })

        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 3)


    def test_filter_or_event_dlp(self):
        """
        Apply study description filter
        """
        filter_pattern = get_or_query(
            {
                "event_dlp__range_min": "15",
                "event_dlp__range_max": "17",
            },
            {
                "event_dlp__range_min": "15",
                "event_dlp__range_max": "17",
            }
        )

        qs = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT").distinct()

        result = json_to_query(filter_pattern, qs, DummyModClass)
        self.assertEqual(result.count(), 1)
        study: GeneralStudyModuleAttr = result[0]

        self.assertEqual(study.study_description, "FACIAL BONES")

