# This Python file uses the following encoding: utf-8
# test_import_px.py

import os
from collections import Counter
import datetime
from decimal import Decimal
from django.contrib.auth.models import User, Group
from django.test import TestCase
from pydicom.dataset import Dataset
from pydicom.dataelem import DataElement
from pydicom.multival import MultiValue
import logging
from testfixtures import LogCapture
from remapp.extractors.dx import _xray_filters_prep
from remapp.models import (
    GeneralStudyModuleAttr,
    ProjectionXRayRadiationDose,
    IrradEventXRayData,
    IrradEventXRaySourceData,
)
from openremproject import settings


settings.LOGGING["loggers"]["remapp"]["level"] = "DEBUG"


class PXFilterTests(TestCase):
    def setUp(self):
        from remapp.extractors import dx
        from remapp.models import PatientIDSettings

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

        px_newtom_go_1 = os.path.join("test_files", "PX-Im-NewTom_Go.dcm")

        root_tests = os.path.dirname(os.path.abspath(__file__))

        with LogCapture(level=logging.DEBUG) as self.log:
            dx.dx(os.path.join(root_tests, px_newtom_go_1))

    def test_newtom_go(self):

        studies = GeneralStudyModuleAttr.objects.order_by("id")

        # Test that one study has been imported
        self.assertEqual(studies.count(), 1)

        # Test that study level data is recorded correctly
        self.assertEqual(studies[0].study_date, datetime.date(2021, 4, 12))
        self.assertEqual(studies[0].study_time, datetime.time(15, 14, 20))
        self.assertEqual(
            studies[0].generalequipmentmoduleattr_set.get().institution_name,
            "A County Hospital",
        )
        self.assertEqual(
            studies[0].generalequipmentmoduleattr_set.get().manufacturer,
            "NNT",
        )
        self.assertEqual(
            studies[0].generalequipmentmoduleattr_set.get().station_name,
            "1210218143451168",
        )
        self.assertEqual(
            studies[0].generalequipmentmoduleattr_set.get().manufacturer_model_name,
            "OPT_EL_CEPH",
        )
        self.assertEqual(
            studies[0].generalequipmentmoduleattr_set.get().software_versions,
            "9_1_0_0",
        )

        # Test the SOP instance UID
        sop_instance_uid_list0 = Counter(
            studies[0].objectuidsprocessed_set.values_list(
                "sop_instance_uid", flat=True
            )
        )
        uid_list0 = Counter(
            [
                "1.76.380.18.0.12102181434511.612104121514200050004",
            ]
        )
        self.assertEqual(sop_instance_uid_list0, uid_list0)

        # Test that patient level data is recorded correctly
        self.assertEqual(
            studies[0].patientmoduleattr_set.get().patient_name,
            "PHYSICS",
        )
        self.assertEqual(
            studies[0].patientmoduleattr_set.get().patient_id,
            None,
        )
        self.assertEqual(
            studies[0].patientmoduleattr_set.get().patient_birth_date,
            datetime.date(1899, 12, 30),
        )

        # Test that irradiation event data is stored correctly
        self.assertEqual(
            studies[0]
            .projectionxrayradiationdose_set.get()
            .irradeventxraydata_set.order_by("id")[0]
            .acquisition_protocol,
            "PANORAMIC",
        )
        self.assertAlmostEqual(
            studies[0]
            .projectionxrayradiationdose_set.get()
            .irradeventxraydata_set.order_by("id")[0]
            .dose_area_product,
            Decimal(0.81 / 100000),
        )

        # Test that irradiation event source data is stored correctly
        self.assertAlmostEqual(
            studies[0]
            .projectionxrayradiationdose_set.get()
            .irradeventxraydata_set.order_by("id")[0]
            .irradeventxraysourcedata_set.get()
            .exposure_time,
            Decimal(6800),
        )
        self.assertAlmostEqual(
            studies[0]
            .projectionxrayradiationdose_set.get()
            .irradeventxraydata_set.order_by("id")[0]
            .irradeventxraysourcedata_set.get()
            .average_xray_tube_current,
            Decimal(9),
        )
        self.assertAlmostEqual(
            studies[0]
            .projectionxrayradiationdose_set.get()
            .irradeventxraydata_set.order_by("id")[0]
            .irradeventxraysourcedata_set.get()
            .kvp_set.get()
            .kvp,
            Decimal(85),
        )

        # Test summary fields
        self.assertAlmostEqual(studies[0].total_dap_a, Decimal(0.81 / 100000))
        self.assertAlmostEqual(studies[0].total_dap, Decimal(0.81 / 100000))
        self.assertEqual(studies[0].number_of_events, 1)
        self.assertEqual(studies[0].number_of_planes, 1)
