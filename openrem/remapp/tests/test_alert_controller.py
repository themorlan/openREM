# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2023  The Royal Marsden NHS Foundation Trust
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

from datetime import datetime
from decimal import Decimal
from django.test import TestCase

from django.contrib.auth.models import User, Group
from remapp.models import (
    PatientIDSettings,
    StandardNames,
    GeneralStudyModuleAttr,
    CtRadiationDose,
    CtIrradiationEventData,
    DiagnosticReferenceLevels,
    Patients,
    VolatilePatientData,
    DiagnosticReferenceLevelAlerts,
)

from remapp.tools.alert_controller import check_for_new_drl_alerts_in_study


def _gen_ct_study(study_description, dlp: Decimal, **kwargs):
    study = GeneralStudyModuleAttr.objects.create(
        study_description=study_description, modality_type="CT", study_date=datetime(2023,5,1), total_dlp=dlp, **kwargs
    )

    dose = CtRadiationDose.objects.create(general_study_module_attributes=study)

    event = CtIrradiationEventData.objects.create(ct_radiation_dose=dose, dlp=dlp)

    return study


class AlertControllerTest(TestCase):
    def setUp(self) -> None:
        PatientIDSettings.objects.create()
        user = User.objects.create_user("temporary", "temporary@gmail.com", "temporary")
        admingroup = Group.objects.create(name="admingroup")
        user.groups.add(admingroup)

        study = _gen_ct_study("STUDY1", Decimal(200.0))

        patient = Patients.objects.create(patient_birth_date=datetime(2000, 1, 1))

        patient.general_study_module_attr.add(study)

        VolatilePatientData(
            patient=patient,
            general_study_module_attr=study,
            patient_size=Decimal(1.8),
            patient_weight=Decimal(80),
        ).save()

        std_name = StandardNames.objects.create(
            standard_name="CT_STANDARDNAME_1",
            study_description="STUDY1",
            modality="CT",
            diagnostic_reference_level_criteria="age",
            drl_alert_factor=Decimal(1.0),
            k_factor_criteria="age",
        )

        drl = DiagnosticReferenceLevels.objects.create(
            lower_bound=18.0, upper_bound=60.0, diagnostic_reference_level=50.0
        )

        drl.standard_name.add(std_name)

    def test_create_new_drl_alert(self):
        study = GeneralStudyModuleAttr.objects.get(study_description="STUDY1")
        check_for_new_drl_alerts_in_study(study)
        alerts = DiagnosticReferenceLevelAlerts.objects.filter(
            general_study_module_attributes=study
        )
        self.assertEqual(alerts.count(), 1)
        alert = alerts[0]
        self.assertEqual(alert.standard_name.standard_name, "CT_STANDARDNAME_1")
        self.assertEqual(alert.diagnostic_reference_level.diagnostic_reference_level, 50.0)
