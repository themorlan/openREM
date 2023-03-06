#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2023 The Royal Marsden NHS Foundation Trust
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

"""
..  module:: alert_controller
    :synopsis: Module with tools to create, update and delete alerts

..  moduleauthor:: Kevin Schärer

"""

from datetime import datetime
from django.core.exceptions import ObjectDoesNotExist

from remapp.models import GeneralStudyModuleAttr, StandardNames, PatientStudyModuleAttr, DiagnosticReferenceLevels, DiagnosticReferenceLevelAlerts


def check_for_new_alerts():
    studies = GeneralStudyModuleAttr.objects.all()
    for study in studies:
        check_for_new_alerts_in_study(study)


def check_for_new_alerts_in_study(study: GeneralStudyModuleAttr):
    std_names = StandardNames.objects.filter(modality__exact=study.modality_type)
    patient = PatientStudyModuleAttr.objects.get(general_study_module_attributes=study)
    check_drv_values_for_study(study, patient, std_names)


def check_drv_values_for_study(study: GeneralStudyModuleAttr, patient, std_names):
    modality = study.modality_type
    if modality == "CT":
        ref_name = "total_dlp"
    elif modality == "RF" or modality == "DX":
        ref_name = "total_dap"
    else:
        return

    field_names = [
        "study_description",
        "requested_procedure_code_meaning",
        "procedure_code_meaning",
    ]

    for field_name in field_names:
        _check_for_new_drv_alert_in_study(std_names, study, patient, field_name, ref_name)


def _check_for_new_drv_alert_in_study(study: GeneralStudyModuleAttr, std_names, patient, field_name, ref_name: str):
    study_dict = study.__dict__
    patient_age = patient.patient_age_decimal
    if study_dict[ref_name] is None:
        return
    if patient_age is None:
        return
    try:
        std_name = std_names.get(**{f"{field_name}__in": [study_dict[field_name]]})
        ref = DiagnosticReferenceLevels.objects.filter(standard_name__in=[std_name]).get(lower_bound__lte=age, upper_bound__gte=age)
    except ObjectDoesNotExist:
        return
    if ref.diagnostic_reference_level is None:
        return
    if study_dict[ref_name] < ref.diagnostic_reference_level:
        return
    if DiagnosticReferenceLevelAlerts.objects.filter(diagnostic_reference_level=ref, general_study_module_attributes=study, standard_name=std_name).exists():
        return
    DiagnosticReferenceLevelAlerts.objects.create(
        date_of_issue=datetime.now(),
        diagnostic_reference_level=ref,
        general_study_module_attributes=study,
        standard_name=std_name
    ).save()