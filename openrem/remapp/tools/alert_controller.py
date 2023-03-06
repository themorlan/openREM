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
from dateutil import relativedelta
from django.core.exceptions import ObjectDoesNotExist

from remapp.models import GeneralStudyModuleAttr, StandardNames, PatientModuleAttr, Patients, VolatilePatientData, DiagnosticReferenceLevels, DiagnosticReferenceLevelAlerts


def check_for_new_alerts():
    studies = GeneralStudyModuleAttr.objects.all()
    for study in studies:
        check_for_new_alerts_in_study(study)


def check_for_new_alerts_in_study(study: GeneralStudyModuleAttr):
    std_names = StandardNames.objects.filter(modality__exact=study.modality_type)
    try:
        patient_id = PatientModuleAttr.objects.get(general_study_module_attributes=study).patient_id
        patient = Patients.objects.get(patient_id=patient_id, general_study_module_attr__in=[study])
    except ObjectDoesNotExist:
        return

    patient_age = None
    if study.study_date and patient.patient_birth_date:
        patient_age = relativedelta.relativedelta(study.study_date, patient.patient_birth_date).years

    try:
        additional_patient_data = VolatilePatientData.objects.get(patient=patient, general_study_module_attr=study)
        patient_bmi = None
        if additional_patient_data.patient_weight and additional_patient_data.patient_size:
            patient_bmi = additional_patient_data.patient_weight / (additional_patient_data.patient_size ** 2)
    except ObjectDoesNotExist:
        # TODO: interpolate with existing data
        return

    check_drv_values_for_study(study, patient_age, patient_bmi, std_names)


def check_drv_values_for_study(study: GeneralStudyModuleAttr, patient_age, patient_bmi, std_names):
    study_dict = study.__dict__
    modality = study.modality_type

    if modality == "CT":
        drl_name = "total_dlp"
    elif modality == "RF" or modality == "DX":
        drl_name = "total_dap"
    else:
        return
    
    if study_dict[drl_name] is None:
        return

    field_names = [
        "study_description",
        "requested_procedure_code_meaning",
        "procedure_code_meaning",
    ]

    for field_name in field_names:
        try:
            std_name = std_names.get(**{f"{field_name}__in": [study_dict[field_name]]})
        except ObjectDoesNotExist:
            continue
        check_drl_for_study_and_std_name(study, std_name, patient_age, patient_bmi, study_dict[drl_name])


def check_drl_for_study_and_std_name(study: GeneralStudyModuleAttr, std_name: StandardNames, patient_age, patient_bmi, value):
    if not std_name.diagnostic_reference_level_criteria:
        return
    
    if std_name.diagnostic_reference_level_criteria == "bmi":
        ref_val = patient_bmi
        print(ref_val)
    else:
        ref_val = patient_age

    if not ref_val:
        return

    try:
        drl = DiagnosticReferenceLevels.objects.filter(standard_name__in=[std_name]).get(lower_bound__lte=ref_val, upper_bound__gte=ref_val)
    except ObjectDoesNotExist:
        return
    if drl.diagnostic_reference_level is None:
        return
    if value < drl.diagnostic_reference_level:
        return
    if DiagnosticReferenceLevelAlerts.objects.filter(diagnostic_reference_level=drl, general_study_module_attributes=study, standard_name=std_name).exists():
        return
    DiagnosticReferenceLevelAlerts.objects.create(
        date_of_issue=datetime.now(),
        diagnostic_reference_level=drl,
        general_study_module_attributes=study,
        standard_name=std_name
    ).save()
