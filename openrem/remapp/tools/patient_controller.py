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
..  module:: patient_controller
    :synopsis: Module with tools to create, update (migrate) and delete patient data

..  moduleauthor:: Kevin Schärer

"""

from django.core.exceptions import ObjectDoesNotExist
from remapp.models import GeneralStudyModuleAttr, Patients, VolatilePatientData, PatientModuleAttr, PatientStudyModuleAttr

def add_or_update_patients():
    studies = GeneralStudyModuleAttr.objects.all().order_by("-study_date")
    for study in studies:
        add_or_update_patient_from_study(study)


def add_or_update_patient_from_study(study: GeneralStudyModuleAttr):
    patient_module_attr = PatientModuleAttr.objects.get(general_study_module_attributes=study)
    patient_study_module_attr = PatientStudyModuleAttr.objects.get(general_study_module_attributes=study)

    if patient_module_attr.patient_id is None:
        # Patient has no ID, thus we cannot use the patient data
        return
    
    try:
        # Patient object already exists, thus we update missing data points
        patient = Patients.objects.get(patient_id__exact=patient_module_attr.patient_id)
        _update_patient(patient, patient_module_attr)
    except ObjectDoesNotExist:
        # Create a new patient object
        patient = _create_patient(patient_module_attr)
    
    if study.study_date is None:
        return
    
    VolatilePatientData.objects.create(
        patient=patient,
        record_date=study.study_date,
        patient_size=patient_study_module_attr.patient_size,
        patient_weight=patient_study_module_attr.patient_weight,
    ).save()


def _update_patient(patient: Patients, patient_module_attr: PatientModuleAttr):
    if patient_module_attr.patient_name:
        patient.patient_name = patient_module_attr.patient_name
    if patient_module_attr.patient_name:
        patient.patient_name = patient_module_attr.patient_name
    if patient_module_attr.patient_name:
        patient.patient_name = patient_module_attr.patient_name
    if patient_module_attr.patient_name:
        patient.patient_name = patient_module_attr.patient_name
    patient.save()


def _create_patient(patient_module_attr: PatientModuleAttr) -> Patients:
    print(patient_module_attr.patient_id)
    patient = Patients(
        patient_id=patient_module_attr.patient_id,
        patient_name=patient_module_attr.patient_name,
        patient_birth_date=patient_module_attr.patient_birth_date,
        patient_sex=patient_module_attr.patient_sex,
    )
    patient.save()
    return patient
