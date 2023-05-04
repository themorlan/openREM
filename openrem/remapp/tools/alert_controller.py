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

from datetime import datetime, timedelta
from dateutil import relativedelta
from decimal import Decimal
from django.core.exceptions import ObjectDoesNotExist
from typing import Union
from remapp.models import (
    GeneralStudyModuleAttr,
    EffectiveDoseAlerts,
    StandardNames,
    Patients,
    VolatilePatientData,
    DiagnosticReferenceLevels,
    DiagnosticReferenceLevelAlerts,
    CumulativeDoseSettings,
    KFactors,
    CtIrradiationEventData,
    IrradEventXRayData,
)

STANDARD_STUDY_NAME_MAPPING_FIELDS = [
    "study_description",
    "requested_procedure_code_meaning",
    "procedure_code_meaning",
]

STANDARD_ACQUISITION_NAME_FIELDS = [
    "acquisition_protocol",
]


def check_for_new_alerts():
    check_for_new_drl_alerts()
    get_cumulative_dose_for_patient()


def check_for_new_drl_alerts():
    studies = GeneralStudyModuleAttr.objects.all()
    for study in studies:
        check_for_new_drl_alerts_in_study(study)


def get_cumulative_dose_for_patient():
    """
    Calculates the cumulative dose of a specific patient in a globally defined time period
    """

    patients = Patients.objects.all()

    for patient in patients:
        patient_studies = patient.general_study_module_attr.all().order_by("study_date")
        cumulative_dose_settings: CumulativeDoseSettings = CumulativeDoseSettings.get_solo()  # type: ignore
        period: timedelta = cumulative_dose_settings.alert_time_period
        threshold: Decimal = cumulative_dose_settings.cumulative_dose_threshold

        sliding_window = []

        for next_study in patient_studies:
            if not next_study.study_date:
                continue
            next_sliding_window = []
            for current_study in sliding_window:
                if next_study.study_date - current_study.study_date <= period:
                    next_sliding_window.append(current_study)
            next_sliding_window.append(next_study)

            cumulative_dose = Decimal(0.0)

            for study in next_sliding_window:
                cumulative_dose += calculate_effective_dose_for_study(study, patient)

            if cumulative_dose > threshold:
                # Let's issue an alert (as long as it does not yet exist)
                if not EffectiveDoseAlerts.objects.filter(
                    cumulative_dose=cumulative_dose,
                    first_study_date=next_sliding_window[0].study_date,
                    last_study_date=next_sliding_window[-1].study_date,
                    patient=patient,
                ).exists():
                    alert = EffectiveDoseAlerts(
                        cumulative_dose=cumulative_dose,
                        first_study_date=next_sliding_window[0].study_date,
                        last_study_date=next_sliding_window[-1].study_date,
                        patient=patient,
                    )
                    alert.save()

            sliding_window = next_sliding_window


def calculate_effective_dose_for_study(
    study: GeneralStudyModuleAttr, patient: Patients
) -> Decimal:
    study_std_names = study.standard_names.all()
    max_effective_dose = Decimal(0.0)

    value = _get_comparison_value_from_study(study)

    if not value:
        return Decimal(0.0)

    for std_name in study_std_names:
        patient_attribute = _get_patient_attribute_for_criteria(
            std_name.diagnostic_reference_level_criteria,
            study,
            patient,
        )

        if not patient_attribute:
            continue

        try:
            k_factor: KFactors = std_name.kfactors_set.get(
                lower_bound__lte=patient_attribute, upper_bound__gte=patient_attribute
            )
        except ObjectDoesNotExist:
            continue
        if not k_factor.k_factor:
            continue
        effective_dose = k_factor.k_factor * value
        max_effective_dose = max(effective_dose, max_effective_dose)
    return max_effective_dose


def _get_comparison_value_from_study(
    study: GeneralStudyModuleAttr,
) -> Union[Decimal, None]:
    study_dict = study.__dict__
    modality = study.modality_type

    if modality == "CT":
        return study_dict["total_dlp"]
    elif modality == "RF" or modality == "DX":
        return study_dict["total_dap"]


def _get_comparison_value_from_event(event) -> Union[Decimal, None]:
    if isinstance(event, CtIrradiationEventData):
        return event.dlp
    elif isinstance(event, IrradEventXRayData):
        return event.dose_area_product


def check_for_new_drl_alerts_in_study(study: GeneralStudyModuleAttr):
    std_names = StandardNames.objects.filter(modality=study.modality_type)
    value = _get_comparison_value_from_study(study)
    modality = study.modality_type

    for field_name in STANDARD_STUDY_NAME_MAPPING_FIELDS:
        try:
            std_name = std_names.get(
                **{f"{field_name}__in": [study.__dict__[field_name]]}
            )
        except ObjectDoesNotExist:
            continue
        check_drl_for_std_name(study, std_name, value)

    try:
        if modality == "CT":
            events = study.ctradiationdose_set.get().ctirradiationeventdata_set.all()  # type: ignore
        else:
            events = study.projectionxrayradiationdose_set.get().irradeventxraydata_set.all()  # type: ignore

        for event in events:
            std_name = std_names.get(
                **{"acquisition_protocol__in": [event.acquisition_protocol]}
            )
            value = _get_comparison_value_from_event(event)
            check_drl_for_std_name(study, std_name, value)
    except ObjectDoesNotExist as e:
        pass


def _get_patient_age(
    study: GeneralStudyModuleAttr, patient: Patients
) -> Union[Decimal, None]:
    if study.study_date and patient.patient_birth_date:
        return Decimal(
            relativedelta.relativedelta(
                study.study_date, patient.patient_birth_date
            ).years
        )


def _get_patient_bmi(
    study: GeneralStudyModuleAttr, patient: Patients
) -> Union[Decimal, None]:
    try:
        additional_patient_data = VolatilePatientData.objects.get(
            patient=patient, general_study_module_attr=study
        )
        return _calculate_patient_bmi(additional_patient_data)
    except:
        pass


def _calculate_patient_bmi(
    additional_patient_data: VolatilePatientData,
) -> Union[Decimal, None]:
    if additional_patient_data.patient_weight and additional_patient_data.patient_size:
        return additional_patient_data.patient_weight / (
            additional_patient_data.patient_size**2
        )


def _get_patient_attribute_for_criteria(
    criteria: str, study: GeneralStudyModuleAttr, patient: Patients
) -> Union[Decimal, None]:
    if criteria == "age":
        return _get_patient_age(study, patient)
    if criteria == "bmi":
        return _get_patient_bmi(study, patient)


def _get_drl(
    std_name: StandardNames, patient_attribute: Decimal
) -> Union[DiagnosticReferenceLevels, None]:
    try:
        return DiagnosticReferenceLevels.objects.filter(
            standard_name__in=[std_name]
        ).get(lower_bound__lte=patient_attribute, upper_bound__gte=patient_attribute)
    except ObjectDoesNotExist:
        pass


def check_drl_for_std_name(
    study: GeneralStudyModuleAttr,
    std_name: StandardNames,
    dose_value: Union[Decimal, None],
):
    try:
        patient = study.patients_set.get()  # type: ignore
    except ObjectDoesNotExist:
        return
    
    patient_attribute = _get_patient_attribute_for_criteria(
        std_name.diagnostic_reference_level_criteria, study, patient
    )

    if not dose_value or not patient_attribute:
        return

    drl = _get_drl(std_name, patient_attribute)

    if not drl:
        return

    if _dose_value_is_exceeding_drl_by_factor_of(
        dose_value, drl.diagnostic_reference_level, std_name.drl_alert_factor
    ):
        _issue_new_drl_alert(std_name, drl, study)


def _dose_value_is_exceeding_drl_by_factor_of(
    dose_value: Decimal, drl: Decimal, factor: Decimal
):
    return dose_value >= drl * factor


def _issue_new_drl_alert(
    std_name: StandardNames,
    drl: DiagnosticReferenceLevels,
    study: GeneralStudyModuleAttr,
):
    if DiagnosticReferenceLevelAlerts.objects.filter(
        diagnostic_reference_level=drl,
        general_study_module_attributes=study,
        standard_name=std_name,
    ).exists():
        return
    new_alert = DiagnosticReferenceLevelAlerts(
        date_of_issue=datetime.now(),
        diagnostic_reference_level=drl,
        general_study_module_attributes=study,
        standard_name=std_name,
    )
    new_alert.save()
