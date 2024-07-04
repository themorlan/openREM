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
..  module:: standard_names
    :synopsis: Module with tools to create, update and delete standard names
        and their mapping to all modalities

..  moduleauthor:: Kevin Schärer

"""

from django.contrib import messages
from django.db.models import Q
from django.core.exceptions import ObjectDoesNotExist
import numpy as np
from remapp.forms import DiagnosticReferenceLevelsFormSet, KFactorsFormSet
from remapp.models import (
    StandardNames,
    GeneralStudyModuleAttr,
    CtIrradiationEventData,
    IrradEventXRayData,
    KFactors,
    DiagnosticReferenceLevels,
)
import operator
from functools import reduce


STANDARD_STUDY_NAME_MAPPING_FIELDS = [
    "study_description",
    "requested_procedure_code_meaning",
    "procedure_code_meaning",
]

STANDARD_ACQUISITION_NAME_FIELDS = [
    "acquisition_protocol",
]


def assign_studies_to_standard_name(standard_name: StandardNames):
    """
    Add references to any matching study entries

    :param standard_name: StandardNames entry
    """

    modality = standard_name.modality
    studies = _get_studies_for_modality(modality)
    acquisitions = _get_acquisitions_for_modality(modality)

    for field_name in STANDARD_STUDY_NAME_MAPPING_FIELDS:
        _assign_standard_study_name_to_studies(
            studies,
            standard_name,
            field_name,
        )

    for field_name in STANDARD_ACQUISITION_NAME_FIELDS:
        _assign_standard_acquisition_name_to_acquisitions(
            acquisitions, standard_name, field_name
        )


def assign_standard_names_to_study(study: GeneralStudyModuleAttr):
    """Add references to any matching standard name entries

    :param g: GeneralStudyModuleAttr database table
    """

    modality = study.modality_type
    if modality in ["CR", "PX"]:
        modality = "DX"

    std_names = StandardNames.objects.filter(modality=modality)

    # Obtain a list of standard name IDs that match this GeneralStudyModuleAttr
    matching_std_name_ids = std_names.filter(
        (
            Q(study_description=study.study_description)
            & Q(study_description__isnull=False)
        )
        | (
            Q(requested_procedure_code_meaning=study.requested_procedure_code_meaning)
            & Q(requested_procedure_code_meaning__isnull=False)
        )
        | (
            Q(procedure_code_meaning=study.procedure_code_meaning)
            & Q(procedure_code_meaning__isnull=False)
        )
    ).values_list("pk", flat=True)

    # Obtain a list of standard name IDs that are already associated with this GeneralStudyModuleAttr
    std_name_ids_already_in_study = study.standard_names.values_list("pk", flat=True)

    # Names that are in the new list, but not in the existing list
    std_name_ids_to_add = np.setdiff1d(
        matching_std_name_ids, std_name_ids_already_in_study
    )

    if std_name_ids_to_add.size:
        study.standard_names.add(*std_name_ids_to_add)

    # Add standard name references to the study irradiation events where the acquisition_protocol values match.
    # Some events will already exist if the new data is adding to an existing study
    try:
        if modality == "CT":
            for (
                event
            ) in study.ctradiationdose_set.get().ctirradiationeventdata_set.all():
                # Only add matching standard name if the event doesn't already have one
                if not event.standard_protocols.values_list("pk", flat=True):
                    pk_value = list(
                        std_names.filter(
                            Q(acquisition_protocol=event.acquisition_protocol)
                            & Q(acquisition_protocol__isnull=False)
                        ).values_list("pk", flat=True)
                    )
                    event.standard_protocols.add(*pk_value)
        else:
            for (
                event
            ) in (
                study.projectionxrayradiationdose_set.get().irradeventxraydata_set.all()
            ):
                # Only add matching standard name if the event doesn't already have one
                if not event.standard_protocols.values_list("pk", flat=True):
                    pk_value = list(
                        std_names.filter(
                            Q(acquisition_protocol=event.acquisition_protocol)
                            & Q(acquisition_protocol__isnull=False)
                        ).values_list("pk", flat=True)
                    )
                    event.standard_protocols.add(*pk_value)

    except ObjectDoesNotExist:
        pass


def add_standard_name(request, form):
    data = form.cleaned_data
    modality = data["modality"]

    if not data["standard_name"]:
        messages.warning(request, "Blank standard name - no update made")
        return

    for field_name in (
        STANDARD_STUDY_NAME_MAPPING_FIELDS + STANDARD_ACQUISITION_NAME_FIELDS
    ):
        for field_value in data[field_name]:
            _add_standard_name(field_name, field_value, **data)

    std_names = StandardNames.objects.filter(modality=modality).filter(
        standard_name__exact=data["standard_name"]
    )

    for standard_name in std_names:
        assign_studies_to_standard_name(standard_name)

    drl_formset = DiagnosticReferenceLevelsFormSet(request.POST, prefix="drl_formset")
    kfactor_formset = KFactorsFormSet(request.POST, prefix="kfactor_formset")
    _save_all_reference_values_for_standard_names(
        modality, std_names, drl_formset, kfactor_formset
    )


def update_all_standard_names_for_modality(modality):
    standard_names = StandardNames.objects.filter(modality=modality)

    for std_name in standard_names:
        std_name.generalstudymoduleattr_set.clear()  # type: ignore

    if modality == "CT":
        for std_name in standard_names:
            std_name.ctirradiationeventdata_set.clear()  # type: ignore
    else:
        for std_name in standard_names:
            std_name.irradeventxraydata_set.clear()  # type: ignore

    for standard_name in standard_names:
        assign_studies_to_standard_name(standard_name)


def update_standard_name(request, form, standard_name: StandardNames):
    data = form.cleaned_data

    studies = _get_studies_for_modality(standard_name.modality)
    acquisitions = _get_acquisitions_for_modality(standard_name.modality)
    standard_names = StandardNames.objects.filter(
        modality=standard_name.modality
    ).filter(standard_name=standard_name.standard_name)

    for field_name in STANDARD_STUDY_NAME_MAPPING_FIELDS:
        if field_name not in data:
            continue
        (names_to_remove, names_to_add) = get_field_values_to_add_and_remove(
            form.initial[field_name], data[field_name]
        )
        filtered_studies = studies.filter(**{field_name + "__in": names_to_remove})
        _remove_studies_from_standard_study_name(filtered_studies, standard_name)
        standard_names.filter(**{field_name + "__in": names_to_remove}).delete()

        for name_to_add in names_to_add:
            _add_standard_name(field_name, name_to_add, **data)

    for field_name in STANDARD_ACQUISITION_NAME_FIELDS:
        if field_name not in data:
            continue
        (names_to_remove, names_to_add) = get_field_values_to_add_and_remove(
            form.initial[field_name], data[field_name]
        )
        filtered_acquisitions = acquisitions.filter(
            **{field_name + "__in": names_to_remove}
        )
        _remove_acquisitions_from_standard_acquisition_name(
            filtered_acquisitions, standard_name
        )

        standard_names.filter(**{field_name + "__in": names_to_remove}).delete()

        for name_to_add in names_to_add:
            _add_standard_name(field_name, name_to_add, **data)

    standard_names = StandardNames.objects.filter(
        modality=standard_name.modality
    ).filter(standard_name=data["standard_name"])

    for standard_name in standard_names:
        assign_studies_to_standard_name(standard_name)

    if "standard_name" in data:
        standard_names.update(standard_name=data["standard_name"])

    if "drl_alert_factor" in data:
        standard_names.update(drl_alert_factor=data["drl_alert_factor"])

    if "national_drl" in data:
        standard_names.update(national_drl=data["national_drl"])

    if "diagnostic_reference_level_criteria" in data:
        standard_names.update(
            diagnostic_reference_level_criteria=data[
                "diagnostic_reference_level_criteria"
            ]
        )

    if "k_factor_criteria" in data:
        standard_names.update(k_factor_criteria=data["k_factor_criteria"])

    drl_formset = DiagnosticReferenceLevelsFormSet(request.POST, prefix="drl_formset")
    kfactor_formset = KFactorsFormSet(request.POST, prefix="kfactor_formset")
    _save_all_reference_values_for_standard_names(
        standard_name.modality, standard_names, drl_formset, kfactor_formset
    )


def get_field_values_to_add_and_remove(initial_values, new_values):
    names_to_remove = np.setdiff1d(initial_values, new_values)
    names_to_add = np.setdiff1d(
        new_values,
        initial_values,
    )
    return (names_to_remove, names_to_add)


def delete_standard_name(standard_name: StandardNames):
    studies = _get_studies_for_modality(standard_name.modality)
    acquisitions = _get_acquisitions_for_modality(standard_name.modality)

    _remove_studies_from_standard_study_name(studies, standard_name)
    _remove_acquisitions_from_standard_acquisition_name(acquisitions, standard_name)

    standard_names = StandardNames.objects.filter(
        modality=standard_name.modality
    ).filter(standard_name=standard_name.standard_name)

    DiagnosticReferenceLevels.objects.filter(standard_name__in=standard_names).delete()
    KFactors.objects.filter(standard_name__in=standard_names).delete()

    standard_names.delete()


def _get_studies_for_modality(modality):
    studies = GeneralStudyModuleAttr.objects
    if modality == "CT":
        studies = studies.filter(modality_type="CT")
    elif modality == "MG":
        studies = studies.filter(modality_type="MG")
    elif modality == "RF":
        studies = studies.filter(modality_type="RF")
    elif modality == "NM":
        studies = studies.filter(modality_type="NM")
    else:
        studies = studies.filter(
            Q(modality_type__exact="DX")
            | Q(modality_type__exact="CR")
            | Q(modality_type__exact="PX")
        )
    return studies


def _get_acquisitions_for_modality(modality):
    if modality == "CT":
        return CtIrradiationEventData.objects
    return _filter_irrad_event_x_ray_data(modality)


def _filter_irrad_event_x_ray_data(modality):
    q = ["DX", "CR", "PX"]
    if modality == "MG":
        q = ["MG"]
    elif modality == "RF":
        q = ["RF"]

    q_criteria = reduce(
        operator.or_,
        (
            Q(
                projection_xray_radiation_dose__general_study_module_attributes__modality_type__icontains=item
            )
            for item in q
        ),
    )
    return IrradEventXRayData.objects.filter(q_criteria)


def _add_standard_name(field_name, field_value, **data):
    return StandardNames.objects.create(
        standard_name=data["standard_name"],
        modality=data["modality"],
        diagnostic_reference_level_criteria=data["diagnostic_reference_level_criteria"],
        national_drl=data["national_drl"],
        drl_alert_factor=data["drl_alert_factor"],
        k_factor_criteria=data["k_factor_criteria"],
        **{field_name: field_value}
    )


def _assign_standard_study_name_to_studies(
    studies, standard_name: StandardNames, field_name
):
    for study in studies.filter(
        **{
            field_name + "__isnull": False,
            field_name: standard_name.__dict__[field_name],
        }
    ):
        print(study.study_description)
        study.standard_names.add(standard_name)


def _assign_standard_acquisition_name_to_acquisitions(
    acquisitions, standard_name: StandardNames, field_name
):
    for acquisition in acquisitions.filter(
        **{field_name: standard_name.__dict__[field_name]}
    ):
        acquisition.standard_protocols.add(standard_name)


def _remove_studies_from_standard_study_name(studies, standard_name: StandardNames):
    standard_name.generalstudymoduleattr_set.remove(  # type: ignore
        *studies.filter(
            standard_names__standard_name=standard_name.standard_name
        ).values_list("pk", flat=True)
    )


def _remove_acquisitions_from_standard_acquisition_name(
    acquisitions, standard_name: StandardNames
):
    if standard_name.modality == "CT":
        standard_name.ctirradiationeventdata_set.remove(  # type: ignore
            *acquisitions.filter(
                standard_protocols__standard_name=standard_name.standard_name
            ).values_list("pk", flat=True)
        )
    else:
        standard_name.irradeventxraydata_set.remove(  # type: ignore
            *acquisitions.filter(
                standard_protocols__standard_name=standard_name.standard_name
            ).values_list("pk", flat=True)
        )


def _save_all_reference_values_for_standard_names(modality, standard_names, *formsets):
    for formset in formsets:
        if not formset.is_valid():
            continue
        _save_reference_values_for_standard_names(formset, standard_names, modality)


def _save_reference_values_for_standard_names(formset, standard_names, modality):
    for item in formset.save(commit=False):
        if modality in ["RF", "DX"]:
            item.diagnostic_reference_level /= 1000000
        item.save()
        item.standard_name.add(*standard_names)

    for item in formset.deleted_objects:
        item.standard_name.remove(*standard_names)
        item.delete()
