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

from django.db import IntegrityError
from django.db.models import Q
import numpy as np
from remapp.forms import DiagnosticReferenceLevelsFormSet, KFactorsFormSet
from remapp.models import StandardNames, GeneralStudyModuleAttr, CtIrradiationEventData, IrradEventXRayData, KFactors, DiagnosticReferenceLevels
import operator
from functools import reduce

def add_standard_name(request, form):
    data = form.cleaned_data

    std_name = data["standard_name"]

    if not std_name:
        pass
        # messages.warning(self.request, "Blank standard name - no update made")
        # return redirect(self.success_url)

    modality = data["modality"]

    field_names = {
        "study_description": [],
        "requested_procedure_code_meaning": [],
        "procedure_code_meaning": [],
        "acquisition_protocol": [],
    }

    for (field, new_ids) in field_names.items():
        # Add new entries to the StandardNames table
        _add_names(data[field], field, modality, std_name, new_ids)

    # Obtain a list of the required studies
    studies = _get_studies(modality)
    
    # Add the standard names to the studies
    _add_multiple_standard_studies(
        studies,
        field_names["study_description"],
        field_names["requested_procedure_code_meaning"],
        field_names["procedure_code_meaning"]
    )

    # Obtain a list of the required acquisitions
    acquisitions = None
    if modality == "CT":
        acquisitions = CtIrradiationEventData.objects
    else:
        # Filter the IrradEventXRayData.objects to just contain the required modality
        acquisitions = _filter_irrad_event_x_ray_data(modality)

    # Add the standard names to the acquisitions
    _add_multiple_standard_acquisitions(acquisitions, field_names["acquisition_protocol"], modality)

    std_names = StandardNames.objects.filter(modality=modality).filter(standard_name__exact=std_name)

    drl_formset = DiagnosticReferenceLevelsFormSet(request.POST, prefix="drl_formset")
    kfactor_formset = KFactorsFormSet(request.POST, prefix="kfactor_formset")
    _save_reference_values(std_names, drl_formset, kfactor_formset)


def update_all(modality):
    # All StandardNames entries for the required modality
    std_names = StandardNames.objects.filter(modality=modality)

    # Obtain a list of relevant studies
    studies = _get_studies(modality)

    # Remove reference to the standard names from entries from generalstudymoduleattr (study level)
    for std_name in std_names:
        std_name.generalstudymoduleattr_set.clear()

    # Remove references to the StandardName entries from the appropriate acquisition table
    if modality == "CT":
        for std_name in std_names:
            std_name.ctirradiationeventdata_set.clear()
    else:
        for std_name in std_names:
            std_name.irradeventxraydata_set.clear()

    # Add the standard names back at the study level
    _add_multiple_standard_studies(
        studies,
        std_names.filter(study_description__isnull=False),
        std_names.filter(requested_procedure_code_meaning__isnull=False),
        std_names.filter(procedure_code_meaning__isnull=False),
    )

    # Add the standard names back at the acquisition level
    acquisitions = None
    if modality == "CT":
        acquisitions = CtIrradiationEventData.objects
    else:
        # Filter the IrradEventXRayData.objects to just contain the required modality
        acquisitions = _filter_irrad_event_x_ray_data(modality)

    _add_multiple_standard_acquisitions(
        acquisitions,
        std_names.filter(acquisition_protocol__isnull=False),
        modality
    )


def update_standard_name(request, form, std_name_obj: StandardNames):

    data = form.cleaned_data

    # All StandardNames entries for the required modality
    std_names = StandardNames.objects.filter(modality=std_name_obj.modality)

    # Obtain a list of relevant studies
    studies = _get_studies(std_name_obj.modality)

    # Remove references to the StandardName entries from generalstudymoduleattr for any study_description,
    # requested_procedure_code_meaning or procedure_code_meaning values which have been removed from this
    # standard name. Then remove the corresponding StandardName entries.
    field_names = [
        "study_description",
        "requested_procedure_code_meaning",
        "procedure_code_meaning",
    ]
    for field in field_names:
        if field in data:
            # Obtain a list of field name values that have been remove from this standard name
            names_to_remove = np.setdiff1d(
                form.initial[field], data[field]
            )

            # Remove reference to these standard names from entries from generalstudymoduleattr
            std_name_obj.generalstudymoduleattr_set.remove(
                *studies.filter(**{field + "__in": names_to_remove})
                .filter(standard_names__standard_name=std_name_obj.standard_name)
                .values_list("pk", flat=True)
            )

            # Remove the corresponding StandardName entries
            std_names.filter(**{field + "__in": names_to_remove}).delete()

    # Remove references to the StandardName entries from the irradiatedevent table for any acquisition_protocol
    # values which have been removed from this standard name. Then remove the StandardName entries.
    acquisitions = None
    field = "acquisition_protocol"
    if field in data:

        # Obtain a list of field name values that have been remove from this standard name
        names_to_remove = np.setdiff1d(
            form.initial[field], data[field]
        )

        if std_name_obj.modality == "CT":
            # Remove reference to these standard names from entries from CtIrradiationEventData
            acquisitions = CtIrradiationEventData.objects
            std_name_obj.ctirradiationeventdata_set.remove(
                *acquisitions.filter(**{field + "__in": names_to_remove})
                .filter(
                    standard_protocols__standard_name=std_name_obj.standard_name
                )
                .values_list("pk", flat=True)
            )
        else:
            # Filter the IrradEventXRayData.objects to just contain the required modality
            acquisitions = _filter_irrad_event_x_ray_data(std_name_obj.modality)

            # Remove reference to these standard names from entries from IrradEventXRayData
            std_name_obj.irradeventxraydata_set.remove(
                *acquisitions.filter(**{field + "__in": names_to_remove})
                .filter(
                    standard_protocols__standard_name=std_name_obj.standard_name
                )
                .values_list("pk", flat=True)
            )

        # Remove the corresponding StandardName entries
        std_names.filter(**{field + "__in": names_to_remove}).delete()

    field_names = {
        "study_description": [],
        "requested_procedure_code_meaning": [],
        "procedure_code_meaning": [],
        "acquisition_protocol": [],
    }

    for (field, new_ids) in field_names.items():
        names_to_add = np.setdiff1d(
            data[field],
            form.initial[field],
        )
        _add_names(names_to_add, field, data["modality"], data["standard_name"], new_ids)

    # Add the new standard names to the studies
    _add_multiple_standard_studies(
        studies,
        field_names["study_description"],
        field_names["requested_procedure_code_meaning"],
        field_names["procedure_code_meaning"]
    )

    # Add the new standard names to the acquisitions
    _add_multiple_standard_acquisitions(acquisitions, field_names["acquisition_protocol"], std_name_obj.modality)

    std_names = std_names.filter(standard_name=form.initial["standard_name"])

    # Update the StandardNames standard name if it has been changed
    if "standard_name" in data:
        std_names.update(standard_name=data["standard_name"])

    drl_formset = DiagnosticReferenceLevelsFormSet(request.POST, prefix="drl_formset")
    kfactor_formset = KFactorsFormSet(request.POST, prefix="kfactor_formset")
    _save_reference_values(std_names, drl_formset, kfactor_formset)


def delete_standard_name(std_name_obj: StandardNames):
    # Obtain a list of relevant studies
    studies = _get_studies(std_name_obj.modality)

    # Remove this standard_name reference to these studies as the standard name may have changed
    std_name_obj.generalstudymoduleattr_set.remove(
        *studies.filter(
            standard_names__standard_name=std_name_obj.standard_name
        ).values_list("pk", flat=True)
    )

    # Remove the standard_names entries from acquisitions
    acquisitions = None
    if std_name_obj.modality == "CT":
        acquisitions = CtIrradiationEventData.objects
        std_name_obj.ctirradiationeventdata_set.remove(
            *acquisitions.filter(
                standard_protocols__standard_name=std_name_obj.standard_name
            ).values_list("pk", flat=True)
        )
    else:
        # Filter the IrradEventXRayData.objects to just contain the required modality
        acquisitions = _filter_irrad_event_x_ray_data(std_name_obj.modality)

        # Remove the standard names from the acquisitions
        std_name_obj.irradeventxraydata_set.remove(
            *acquisitions.filter(
                standard_protocols__standard_name=std_name_obj.standard_name
            ).values_list("pk", flat=True)
        )

    # Remove entries with standard_name = self.object.standard_name from the StandardNames table
    std_names = StandardNames.objects.filter(modality=std_name_obj.modality).filter(
        standard_name=std_name_obj.standard_name
    )

    DiagnosticReferenceLevels.objects.filter(standard_name__in=std_names).delete()
    KFactors.objects.filter(standard_name__in=std_names).delete()

    std_names.delete()


def _add_multiple_standard_studies(
    studies, std_name_study_ids, std_name_request_ids, std_name_procedure_ids
):
    for standard_name in StandardNames.objects.filter(pk__in=std_name_study_ids):
        standard_name.generalstudymoduleattr_set.add(
            *studies.filter(
                study_description=standard_name.study_description
            ).values_list("pk", flat=True)
        )

    for standard_name in StandardNames.objects.filter(pk__in=std_name_request_ids):
        standard_name.generalstudymoduleattr_set.add(
            *studies.filter(
                requested_procedure_code_meaning=standard_name.requested_procedure_code_meaning
            ).values_list("pk", flat=True)
        )

    for standard_name in StandardNames.objects.filter(
        pk__in=std_name_procedure_ids
    ):
        standard_name.generalstudymoduleattr_set.add(
            *studies.filter(
                procedure_code_meaning=standard_name.procedure_code_meaning
            ).values_list("pk", flat=True)
        )


def _add_multiple_standard_acquisitions(
    acquisitions, std_name_acquisition_ids, modality
):
    for standard_name in StandardNames.objects.filter(
        pk__in=std_name_acquisition_ids
    ):
        if modality == "CT":
            standard_name.ctirradiationeventdata_set.add(
                *acquisitions.filter(
                    acquisition_protocol=standard_name.acquisition_protocol
                ).values_list("pk", flat=True)
            )
        else:
            standard_name.irradeventxraydata_set.add(
                *acquisitions.filter(
                    acquisition_protocol=standard_name.acquisition_protocol
                ).values_list("pk", flat=True)
            )


def _get_studies(modality):
    studies = GeneralStudyModuleAttr.objects
    if modality == "CT":
        studies = studies.filter(modality_type="CT")
    elif modality == "MG":
        studies = studies.filter(modality_type="MG")
    elif modality == "RF":
        studies = studies.filter(modality_type="RF")
    else:
        studies = studies.filter(
            Q(modality_type__exact="DX")
            | Q(modality_type__exact="CR")
            | Q(modality_type__exact="PX")
        )
    return studies


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


def _add_names(names_to_add, field, modality, standard_name, new_ids):
    for item in names_to_add:
        new_entry = StandardNames(
            standard_name=standard_name,
            modality=modality,
            **{ field: item }
        )
        try:
            new_entry.save()
            new_ids.append(new_entry.pk)
        except IntegrityError as e:
            pass
            # messages.warning(self.request, mark_safe("Error adding name: {0}".format(e.args)))
            # return redirect(self.success_url)


def _save_reference_values(std_names, *formsets):
    for formset in formsets:
        if formset.is_valid():
            for item in formset.save(commit=False):
                item.save()
                item.standard_name.add(*std_names)

            for item in formset.deleted_objects:
                item.standard_name.remove(*std_names)
                item.delete()


