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
..  module:: mod_adv_filters
    :synopsis: Module to do advanced filtering

..  moduleauthor:: Kevin Schärer

"""

from django import forms
from django.db.models import Q
from remapp.models import (
    GeneralStudyModuleAttr,
)
import django_filters
from .mod_filters import custom_id_filter, custom_name_filter, _custom_acc_filter, TEST_CHOICES, DateTimeOrderingFilter
import json

ALLOWED_LOOKUP_TYPES = ["iexact", "icontains", "iregex"]

class InvalidQuery(Exception):
    "Raised when the given query is invalid"
    pass


def json_to_query(pattern, group="root") -> Q:
    """
        Transforms the JSON pattern into a Q object
    """
    print(pattern)
    q = Q()
    operator = Q.AND
    nextEntryId = pattern[group]["first"]
    
    while nextEntryId != None:
        nextEntry = pattern[nextEntryId]
        if not "type" in nextEntry:
            raise InvalidQuery
        q_type = nextEntry["type"]
        if q_type == "filter":
            try:
                q.add(get_filter(nextEntry["fields"]), operator)
            except KeyError:
                raise InvalidQuery
        if q_type == "operator":
            try:
                operator = nextEntry["operator"]
            except KeyError:
                raise InvalidQuery
        pass
        nextEntryId = nextEntry["next"]
    return q

def get_filter(fields: dict) -> Q:
    q = Q()
    for field, value in fields.items():
        if value[1] in ALLOWED_LOOKUP_TYPES:
            field = field+"__"+value[1]
        q.add(Q(**{field: value[0]}), Q.AND)
    return q

class NMSummaryListFilter(django_filters.FilterSet):
    """
    Filter for NM studies to display in web interface.
    """

    study_date__gt = django_filters.DateFilter(
        lookup_expr="gte",
        label="Date from",
        field_name="study_date",
        widget=forms.TextInput(attrs={"class": "datepicker"}),
    )
    study_date__lt = django_filters.DateFilter(
        lookup_expr="lte",
        label="Date until",
        field_name="study_date",
        widget=forms.TextInput(attrs={"class": "datepicker"}),
    )
    study_description = django_filters.CharFilter(
        lookup_expr="icontains", label="Study description"
    )
    procedure_code_meaning = django_filters.CharFilter(
        lookup_expr="icontains", label="Procedure"
    )
    requested_procedure_code_meaning = django_filters.CharFilter(
        lookup_expr="icontains", label="Requested procedure"
    )
    patientstudymoduleattr__patient_age_decimal__gte = django_filters.NumberFilter(
        lookup_expr="gte",
        label="Min age (yrs)",
        field_name="patientstudymoduleattr__patient_age_decimal",
    )
    patientstudymoduleattr__patient_age_decimal__lte = django_filters.NumberFilter(
        lookup_expr="lte",
        label="Max age (yrs)",
        field_name="patientstudymoduleattr__patient_age_decimal",
    )
    patientstudymoduleattr__patient_weight__gte = django_filters.NumberFilter(
        lookup_expr="gte",
        label="Min weight (kg)",
        field_name="patientstudymoduleattr__patient_weight",
    )
    patientstudymoduleattr__patient_weight__lte = django_filters.NumberFilter(
        lookup_expr="lte",
        label="Max weight (kg)",
        field_name="patientstudymoduleattr__patient_weight",
    )
    generalequipmentmoduleattr__institution_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Hospital"
    )
    generalequipmentmoduleattr__manufacturer = django_filters.CharFilter(
        lookup_expr="icontains", label="Make"
    )
    generalequipmentmoduleattr__manufacturer_model_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Model"
    )
    generalequipmentmoduleattr__station_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Station name"
    )
    accession_number = django_filters.CharFilter(
        method=_custom_acc_filter, label="Accession number"
    )
    radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity_gte = django_filters.NumberFilter(
        lookup_expr="gte",
        label="Min administered dose (MBq)",
        field_name="radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity",
    )
    radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity_lte = django_filters.NumberFilter(
        lookup_expr="lte",
        label="Max administered dose (MBq)",
        field_name="radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity",
    )
    generalequipmentmoduleattr__unique_equipment_name__display_name = (
        django_filters.CharFilter(lookup_expr="icontains", label="Display name")
    )
    test_data = django_filters.ChoiceFilter(
        lookup_expr="isnull",
        label="Include possible test data",
        field_name="patientmoduleattr__not_patient_indicator",
        choices=TEST_CHOICES,
        widget=forms.Select,
    )

    class Meta:
        """
        Lists fields and order-by information for django-filter filtering
        """

        model = GeneralStudyModuleAttr
        fields = [
            "study_date__gt",
            "study_date__lt",
            "study_description",
            "procedure_code_meaning",
            "requested_procedure_code_meaning",
            "generalequipmentmoduleattr__institution_name",
            "generalequipmentmoduleattr__manufacturer",
            "generalequipmentmoduleattr__manufacturer_model_name",
            "generalequipmentmoduleattr__station_name",
            "patientstudymoduleattr__patient_age_decimal__gte",
            "patientstudymoduleattr__patient_age_decimal__lte",
            "patientstudymoduleattr__patient_weight__gte",
            "patientstudymoduleattr__patient_weight__lte",
            "accession_number",
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity_gte",
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity_lte",
            "generalequipmentmoduleattr__unique_equipment_name__display_name",
            "test_data",
        ]

    o = DateTimeOrderingFilter(
        choices=(
            ("study_description", "Study Description"),
            ("generalequipmentmoduleattr__institution_name", "Hospital"),
            ("generalequipmentmoduleattr__manufacturer", "Make"),
            ("generalequipmentmoduleattr__manufacturer_model_name", "Model"),
            (
                "generalequipmentmoduleattr__unique_equipment_name__display_name",
                "Display name",
            ),
            ("study_description", "Study description"),
        ),
        fields=(
            ("study_description", "study_description"),
            (
                "generalequipmentmoduleattr__institution_name",
                "generalequipmentmoduleattr__institution_name",
            ),
            (
                "generalequipmentmoduleattr__manufacturer",
                "generalequipmentmoduleattr__manufacturer",
            ),
            (
                "generalequipmentmoduleattr__manufacturer_model_name",
                "generalequipmentmoduleattr__manufacturer_model_name",
            ),
            (
                "generalequipmentmoduleattr__unique_equipment_name__display_name",
                "generalequipmentmoduleattr__unique_equipment_name__display_name",
            ),
            ("study_description", "study_description"),
        ),
    )


class NMFilterPlusPid(NMSummaryListFilter):

    """Adding patient name and ID to filter if permissions allow"""

    def __init__(self, *args, **kwargs):
        super(NMFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters["patient_name"] = django_filters.CharFilter(
            method=custom_name_filter, label="Patient name"
        )
        self.filters["patient_id"] = django_filters.CharFilter(
            method=custom_id_filter, label="Patient ID"
        )


def nm_filter(filters, pid=False):
    studies = GeneralStudyModuleAttr.objects.filter(modality_type__exact="NM")
    a = filters.get("filterQuery")
    if a != None and a != "":
        import urllib.parse
        data = urllib.parse.unquote(a)
        data = json.loads(data)
        print(data)
        try:
            q = json_to_query(data)
            studies = studies.filter(q)
        except InvalidQuery:
            pass
    if pid:
        return NMFilterPlusPid(filters, queryset=studies.order_by("-study_date", "-study_time").distinct()
        )
    return NMSummaryListFilter(filters, queryset=studies.order_by("-study_date", "-study_time").distinct()
    )
