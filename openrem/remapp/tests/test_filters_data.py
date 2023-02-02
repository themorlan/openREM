import copy
import django_filters
from django_filters import FilterSet
from remapp.models import GeneralStudyModuleAttr


TEMPLATE_SIMPLE = {
    "root": {"prev": None, "next": None, "parent": None, "type": "group", "first": "1"},
    "1": {"prev": None, "next": None, "parent": "root", "type": "filter", "fields": {}},
}

TEMPLATE_OR = {
    "root": {"prev": None, "next": None, "parent": None, "type": "group", "first": "1"},
    "1": {"prev": None, "next": "2", "parent": "root", "type": "filter", "fields": {}},
    "2": {
        "prev": "1",
        "next": "3",
        "parent": "root",
        "type": "operator",
        "operator": "OR",
    },
    "3": {"prev": "2", "next": None, "parent": "root", "type": "filter", "fields": {}},
}


def get_simple_query(field, value):
    return get_simple_multiple_query({field: value})


def get_simple_multiple_query(fields: dict):
    return fill_fields(TEMPLATE_SIMPLE, ("1", fields))


def get_or_query(fields_a: dict, fields_b: dict):
    return fill_fields(TEMPLATE_OR, ("1", fields_a), ("3", fields_b))


def fill_fields(template, *filters):
    temp = copy.deepcopy(template)
    for (id, fields) in filters:
        for (field, value) in fields.items():
            if type(value) is list:
                temp[id]["fields"][field] = value
            else:
                temp[id]["fields"][field] = [value, None, False]
    return temp


class DummyModClass(FilterSet):
    study_date__gt = django_filters.DateFilter(
        lookup_expr="gte",
        field_name="study_date",
    )
    study_date__lt = django_filters.DateFilter(
        lookup_expr="lte",
        field_name="study_date",
    )
    study_description = django_filters.CharFilter(
        lookup_expr="icontains", label="Study description"
    )
    generalequipmentmoduleattr__manufacturer_model_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Model"
    )
    patientstudymoduleattr__patient_age_decimal__gte = django_filters.NumberFilter(
        lookup_expr="gte",
        field_name="patientstudymoduleattr__patient_age_decimal",
    )
    patientstudymoduleattr__patient_age_decimal__lte = django_filters.NumberFilter(
        lookup_expr="lte",
        field_name="patientstudymoduleattr__patient_age_decimal",
    )
    generalequipmentmoduleattr__manufacturer = django_filters.CharFilter(
        lookup_expr="icontains", label="Make"
    )

    class Meta:
        model = GeneralStudyModuleAttr
        fields = [
            "study_date__gt",
            "study_date__lt",
            "study_description",
            "generalequipmentmoduleattr__manufacturer_model_name",
            "patientstudymoduleattr__patient_age_decimal__gte",
            "patientstudymoduleattr__patient_age_decimal__lte",
            "generalequipmentmoduleattr__manufacturer",
        ]
