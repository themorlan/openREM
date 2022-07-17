# pylint: disable=line-too-long, too-many-lines
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2012,2013  The Royal Marsden NHS Foundation Trust
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
..  module:: mod_filters.
    :synopsis: Module for filtering studies on the summary filter pages.

..  moduleauthor:: Ed McDonagh

"""

from decimal import Decimal, InvalidOperation
import logging

from django import forms
from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Q
import django_filters

from remapp.models import (
    GeneralStudyModuleAttr,
    CtIrradiationEventData,
    StandardNames,
    StandardNameSettings,
)
from ..tools.hash_id import hash_id

logger = logging.getLogger(__name__)

TEST_CHOICES = (("", "Yes (default)"), (2, "No (caution)"))


def custom_name_filter(queryset, name, value):
    """Search for name as plain text and encrypted"""
    if not value:
        return queryset
    filtered = queryset.filter(
        (
            Q(patientmoduleattr__name_hashed=False)
            & Q(patientmoduleattr__patient_name__icontains=value)
        )
        | (
            Q(patientmoduleattr__name_hashed=True)
            & Q(patientmoduleattr__patient_name__exact=hash_id(value))
        )
    )
    return filtered


def custom_id_filter(queryset, name, value):
    """Search for ID as plain text and encrypted"""
    if not value:
        return queryset
    filtered = queryset.filter(
        (
            Q(patientmoduleattr__id_hashed=False)
            & Q(patientmoduleattr__patient_id__icontains=value)
        )
        | (
            Q(patientmoduleattr__id_hashed=True)
            & Q(patientmoduleattr__patient_id__exact=hash_id(value))
        )
    )
    return filtered


def _custom_acc_filter(queryset, name, value):
    """Search for accession number as plain text and encrypted"""
    if not value:
        return queryset
    filtered = queryset.filter(
        (Q(accession_hashed=False) & Q(accession_number__icontains=value))
        | (Q(accession_hashed=True) & Q(accession_number__exact=hash_id(value)))
    )
    return filtered


def _dap_filter(queryset, name, value):
    """Modify DAP to Gy.m2 before filtering"""
    if not value or not name:
        return queryset
    try:
        value_gy_m2 = Decimal(value) / Decimal(1000000)
    except InvalidOperation:
        return queryset
    if "study_dap_min" in name:
        filtered = queryset.filter(total_dap__gte=value_gy_m2)
    elif "study_dap_max" in name:
        filtered = queryset.filter(total_dap__lte=value_gy_m2)
    elif "event_dap_min" in name:
        filtered = queryset.filter(
            projectionxrayradiationdose__irradeventxraydata__dose_area_product__gte=value_gy_m2
        )
    elif "event_dap_max" in name:
        filtered = queryset.filter(
            projectionxrayradiationdose__irradeventxraydata__dose_area_product__lte=value_gy_m2
        )
    else:
        return queryset
    return filtered


class DateTimeOrderingFilter(django_filters.OrderingFilter):

    """Custom filter to order by date and time as they are two seperate fields"""

    def __init__(self, *args, **kwargs):
        super(DateTimeOrderingFilter, self).__init__(*args, **kwargs)
        self.extra["choices"] += (
            ("-time_date", "Exam date ⬇"),
            ("time_date", "Exam date ⬆"),
        )

    def filter(self, qs, value):
        """Sets order_by to date then time and returns queryset

        :param qs: queryset
        :param value: list containing ordering type as string
        :return: ordered queryset
        """
        # OrderingFilter is CSV-based, so `value` is a list
        if value and any(v in ["time_date", "-time_date"] for v in value):
            if "-time_date" in value:
                return qs.order_by("-study_date", "-study_time")
            if "time_date" in value:
                return qs.order_by("study_date", "study_time")

        return super(DateTimeOrderingFilter, self).filter(qs, value)


class RFSummaryListFilter(django_filters.FilterSet):

    """Filter for fluoroscopy studies to display in web interface."""

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
    projectionxrayradiationdose__irradeventxraydata__acquisition_protocol = (
        django_filters.CharFilter(lookup_expr="icontains", label="Acquisition protocol")
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
    performing_physician_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Physician"
    )
    accession_number = django_filters.CharFilter(
        method=_custom_acc_filter, label="Accession number"
    )
    study_dap_min = django_filters.NumberFilter(
        method=_dap_filter, label="Min study DAP (cGy·cm²)"
    )
    study_dap_max = django_filters.NumberFilter(
        method=_dap_filter, label="Max study DAP (cGy·cm²)"
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
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "generalequipmentmoduleattr__institution_name",
            "generalequipmentmoduleattr__manufacturer",
            "generalequipmentmoduleattr__manufacturer_model_name",
            "generalequipmentmoduleattr__station_name",
            "patientstudymoduleattr__patient_age_decimal__gte",
            "patientstudymoduleattr__patient_age_decimal__lte",
            "patientstudymoduleattr__patient_weight__gte",
            "patientstudymoduleattr__patient_weight__lte",
            "performing_physician_name",
            "accession_number",
            "study_dap_min",
            "study_dap_max",
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
            ("-total_dap", "Total DAP"),
            ("-total_rp_dose_a", "Total RP Dose (A)"),
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
            ("total_dap", "-total_dap"),
            ("total_rp_dose_a", "-total_rp_dose_a"),
        ),
    )


class RFFilterPlusStdNames(RFSummaryListFilter):
    """Adding standard name fields"""

    standard_names__standard_name = django_filters.CharFilter(
        lookup_expr="icontains",
        label="Standard study name",
    )
    projectionxrayradiationdose__irradeventxraydata__standard_protocols__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard acquisition name"
    )


class RFFilterPlusPid(RFSummaryListFilter):

    """Adding patient name and ID to filter if permissions allow"""

    def __init__(self, *args, **kwargs):
        super(RFFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters["patient_name"] = django_filters.CharFilter(
            method=custom_name_filter, label="Patient name"
        )
        self.filters["patient_id"] = django_filters.CharFilter(
            method=custom_id_filter, label="Patient ID"
        )


class RFFilterPlusPidPlusStdNames(RFFilterPlusPid):
    """Adding standard name fields"""

    standard_names__standard_name = django_filters.CharFilter(
        lookup_expr="icontains",
        label="Standard study name",
    )
    projectionxrayradiationdose__irradeventxraydata__standard_protocols__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard acquisition name"
    )


EVENT_NUMBER_CHOICES = (
    (None, "Any"),
    (0, "None"),
    ("some", ">0"),
    (1, "1"),
    (2, "2"),
    (3, "3"),
    (4, "4"),
    (5, "5"),
    (6, "6"),
    (7, "7"),
    (8, "8"),
    (9, "9"),
    (10, "10"),
    ("more", ">10"),
)


def _specify_event_numbers(queryset, name, value):

    """Method filter for specifying number of events in each study

    :param queryset: Study list
    :param name: field name (not used)
    :param value: number of events
    :return: filtered queryset
    """
    try:
        value = int(value)
    except ValueError:
        if value == "more":
            min_value = 10
        elif value == "some":
            min_value = 0
        else:
            return queryset
        if "num_events" in name:
            filtered = queryset.filter(number_of_events__gt=min_value)
        elif "num_spiral_events" in name:
            filtered = queryset.filter(number_of_spiral__gt=min_value)
        elif "num_axial_events" in name:
            filtered = queryset.filter(number_of_axial__gt=min_value)
        elif "num_spr_events" in name:
            filtered = queryset.filter(number_of_const_angle__gt=min_value)
        elif "num_stationary_events" in name:
            filtered = queryset.filter(number_of_stationary__gt=min_value)
        else:
            return queryset
        return filtered
    if "num_events" in name:
        filtered = queryset.filter(number_of_events__exact=value)
    elif "num_spiral_events" in name:
        filtered = queryset.filter(number_of_spiral__exact=value)
    elif "num_axial_events" in name:
        filtered = queryset.filter(number_of_axial__exact=value)
    elif "num_spr_events" in name:
        filtered = queryset.filter(number_of_const_angle__exact=value)
    elif "num_stationary_events" in name:
        filtered = queryset.filter(number_of_stationary__exact=value)
    else:
        return queryset
    return filtered


class CTSummaryListFilter(django_filters.FilterSet):

    """Filter for CT studies to display in web interface."""

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
    ctradiationdose__ctirradiationeventdata__acquisition_protocol = (
        django_filters.CharFilter(lookup_expr="icontains", label="Acquisition protocol")
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
    total_dlp__gte = django_filters.NumberFilter(
        lookup_expr="gte", field_name="total_dlp", label="Min study DLP"
    )
    total_dlp__lte = django_filters.NumberFilter(
        lookup_expr="lte", field_name="total_dlp", label="Max study DLP"
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
    num_events = django_filters.ChoiceFilter(
        method=_specify_event_numbers,
        label="Num. events total",
        choices=EVENT_NUMBER_CHOICES,
        widget=forms.Select,
    )
    num_spiral_events = django_filters.ChoiceFilter(
        method=_specify_event_numbers,
        label="Num. spiral events",
        choices=EVENT_NUMBER_CHOICES,
        widget=forms.Select,
    )
    num_axial_events = django_filters.ChoiceFilter(
        method=_specify_event_numbers,
        label="Num. axial events",
        choices=EVENT_NUMBER_CHOICES,
        widget=forms.Select,
    )
    num_spr_events = django_filters.ChoiceFilter(
        method=_specify_event_numbers,
        label="Num. localisers",
        choices=EVENT_NUMBER_CHOICES,
        widget=forms.Select,
    )
    num_stationary_events = django_filters.ChoiceFilter(
        method=_specify_event_numbers,
        label="Num. stationary events",
        choices=EVENT_NUMBER_CHOICES,
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
            "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
            "generalequipmentmoduleattr__institution_name",
            "generalequipmentmoduleattr__manufacturer",
            "generalequipmentmoduleattr__manufacturer_model_name",
            "generalequipmentmoduleattr__station_name",
            "patientstudymoduleattr__patient_age_decimal__gte",
            "patientstudymoduleattr__patient_age_decimal__lte",
            "patientstudymoduleattr__patient_weight__gte",
            "patientstudymoduleattr__patient_weight__lte",
            "accession_number",
            "total_dlp__gte",
            "total_dlp__lte",
            "generalequipmentmoduleattr__unique_equipment_name__display_name",
            "test_data",
            "num_events",
            "num_spiral_events",
            "num_axial_events",
            "num_spr_events",
            "num_stationary_events",
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
            ("-total_dlp", "Total DLP"),
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
            ("total_dlp", "-total_dlp"),
        ),
    )


class CTFilterPlusStdNames(CTSummaryListFilter):
    """Adding standard name fields"""

    standard_names__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard study name"
    )
    ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name = (
        django_filters.CharFilter(
            lookup_expr="icontains", label="Standard acquisition name"
        )
    )


class CTFilterPlusPid(CTSummaryListFilter):

    """Adding patient name and ID to filter if permissions allow"""

    def __init__(self, *args, **kwargs):
        super(CTFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters["patient_name"] = django_filters.CharFilter(
            method=custom_name_filter, label="Patient name"
        )
        self.filters["patient_id"] = django_filters.CharFilter(
            method=custom_id_filter, label="Patient ID"
        )


class CTFilterPlusPidPlusStdNames(CTFilterPlusPid):
    """Adding standard name fields"""

    standard_names__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard study name"
    )
    ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name = (
        django_filters.CharFilter(
            lookup_expr="icontains", label="Standard acquisition name"
        )
    )


def ct_acq_filter(filters, pid=False):

    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

    studies = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT")

    if pid:
        if enable_standard_names:
            return CTFilterPlusPidPlusStdNames(
                filters, studies.order_by("-study_date", "-study_time").distinct()
            )
        else:
            return CTFilterPlusPid(
                filters, studies.order_by("-study_date", "-study_time").distinct()
            )
    if enable_standard_names:
        return CTFilterPlusStdNames(
            filters, studies.order_by("-study_date", "-study_time").distinct()
        )
    else:
        return CTSummaryListFilter(
            filters, studies.order_by("-study_date", "-study_time").distinct()
        )


class MGSummaryListFilter(django_filters.FilterSet):

    """Filter for mammography studies to display in web interface."""

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
    projectionxrayradiationdose__irradeventxraydata__acquisition_protocol = (
        django_filters.CharFilter(lookup_expr="icontains", label="Acquisition protocol")
    )
    projectionxrayradiationdose__irradeventxraydata__image_view__code_meaning = (
        django_filters.CharFilter(lookup_expr="icontains", label="View code")
    )
    projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness__range = django_filters.NumericRangeFilter(
        lookup_expr="range",
        label="Breast thickness range (mm)",
        field_name="projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
    )
    projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure_control_mode = django_filters.CharFilter(
        lookup_expr="icontains", label="Exposure control mode"
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
    generalequipmentmoduleattr__unique_equipment_name__display_name = (
        django_filters.CharFilter(lookup_expr="icontains", label="Display name")
    )
    num_events = django_filters.ChoiceFilter(
        method=_specify_event_numbers,
        label="Num. events total",
        choices=EVENT_NUMBER_CHOICES,
        widget=forms.Select,
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
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "projectionxrayradiationdose__irradeventxraydata__image_view__code_meaning",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness__range",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure_control_mode",
            "generalequipmentmoduleattr__institution_name",
            "generalequipmentmoduleattr__manufacturer",
            "generalequipmentmoduleattr__manufacturer_model_name",
            "generalequipmentmoduleattr__station_name",
            "patientstudymoduleattr__patient_age_decimal__gte",
            "patientstudymoduleattr__patient_age_decimal__lte",
            "accession_number",
            "generalequipmentmoduleattr__unique_equipment_name__display_name",
            "num_events",
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
            ("procedure_code_meaning", "Procedure"),
            ("-total_agd_left", "AGD (left)"),
            ("-total_agd_right", "AGD (right)"),
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
            ("procedure_doce_meaning", "procedure_code_menaing"),
            ("total_agd_left", "-total_agd_left"),
            ("total_agd_right", "-total_agd_right"),
        ),
    )


class MGFilterPlusStdNames(MGSummaryListFilter):
    """Adding standard name fields"""

    standard_names__standard_name = django_filters.CharFilter(
        lookup_expr="icontains",
        label="Standard study name",
    )
    projectionxrayradiationdose__irradeventxraydata__standard_protocols__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard acquisition name"
    )


class MGFilterPlusPid(MGSummaryListFilter):
    """Adding patient name and ID to filter if permissions allow"""

    def __init__(self, *args, **kwargs):
        super(MGFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters["patient_name"] = django_filters.CharFilter(
            method=custom_name_filter, label="Patient name"
        )
        self.filters["patient_id"] = django_filters.CharFilter(
            method=custom_id_filter, label="Patient ID"
        )


class MGFilterPlusPidPlusStdNames(MGFilterPlusPid):
    """Adding standard name fields"""

    standard_names__standard_name = django_filters.CharFilter(
        lookup_expr="icontains",
        label="Standard study name",
    )
    projectionxrayradiationdose__irradeventxraydata__standard_protocols__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard acquisition name"
    )


class DXSummaryListFilter(django_filters.FilterSet):

    """Filter for DX studies to display in web interface."""

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
    projectionxrayradiationdose__irradeventxraydata__acquisition_protocol = (
        django_filters.CharFilter(lookup_expr="icontains", label="Acquisition protocol")
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
    study_dap_min = django_filters.NumberFilter(
        method=_dap_filter, label="Min study DAP (cGy·cm²)"
    )
    study_dap_max = django_filters.NumberFilter(
        method=_dap_filter, label="Max study DAP (cGy·cm²)"
    )
    event_dap_min = django_filters.NumberFilter(
        method=_dap_filter, label="Min acquisition DAP (cGy·cm²)"
    )
    event_dap_max = django_filters.NumberFilter(
        method=_dap_filter, label="Max acquisition DAP (cGy·cm²)"
    )
    generalequipmentmoduleattr__unique_equipment_name__display_name = (
        django_filters.CharFilter(lookup_expr="icontains", label="Display name")
    )
    num_events = django_filters.ChoiceFilter(
        method=_specify_event_numbers,
        label="Num. events total",
        choices=EVENT_NUMBER_CHOICES,
        widget=forms.Select,
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
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "generalequipmentmoduleattr__institution_name",
            "generalequipmentmoduleattr__manufacturer",
            "generalequipmentmoduleattr__manufacturer_model_name",
            "generalequipmentmoduleattr__station_name",
            "patientstudymoduleattr__patient_age_decimal__gte",
            "patientstudymoduleattr__patient_age_decimal__lte",
            "patientstudymoduleattr__patient_weight__gte",
            "patientstudymoduleattr__patient_weight__lte",
            "accession_number",
            "study_dap_min",
            "study_dap_max",
            "event_dap_min",
            "event_dap_max",
            "generalequipmentmoduleattr__unique_equipment_name__display_name",
            "num_events",
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
            ("-total_dap", "Total DAP"),
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
            ("total_dap", "-total_dap"),
        ),
    )


class DXFilterPlusStdNames(DXSummaryListFilter):
    """Adding standard name fields"""

    standard_names__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard study name"
    )
    projectionxrayradiationdose__irradeventxraydata__standard_protocols__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard acquisition name"
    )


class DXFilterPlusPid(DXSummaryListFilter):

    """Adding patient name and ID to filter if permissions allow"""

    def __init__(self, *args, **kwargs):
        super(DXFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters["patient_name"] = django_filters.CharFilter(
            method=custom_name_filter, label="Patient name"
        )
        self.filters["patient_id"] = django_filters.CharFilter(
            method=custom_id_filter, label="Patient ID"
        )


class DXFilterPlusPidPlusStdNames(DXFilterPlusPid):
    """Adding standard name fields"""

    standard_names__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard study name"
    )
    projectionxrayradiationdose__irradeventxraydata__standard_protocols__standard_name = django_filters.CharFilter(
        lookup_expr="icontains", label="Standard acquisition name"
    )


def dx_acq_filter(filters, pid=False):

    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

    studies = GeneralStudyModuleAttr.objects.filter(
        Q(modality_type__exact="DX")
        | Q(modality_type__exact="CR")
        | Q(modality_type__exact="PX")
    )

    if pid:
        if enable_standard_names:
            return DXFilterPlusPidPlusStdNames(
                filters,
                queryset=studies.order_by("-study_date", "-study_time").distinct(),
            )
        else:
            return DXFilterPlusPid(
                filters,
                queryset=studies.order_by("-study_date", "-study_time").distinct(),
            )
    if enable_standard_names:
        return DXFilterPlusStdNames(
            filters, queryset=studies.order_by("-study_date", "-study_time").distinct()
        )
    else:
        return DXSummaryListFilter(
            filters, queryset=studies.order_by("-study_date", "-study_time").distinct()
        )


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
    if pid:
        return NMFilterPlusPid(
            filters, studies.order_by("-study_date", "-study_time").distinct()
        )
    return NMSummaryListFilter(
        filters, studies.order_by("-study_date", "-study_time").distinct()
    )
