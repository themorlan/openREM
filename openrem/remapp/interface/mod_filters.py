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
from __future__ import division

from builtins import object  # pylint: disable=redefined-builtin
from past.utils import old_div
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'

import django_filters
from django import forms
from remapp.models import GeneralStudyModuleAttr
from django.utils.safestring import mark_safe

TEST_CHOICES = ((u'', u'Yes (default)'), (2, u'No (caution)'),)


def custom_name_filter(queryset, value):
    if not value:
        return queryset

    from django.db.models import Q
    from remapp.tools.hash_id import hash_id
    filtered = queryset.filter(
        (
                Q(patientmoduleattr__name_hashed=False) & Q(patientmoduleattr__patient_name__icontains=value)
        ) | (
                Q(patientmoduleattr__name_hashed=True) & Q(patientmoduleattr__patient_name__exact=hash_id(value))
        )
    )
    return filtered


def custom_id_filter(queryset, value):
    if not value:
        return queryset

    from django.db.models import Q
    from remapp.tools.hash_id import hash_id
    filtered = queryset.filter(
        (
                Q(patientmoduleattr__id_hashed=False) & Q(patientmoduleattr__patient_id__icontains=value)
        ) | (
                Q(patientmoduleattr__id_hashed=True) & Q(patientmoduleattr__patient_id__exact=hash_id(value))
        )
    )
    return filtered


def _custom_acc_filter(queryset, name, value):
    if not value:
        return queryset

    from django.db.models import Q
    from remapp.tools.hash_id import hash_id
    filtered = queryset.filter(
        (
                Q(accession_hashed=False) & Q(accession_number__icontains=value)
        ) | (
                Q(accession_hashed=True) & Q(accession_number__exact=hash_id(value))
        )
    )
    return filtered


def _total_dap_filter(queryset, name, value):
    if not value or not name:
        return queryset

    from decimal import Decimal, InvalidOperation
    try:
        value_gy_m2 = old_div(Decimal(value), Decimal(1000000))
    except InvalidOperation:
        return queryset
    if 'study_dap_min' in name:
        return queryset.filter(total_dap__gte=value_gy_m2)
    elif 'study_dap_max' in name:
        return queryset.filter(total_dap__lte=value_gy_m2)
    else:
        return queryset


def _event_dap_filter(queryset, name, value):
    if not value or not name:
        return queryset

    from decimal import Decimal, InvalidOperation
    try:
        value_gy_m2 = old_div(Decimal(value), Decimal(1000000))
    except InvalidOperation:
        return queryset
    if 'event_dap_min' in name:
        return queryset.filter(projectionxrayradiationdose__irradeventxraydata__dose_area_product__gte=value_gy_m2)
    elif 'event_dap_max' in name:
        return queryset.filter(projectionxrayradiationdose__irradeventxraydata__dose_area_product__lte=value_gy_m2)
    else:
        return queryset


class DateTimeOrderingFilter(django_filters.OrderingFilter):

    def __init__(self, *args, **kwargs):
        super(DateTimeOrderingFilter, self).__init__(*args, **kwargs)
        self.extra['choices'] += (
            ('-time_date', 'Exam date ⬇'),
            ('time_date', 'Exam date ⬆'),
        )

    def filter(self, qs, value):
        # OrderingFilter is CSV-based, so `value` is a list
        if value and any(v in ['time_date', '-time_date'] for v in value):
            if '-time_date' in value:
                return qs.order_by('-study_date', '-study_time')
            if 'time_date' in value:
                return qs.order_by('study_date', 'study_time')

        return super(DateTimeOrderingFilter, self).filter(qs, value)


class RFSummaryListFilter(django_filters.FilterSet):
    """Filter for fluoroscopy studies to display in web interface.

    """
    study_date__gt = django_filters.DateFilter(lookup_expr='gte', label='Date from', field_name='study_date',
                                               widget=forms.TextInput(attrs={'class': 'datepicker'}))
    study_date__lt = django_filters.DateFilter(lookup_expr='lte', label='Date until', field_name='study_date',
                                               widget=forms.TextInput(attrs={'class': 'datepicker'}))
    study_description = django_filters.CharFilter(lookup_expr='icontains', label='Study description')
    procedure_code_meaning = django_filters.CharFilter(lookup_expr='icontains', label='Procedure')
    requested_procedure_code_meaning = django_filters.CharFilter(lookup_expr='icontains', label='Requested procedure')
    projectionxrayradiationdose__irradeventxraydata__acquisition_protocol = django_filters.CharFilter(
        lookup_expr='icontains', label='Acquisition protocol')
    patientstudymoduleattr__patient_age_decimal__gte = django_filters.NumberFilter(lookup_expr='gte',
                                                                                   label=u'Min age (yrs)',
                                                                                   field_name='patientstudymoduleattr'
                                                                                        '__patient_age_decimal')
    patientstudymoduleattr__patient_age_decimal__lte = django_filters.NumberFilter(lookup_expr='lte',
                                                                                   label=u'Max age (yrs)',
                                                                                   field_name='patientstudymoduleattr'
                                                                                        '__patient_age_decimal')
    generalequipmentmoduleattr__institution_name = django_filters.CharFilter(lookup_expr='icontains', label=u'Hospital')
    generalequipmentmoduleattr__manufacturer = django_filters.CharFilter(lookup_expr='icontains', label=u'Make')
    generalequipmentmoduleattr__manufacturer_model_name = django_filters.CharFilter(
        lookup_expr='icontains', label='Model')
    generalequipmentmoduleattr__station_name = django_filters.CharFilter(lookup_expr='icontains', label=u'Station name')
    performing_physician_name = django_filters.CharFilter(lookup_expr='icontains', label=u'Physician')
    accession_number = django_filters.CharFilter(method=_custom_acc_filter, label='Accession number')
    study_dap_min = django_filters.NumberFilter(method=_total_dap_filter, label='Min study DAP (cGy·cm²)')
    study_dap_max = django_filters.NumberFilter(method=_total_dap_filter, label='Max study DAP (cGy·cm²)')
    generalequipmentmoduleattr__unique_equipment_name__display_name = django_filters.CharFilter(
        lookup_expr='icontains', label='Display name')
    test_data = django_filters.ChoiceFilter(lookup_expr='isnull', label=u"Include possible test data",
                                            field_name='patientmoduleattr__not_patient_indicator', choices=TEST_CHOICES,
                                            widget=forms.Select)

    class Meta:
        """
        Lists fields and order-by information for django-filter filtering
        """
        model = GeneralStudyModuleAttr
        fields = [
            'study_date__gt', 'study_date__lt',
            'study_description', 'procedure_code_meaning', 'requested_procedure_code_meaning',
            'projectionxrayradiationdose__irradeventxraydata__acquisition_protocol',
            'generalequipmentmoduleattr__institution_name', 'generalequipmentmoduleattr__manufacturer',
            'generalequipmentmoduleattr__manufacturer_model_name', 'generalequipmentmoduleattr__station_name',
            'patientstudymoduleattr__patient_age_decimal__gte', 'patientstudymoduleattr__patient_age_decimal__lte',
            'performing_physician_name', 'accession_number', 'study_dap_min', 'study_dap_max',
            'generalequipmentmoduleattr__unique_equipment_name__display_name', 'test_data',
        ]

    o = DateTimeOrderingFilter(
        choices=(
            ('study_description', 'Study Description'),
            ('generalequipmentmoduleattr__institution_name', 'Hospital'),
            ('generalequipmentmoduleattr__manufacturer', 'Make'),
            ('generalequipmentmoduleattr__manufacturer_model_name', 'Model'),
            ('generalequipmentmoduleattr__unique_equipment_name__display_name', 'Display name'),
            ('study_description', 'Study description'),
            ('-total_dap', 'Total DAP'),
            ('-total_rp_dose_a', 'Total RP Dose (A)'),
        ),
        fields=(
            ('study_description', 'study_description'),
            ('generalequipmentmoduleattr__institution_name', 'generalequipmentmoduleattr__institution_name'),
            ('generalequipmentmoduleattr__manufacturer', 'generalequipmentmoduleattr__manufacturer'),
            ('generalequipmentmoduleattr__manufacturer_model_name',
             'generalequipmentmoduleattr__manufacturer_model_name'),
            ('generalequipmentmoduleattr__unique_equipment_name__display_name',
             'generalequipmentmoduleattr__unique_equipment_name__display_name'),
            ('study_description', 'study_description'),
            ('total_dap', '-total_dap'),
            ('total_rp_dose_a', '-total_rp_dose_a'),
        ),
    )


class RFFilterPlusPid(RFSummaryListFilter):
    def __init__(self, *args, **kwargs):
        super(RFFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters['patient_name'] = django_filters.CharFilter(method='custom_name_filter', label=u'Patient name')
        self.filters['patient_id'] = django_filters.CharFilter(method='custom_id_filter', label=u'Patient ID')


# Values from DICOM CID 10013 CT Acquisition Type
CT_ACQ_TYPE_CHOICES = (
    ('Spiral Acquisition', 'Spiral'),
    ('Sequenced Acquisition', 'Axial'),
    ('Constant Angle Acquisition', 'Localiser'),
    ('Stationary Acquisition', 'Stationary acquisition'),
    ('Free Acquisition', 'Free acquisition'),
)


EVENT_NUMBER_CHOICES = (
    (None, 'Any'),
    (0, 'None'),
    (1, '1'),
    (2, '2'),
    (3, '3'),
    (4, '4'),
    (5, '5'),
    (6, '6'),
    (7, '7'),
    (8, '8'),
    (9, '9'),
    (10, '10'),
    ('more', '>10'),
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
        if value == 'more':
            filtered = queryset.filter(number_of_events__gt=10)
            return filtered
        return queryset
    filtered = queryset.filter(number_of_events__exact=value)
    return filtered


def _specify_event_numbers_spiral(queryset, name, value):
    """Method filter for specifying number of spiral (helical) events in each study

    :param queryset: Study list
    :param name: field name (not used)
    :param value: number of events
    :return: filtered queryset
    """
    try:
        value = int(value)
    except ValueError:
        if value == 'more':
            filtered = queryset.filter(number_of_spiral__gt=10)
            return filtered
        return queryset
    filtered = queryset.filter(number_of_spiral__exact=value)
    return filtered


def _specify_event_numbers_axial(queryset, name, value):
    """Method filter for specifying number of axial events in each study

    :param queryset: Study list
    :param name: field name (not used)
    :param value: number of events
    :return: filtered queryset
    """
    try:
        value = int(value)
    except ValueError:
        if value == 'more':
            filtered = queryset.filter(number_of_axial__gt=10)
            return filtered
        return queryset
    filtered = queryset.filter(number_of_axial__exact=value)
    return filtered


def _specify_event_numbers_spr(queryset, name, value):
    """Method filter for specifying number of scan projection radiograph events in each study

    :param queryset: Study list
    :param name: field name (not used)
    :param value: number of events
    :return: filtered queryset
    """
    try:
        value = int(value)
    except ValueError:
        if value == 'more':
            filtered = queryset.filter(number_of_const_angle__gt=10)
            return filtered
        return queryset
    filtered = queryset.filter(number_of_const_angle__exact=value)
    return filtered


def _specify_event_numbers_stationary(queryset, name, value):
    """Method filter for specifying number of scan projection radiograph events in each study

    :param queryset: Study list
    :param name: field name (not used)
    :param value: number of events
    :return: filtered queryset
    """
    try:
        value = int(value)
    except ValueError:
        if value == 'more':
            filtered = queryset.filter(number_of_stationary__gt=10)
            return filtered
        return queryset
    filtered = queryset.filter(number_of_stationary__exact=value)
    return filtered


class CTSummaryListFilter(django_filters.FilterSet):
    """Filter for CT studies to display in web interface.

    """
    study_date__gt = django_filters.DateFilter(lookup_expr='gte', label='Date from', field_name='study_date',
                                               widget=forms.TextInput(attrs={'class': 'datepicker'}))
    study_date__lt = django_filters.DateFilter(lookup_expr='lte', label='Date until', field_name='study_date',
                                               widget=forms.TextInput(attrs={'class': 'datepicker'}))
    study_description = django_filters.CharFilter(lookup_expr='icontains', label='Study description')
    procedure_code_meaning = django_filters.CharFilter(lookup_expr='icontains', label='Procedure')
    requested_procedure_code_meaning = django_filters.CharFilter(lookup_expr='icontains', label='Requested procedure')
    ctradiationdose__ctirradiationeventdata__acquisition_protocol = django_filters.CharFilter(
        lookup_expr='icontains', label='Acquisition protocol')
    patientstudymoduleattr__patient_age_decimal__gte = django_filters.NumberFilter(lookup_expr='gte',
                                                                                   label=u'Min age (yrs)',
                                                                                   field_name='patientstudymoduleattr'
                                                                                        '__patient_age_decimal')
    patientstudymoduleattr__patient_age_decimal__lte = django_filters.NumberFilter(lookup_expr='lte',
                                                                                   label=u'Max age (yrs)',
                                                                                   field_name='patientstudymoduleattr'
                                                                                        '__patient_age_decimal')
    generalequipmentmoduleattr__institution_name = django_filters.CharFilter(lookup_expr='icontains', label=u'Hospital')
    generalequipmentmoduleattr__manufacturer = django_filters.CharFilter(lookup_expr='icontains', label=u'Make')
    generalequipmentmoduleattr__manufacturer_model_name = django_filters.CharFilter(
        lookup_expr='icontains', label='Model')
    generalequipmentmoduleattr__station_name = django_filters.CharFilter(lookup_expr='icontains', label=u'Station name')
    accession_number = django_filters.CharFilter(method=_custom_acc_filter, label='Accession number')
    total_dlp__gte = django_filters.NumberFilter(lookup_expr='gte', field_name='total_dlp', label="Min study DLP")
    total_dlp__lte = django_filters.NumberFilter(lookup_expr='lte', field_name='total_dlp', label="Max study DLP")
    generalequipmentmoduleattr__unique_equipment_name__display_name = django_filters.CharFilter(
        lookup_expr='icontains', label='Display name')
    test_data = django_filters.ChoiceFilter(lookup_expr='isnull', label=u"Include possible test data",
                                            field_name='patientmoduleattr__not_patient_indicator', choices=TEST_CHOICES,
                                            widget=forms.Select)
    ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning = django_filters.MultipleChoiceFilter(
        lookup_expr='iexact', label=u'Acquisition type restriction',
        field_name='ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning', choices=CT_ACQ_TYPE_CHOICES,
        widget=forms.CheckboxSelectMultiple)
    num_events = django_filters.ChoiceFilter(method=_specify_event_numbers, label=u'Num. events total',
                                             choices=EVENT_NUMBER_CHOICES, widget=forms.Select)
    num_spiral_events = django_filters.ChoiceFilter(method=_specify_event_numbers_spiral, label=u'Num. spiral events',
                                                    choices=EVENT_NUMBER_CHOICES, widget=forms.Select)
    num_axial_events = django_filters.ChoiceFilter(method=_specify_event_numbers_axial, label=u'Num. axial events',
                                                   choices=EVENT_NUMBER_CHOICES, widget=forms.Select)
    num_spr_events = django_filters.ChoiceFilter(method=_specify_event_numbers_spr, label=u'Num. localisers',
                                                 choices=EVENT_NUMBER_CHOICES, widget=forms.Select)
    num_stationary_events = django_filters.ChoiceFilter(method=_specify_event_numbers_stationary,
                                                        label=u'Num. stationary events', choices=EVENT_NUMBER_CHOICES,
                                                        widget=forms.Select)

    class Meta:
        """
        Lists fields and order-by information for django-filter filtering
        """
        model = GeneralStudyModuleAttr
        fields = [
            'study_date__gt', 'study_date__lt',
            'study_description', 'procedure_code_meaning', 'requested_procedure_code_meaning',
            'ctradiationdose__ctirradiationeventdata__acquisition_protocol',
            'generalequipmentmoduleattr__institution_name', 'generalequipmentmoduleattr__manufacturer',
            'generalequipmentmoduleattr__manufacturer_model_name', 'generalequipmentmoduleattr__station_name',
            'patientstudymoduleattr__patient_age_decimal__gte', 'patientstudymoduleattr__patient_age_decimal__lte',
            'accession_number', 'total_dlp__gte', 'total_dlp__lte',
            'generalequipmentmoduleattr__unique_equipment_name__display_name', 'test_data',
            'ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning', 'num_events',
            'num_spiral_events', 'num_axial_events', 'num_spr_events', 'num_stationary_events',
            ]

    o = DateTimeOrderingFilter(
        choices=(
            ('study_description', 'Study Description'),
            ('generalequipmentmoduleattr__institution_name', 'Hospital'),
            ('generalequipmentmoduleattr__manufacturer', 'Make'),
            ('generalequipmentmoduleattr__manufacturer_model_name', 'Model'),
            ('generalequipmentmoduleattr__unique_equipment_name__display_name', 'Display name'),
            ('study_description', 'Study description'),
            ('-total_dlp', 'Total DLP'),
        ),
        fields=(
            ('study_description', 'study_description'),
            ('generalequipmentmoduleattr__institution_name', 'generalequipmentmoduleattr__institution_name'),
            ('generalequipmentmoduleattr__manufacturer', 'generalequipmentmoduleattr__manufacturer'),
            ('generalequipmentmoduleattr__manufacturer_model_name',
             'generalequipmentmoduleattr__manufacturer_model_name'),
            ('generalequipmentmoduleattr__unique_equipment_name__display_name',
             'generalequipmentmoduleattr__unique_equipment_name__display_name'),
            ('study_description', 'study_description'),
            ('total_dlp', '-total_dlp'),
        ),
    )


class CTFilterPlusPid(CTSummaryListFilter):
    def __init__(self, *args, **kwargs):
        super(CTFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters['patient_name'] = django_filters.CharFilter(method='custom_name_filter', label=u'Patient name')
        self.filters['patient_id'] = django_filters.CharFilter(method='custom_id_filter', label=u'Patient ID')


def ct_acq_filter(filters, pid=False):
    from decimal import Decimal, InvalidOperation
    from ..models import CtIrradiationEventData
    filteredInclude = []
    if 'acquisition_protocol' in filters and (
            'acquisition_ctdi_min' in filters or 'acquisition_ctdi_max' in filters or
            'acquisition_dlp_min' in filters or 'acquisition_dlp_max' in filters
    ):
        if ('studyhist' in filters) and ('study_description' in filters):
            events = CtIrradiationEventData.objects.select_related().filter(
                ct_radiation_dose_id__general_study_module_attributes__study_description=filters['study_description'])
        else:
            events = CtIrradiationEventData.objects.filter(acquisition_protocol__iexact=filters['acquisition_protocol'])
        if 'acquisition_ctdi_min' in filters:
            try:
                Decimal(filters['acquisition_ctdi_min'])
                events = events.filter(mean_ctdivol__gte=filters['acquisition_ctdi_min'])
            except InvalidOperation:
                pass
        if 'acquisition_ctdi_max' in filters:
            try:
                Decimal(filters['acquisition_ctdi_max'])
                events = events.filter(mean_ctdivol__lte=filters['acquisition_ctdi_max'])
            except InvalidOperation:
                pass
        if 'acquisition_dlp_min' in filters:
            try:
                Decimal(filters['acquisition_dlp_min'])
                events = events.filter(dlp__gte=filters['acquisition_dlp_min'])
            except InvalidOperation:
                pass
        if 'acquisition_dlp_max' in filters:
            try:
                Decimal(filters['acquisition_dlp_max'])
                events = events.filter(dlp__lte=filters['acquisition_dlp_max'])
            except InvalidOperation:
                pass
        if 'ct_acquisition_type' in filters:
            try:
                events = events.filter(ct_acquisition_type__code_meaning__iexact=filters['ct_acquisition_type'])
            except InvalidOperation:
                pass
        filteredInclude = events.values_list(
            'ct_radiation_dose__general_study_module_attributes__study_instance_uid').distinct()

    elif ('study_description' in filters) and ('acquisition_ctdi_min' in filters) and (
            'acquisition_ctdi_max' in filters):
        events = CtIrradiationEventData.objects.select_related().filter(
            ct_radiation_dose_id__general_study_module_attributes__study_description=filters['study_description'])
        if 'acquisition_ctdi_min' in filters:
            try:
                Decimal(filters['acquisition_ctdi_min'])
                events = events.filter(mean_ctdivol__gte=filters['acquisition_ctdi_min'])
            except InvalidOperation:
                pass
        if 'acquisition_ctdi_max' in filters:
            try:
                Decimal(filters['acquisition_ctdi_max'])
                events = events.filter(mean_ctdivol__lte=filters['acquisition_ctdi_max'])
            except InvalidOperation:
                pass
        if 'ct_acquisition_type' in filters:
            try:
                events = events.filter(ct_acquisition_type__code_meaning__iexact=filters['ct_acquisition_type'])
            except InvalidOperation:
                pass
        filteredInclude = events.values_list(
            'ct_radiation_dose__general_study_module_attributes__study_instance_uid').distinct()

    studies = GeneralStudyModuleAttr.objects.filter(modality_type__exact='CT')
    if filteredInclude:
        studies = studies.filter(study_instance_uid__in=filteredInclude)
    if pid:
        return CTFilterPlusPid(filters, studies.order_by('-study_date', '-study_time').distinct())
    return CTSummaryListFilter(filters, studies.order_by('-study_date', '-study_time').distinct())


class MGSummaryListFilter(django_filters.FilterSet):
    """Filter for mammography studies to display in web interface.

    """
    study_date__gt = django_filters.DateFilter(lookup_expr='gte', label='Date from', field_name='study_date',
                                               widget=forms.TextInput(attrs={'class': 'datepicker'}))
    study_date__lt = django_filters.DateFilter(lookup_expr='lte', label='Date until', field_name='study_date',
                                               widget=forms.TextInput(attrs={'class': 'datepicker'}))
    study_description = django_filters.CharFilter(lookup_expr='icontains', label='Study description')
    procedure_code_meaning = django_filters.CharFilter(lookup_expr='icontains', label='Procedure')
    requested_procedure_code_meaning = django_filters.CharFilter(lookup_expr='icontains', label='Requested procedure')
    projectionxrayradiationdose__irradeventxraydata__acquisition_protocol = django_filters.CharFilter(
        lookup_expr='icontains', label='Acquisition protocol')
    patientstudymoduleattr__patient_age_decimal__gte = django_filters.NumberFilter(lookup_expr='gte',
                                                                                   label=u'Min age (yrs)',
                                                                                   field_name='patientstudymoduleattr'
                                                                                        '__patient_age_decimal')
    patientstudymoduleattr__patient_age_decimal__lte = django_filters.NumberFilter(lookup_expr='lte',
                                                                                   label=u'Max age (yrs)',
                                                                                   field_name='patientstudymoduleattr'
                                                                                        '__patient_age_decimal')
    generalequipmentmoduleattr__institution_name = django_filters.CharFilter(lookup_expr='icontains', label=u'Hospital')
    generalequipmentmoduleattr__manufacturer = django_filters.CharFilter(lookup_expr='icontains', label=u'Make')
    generalequipmentmoduleattr__manufacturer_model_name = django_filters.CharFilter(
        lookup_expr='icontains', label='Model')
    generalequipmentmoduleattr__station_name = django_filters.CharFilter(lookup_expr='icontains', label=u'Station name')
    accession_number = django_filters.CharFilter(method=_custom_acc_filter, label='Accession number')
    generalequipmentmoduleattr__unique_equipment_name__display_name = django_filters.CharFilter(
        lookup_expr='icontains', label='Display name')
    num_events = django_filters.ChoiceFilter(method=_specify_event_numbers, label='Num. events total',
                                             choices=EVENT_NUMBER_CHOICES, widget=forms.Select)
    test_data = django_filters.ChoiceFilter(lookup_expr='isnull', label=u"Include possible test data",
                                            field_name='patientmoduleattr__not_patient_indicator', choices=TEST_CHOICES,
                                            widget=forms.Select)

    class Meta:
        """
        Lists fields and order-by information for django-filter filtering
        """

        model = GeneralStudyModuleAttr
        fields = [
            'study_date__gt', 'study_date__lt',
            'study_description', 'procedure_code_meaning', 'requested_procedure_code_meaning',
            'projectionxrayradiationdose__irradeventxraydata__acquisition_protocol',
            'generalequipmentmoduleattr__institution_name', 'generalequipmentmoduleattr__manufacturer',
            'generalequipmentmoduleattr__manufacturer_model_name', 'generalequipmentmoduleattr__station_name',
            'patientstudymoduleattr__patient_age_decimal__gte', 'patientstudymoduleattr__patient_age_decimal__lte',
            'accession_number',
            'generalequipmentmoduleattr__unique_equipment_name__display_name', 'num_events', 'test_data',
        ]

    o = DateTimeOrderingFilter(
        choices=(
            ('study_description', 'Study Description'),
            ('generalequipmentmoduleattr__institution_name', 'Hospital'),
            ('generalequipmentmoduleattr__manufacturer', 'Make'),
            ('generalequipmentmoduleattr__manufacturer_model_name', 'Model'),
            ('generalequipmentmoduleattr__unique_equipment_name__display_name', 'Display name'),
            ('study_description', 'Study description'),
            ('procedure_code_meaning', 'Procedure'),
            ('-total_agd_left', 'AGD (left)'),
            ('-total_agd_right', 'AGD (right)'),
        ),
        fields=(
            ('study_description', 'study_description'),
            ('generalequipmentmoduleattr__institution_name', 'generalequipmentmoduleattr__institution_name'),
            ('generalequipmentmoduleattr__manufacturer', 'generalequipmentmoduleattr__manufacturer'),
            ('generalequipmentmoduleattr__manufacturer_model_name',
             'generalequipmentmoduleattr__manufacturer_model_name'),
            ('generalequipmentmoduleattr__unique_equipment_name__display_name',
             'generalequipmentmoduleattr__unique_equipment_name__display_name'),
            ('study_description', 'study_description'),
            ('procedure_doce_meaning', 'procedure_code_menaing'),
            ('total_agd_left', '-total_agd_left'),
            ('total_agd_right', '-total_agd_right'),
        ),
    )


class MGFilterPlusPid(MGSummaryListFilter):
    def __init__(self, *args, **kwargs):
        super(MGFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters['patient_name'] = django_filters.CharFilter(method='custom_name_filter', label=u'Patient name')
        self.filters['patient_id'] = django_filters.CharFilter(method='custom_id_filter', label=u'Patient ID')


class DXSummaryListFilter(django_filters.FilterSet):
    """Filter for DX studies to display in web interface.

    """

    study_date__gt = django_filters.DateFilter(lookup_expr='gte', label='Date from', field_name='study_date',
                                               widget=forms.TextInput(attrs={'class': 'datepicker'}))
    study_date__lt = django_filters.DateFilter(lookup_expr='lte', label='Date until', field_name='study_date',
                                               widget=forms.TextInput(attrs={'class': 'datepicker'}))
    study_description = django_filters.CharFilter(lookup_expr='icontains', label='Study description')
    procedure_code_meaning = django_filters.CharFilter(lookup_expr='icontains', label='Procedure')
    requested_procedure_code_meaning = django_filters.CharFilter(lookup_expr='icontains', label='Requested procedure')
    projectionxrayradiationdose__irradeventxraydata__acquisition_protocol = django_filters.CharFilter(
        lookup_expr='icontains', label='Acquisition protocol')
    patientstudymoduleattr__patient_age_decimal__gte = django_filters.NumberFilter(lookup_expr='gte',
                                                                                   label=u'Min age (yrs)',
                                                                                   field_name='patientstudymoduleattr'
                                                                                        '__patient_age_decimal')
    patientstudymoduleattr__patient_age_decimal__lte = django_filters.NumberFilter(lookup_expr='lte',
                                                                                   label=u'Max age (yrs)',
                                                                                   field_name='patientstudymoduleattr'
                                                                                        '__patient_age_decimal')
    generalequipmentmoduleattr__institution_name = django_filters.CharFilter(lookup_expr='icontains', label=u'Hospital')
    generalequipmentmoduleattr__manufacturer = django_filters.CharFilter(lookup_expr='icontains', label=u'Make')
    generalequipmentmoduleattr__manufacturer_model_name = django_filters.CharFilter(
        lookup_expr='icontains', label='Model')
    generalequipmentmoduleattr__station_name = django_filters.CharFilter(lookup_expr='icontains', label=u'Station name')
    accession_number = django_filters.CharFilter(method=_custom_acc_filter, label='Accession number')
    study_dap_min = django_filters.NumberFilter(method=_total_dap_filter, label='Min study DAP (cGy·cm²)')
    study_dap_max = django_filters.NumberFilter(method=_total_dap_filter, label='Max study DAP (cGy·cm²)')
    event_dap_min = django_filters.NumberFilter(method=_event_dap_filter, label='Min acquisition DAP (cGy·cm²)',)
    event_dap_max = django_filters.NumberFilter(method=_event_dap_filter, label='Max acquisition DAP (cGy·cm²)',)
    generalequipmentmoduleattr__unique_equipment_name__display_name = django_filters.CharFilter(
        lookup_expr='icontains', label='Display name')
    num_events = django_filters.ChoiceFilter(method=_specify_event_numbers, label='Num. events total',
                                             choices=EVENT_NUMBER_CHOICES, widget=forms.Select)
    test_data = django_filters.ChoiceFilter(lookup_expr='isnull', label=u"Include possible test data",
                                            field_name='patientmoduleattr__not_patient_indicator', choices=TEST_CHOICES,
                                            widget=forms.Select)

    class Meta:
        """
        Lists fields and order-by information for django-filter filtering
        """
        model = GeneralStudyModuleAttr
        fields = [
            'study_date__gt', 'study_date__lt',
            'study_description', 'procedure_code_meaning', 'requested_procedure_code_meaning',
            'projectionxrayradiationdose__irradeventxraydata__acquisition_protocol',
            'generalequipmentmoduleattr__institution_name', 'generalequipmentmoduleattr__manufacturer',
            'generalequipmentmoduleattr__manufacturer_model_name', 'generalequipmentmoduleattr__station_name',
            'patientstudymoduleattr__patient_age_decimal__gte', 'patientstudymoduleattr__patient_age_decimal__lte',
            'accession_number', 'study_dap_min', 'study_dap_max',
            'event_dap_min', 'event_dap_max',
            'generalequipmentmoduleattr__unique_equipment_name__display_name', 'num_events', 'test_data',
        ]

    o = DateTimeOrderingFilter(
        choices=(
            ('study_description', 'Study Description'),
            ('generalequipmentmoduleattr__institution_name', 'Hospital'),
            ('generalequipmentmoduleattr__manufacturer', 'Make'),
            ('generalequipmentmoduleattr__manufacturer_model_name', 'Model'),
            ('generalequipmentmoduleattr__unique_equipment_name__display_name', 'Display name'),
            ('study_description', 'Study description'),
            ('-total_dap', 'Total DAP'),
        ),
        fields=(
            ('study_description', 'study_description'),
            ('generalequipmentmoduleattr__institution_name', 'generalequipmentmoduleattr__institution_name'),
            ('generalequipmentmoduleattr__manufacturer', 'generalequipmentmoduleattr__manufacturer'),
            ('generalequipmentmoduleattr__manufacturer_model_name',
             'generalequipmentmoduleattr__manufacturer_model_name'),
            ('generalequipmentmoduleattr__unique_equipment_name__display_name',
             'generalequipmentmoduleattr__unique_equipment_name__display_name'),
            ('study_description', 'study_description'),
            ('total_dap', '-total_dap'),
        ),
    )


class DXFilterPlusPid(DXSummaryListFilter):
    def __init__(self, *args, **kwargs):
        super(DXFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters['patient_name'] = django_filters.CharFilter(method='custom_name_filter', label=u'Patient name')
        self.filters['patient_id'] = django_filters.CharFilter(method='custom_id_filter', label=u'Patient ID')


def dx_acq_filter(filters, pid=False):
    from decimal import Decimal, InvalidOperation
    from django.db.models import Q
    from remapp.models import IrradEventXRayData
    filteredInclude = []
    if 'acquisition_protocol' in filters and (
            'acquisition_dap_min' in filters or 'acquisition_dap_max' in filters or
            'acquisition_kvp_min' in filters or 'acquisition_kvp_max' in filters or
            'acquisition_mas_min' in filters or 'acquisition_mas_max' in filters
    ):
        events = IrradEventXRayData.objects.filter(acquisition_protocol__iexact=filters['acquisition_protocol'])
        if 'acquisition_dap_min' in filters:
            try:
                Decimal(filters['acquisition_dap_min'])
                events = events.filter(dose_area_product__gte=filters['acquisition_dap_min'])
            except InvalidOperation:
                pass
        if 'acquisition_dap_max' in filters:
            try:
                Decimal(filters['acquisition_dap_max'])
                events = events.filter(dose_area_product__lte=filters['acquisition_dap_max'])
            except InvalidOperation:
                pass
        if 'acquisition_kvp_min' in filters:
            try:
                Decimal(filters['acquisition_kvp_min'])
                events = events.filter(irradeventxraysourcedata__kvp__kvp__gte=filters['acquisition_kvp_min'])
            except InvalidOperation:
                pass
        if 'acquisition_kvp_max' in filters:
            try:
                Decimal(filters['acquisition_kvp_max'])
                events = events.filter(irradeventxraysourcedata__kvp__kvp__lte=filters['acquisition_kvp_max'])
            except InvalidOperation:
                pass
        if 'acquisition_mas_min' in filters:
            try:
                Decimal(filters['acquisition_mas_min'])
                events = events.filter(irradeventxraysourcedata__exposure__exposure__gte=filters['acquisition_mas_min'])
            except InvalidOperation:
                pass
        if 'acquisition_mas_max' in filters:
            try:
                Decimal(filters['acquisition_mas_max'])
                events = events.filter(irradeventxraysourcedata__exposure__exposure__lte=filters['acquisition_mas_max'])
            except InvalidOperation:
                pass
        filteredInclude = events.values_list(
            'projection_xray_radiation_dose__general_study_module_attributes__study_instance_uid').distinct()

    studies = GeneralStudyModuleAttr.objects.filter(
        Q(modality_type__exact='DX') | Q(modality_type__exact='CR'))

    if filteredInclude:
        studies = studies.filter(study_instance_uid__in=filteredInclude)
    if pid:
        return DXFilterPlusPid(filters, queryset=studies.order_by('-study_date', '-study_time').distinct())
    return DXSummaryListFilter(filters, queryset=studies.order_by('-study_date', '-study_time').distinct())
