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

# Following three lines added so that sphinx autodocumentation works.
import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'
from django.db import models

import logging
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
            Q(patientmoduleattr__name_hashed = False) & Q(patientmoduleattr__patient_name__icontains = value)
         ) | (
            Q(patientmoduleattr__name_hashed = True) & Q(patientmoduleattr__patient_name__exact = hash_id(value))
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
            Q(patientmoduleattr__id_hashed = False) & Q(patientmoduleattr__patient_id__icontains = value)
         ) | (
            Q(patientmoduleattr__id_hashed = True) & Q(patientmoduleattr__patient_id__exact = hash_id(value))
        )
    )
    return filtered

def custom_acc_filter(queryset, value):
    if not value:
        return queryset

    from django.db.models import Q
    from remapp.tools.hash_id import hash_id
    filtered = queryset.filter(
        (
            Q(accession_hashed = False) & Q(accession_number__icontains = value)
         ) | (
            Q(accession_hashed = True) & Q(accession_number__exact = hash_id(value))
        )
    )
    return filtered


def dap_min_filter(queryset, value):
    if not value:
        return queryset

    from decimal import Decimal, InvalidOperation
    try:
        value_gy_m2 = Decimal(value) / Decimal(1000000)
    except InvalidOperation:
        return queryset
    filtered = queryset.filter(
        projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_area_product_total__gte=
        value_gy_m2)
    return filtered


def dap_max_filter(queryset, value):
    if not value:
        return queryset

    from decimal import Decimal, InvalidOperation
    try:
        value_gy_m2 = Decimal(value) / Decimal(1000000)
    except InvalidOperation:
        return queryset
    filtered = queryset.filter(
        projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_area_product_total__lte=
        value_gy_m2)
    return filtered


def get_database_field(json_filter_options, label):
    for item in json_filter_options:
        if item['label'] == label:
            return item['db_field']
    # we should never end up here, but let's return None
    return None


def parse_advanced_search(search_string, json_filter_options):
    """
    Filter model based on the advanced search_string
    adapted from: https://www.djangosnippets.org/snippets/1700/
    :param model: the model to filter
    :param search_string: human readable advanced search string
    :param json_filter_options: json search template
    :return: Queryset or None if no filter was found
    """
    from django.db.models import Q

    # Replace human readable parameters by database (model) fields
    start_pos_parameter = search_string.find("[")
    end_pos_parameter = search_string.find("]")
    while (start_pos_parameter < end_pos_parameter):
        # if this is the case both are found and start_pos < end_pos
        parameter = search_string[start_pos_parameter + 1:end_pos_parameter]
        db_field = get_database_field(json_filter_options, parameter)
        parameter = '[' + parameter + ']'
        search_string = search_string.replace(parameter, db_field)
        start_pos_parameter = search_string.find("[")
        end_pos_parameter = search_string.find("]")

    # Replace human readable operators by django-operators
    search_string = search_string.replace(" is ", "=")
    search_string = search_string.replace(" contains ", "__icontains=")
    search_string = search_string.replace(" begins with ", "__startswith=")
    search_string = search_string.replace(" ends with ", "__endswith=")
    search_string = search_string.replace(" greater than or equal to ", "__gte=")
    search_string = search_string.replace(" smaller than or equal to ", "__lte=")
    search_string = search_string.replace(" greater than ", "__gt=")
    search_string = search_string.replace(" smaller than ", "__lt=")

    # Create list of Q objects and find operators
    queries = []
    and_or_operators = []
    brace_start_pos = search_string.find("{")
    operator_pos = search_string.find("=")
    brace_end_pos = search_string.find("}")
    while brace_start_pos < brace_end_pos:
        parameter = search_string[brace_start_pos+1:operator_pos]
        value = search_string[operator_pos+2:brace_end_pos-1]  # +/-2 to remove outer quotation marks
        value.replace("'", "")
        if value != "":
            if (brace_start_pos > 4) and (search_string[brace_start_pos-4:brace_start_pos-1] == 'NOT'):
                queries.append(~Q(**{parameter: value}))
            else:
                queries.append(Q(**{parameter: value}))
        brace_start_pos = search_string.find("{", brace_end_pos)
        if brace_start_pos > -1:
            and_or_operators.append(search_string[brace_end_pos+1:brace_start_pos].replace('NOT', '').strip())
        operator_pos = search_string.find("=", brace_end_pos+1)
        brace_end_pos = search_string.find("}", brace_end_pos+1)

    if len(queries) == 0:
        return None
    q_object = queries[0]
    if len(queries) > 1:
        for query, and_or_operator in zip(queries[1:], and_or_operators):
            # Q.AND is defined as "AND" and Q.OR as "OR", so the text "AND" or "OR" will work
            # but if we gonna localise the human readable string, we should be aware of this
            q_object.add(query, and_or_operator)
    return q_object


def get_advanced_search_objects(model, search_string, json_search_object):

    from django.db.models import Q
    from copy import deepcopy

    prev_q_object = Q()
    search_string = search_string.strip()
    closing_bracket_pos = search_string.find(')')
    last_closing_bracket_pos = -1
    while closing_bracket_pos > -1:
        opening_bracket_pos = search_string.rfind('(', 0, closing_bracket_pos)
        if opening_bracket_pos == -1:
            return None
        inner_search_string = search_string[opening_bracket_pos + 1:closing_bracket_pos].strip()
        # operator should be just before opening_bracket_pos, if it is there (first or the inner of double brackets
        # doesn't have one)
        operator = ''
        start_pos_operator = -2
        if not((opening_bracket_pos == 0) or (search_string[opening_bracket_pos-1] == '(')):
            start_pos_operator = search_string.rfind('}', 0, opening_bracket_pos)
            if search_string.rfind('(', 0, opening_bracket_pos) > start_pos_operator:
                start_pos_operator = search_string.rfind('(', 0, opening_bracket_pos)
            operator = search_string[start_pos_operator+1:opening_bracket_pos - 1].strip()
        cur_q_object = parse_advanced_search(inner_search_string, json_search_object)
        if operator.find('NOT') > -1:
            cur_q_object.negate()
            operator = operator.replace('NOT', '').strip()
        if operator != '':
            if opening_bracket_pos > last_closing_bracket_pos:
                cur_q_object.add(prev_q_object, operator)
            else:
                # Add as child
                cur_q_object.add(prev_q_object, operator, squash=False)
        prev_q_object = deepcopy(cur_q_object)
        if start_pos_operator > -2:
            search_string = (search_string[0:start_pos_operator+1] + search_string[closing_bracket_pos + 1:]).strip()
        else:
            search_string = (search_string[0:opening_bracket_pos] + search_string[closing_bracket_pos + 1:]).strip()
        last_closing_bracket_pos = opening_bracket_pos
        while '()' in search_string:
            # if position before last_closing_bracket_pos lower this variable with 2.
            pos_brackets = search_string.find('()')
            search_string = search_string[0:pos_brackets] + search_string[pos_brackets+2:]
            if pos_brackets < last_closing_bracket_pos:
                last_closing_bracket_pos = last_closing_bracket_pos - 2
        closing_bracket_pos = search_string.find(')', last_closing_bracket_pos)

    # Now parse the (last) part without brackets, if it is there
    if len(search_string) > 0:
        if len(prev_q_object.children) == 0:
            # it should be a simple query without brackets
            cur_q_object = parse_advanced_search(search_string, json_search_object)
        else:
            if search_string[0] != '{':
                # operator is at the start of the search string
                operator = search_string[0:search_string.find('{')].strip()
                search_string = search_string[search_string.find('{'):]
            else:
                # operator is at the end of the search string
                operator = search_string[search_string.find('}')+1:].strip()
                search_string = search_string[:search_string.find('}')+1]
            cur_q_object = parse_advanced_search(search_string, json_search_object)
            if operator.find('NOT') > -1:
                cur_q_object.negate()
                operator = operator.replace('NOT', '').strip()
            if operator != '':
                cur_q_object.add(prev_q_object, operator)

    return model.objects.filter(cur_q_object)


class RFSummaryListFilter(django_filters.FilterSet):
    """Filter for fluoroscopy studies to display in web interface.

    """
    date_after = django_filters.DateFilter(lookup_type='gte', label=u'Date from', name='study_date', widget=forms.TextInput(attrs={'class':'datepicker'}))
    date_before = django_filters.DateFilter(lookup_type='lte', label=u'Date until', name='study_date', widget=forms.TextInput(attrs={'class':'datepicker'}))
    study_description = django_filters.CharFilter(lookup_type='icontains', label=u'Study description')
    procedure_code_meaning = django_filters.CharFilter(lookup_type='icontains', label=u'Procedure', name='procedure_code_meaning')
    requested_procedure = django_filters.CharFilter(lookup_type='icontains', label=u'Requested procedure', name='requested_procedure_code_meaning')
    acquisition_protocol = django_filters.CharFilter(lookup_type='icontains', label=u'Acquisition protocol', name='projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')
    patient_age_min = django_filters.NumberFilter(lookup_type='gt', label=u'Min age (yrs)', name='patientstudymoduleattr__patient_age_decimal')
    patient_age_max = django_filters.NumberFilter(lookup_type='lt', label=u'Max age (yrs)', name='patientstudymoduleattr__patient_age_decimal')
    institution_name = django_filters.CharFilter(lookup_type='icontains', label=u'Hospital', name='generalequipmentmoduleattr__institution_name')
    manufacturer = django_filters.CharFilter(lookup_type='icontains', label=u'Manufacturer', name='generalequipmentmoduleattr__manufacturer')
    model_name = django_filters.CharFilter(lookup_type='icontains', label=u'Model', name='generalequipmentmoduleattr__manufacturer_model_name')
    station_name = django_filters.CharFilter(lookup_type='icontains', label=u'Station name', name='generalequipmentmoduleattr__station_name')
    performing_physician_name = django_filters.CharFilter(lookup_type='icontains', label=u'Physician')
    accession_number = django_filters.MethodFilter(action=custom_acc_filter, label=u'Accession number')
    display_name = django_filters.CharFilter(lookup_type='icontains', label=u'Display name', name='generalequipmentmoduleattr__unique_equipment_name__display_name')
    study_dap_min = django_filters.MethodFilter(action=dap_min_filter, label=mark_safe(u'Min study DAP (cGy.cm<sup>2</sup>)'))
    study_dap_max = django_filters.MethodFilter(action=dap_max_filter, label=mark_safe(u'Max study DAP (cGy.cm<sup>2</sup>)'))
    test_data = django_filters.ChoiceFilter(lookup_type='isnull', label=u"Include possible test data", name='patientmoduleattr__not_patient_indicator', choices=TEST_CHOICES, widget=forms.Select)

    class Meta:
        model = GeneralStudyModuleAttr
        fields = []
        order_by = (
            ('-study_date', mark_safe('Exam date &darr;')),
            ('study_date', mark_safe('Exam date &uarr;')),
            ('generalequipmentmoduleattr__institution_name', 'Hospital'),
            ('generalequipmentmoduleattr__manufacturer', 'Make'),
            ('generalequipmentmoduleattr__manufacturer_model_name', 'Model name'),
            ('generalequipmentmoduleattr__station_name', 'Station name'),
            ('study_description', 'Study description'),
            ('-projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_area_product_total','Total DAP'),
            ('-projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_rp_total','Total RP Dose'),
            )
    def get_order_by(self, order_value):
        if order_value == 'study_date':
            return ['study_date', 'study_time']
        elif order_value == '-study_date':
            return ['-study_date','-study_time']
        return super(RFSummaryListFilter, self).get_order_by(order_value)


class RFFilterPlusPid(RFSummaryListFilter):
    def __init__(self, *args, **kwargs):
        super(RFFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters['patient_name'] = django_filters.MethodFilter(action=custom_name_filter, label=u'Patient name')
        self.filters['patient_id'] = django_filters.MethodFilter(action=custom_id_filter, label=u'Patient ID')


class CTSummaryListFilter(django_filters.FilterSet):
    """Filter for CT studies to display in web interface.

    """
    date_after = django_filters.DateFilter(lookup_type='gte', label=u'Date from', name='study_date', widget=forms.TextInput(attrs={'class':'datepicker'}))
    date_before = django_filters.DateFilter(lookup_type='lte', label=u'Date until', name='study_date', widget=forms.TextInput(attrs={'class':'datepicker'}))
    study_description = django_filters.CharFilter(lookup_type='icontains', label=u'Study description')
    procedure_code_meaning = django_filters.CharFilter(lookup_type='icontains', label=u'Procedure', name='procedure_code_meaning')
    requested_procedure = django_filters.CharFilter(lookup_type='icontains', label=u'Requested procedure', name='requested_procedure_code_meaning')
    acquisition_protocol = django_filters.CharFilter(lookup_type='icontains', label=u'Acquisition protocol', name='ctradiationdose__ctirradiationeventdata__acquisition_protocol')
    patient_age_min = django_filters.NumberFilter(lookup_type='gt', label=u'Min age (yrs)', name='patientstudymoduleattr__patient_age_decimal')
    patient_age_max = django_filters.NumberFilter(lookup_type='lt', label=u'Max age (yrs)', name='patientstudymoduleattr__patient_age_decimal')
    institution_name = django_filters.CharFilter(lookup_type='icontains', label=u'Hospital', name='generalequipmentmoduleattr__institution_name')
    manufacturer = django_filters.CharFilter(lookup_type='icontains', label=u'Make', name='generalequipmentmoduleattr__manufacturer')
    model_name = django_filters.CharFilter(lookup_type='icontains', label=u'Model', name='generalequipmentmoduleattr__manufacturer_model_name')
    station_name = django_filters.CharFilter(lookup_type='icontains', label=u'Station name', name='generalequipmentmoduleattr__station_name')
    accession_number = django_filters.MethodFilter(action=custom_acc_filter, label=u'Accession number')
    study_dlp_min = django_filters.NumberFilter(lookup_type='gte', label=u'Min study DLP', name='ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total')
    study_dlp_max = django_filters.NumberFilter(lookup_type='lte', label=u'Max study DLP', name='ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total')
    display_name = django_filters.CharFilter(lookup_type='icontains', label=u'Display name', name='generalequipmentmoduleattr__unique_equipment_name__display_name')
    test_data = django_filters.ChoiceFilter(lookup_type='isnull', label=u"Include possible test data", name='patientmoduleattr__not_patient_indicator', choices=TEST_CHOICES, widget=forms.Select)

    class Meta:
        model = GeneralStudyModuleAttr
        fields = []
        order_by = (
            ('-study_date', mark_safe('Exam date &darr;')),
            ('study_date', mark_safe('Exam date &uarr;')),
            ('generalequipmentmoduleattr__institution_name', 'Hospital'),
            ('generalequipmentmoduleattr__manufacturer', 'Make'),
            ('generalequipmentmoduleattr__manufacturer_model_name', 'Model name'),
            ('generalequipmentmoduleattr__station_name', 'Station name'),
            ('study_description', 'Study description'),
            ('-ctradiationdose__ctaccumulateddosedata__ct_dose_length_product_total', 'Total DLP'),
            )

    def get_order_by(self, order_value):
        if order_value == 'study_date':
            return ['study_date', 'study_time']
        elif order_value == '-study_date':
            return ['-study_date','-study_time']
        return super(CTSummaryListFilter, self).get_order_by(order_value)


class CTFilterPlusPid(CTSummaryListFilter):
    def __init__(self, *args, **kwargs):
        super(CTFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters['patient_name'] = django_filters.MethodFilter(action=custom_name_filter, label=u'Patient name')
        self.filters['patient_id'] = django_filters.MethodFilter(action=custom_id_filter, label=u'Patient ID')


def ct_acq_filter(filters, pid=False):
    from decimal import Decimal, InvalidOperation
    from remapp.models import GeneralStudyModuleAttr, CtIrradiationEventData
    filteredInclude = []
    if 'acquisition_protocol' in filters and (
                    'acquisition_ctdi_min' in filters or 'acquisition_ctdi_max' in filters or
                        'acquisition_dlp_min' in filters or 'acquisition_dlp_max' in filters
    ):
        if ('studyhist' in filters) and ('study_description' in filters):
            events = CtIrradiationEventData.objects.select_related().filter(ct_radiation_dose_id__general_study_module_attributes__study_description=filters['study_description'])
        else:
            events = CtIrradiationEventData.objects.filter(acquisition_protocol__exact = filters['acquisition_protocol'])
        if 'acquisition_ctdi_min' in filters:
            try:
                Decimal(filters['acquisition_ctdi_min'])
                events = events.filter(mean_ctdivol__gte = filters['acquisition_ctdi_min'])
            except InvalidOperation:
                pass
        if 'acquisition_ctdi_max' in filters:
            try:
                Decimal(filters['acquisition_ctdi_max'])
                events = events.filter(mean_ctdivol__lte = filters['acquisition_ctdi_max'])
            except InvalidOperation:
                pass
        if 'acquisition_dlp_min' in filters:
            try:
                Decimal(filters['acquisition_dlp_min'])
                events = events.filter(dlp__gte = filters['acquisition_dlp_min'])
            except InvalidOperation:
                pass
        if 'acquisition_dlp_max' in filters:
            try:
                Decimal(filters['acquisition_dlp_max'])
                events = events.filter(dlp__lte = filters['acquisition_dlp_max'])
            except InvalidOperation:
                pass
        filteredInclude = list(set(
            [o.ct_radiation_dose.general_study_module_attributes.study_instance_uid for o in events]))

    elif ('study_description' in filters) and ('acquisition_ctdi_min' in filters) and ('acquisition_ctdi_max' in filters):
        events = CtIrradiationEventData.objects.select_related().filter(ct_radiation_dose_id__general_study_module_attributes__study_description=filters['study_description'])
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
        filteredInclude = list(set(
            [o.ct_radiation_dose.general_study_module_attributes.study_instance_uid for o in events]))

    studies = GeneralStudyModuleAttr.objects.filter(modality_type__exact = 'CT')
    if filteredInclude:
        studies = studies.filter(study_instance_uid__in = filteredInclude)
    if pid:
        return CTFilterPlusPid(filters, studies.order_by().distinct())
    return CTSummaryListFilter(filters, studies.order_by().distinct())


class MGSummaryListFilter(django_filters.FilterSet):
    """Filter for mammography studies to display in web interface.

    """
    date_after = django_filters.DateFilter(lookup_type='gte', label=u'Date from', name='study_date', widget=forms.TextInput(attrs={'class':'datepicker'}))
    date_before = django_filters.DateFilter(lookup_type='lte', label=u'Date until', name='study_date', widget=forms.TextInput(attrs={'class':'datepicker'}))
    study_description = django_filters.CharFilter(lookup_type='icontains', label=u'Study description')
    procedure_code_meaning = django_filters.CharFilter(lookup_type='icontains', label=u'Procedure', name='procedure_code_meaning')
    requested_procedure = django_filters.CharFilter(lookup_type='icontains', label=u'Requested procedure', name='requested_procedure_code_meaning')
    acquisition_protocol = django_filters.CharFilter(lookup_type='icontains', label=u'Acquisition protocol', name='projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')
    patient_age_min = django_filters.NumberFilter(lookup_type='gt', label=u'Min age (yrs)', name='patientstudymoduleattr__patient_age_decimal')
    patient_age_max = django_filters.NumberFilter(lookup_type='lt', label=u'Max age (yrs)', name='patientstudymoduleattr__patient_age_decimal')
    institution_name = django_filters.CharFilter(lookup_type='icontains', label=u'Hospital', name='generalequipmentmoduleattr__institution_name')
    manufacturer = django_filters.CharFilter(lookup_type='icontains', label=u'Manufacturer', name='generalequipmentmoduleattr__manufacturer')
    model_name = django_filters.CharFilter(lookup_type='icontains', label=u'Model', name='generalequipmentmoduleattr__manufacturer_model_name')
    station_name = django_filters.CharFilter(lookup_type='icontains', label=u'Station name', name='generalequipmentmoduleattr__station_name')
    accession_number = django_filters.MethodFilter(action=custom_acc_filter, label=u'Accession number')
    display_name = django_filters.CharFilter(lookup_type='icontains', label=u'Display name', name='generalequipmentmoduleattr__unique_equipment_name__display_name')
    test_data = django_filters.ChoiceFilter(lookup_type='isnull', label=u"Include possible test data", name='patientmoduleattr__not_patient_indicator', choices=TEST_CHOICES, widget=forms.Select)

    class Meta:
        model = GeneralStudyModuleAttr
        fields = [
            ]

        order_by = (
            ('-study_date', mark_safe('Exam date &darr;')),
            ('study_date', mark_safe('Exam date &uarr;')),
            ('generalequipmentmoduleattr__institution_name', 'Hospital'),
            ('generalequipmentmoduleattr__manufacturer', 'Make'),
            ('generalequipmentmoduleattr__manufacturer_model_name', 'Model name'),
            ('generalequipmentmoduleattr__station_name', 'Station name'),
            ('procedure_code_meaning', 'Procedure'),
            ('-projectionxrayradiationdose__accumxraydose__accummammographyxraydose__accumulated_average_glandular_dose', 'Accumulated AGD'),
            )

    def get_order_by(self, order_value):
        if order_value == 'study_date':
            return ['study_date', 'study_time']
        elif order_value == '-study_date':
            return ['-study_date','-study_time']
        return super(MGSummaryListFilter, self).get_order_by(order_value)

class MGFilterPlusPid(MGSummaryListFilter):
    def __init__(self, *args, **kwargs):
        super(MGFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters['patient_name'] = django_filters.MethodFilter(action=custom_name_filter, label=u'Patient name')
        self.filters['patient_id'] = django_filters.MethodFilter(action=custom_id_filter, label=u'Patient ID')


class DXSummaryListFilter(django_filters.FilterSet):
    """Filter for DX studies to display in web interface.

    """
    date_after = django_filters.DateFilter(lookup_type='gte', label=u'Date from', name='study_date', widget=forms.TextInput(attrs={'class':'datepicker'}))
    date_before = django_filters.DateFilter(lookup_type='lte', label=u'Date until', name='study_date', widget=forms.TextInput(attrs={'class':'datepicker'}))
    study_description = django_filters.CharFilter(lookup_type='icontains', label=u'Study description')
    procedure_code_meaning = django_filters.CharFilter(lookup_type='icontains', label=u'Procedure', name='procedure_code_meaning')
    requested_procedure = django_filters.CharFilter(lookup_type='icontains', label=u'Requested procedure', name='requested_procedure_code_meaning')
    acquisition_protocol = django_filters.CharFilter(lookup_type='icontains', label=u'Acquisition protocol', name='projectionxrayradiationdose__irradeventxraydata__acquisition_protocol')
    patient_age_min = django_filters.NumberFilter(lookup_type='gt', label=u'Min age (yrs)', name='patientstudymoduleattr__patient_age_decimal')
    patient_age_max = django_filters.NumberFilter(lookup_type='lt', label=u'Max age (yrs)', name='patientstudymoduleattr__patient_age_decimal')
    institution_name = django_filters.CharFilter(lookup_type='icontains', label=u'Hospital', name='generalequipmentmoduleattr__institution_name')
    manufacturer = django_filters.CharFilter(lookup_type='icontains', label=u'Make', name='generalequipmentmoduleattr__manufacturer')
    model_name = django_filters.CharFilter(lookup_type='icontains', label=u'Model', name='generalequipmentmoduleattr__manufacturer_model_name')
    station_name = django_filters.CharFilter(lookup_type='icontains', label=u'Station name', name='generalequipmentmoduleattr__station_name')
    accession_number = django_filters.MethodFilter(action=custom_acc_filter, label=u'Accession number')
    study_dap_min = django_filters.MethodFilter(action=dap_min_filter, label=mark_safe(u'Min study DAP (cGy.cm<sup>2</sup>)'))
    study_dap_max = django_filters.MethodFilter(action=dap_max_filter, label=mark_safe(u'Max study DAP (cGy.cm<sup>2</sup>)'))
    # acquisition_dap_max = django_filters.NumberFilter(lookup_type='lte', label=mark_safe('Max acquisition DAP (Gy.m<sup>2</sup>)'), name='projectionxrayradiationdose__irradeventxraydata__dose_area_product')
    # acquisition_dap_min = django_filters.NumberFilter(lookup_type='gte', label=mark_safe('Min acquisition DAP (Gy.m<sup>2</sup>)'), name='projectionxrayradiationdose__irradeventxraydata__dose_area_product')
    display_name = django_filters.CharFilter(lookup_type='icontains', label=u'Display name', name='generalequipmentmoduleattr__unique_equipment_name__display_name')
    test_data = django_filters.ChoiceFilter(lookup_type='isnull', label=u"Include possible test data", name='patientmoduleattr__not_patient_indicator', choices=TEST_CHOICES, widget=forms.Select)

    class Meta:
        model = GeneralStudyModuleAttr
        fields = [
            'date_after',
            'date_before',
            'institution_name',
            'study_description',
            'procedure_code_meaning',
            'requested_procedure',
            'acquisition_protocol',
            'patient_age_min',
            'patient_age_max',
            'manufacturer',
            'model_name',
            'station_name',
            'display_name',
            'accession_number',
            #'study_dap_min',
            #'study_dap_max',
            'test_data',
            ]
        order_by = (
            ('-study_date', mark_safe('Exam date &darr;')),
            ('study_date', mark_safe('Exam date &uarr;')),
            ('generalequipmentmoduleattr__institution_name', 'Hospital'),
            ('generalequipmentmoduleattr__manufacturer', 'Make'),
            ('generalequipmentmoduleattr__manufacturer_model_name', 'Model name'),
            ('generalequipmentmoduleattr__station_name', 'Station name'),
            ('study_description', 'Study description'),
            ('-projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_area_product_total', 'Total DAP'),
            )

    def get_order_by(self, order_value):
        if order_value == 'study_date':
            return ['study_date', 'study_time']
        elif order_value == '-study_date':
            return ['-study_date','-study_time']
        return super(DXSummaryListFilter, self).get_order_by(order_value)

class DXFilterPlusPid(DXSummaryListFilter):
    def __init__(self, *args, **kwargs):
        super(DXFilterPlusPid, self).__init__(*args, **kwargs)
        self.filters['patient_name'] = django_filters.MethodFilter(action=custom_name_filter, label=u'Patient name')
        self.filters['patient_id'] = django_filters.MethodFilter(action=custom_id_filter, label=u'Patient ID')


def dx_acq_filter(filters, pid=False):
    from decimal import Decimal, InvalidOperation
    from django.db.models import Q
    from remapp.models import GeneralStudyModuleAttr, IrradEventXRayData
    filteredInclude = []
    if 'acquisition_protocol' in filters and (
        'acquisition_dap_min' in filters or 'acquisition_dap_max' in filters or
        'acquisition_kvp_min' in filters or 'acquisition_kvp_max' in filters or
        'acquisition_mas_min' in filters or 'acquisition_mas_max' in filters
    ):
        events = IrradEventXRayData.objects.filter(acquisition_protocol__exact = filters['acquisition_protocol'])
        if 'acquisition_dap_min' in filters:
            try:
                Decimal(filters['acquisition_dap_min'])
                events = events.filter(dose_area_product__gte = filters['acquisition_dap_min'])
            except InvalidOperation:
                pass
        if 'acquisition_dap_max' in filters:
            try:
                Decimal(filters['acquisition_dap_max'])
                events = events.filter(dose_area_product__lte = filters['acquisition_dap_max'])
            except InvalidOperation:
                pass
        if 'acquisition_kvp_min' in filters:
            try:
                Decimal(filters['acquisition_kvp_min'])
                events = events.filter(irradeventxraysourcedata__kvp__kvp__gte = filters['acquisition_kvp_min'])
            except InvalidOperation:
                pass
        if 'acquisition_kvp_max' in filters:
            try:
                Decimal(filters['acquisition_kvp_max'])
                events = events.filter(irradeventxraysourcedata__kvp__kvp__lte = filters['acquisition_kvp_max'])
            except InvalidOperation:
                pass
        if 'acquisition_mas_min' in filters:
            try:
                Decimal(filters['acquisition_mas_min'])
                events = events.filter(irradeventxraysourcedata__exposure__exposure__gte = filters['acquisition_mas_min'])
            except InvalidOperation:
                pass
        if 'acquisition_mas_max' in filters:
            try:
                Decimal(filters['acquisition_mas_max'])
                events = events.filter(irradeventxraysourcedata__exposure__exposure__lte = filters['acquisition_mas_max'])
            except InvalidOperation:
                pass
        filteredInclude = list(set(
            [o.projection_xray_radiation_dose.general_study_module_attributes.study_instance_uid for o in events]
        ))
    studies = GeneralStudyModuleAttr.objects.filter(
        Q(modality_type__exact='DX') | Q(modality_type__exact='CR'))
    if filteredInclude:
        studies = studies.filter(study_instance_uid__in = filteredInclude)
    if pid:
        return DXFilterPlusPid(filters, studies.order_by().distinct())
    return DXSummaryListFilter(filters, studies.order_by().distinct())