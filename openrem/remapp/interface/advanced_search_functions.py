"""
..  module:: advanced_search_functions
    :synopsis: Helper functions for advanced search

..  moduleauthor:: Luuk Oostveen

"""
import django_filters
from django import forms


def add2json_search_string(json_string, dbfield, label=None, comparison=None):
    """
    adds a databasefield to the json-search-options
    :param json_string: initial json-options
    :param dbfield: databasefield to add
    :param label: text for user, if not given, it is build from databasefield name
    :param comparison: comparisons that are possible, if not given, it is build from databasefield type
    :return: json-search-options including added databasefield
    """
    from remapp.models import GeneralStudyModuleAttr

    db_field_list = dbfield.split('__')
    if comparison is None:
        child_model = GeneralStudyModuleAttr
        for field in db_field_list[:-1]:
            child_model = child_model._meta.get_field(field).related_model
        field_type = child_model._meta.get_field(db_field_list[-1]).get_internal_type()
        if field_type == 'TextField':
            comparison = '["is", "contains", "begins with", "ends with"]'
        elif field_type == 'BooleanField':
            comparison = '["is"]'
        else:
            # at least: DateField, TimeField, DateTimeField, TimeField, DecimalField
            comparison = \
                '["is", "greater than", "smaller than", "greater than or equal to", "smaller than or equal to"]'
    if label is None or len(label) == 0:
        label = db_field_list[-1].replace('_', ' ')
    if dbfield and label and comparison:
        # assumes labels with at least 2 characters
        label = label[0].upper() + label[1:]
        return json_string + '{"db_field":"' + dbfield + '",' + \
               '"label":"' + label + '",' + \
               '"comparison":' + comparison + '},'
    else:
        return json_string


def add_model2json_search(model, json_search, field_prefix='', pid=True):
    """
    :param model: add a full django-model (database-table) to the json-search-options
    :param json_search: the inital json-search-options
    :param field_prefix: prefix for each field
    :param pid: if true: add patientid and patientname to search options
    :return: json-search-options including the model
    """
    from remapp.models import ContextID

    # We need to access protected member, not so nice, but don't see another way.
    for field in model._meta.get_fields():
        if not field.is_relation:
            if not ((field.name == 'id') or ('hash' in field.name)):
                if pid or not ((field.name == 'patient_name') or (field.name == 'patient_id')):
                    json_search = add2json_search_string(json_search, field_prefix + field.name, field.verbose_name)
        else:
            if field.related_model == ContextID:
                json_search = add2json_search_string(json_search, field_prefix + field.name + '__code_meaning',
                                                     field.verbose_name)
    return json_search


class AdvancedSearchFilter(django_filters.FilterSet):
    """
    Filter based on string containing fieldnames, operators and values.
    String can contain AND/ AND NOT / OR / OR NOT and braces.
    """
    # https://github.com/carltongibson/django-filter/issues/137, almost at the bottom (visigoth)
    # https://stackoverflow.com/questions/28701202/create-or-filter-with-django-filters

    advanced_search_string = django_filters.CharFilter(
        widget=forms.TextInput(attrs={'width': '100%', 'readonly': 'False', 'class': 'form-control',
                                      'placeholder': 'Build your search string using the builder below.'}),
        required=False,
        label='',
        help_text='')

    def __init__(self, data, json_search_object, modality=None):
        """
        :param data: default __init__ data argument for CharFilter
        :param json_search_object: Object containing all possible search options
        :param modality: Modality that should be filtered on. If None, cross modality search will be performed
        """
        super(AdvancedSearchFilter, self).__init__(data=data)
        self.json_search_object = json_search_object
        self.modality = modality

    @property
    def qs(self):
        """
        Overrides BaseFilterSet.qs
        :return: QuerySet
        """
        from django.db.models import Q
        from copy import deepcopy

        # start with all the results and filter from there
        _qs = self.queryset.all()

        prev_q_object = Q()
        cur_q_object = Q()
        local_search_string = ''
        if self.form.is_valid() and 'advanced_search_string' in self.filters:
            local_search_string = self.form['advanced_search_string'].value()
        closing_bracket_pos = local_search_string.find(')')
        last_closing_bracket_pos = -1
        while closing_bracket_pos > -1:
            opening_bracket_pos = local_search_string.rfind('(', 0, closing_bracket_pos)
            if opening_bracket_pos == -1:
                return None
            inner_search_string = local_search_string[opening_bracket_pos + 1:closing_bracket_pos].strip()
            # operator should be just before opening_bracket_pos, if it is there (first or the inner of double brackets
            # doesn't have one)
            operator = ''
            start_pos_operator = -2
            if not ((opening_bracket_pos == 0) or (local_search_string[opening_bracket_pos - 1] == '(')):
                start_pos_operator = local_search_string.rfind('}', 0, opening_bracket_pos)
                if local_search_string.rfind('(', 0, opening_bracket_pos) > start_pos_operator:
                    start_pos_operator = local_search_string.rfind('(', 0, opening_bracket_pos)
                operator = local_search_string[start_pos_operator + 1:opening_bracket_pos - 1].strip()
            temp_q_object = self.__parse_advanced_search(inner_search_string)
            if operator.find('NOT') > -1:
                temp_q_object.negate()
                operator = operator.replace('NOT', '').strip()
            cur_q_object.add(temp_q_object, "AND")
            if operator != '':
                if opening_bracket_pos > last_closing_bracket_pos:
                    cur_q_object.add(prev_q_object, operator)
                else:
                    # Add as child
                    cur_q_object.add(prev_q_object, operator, squash=False)
            prev_q_object = deepcopy(cur_q_object)
            if start_pos_operator > -2:
                local_search_string = (local_search_string[0:start_pos_operator + 1] +
                                       local_search_string[closing_bracket_pos + 1:]).strip()
            else:
                local_search_string = (local_search_string[0:opening_bracket_pos] +
                                       local_search_string[closing_bracket_pos + 1:]).strip()
            last_closing_bracket_pos = opening_bracket_pos
            while '()' in local_search_string:
                # if position before last_closing_bracket_pos lower this variable with 2.
                pos_brackets = local_search_string.find('()')
                local_search_string = local_search_string[0:pos_brackets] + local_search_string[pos_brackets + 2:]
                if pos_brackets < last_closing_bracket_pos:
                    last_closing_bracket_pos = last_closing_bracket_pos - 2
            closing_bracket_pos = local_search_string.find(')', last_closing_bracket_pos)

        # Now parse the (last) part without brackets, if it is there
        if len(local_search_string) > 0:
            if local_search_string[:3] == 'NOT' and len(cur_q_object) > 0:
                # We had a search_string with multiple round brackets just after the first "NOT"
                # we need this difficult way of "negate", because if the root node is "negated", also the modality
                # that we add later, will be negated.
                temp_q_object = deepcopy(cur_q_object)
                temp_q_object.negate()
                cur_q_object = Q()
                cur_q_object.add(temp_q_object, "AND")
            elif len(prev_q_object.children) == 0:
                # it should be a simple query without brackets
                cur_q_object = self.__parse_advanced_search(local_search_string)
            else:
                if local_search_string[0] != '{':
                    # operator is at the start of the search string
                    operator = local_search_string[0:local_search_string.find('{')].strip()
                    _search_string = local_search_string[local_search_string.find('{'):]
                else:
                    # operator is at the end of the search string
                    operator = local_search_string[local_search_string.find('}') + 1:].strip()
                    _search_string = local_search_string[:local_search_string.find('}') + 1]
                cur_q_object = self.__parse_advanced_search(local_search_string)
                if operator.find('NOT') > -1:
                    cur_q_object.negate()
                    operator = operator.replace('NOT', '').strip()
                if operator != '':
                    cur_q_object.add(prev_q_object, operator)
        if self.modality:
            # TODO: not so nice: hardcoded 'Modality type'. If someone changes verbose_name in models.py,
            #  it doesn't work
            cur_q_object.add(self.__parse_advanced_search("{{[Modality type] is '{0}'}}".format(self.modality.upper())),
                             "AND")
        if cur_q_object:
            _qs = _qs.filter(cur_q_object)
        return _qs

    class Meta:
        from remapp.models import GeneralStudyModuleAttr
        model = GeneralStudyModuleAttr
        fields = []

    def __get_database_field(self, label):
        """
        return databasefield belonging to label
        :param label: user readable label
        :return: databasedfield belonging to label
        """
        for item in self.json_search_object:
            if item['label'] == label:
                return item['db_field']
        # we should never end up here, but let's return None
        return None

    def __parse_advanced_search(self, search_string):
        """
        Filter model based on the advanced search_string
        adapted from: https://www.djangosnippets.org/snippets/1700/
        :param search_string: human readable advanced search string
        :return: Queryset or None if no filter was found
        """
        from django.db.models import Q

        # Replace human readable parameters by database (model) fields
        start_pos_parameter = search_string.find("[")
        end_pos_parameter = search_string.find("]")
        while start_pos_parameter < end_pos_parameter:
            # if this is the case both are found and start_pos < end_pos
            parameter = search_string[start_pos_parameter + 1:end_pos_parameter]
            db_field = self.__get_database_field(parameter)
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
            parameter = search_string[brace_start_pos + 1:operator_pos]
            value = search_string[operator_pos + 2:brace_end_pos - 1]  # +/-2 to remove outer quotation marks
            value.replace("'", "")
            if value != "":
                if (brace_start_pos >= 4) and (search_string[brace_start_pos - 4:brace_start_pos - 1] == 'NOT'):
                    queries.append(~Q(**{parameter: value}))
                else:
                    queries.append(Q(**{parameter: value}))
            brace_start_pos = search_string.find("{", brace_end_pos)
            if brace_start_pos > -1:
                and_or_operators.append(search_string[brace_end_pos + 1:brace_start_pos].replace('NOT', '').strip())
            operator_pos = search_string.find("=", brace_end_pos + 1)
            brace_end_pos = search_string.find("}", brace_end_pos + 1)

        if len(queries) == 0:
            return None
        q_object = Q()
        q_object.add(queries[0], "AND")
        # if first three characters of search_string are "NOT" we have to negate q_object
        if len(queries) > 1:
            for query, and_or_operator in zip(queries[1:], and_or_operators):
                # Q.AND is defined as "AND" and Q.OR as "OR", so the text "AND" or "OR" will work
                # but if we gonna localise the human readable string, we should be aware of this
                q_object.add(query, and_or_operator)
        return q_object


def get_advanced_search_options_ct(pid=True):
    """
    Adds all advanced search options for CT to the json-search-options
    :return: json-search-options for modality CT
    """
    from remapp.models import GeneralStudyModuleAttr, PatientModuleAttr, PatientStudyModuleAttr, \
        GeneralEquipmentModuleAttr, CtRadiationDose, CtAccumulatedDoseData, CtIrradiationEventData, \
        CtXRaySourceParameters, ScanningLength, SizeSpecificDoseEstimation, CtDoseCheckDetails, UniqueEquipmentNames

    json_search = ''
    json_search = add_model2json_search(GeneralStudyModuleAttr, json_search)
    json_search = add_model2json_search(PatientModuleAttr, json_search, 'patientmoduleattr__')
    json_search = add_model2json_search(PatientStudyModuleAttr, json_search, 'patientstudymoduleattr__', pid)
    json_search = add_model2json_search(GeneralEquipmentModuleAttr, json_search, 'generalequipmentmoduleattr__')
    json_search = add_model2json_search(CtRadiationDose, json_search, 'ctradiationdose__')
    json_search = add_model2json_search(CtAccumulatedDoseData, json_search, 'ctradiationdose__ctaccumulateddosedata__')
    json_search = add_model2json_search(CtIrradiationEventData, json_search,
                                        'ctradiationdose__ctirradiationeventdata__')
    json_search = add_model2json_search(CtXRaySourceParameters, json_search,
                                        'ctradiationdose__ctirradiationeventdata__ctxraysourceparameters__')
    json_search = add_model2json_search(ScanningLength, json_search,
                                        'ctradiationdose__ctirradiationeventdata__scanninglength__')
    json_search = add_model2json_search(SizeSpecificDoseEstimation, json_search,
                                        'ctradiationdose__ctirradiationeventdata__sizespecificdoseestimation__')
    json_search = add_model2json_search(CtDoseCheckDetails, json_search,
                                        'ctradiationdose__ctirradiationeventdata__ctdosecheckdetails__')
    json_search = '[' + json_search[:-1] + ']'

    return json_search


def get_advanced_search_options_2dplane(pid=True):
    """
    Adds all advanced search options for DX to the json-search-options
    :return: json-search-options for modality DX
    """
    from remapp.models import GeneralStudyModuleAttr, PatientModuleAttr, PatientStudyModuleAttr, \
        GeneralEquipmentModuleAttr, ProjectionXRayRadiationDose, IrradEventXRayData, ImageViewModifier, \
        IrradEventXRayData, IrradEventXRaySourceData, XrayGrid, Kvp, XrayTubeCurrent, Exposure, XrayFilters, \
        IrradEventXRayMechanicalData, DoseRelatedDistanceMeasurements

    json_search = ''
    json_search = add_model2json_search(GeneralStudyModuleAttr, json_search)
    json_search = add_model2json_search(PatientModuleAttr, json_search, 'patientmoduleattr__')
    json_search = add_model2json_search(PatientStudyModuleAttr, json_search, 'patientstudymoduleattr__', pid)
    json_search = add_model2json_search(GeneralEquipmentModuleAttr, json_search, 'generalequipmentmoduleattr__')
    json_search = add_model2json_search(ProjectionXRayRadiationDose, json_search, 'projectionxrayradiationdose__')
    json_search = add_model2json_search(IrradEventXRayData, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__')
    json_search = add_model2json_search(ImageViewModifier, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__imageviewmodifier__')
    json_search = add_model2json_search(IrradEventXRaySourceData, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__')
    json_search = add_model2json_search(XrayGrid, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata_'
                                        'xraygrid__')
    json_search = add_model2json_search(Kvp, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__'
                                        'irradeventxrayxourcedata__kvp__')
    json_search = add_model2json_search(XrayTubeCurrent, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__'
                                        'irradeventxrayxourcedata__xraytubecurrent__')
    json_search = add_model2json_search(Exposure, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__'
                                        'irradeventxrayxourcedata__exposure__')
    json_search = add_model2json_search(XrayFilters, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__'
                                        'irradeventxrayxourcedata__xrayfilters__')
    json_search = add_model2json_search(IrradEventXRayMechanicalData, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__'
                                        'irradeventxraymechanicaldata__')
    json_search = add_model2json_search(DoseRelatedDistanceMeasurements, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__'
                                        'irradeventxraymechanicaldata__doserelateddistancemeasurements')

    return json_search


def get_advanced_search_options_dx(pid=True):
    from remapp.models import AccumXRayDose, AccumProjXRayDose, AccumCassetteBsdProjRadiogDose, \
        AccumIntegratedProjRadiogDose

    json_search = get_advanced_search_options_2dplane(pid)
    json_search = add_model2json_search(AccumXRayDose, json_search,
                                        'projectionxrayradiationdose__accumxraydose__')
    json_search = add_model2json_search(AccumProjXRayDose, json_search,
                                        'projectionxrayradiationdose__accumxraydose__accumprojxraydose')
    json_search = add_model2json_search(AccumCassetteBsdProjRadiogDose, json_search,
                                        'projectionxrayradiationdose__accumxraydose__accumcassettebsdprojradiogdose')
    json_search = add_model2json_search(AccumIntegratedProjRadiogDose, json_search,
                                        'projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose')

    return json_search


def get_advanced_search_options_rf(pid=True):
    from remapp.models import PulseWidth

    json_search = get_advanced_search_options_dx(pid)  # is this correct?
    json_search = add_model2json_search(PulseWidth, json_search,
                                        'projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata_'
                                        'pulsewidth__')
    return json_search


def get_advanced_search_options_mg(pid=True):
    from remapp.models import AccumMammographyXRayDose

    json_search = get_advanced_search_options_2dplane(pid)  # is this correct?
    json_search = add_model2json_search(AccumMammographyXRayDose, json_search,
                                        'projectionxrayradiationdose__accumxraydose__accummammographyxraydose__')
    return json_search
