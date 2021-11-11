# pylint: disable=too-many-lines
import logging
from datetime import datetime
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import JsonResponse
from remapp.forms import DXChartOptionsForm
from remapp.interface.mod_filters import dx_acq_filter
from remapp.models import create_user_profile
from remapp.views_admin import (
    set_average_chart_options,
    required_average_choices,
    initialise_dx_form_data,
    set_dx_chart_options,
    set_common_chart_options,
)
from .interface.chart_functions import (
    create_dataframe,
    create_dataframe_weekdays,
    create_dataframe_aggregates,
    plotly_boxplot,
    plotly_barchart,
    plotly_histogram_barchart,
    plotly_barchart_weekdays,
    plotly_set_default_theme,
    plotly_frequency_barchart,
    plotly_scatter,
    construct_over_time_charts,
    generate_average_chart_group,
)

logger = logging.getLogger(__name__)


def generate_required_dx_charts_list(profile):
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    charts_of_interest = [
        profile.plotDXAcquisitionMeanDAPOverTime,
        profile.plotDXAcquisitionMeanmAsOverTime,
        profile.plotDXAcquisitionMeankVpOverTime,
        profile.plotDXAcquisitionMeankVpOverTime,
    ]
    if any(charts_of_interest):
        keys = list(dict(profile.TIME_PERIOD).keys())
        values = list(dict(profile.TIME_PERIOD).values())
        time_period = (
            values[keys.index(profile.plotDXAcquisitionMeanDAPOverTimePeriod)]
        ).lower()

    if profile.plotDXAcquisitionMeanDAP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DAP for each acquisition protocol",
                    "var_name": "acquisitionMeanDAP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DAP for each acquisition protocol",
                    "var_name": "acquisitionMedianDAP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of DAP for each acquisition protocol",
                    "var_name": "acquisitionBoxplotDAP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of DAP for each acquisition protocol",
                    "var_name": "acquisitionHistogramDAP",
                }
            )

    if profile.plotDXAcquisitionFreq:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol frequency",
                "var_name": "acquisitionFrequency",
            }
        )

    if profile.plotDXAcquisitionMeankVp:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean kVp for each acquisition protocol",
                    "var_name": "acquisitionMeankVp",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median kVp for each acquisition protocol",
                    "var_name": "acquisitionMediankVp",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of kVp for each acquisition protocol",
                    "var_name": "acquisitionBoxplotkVp",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of kVp for each acquisition protocol",
                    "var_name": "acquisitionHistogramkVp",
                }
            )

    if profile.plotDXAcquisitionMeanmAs:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean mAs for each acquisition protocol",
                    "var_name": "acquisitionMeanmAs",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median mAs for each acquisition protocol",
                    "var_name": "acquisitionMedianmAs",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of mAs for each acquisition protocol",
                    "var_name": "acquisitionBoxplotmAs",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of mAs for each acquisition protocol",
                    "var_name": "acquisitionHistogrammAs",
                }
            )

    if profile.plotDXAcquisitionMeanDAPOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DAP per acquisition protocol over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMeanDAPOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DAP per acquisition protocol over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMedianDAPOverTime",
                }
            )

    if profile.plotDXAcquisitionMeankVpOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean kVp per acquisition protocol over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMeankVpOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median kVp per acquisition protocol over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMediankVpOverTime",
                }
            )

    if profile.plotDXAcquisitionMeanmAsOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean mAs per acquisition protocol over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMeanmAsOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median mAs per acquisition protocol over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMedianmAsOverTime",
                }
            )

    if profile.plotDXAcquisitionDAPvsMass:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol DAP vs patient mass",
                "var_name": "acquisitionDAPvsMass",
            }
        )

    if profile.plotDXStudyMeanDAP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DAP for each study description",
                    "var_name": "studyMeanDAP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DAP for each study description",
                    "var_name": "studyMedianDAP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of DAP for each study description",
                    "var_name": "studyBoxplotDAP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of DAP for each study description",
                    "var_name": "studyHistogramDAP",
                }
            )

    if profile.plotDXStudyFreq:
        required_charts.append(
            {
                "title": "Chart of study description frequency",
                "var_name": "studyFrequency",
            }
        )

    if profile.plotDXStudyPerDayAndHour:
        required_charts.append(
            {
                "title": "Chart of study description workload",
                "var_name": "studyWorkload",
            }
        )

    if profile.plotDXStudyDAPvsMass:
        required_charts.append(
            {
                "title": "Chart of study description DAP vs patient mass",
                "var_name": "studyDAPvsMass",
            }
        )

    if profile.plotDXRequestMeanDAP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DAP for each requested procedure",
                    "var_name": "requestMeanDAP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DAP for each requested procedure",
                    "var_name": "requestMedianDAP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of DAP for each requested procedure",
                    "var_name": "requestBoxplotDAP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of DAP for each requested procedure",
                    "var_name": "requestHistogramDAP",
                }
            )

    if profile.plotDXRequestFreq:
        required_charts.append(
            {
                "title": "Chart of requested procedure frequency",
                "var_name": "requestFrequency",
            }
        )

    if profile.plotDXRequestDAPvsMass:
        required_charts.append(
            {
                "title": "Chart of requested procedure DAP vs patient mass",
                "var_name": "requestDAPvsMass",
            }
        )

    return required_charts


@login_required
def dx_summary_chart_data(request):
    """Obtain data for Ajax chart call"""
    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = dx_acq_filter(request.GET, pid=pid)

    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile

    if settings.DEBUG:
        start_time = datetime.now()

    return_structure = dx_plot_calculations(f, user_profile)

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def dx_plot_calculations(f, user_profile, return_as_dict=False):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """Calculations for radiographic charts"""
    # Return an empty structure if the queryset is empty
    if not f.qs:
        return {}

    charts_of_interest = [
        user_profile.plotDXAcquisitionMeanDAPOverTime,
        user_profile.plotDXAcquisitionMeankVpOverTime,
        user_profile.plotDXAcquisitionMeanmAsOverTime,
    ]
    if any(charts_of_interest):
        plot_timeunit_period = user_profile.plotDXAcquisitionMeanDAPOverTimePeriod

    # Set the Plotly chart theme
    plotly_set_default_theme(user_profile.plotThemeChoice)

    return_structure = {}

    average_choices = []
    if user_profile.plotMean:
        average_choices.append("mean")
    if user_profile.plotMedian:
        average_choices.append("median")

    #######################################################################
    # Prepare acquisition-level Pandas DataFrame to use for charts
    charts_of_interest = [
        user_profile.plotDXAcquisitionMeanDAP,
        user_profile.plotDXAcquisitionFreq,
        user_profile.plotDXAcquisitionMeankVp,
        user_profile.plotDXAcquisitionMeanmAs,
        user_profile.plotDXAcquisitionMeankVpOverTime,
        user_profile.plotDXAcquisitionMeanmAsOverTime,
        user_profile.plotDXAcquisitionMeanDAPOverTime,
        user_profile.plotDXAcquisitionDAPvsMass,
    ]
    if any(charts_of_interest):

        name_fields = [
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
        ]

        value_fields = []
        value_multipliers = []
        charts_of_interest = [
            user_profile.plotDXAcquisitionMeanDAP,
            user_profile.plotDXAcquisitionMeanDAPOverTime,
            user_profile.plotDXAcquisitionDAPvsMass,
        ]
        if any(charts_of_interest):
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__dose_area_product"
            )
            value_multipliers.append(1000000)
        if (
            user_profile.plotDXAcquisitionMeankVp
            or user_profile.plotDXAcquisitionMeankVpOverTime
        ):
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp"
            )
            value_multipliers.append(1)
        if (
            user_profile.plotDXAcquisitionMeanmAs
            or user_profile.plotDXAcquisitionMeanmAsOverTime
        ):
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure"
            )
            value_multipliers.append(0.001)
        if user_profile.plotDXAcquisitionDAPvsMass:
            value_fields.append("patientstudymoduleattr__patient_weight")
            value_multipliers.append(1)

        time_fields = []
        date_fields = []
        charts_of_interest = [
            user_profile.plotDXAcquisitionMeanDAPOverTime,
            user_profile.plotDXAcquisitionMeankVpOverTime,
            user_profile.plotDXAcquisitionMeanmAsOverTime,
        ]
        if any(charts_of_interest):
            date_fields.append("study_date")

        system_field = []
        if user_profile.plotSeriesPerSystem:
            system_field.append(
                "generalequipmentmoduleattr__unique_equipment_name_id__display_name"
            )

        fields = {
            "names": name_fields,
            "values": value_fields,
            "dates": date_fields,
            "times": time_fields,
            "system": system_field,
        }
        df = create_dataframe(
            f.qs,
            fields,
            data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
            data_point_name_remove_whitespace_padding=user_profile.plotRemoveCategoryWhitespacePadding,
            data_point_value_multipliers=value_multipliers,
            char_wrap=user_profile.plotLabelCharWrap,
            uid="projectionxrayradiationdose__irradeventxraydata__pk",
        )
        #######################################################################

        if user_profile.plotDXAcquisitionMeanDAP:

            name_field = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
            value_field = "projectionxrayradiationdose__irradeventxraydata__dose_area_product"
            value_text = "DAP"
            units_text = "(cGy.cm<sup>2</sup>)"
            name_text = "Acquisition protocol"
            variable_name_start = "acquisition"
            variable_value_name = "DAP"
            modality_text = "DX"
            chart_message = ""

            new_charts = generate_average_chart_group(average_choices, chart_message, df, modality_text,
                                                      name_field, name_text, return_as_dict, return_structure,
                                                      units_text, user_profile, value_field, value_text,
                                                      variable_name_start, variable_value_name,
                                                      user_profile.plotDXInitialSortingChoice)

            return_structure = {**return_structure, **new_charts}

        if user_profile.plotDXAcquisitionMeankVp:

            name_field = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
            value_field = "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp"
            value_text = "kVp"
            units_text = ""
            name_text = "Acquisition protocol"
            variable_name_start = "acquisition"
            variable_value_name = "kVp"
            modality_text = "DX"
            chart_message = ""

            new_charts = generate_average_chart_group(average_choices, chart_message, df, modality_text,
                                                      name_field, name_text, return_as_dict, return_structure,
                                                      units_text, user_profile, value_field, value_text,
                                                      variable_name_start, variable_value_name,
                                                      user_profile.plotDXInitialSortingChoice)

            return_structure = {**return_structure, **new_charts}

        if user_profile.plotDXAcquisitionMeanmAs:

            name_field = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
            value_field = "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure"
            value_text = "mAs"
            units_text = ""
            name_text = "Acquisition protocol"
            variable_name_start = "acquisition"
            variable_value_name = "mAs"
            modality_text = "DX"
            chart_message = ""

            new_charts = generate_average_chart_group(average_choices, chart_message, df, modality_text,
                                                      name_field, name_text, return_as_dict, return_structure,
                                                      units_text, user_profile, value_field, value_text,
                                                      variable_name_start, variable_value_name,
                                                      user_profile.plotDXInitialSortingChoice)

            return_structure = {**return_structure, **new_charts}

        if user_profile.plotDXAcquisitionFreq:
            parameter_dict = {
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                "legend_title": "Acquisition protocol",
                "df_x_axis_col": "x_ray_system_name",
                "x_axis_title": "System",
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "filename": "OpenREM DX acquisition protocol frequency",
                "groupby_cols": None,
                "facet_col": None,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "return_as_dict": return_as_dict,
            }
            (
                return_structure["acquisitionFrequencyData"],
                return_structure["acquisitionFrequencyDataCSV"],
            ) = plotly_frequency_barchart(  # pylint: disable=line-too-long
                df,
                parameter_dict,
                csv_name="acquisitionFrequencyData.csv",
            )

        if user_profile.plotDXAcquisitionMeanDAPOverTime:
            facet_title = "System"

            if user_profile.plotGroupingChoice == "series":
                facet_title = "Acquisition protocol"

            parameter_dict = {
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "df_value_col": "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                "df_date_col": "study_date",
                "name_title": "Acquisition protocol",
                "value_title": "DAP (cGy.cm<sup>2</sup>)",
                "date_title": "Study date",
                "facet_title": facet_title,
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                "time_period": plot_timeunit_period,
                "average_choices": average_choices + ["count"],
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "filename": "OpenREM DX acquisition protocol DAP over time",
                "return_as_dict": return_as_dict,
            }
            result = construct_over_time_charts(
                df,
                parameter_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanDAPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianDAPOverTime"] = result["median"]

        if user_profile.plotDXAcquisitionMeankVpOverTime:
            facet_title = "System"

            if user_profile.plotGroupingChoice == "series":
                facet_title = "Acquisition protocol"

            parameter_dict = {
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "df_value_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                "df_date_col": "study_date",
                "name_title": "Acquisition protocol",
                "value_title": "kVp",
                "date_title": "Study date",
                "facet_title": facet_title,
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                "time_period": plot_timeunit_period,
                "average_choices": average_choices + ["count"],
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "filename": "OpenREM DX acquisition protocol kVp over time",
                "return_as_dict": return_as_dict,
            }
            result = construct_over_time_charts(
                df,
                parameter_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeankVpOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMediankVpOverTime"] = result["median"]

        if user_profile.plotDXAcquisitionMeanmAsOverTime:
            facet_title = "System"

            if user_profile.plotGroupingChoice == "series":
                facet_title = "Acquisition protocol"

            parameter_dict = {  # pylint: disable=line-too-long
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "df_value_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",  # pylint: disable=line-too-long
                "df_date_col": "study_date",
                "name_title": "Acquisition protocol",
                "value_title": "mAs",
                "date_title": "Study date",
                "facet_title": facet_title,
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                "time_period": plot_timeunit_period,
                "average_choices": average_choices + ["count"],
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "filename": "OpenREM DX acquisition protocol mAs over time",
                "return_as_dict": return_as_dict,
            }
            result = construct_over_time_charts(
                df,
                parameter_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanmAsOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianmAsOverTime"] = result["median"]

        if user_profile.plotDXAcquisitionDAPvsMass:
            parameter_dict = {
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "df_x_col": "patientstudymoduleattr__patient_weight",
                "df_y_col": "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                "grouping_choice": user_profile.plotGroupingChoice,
                "legend_title": "Acquisition protocol",
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "x_axis_title": "Patient mass (kg)",
                "y_axis_title": "DAP (mGy.cm<sup>2</sub>)",
                "filename": "OpenREM DX acquisition protocol DAP vs patient mass",
                "return_as_dict": return_as_dict,
            }
            return_structure["acquisitionDAPvsMass"] = plotly_scatter(
                df,
                parameter_dict,
            )

    #######################################################################
    # Prepare study- and request-level Pandas DataFrame to use for charts
    charts_of_interest = [
        user_profile.plotDXStudyMeanDAP,
        user_profile.plotDXStudyFreq,
        user_profile.plotDXStudyPerDayAndHour,
        user_profile.plotDXStudyDAPvsMass,
        user_profile.plotDXRequestMeanDAP,
        user_profile.plotDXRequestFreq,
        user_profile.plotDXRequestDAPvsMass,
    ]
    if any(charts_of_interest):

        name_fields = []
        charts_of_interest = [
            user_profile.plotDXStudyMeanDAP,
            user_profile.plotDXStudyFreq,
            user_profile.plotDXStudyPerDayAndHour,
            user_profile.plotDXStudyDAPvsMass,
        ]
        if any(charts_of_interest):
            name_fields.append("study_description")

        charts_of_interest = [
            user_profile.plotDXRequestMeanDAP,
            user_profile.plotDXRequestFreq,
            user_profile.plotDXRequestDAPvsMass,
        ]
        if any(charts_of_interest):
            name_fields.append("requested_procedure_code_meaning")

        value_fields = []
        value_multipliers = []
        charts_of_interest = [
            user_profile.plotDXStudyMeanDAP,
            user_profile.plotDXRequestMeanDAP,
            user_profile.plotDXStudyDAPvsMass,
            user_profile.plotDXRequestDAPvsMass,
        ]
        if any(charts_of_interest):
            value_fields.append("total_dap")
            value_multipliers.append(1000000)
        if user_profile.plotDXStudyDAPvsMass or user_profile.plotDXRequestDAPvsMass:
            value_fields.append("patientstudymoduleattr__patient_weight")
            value_multipliers.append(1)

        date_fields = []
        time_fields = []
        if user_profile.plotDXStudyPerDayAndHour:
            date_fields.append("study_date")
            time_fields.append("study_time")

        system_field = []
        if user_profile.plotSeriesPerSystem:
            system_field.append(
                "generalequipmentmoduleattr__unique_equipment_name_id__display_name"
            )

        fields = {
            "names": name_fields,
            "values": value_fields,
            "dates": date_fields,
            "times": time_fields,
            "system": system_field,
        }
        df = create_dataframe(
            f.qs,
            fields,
            data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
            data_point_name_remove_whitespace_padding=user_profile.plotRemoveCategoryWhitespacePadding,
            data_point_value_multipliers=value_multipliers,
            char_wrap=user_profile.plotLabelCharWrap,
            uid="pk",
        )
        #######################################################################

        if user_profile.plotDXStudyMeanDAP:

            name_field = "study_description"
            value_field = "total_dap"
            value_text = "DAP"
            units_text = "(cGy.cm<sup>2</sup>)"
            name_text = "Study description"
            variable_name_start = "study"
            variable_value_name = "DAP"
            modality_text = "DX"
            chart_message = ""

            new_charts = generate_average_chart_group(average_choices, chart_message, df, modality_text,
                                                      name_field, name_text, return_as_dict, return_structure,
                                                      units_text, user_profile, value_field, value_text,
                                                      variable_name_start, variable_value_name,
                                                      user_profile.plotDXInitialSortingChoice)

            return_structure = {**return_structure, **new_charts}

        if user_profile.plotDXStudyFreq:
            parameter_dict = {
                "df_name_col": "study_description",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                "legend_title": "Study description",
                "df_x_axis_col": "x_ray_system_name",
                "x_axis_title": "System",
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "filename": "OpenREM DX study description frequency",
                "groupby_cols": None,
                "facet_col": None,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "return_as_dict": return_as_dict,
            }
            (
                return_structure["studyFrequencyData"],
                return_structure["studyFrequencyDataCSV"],
            ) = plotly_frequency_barchart(  # pylint: disable=line-too-long
                df,
                parameter_dict,
                csv_name="studyFrequencyData.csv",
            )

        if user_profile.plotDXRequestMeanDAP:

            name_field = "requested_procedure_code_meaning"
            value_field = "total_dap"
            value_text = "DAP"
            units_text = "(cGy.cm<sup>2</sup>)"
            name_text = "Requested procedure"
            variable_name_start = "request"
            variable_value_name = "DAP"
            modality_text = "DX"
            chart_message = ""

            new_charts = generate_average_chart_group(average_choices, chart_message, df, modality_text,
                                                      name_field, name_text, return_as_dict, return_structure,
                                                      units_text, user_profile, value_field, value_text,
                                                      variable_name_start, variable_value_name,
                                                      user_profile.plotDXInitialSortingChoice)

            return_structure = {**return_structure, **new_charts}

        if user_profile.plotDXRequestFreq:
            parameter_dict = {
                "df_name_col": "requested_procedure_code_meaning",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                "legend_title": "Requested procedure",
                "df_x_axis_col": "x_ray_system_name",
                "x_axis_title": "System",
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "filename": "OpenREM DX requested procedure frequency",
                "groupby_cols": None,
                "facet_col": None,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "return_as_dict": return_as_dict,
            }
            (
                return_structure["requestFrequencyData"],
                return_structure["requestFrequencyDataCSV"],
            ) = plotly_frequency_barchart(  # pylint: disable=line-too-long
                df,
                parameter_dict,
                csv_name="requestFrequencyData.csv",
            )

        if user_profile.plotDXStudyPerDayAndHour:
            df_time_series_per_weekday = create_dataframe_weekdays(
                df, "study_description", df_date_col="study_date"
            )

            return_structure["studyWorkloadData"] = plotly_barchart_weekdays(
                df_time_series_per_weekday,
                "weekday",
                "study_description",
                name_axis_title="Weekday",
                value_axis_title="Frequency",
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM DX study description workload",
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                sorting_choice=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                return_as_dict=return_as_dict,
            )

        if user_profile.plotDXStudyDAPvsMass:
            parameter_dict = {
                "df_name_col": "study_description",
                "df_x_col": "patientstudymoduleattr__patient_weight",
                "df_y_col": "total_dap",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                "grouping_choice": user_profile.plotGroupingChoice,
                "legend_title": "Study description",
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "x_axis_title": "Patient mass (kg)",
                "y_axis_title": "DAP (mGy.cm<sup>2</sub>)",
                "filename": "OpenREM DX study description DAP vs patient mass",
                "return_as_dict": return_as_dict,
            }
            return_structure["studyDAPvsMass"] = plotly_scatter(
                df,
                parameter_dict,
            )

        if user_profile.plotDXRequestDAPvsMass:
            parameter_dict = {
                "df_name_col": "requested_procedure_code_meaning",
                "df_x_col": "patientstudymoduleattr__patient_weight",
                "df_y_col": "total_dap",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                "grouping_choice": user_profile.plotGroupingChoice,
                "legend_title": "Requested procedure",
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "x_axis_title": "Patient mass (kg)",
                "y_axis_title": "DAP (mGy.cm<sup>2</sub>)",
                "filename": "OpenREM DX requested procedure DAP vs patient mass",
                "return_as_dict": return_as_dict,
            }
            return_structure["requestDAPvsMass"] = plotly_scatter(
                df,
                parameter_dict,
            )

    return return_structure


def dx_chart_form_processing(request, user_profile):
    # pylint: disable=too-many-statements
    # Obtain the chart options from the request
    chart_options_form = DXChartOptionsForm(request.GET)
    # check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required

            set_common_chart_options(chart_options_form, user_profile)

            set_average_chart_options(chart_options_form, user_profile)

            set_dx_chart_options(chart_options_form, user_profile)

            user_profile.save()

        # If submit was not clicked then use the settings already stored in the user's profile
        else:
            average_choices = required_average_choices(user_profile)

            dx_form_data = initialise_dx_form_data(user_profile)

            form_data = {
                "plotCharts": user_profile.plotCharts,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices,
            }

            form_data = {**form_data, **dx_form_data}

            chart_options_form = DXChartOptionsForm(form_data)
    return chart_options_form
