from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from remapp.models import GeneralStudyModuleAttr, create_user_profile
import logging

logger = logging.getLogger(__name__)


def generate_required_rf_charts_list(profile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    if profile.plotRFStudyDAPOverTime or profile.plotRFRequestDAPOverTime:
        keys = list(dict(profile.TIME_PERIOD).keys())
        values = list(dict(profile.TIME_PERIOD).values())
        time_period = (values[keys.index(profile.plotRFOverTimePeriod)]).lower()

    if profile.plotRFStudyDAP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of study description mean DAP",
                    "var_name": "studyMeanDAP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of study description median DAP",
                    "var_name": "studyMedianDAP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of study description DAP",
                    "var_name": "studyBoxplotDAP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of study description DAP",
                    "var_name": "studyHistogramDAP",
                }
            )

    if profile.plotRFRequestDAP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of requested procedure mean DAP",
                    "var_name": "requestMeanDAP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of requested procedure median DAP",
                    "var_name": "requestMedianDAP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of requested procedure DAP",
                    "var_name": "requestBoxplotDAP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of requested procedure DAP",
                    "var_name": "requestHistogramDAP",
                }
            )

    if profile.plotRFStudyFreq:
        required_charts.append(
            {
                "title": "Chart of study description frequency",
                "var_name": "studyFrequency",
            }
        )

    if profile.plotRFRequestFreq:
        required_charts.append(
            {
                "title": "Chart of requested procedure frequency",
                "var_name": "requestFrequency",
            }
        )

    if profile.plotRFStudyPerDayAndHour:
        required_charts.append(
            {
                "title": "Chart of study description workload",
                "var_name": "studyWorkload",
            }
        )

    if profile.plotRFStudyDAPOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of study description mean DAP over time ("
                    + time_period
                    + ")",
                    "var_name": "studyMeanDAPOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of study description median DAP over time ("
                    + time_period
                    + ")",
                    "var_name": "studyMedianDAPOverTime",
                }
            )

    if profile.plotRFRequestDAPOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of requested procedure mean DAP over time ("
                    + time_period
                    + ")",
                    "var_name": "requestMeanDAPOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of requested procedure median DAP over time ("
                    + time_period
                    + ")",
                    "var_name": "requestMedianDAPOverTime",
                }
            )

    return required_charts


@login_required
def rf_summary_chart_data(request):
    """Obtain data for Ajax chart call"""
    from remapp.interface.mod_filters import RFSummaryListFilter, RFFilterPlusPid
    from openremproject import settings
    from django.http import JsonResponse

    if request.user.groups.filter(name="pidgroup"):
        f = RFFilterPlusPid(
            request.GET,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="RF")
            .order_by()
            .distinct(),
        )
    else:
        f = RFSummaryListFilter(
            request.GET,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="RF")
            .order_by()
            .distinct(),
        )

    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile

    if settings.DEBUG:
        from datetime import datetime

        start_time = datetime.now()

    return_structure = rf_plot_calculations(f, user_profile)

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def rf_plot_calculations(f, user_profile, return_as_dict=False):
    """Calculations for fluoroscopy charts"""
    from .interface.chart_functions import (
        create_dataframe,
        create_dataframe_weekdays,
        create_dataframe_aggregates,
        create_sorted_category_list,
        plotly_boxplot,
        plotly_barchart,
        plotly_histogram_barchart,
        plotly_barchart_weekdays,
        plotly_set_default_theme,
        construct_frequency_chart,
        construct_over_time_charts,
    )

    # Return an empty structure if the queryset is empty
    if not f.qs:
        return {}

    # Set the Plotly chart theme
    plotly_set_default_theme(user_profile.plotThemeChoice)

    return_structure = {}

    average_choices = []
    if user_profile.plotMean:
        average_choices.append("mean")
    if user_profile.plotMedian:
        average_choices.append("median")

    if user_profile.plotRFStudyDAPOverTime or user_profile.plotRFRequestDAPOverTime:
        plot_timeunit_period = user_profile.plotRFOverTimePeriod

    #######################################################################
    # Prepare Pandas DataFrame to use for charts
    name_fields = []
    if (
        user_profile.plotRFStudyFreq
        or user_profile.plotRFStudyDAP
        or user_profile.plotRFStudyPerDayAndHour
        or user_profile.plotRFStudyDAPOverTime
    ):
        name_fields.append("study_description")
    if user_profile.plotRFSplitByPhysician:
        name_fields.append("performing_physician_name")
    if (
        user_profile.plotRFRequestFreq
        or user_profile.plotRFRequestDAP
        or user_profile.plotRFRequestDAPOverTime
    ):
        name_fields.append("requested_procedure_code_meaning")

    value_fields = []
    value_multipliers = []
    if (
        user_profile.plotRFStudyDAP
        or user_profile.plotRFRequestDAP
        or user_profile.plotRFStudyDAPOverTime
        or user_profile.plotRFRequestDAPOverTime
    ):
        value_fields.append("total_dap")
        value_multipliers.append(1000000)

    date_fields = []
    if (
        user_profile.plotRFStudyPerDayAndHour
        or user_profile.plotRFStudyDAPOverTime
        or user_profile.plotRFRequestDAPOverTime
    ):
        date_fields.append("study_date")

    time_fields = []
    if user_profile.plotRFStudyPerDayAndHour:
        time_fields.append("study_time")

    system_field = None
    if user_profile.plotSeriesPerSystem:
        system_field = (
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name"
        )

    df = create_dataframe(
        f.qs,
        data_point_name_fields=name_fields,
        data_point_value_fields=value_fields,
        data_point_date_fields=date_fields,
        data_point_time_fields=time_fields,
        system_name_field=system_field,
        data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
        data_point_value_multipliers=value_multipliers,
        uid="pk",
    )
    #######################################################################

    if user_profile.plotRFStudyPerDayAndHour:
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
            filename="OpenREM RF study description workload",
            facet_col_wrap=user_profile.plotFacetColWrapVal,
            return_as_dict=return_as_dict,
        )

    stats_to_include = ["count"]
    if user_profile.plotMean:
        stats_to_include.append("mean")
    if user_profile.plotMedian:
        stats_to_include.append("median")

    sorted_study_categories = None
    if user_profile.plotRFStudyDAP:
        sorted_study_categories = create_sorted_category_list(
            df,
            "study_description",
            "total_dap",
            [
                user_profile.plotInitialSortingDirection,
                user_profile.plotRFInitialSortingChoice,
            ],
        )

        if user_profile.plotMean or user_profile.plotMedian:

            x_col = "study_description"
            groupby_cols = ["study_description"]
            facet_col = None

            if user_profile.plotRFSplitByPhysician:
                groupby_cols = groupby_cols + ["performing_physician_name"]
                x_col = "performing_physician_name"
                facet_col = "study_description"

            df_aggregated = create_dataframe_aggregates(
                df, groupby_cols, "total_dap", stats=stats_to_include
            )

            if user_profile.plotMean:
                return_structure["studyMeanData"] = plotly_barchart(
                    df_aggregated,
                    x_col,
                    value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Study description",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM RF study description DAP mean",
                    sorted_category_list=sorted_study_categories,
                    average_choice="mean",
                    facet_col=facet_col,
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    return_as_dict=return_as_dict,
                )

            if user_profile.plotMedian:
                return_structure["studyMedianData"] = plotly_barchart(
                    df_aggregated,
                    x_col,
                    value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Study description",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM RF study description DAP median",
                    sorted_category_list=sorted_study_categories,
                    average_choice="median",
                    facet_col=facet_col,
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    return_as_dict=return_as_dict,
                )

        if user_profile.plotBoxplots:
            x_col = "study_description"
            facet_col = None

            if user_profile.plotRFSplitByPhysician:
                x_col = "performing_physician_name"
                facet_col = "study_description"

            return_structure["studyBoxplotData"] = plotly_boxplot(
                df,
                x_col,
                "total_dap",
                facet_col=facet_col,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                name_axis_title="Study description",
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF study description DAP boxplot",
                sorted_category_list=sorted_study_categories,
                return_as_dict=return_as_dict,
            )

        if user_profile.plotHistograms:
            category_names_col = "study_description"
            group_by_col = "x_ray_system_name"
            legend_title = "Study description"
            facet_names = list(df[group_by_col].unique())
            category_names = list(sorted_study_categories.values())[0]

            if user_profile.plotGroupingChoice == "series":
                category_names_col = "x_ray_system_name"
                group_by_col = "study_description"
                legend_title = "System"
                category_names = facet_names
                facet_names = list(sorted_study_categories.values())[0]

            return_structure["studyHistogramData"] = plotly_histogram_barchart(
                df,
                group_by_col,
                category_names_col,
                "total_dap",
                value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                legend_title=legend_title,
                n_bins=user_profile.plotHistogramBins,
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF study description DAP histogram",
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                df_facet_category_list=facet_names,
                df_category_name_list=category_names,
                global_max_min=user_profile.plotHistogramGlobalBins,
                return_as_dict=return_as_dict,
            )

    if user_profile.plotRFStudyFreq:
        x_col = "study_description"
        groupby_cols = ["study_description"]
        facet_col = None

        if user_profile.plotRFSplitByPhysician:
            groupby_cols = groupby_cols + ["performing_physician_name"]
            x_col = "performing_physician_name"
            facet_col = "study_description"

        return_structure["studyFrequencyData"] = construct_frequency_chart(
            df=df,
            df_name_col=x_col,
            groupby_cols=groupby_cols,
            facet_col=facet_col,
            facet_col_wrap=user_profile.plotFacetColWrapVal,
            sorting_choice=[
                user_profile.plotInitialSortingDirection,
                user_profile.plotRFInitialSortingChoice,
            ],
            legend_title="Study description",
            df_x_axis_col="x_ray_system_name",
            x_axis_title="System",
            grouping_choice=user_profile.plotGroupingChoice,
            colour_map=user_profile.plotColourMapChoice,
            file_name="OpenREM RF study description frequency",
            sorted_categories=sorted_study_categories,
            return_as_dict=return_as_dict,
        )

    sorted_request_categories = None
    if user_profile.plotRFRequestDAP:
        sorted_request_categories = create_sorted_category_list(
            df,
            "requested_procedure_code_meaning",
            "total_dap",
            [
                user_profile.plotInitialSortingDirection,
                user_profile.plotRFInitialSortingChoice,
            ],
        )

        if user_profile.plotMean or user_profile.plotMedian:

            x_col = "requested_procedure_code_meaning"
            groupby_cols = ["requested_procedure_code_meaning"]
            facet_col = None

            if user_profile.plotRFSplitByPhysician:
                groupby_cols = groupby_cols + ["performing_physician_name"]
                x_col = "performing_physician_name"
                facet_col = "requested_procedure_code_meaning"

            df_aggregated = create_dataframe_aggregates(
                df, groupby_cols, "total_dap", stats=stats_to_include
            )

            if user_profile.plotMean:
                return_structure["requestMeanData"] = plotly_barchart(
                    df_aggregated,
                    x_col,
                    value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Requested procedure",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM RF requested procedure DAP mean",
                    sorted_category_list=sorted_request_categories,
                    average_choice="mean",
                    facet_col=facet_col,
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    return_as_dict=return_as_dict,
                )

            if user_profile.plotMedian:
                return_structure["requestMedianData"] = plotly_barchart(
                    df_aggregated,
                    x_col,
                    value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Requested procedure",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM RF requested procedure DAP median",
                    sorted_category_list=sorted_request_categories,
                    average_choice="median",
                    facet_col=facet_col,
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    return_as_dict=return_as_dict,
                )

        if user_profile.plotBoxplots:
            return_structure["requestBoxplotData"] = plotly_boxplot(
                df,
                "requested_procedure_code_meaning",
                "total_dap",
                value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                name_axis_title="Requested procedure",
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF requested procedure DAP boxplot",
                sorted_category_list=sorted_request_categories,
                return_as_dict=return_as_dict,
            )

        if user_profile.plotHistograms:
            category_names_col = "requested_procedure_code_meaning"
            group_by_col = "x_ray_system_name"
            legend_title = "Requested procedure"
            facet_names = list(df[group_by_col].unique())
            category_names = list(sorted_request_categories.values())[0]

            if user_profile.plotGroupingChoice == "series":
                category_names_col = "x_ray_system_name"
                group_by_col = "requested_procedure_code_meaning"
                legend_title = "System"
                category_names = facet_names
                facet_names = list(sorted_request_categories.values())[0]

            return_structure["requestHistogramData"] = plotly_histogram_barchart(
                df,
                group_by_col,
                category_names_col,
                "total_dap",
                value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                legend_title=legend_title,
                n_bins=user_profile.plotHistogramBins,
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF requested procedure DAP histogram",
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                df_facet_category_list=facet_names,
                df_category_name_list=category_names,
                global_max_min=user_profile.plotHistogramGlobalBins,
                return_as_dict=return_as_dict,
            )

    if user_profile.plotRFRequestFreq:
        x_col = "requested_procedure_code_meaning"
        groupby_cols = ["requested_procedure_code_meaning"]
        facet_col = None

        if user_profile.plotRFSplitByPhysician:
            groupby_cols = groupby_cols + ["performing_physician_name"]
            x_col = "performing_physician_name"
            facet_col = "requested_procedure_code_meaning"

        return_structure["requestFrequencyData"] = construct_frequency_chart(
            df=df,
            df_name_col=x_col,
            groupby_cols=groupby_cols,
            facet_col=facet_col,
            facet_col_wrap=user_profile.plotFacetColWrapVal,
            sorting_choice=[
                user_profile.plotInitialSortingDirection,
                user_profile.plotRFInitialSortingChoice,
            ],
            legend_title="Requested procedure",
            df_x_axis_col="x_ray_system_name",
            x_axis_title="System",
            grouping_choice=user_profile.plotGroupingChoice,
            colour_map=user_profile.plotColourMapChoice,
            file_name="OpenREM RF requested procedure frequency",
            sorted_categories=sorted_request_categories,
            return_as_dict=return_as_dict,
        )

    if user_profile.plotRFStudyDAPOverTime:
        facet_title = "System"

        if user_profile.plotGroupingChoice == "series":
            facet_title = "Study description"

        result = construct_over_time_charts(
            df=df,
            df_name_col="study_description",
            df_value_col="total_dap",
            df_date_col="study_date",
            name_title="Study description",
            value_title="DAP (cGy.cm<sup>2</sup>)",
            date_title="Study date",
            facet_title=facet_title,
            sorting=[
                user_profile.plotInitialSortingDirection,
                user_profile.plotRFInitialSortingChoice,
            ],
            time_period=plot_timeunit_period,
            average_choices=average_choices + ["count"],
            grouping_choice=user_profile.plotGroupingChoice,
            colour_map=user_profile.plotColourMapChoice,
            facet_col_wrap=user_profile.plotFacetColWrapVal,
            file_name="OpenREM RF study DAP over time",
            return_as_dict=return_as_dict,
        )

        if user_profile.plotMean:
            return_structure["studyMeanDAPOverTime"] = result["mean"]
        if user_profile.plotMedian:
            return_structure["studyMedianDAPOverTime"] = result["median"]

    if user_profile.plotRFRequestDAPOverTime:
        facet_title = "System"

        if user_profile.plotGroupingChoice == "series":
            facet_title = "Requested procedure"

        result = construct_over_time_charts(
            df=df,
            df_name_col="requested_procedure_code_meaning",
            df_value_col="total_dap",
            df_date_col="study_date",
            name_title="Requested procedure",
            value_title="DAP (cGy.cm<sup>2</sup>)",
            date_title="Study date",
            facet_title=facet_title,
            sorting=[
                user_profile.plotInitialSortingDirection,
                user_profile.plotRFInitialSortingChoice,
            ],
            time_period=plot_timeunit_period,
            average_choices=average_choices + ["count"],
            grouping_choice=user_profile.plotGroupingChoice,
            colour_map=user_profile.plotColourMapChoice,
            facet_col_wrap=user_profile.plotFacetColWrapVal,
            file_name="OpenREM RF requested procedure DAP over time",
            return_as_dict=return_as_dict,
        )

        if user_profile.plotMean:
            return_structure["requestMeanDAPOverTime"] = result["mean"]
        if user_profile.plotMedian:
            return_structure["requestMedianDAPOverTime"] = result["median"]

    return return_structure


def rf_chart_form_processing(request, user_profile):
    from remapp.forms import RFChartOptionsForm

    # Obtain the chart options from the request
    chart_options_form = RFChartOptionsForm(request.GET)
    # Check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.plotCharts = chart_options_form.cleaned_data["plotCharts"]
            user_profile.plotRFStudyPerDayAndHour = chart_options_form.cleaned_data[
                "plotRFStudyPerDayAndHour"
            ]
            user_profile.plotRFStudyFreq = chart_options_form.cleaned_data[
                "plotRFStudyFreq"
            ]
            user_profile.plotRFStudyDAP = chart_options_form.cleaned_data[
                "plotRFStudyDAP"
            ]
            user_profile.plotRFStudyDAPOverTime = chart_options_form.cleaned_data[
                "plotRFStudyDAPOverTime"
            ]
            user_profile.plotRFRequestFreq = chart_options_form.cleaned_data[
                "plotRFRequestFreq"
            ]
            user_profile.plotRFRequestDAP = chart_options_form.cleaned_data[
                "plotRFRequestDAP"
            ]
            user_profile.plotRFRequestDAPOverTime = chart_options_form.cleaned_data[
                "plotRFRequestDAPOverTime"
            ]
            user_profile.plotRFOverTimePeriod = chart_options_form.cleaned_data[
                "plotRFOverTimePeriod"
            ]
            user_profile.plotRFSplitByPhysician = chart_options_form.cleaned_data[
                "plotRFSplitByPhysician"
            ]
            user_profile.plotGroupingChoice = chart_options_form.cleaned_data[
                "plotGrouping"
            ]
            user_profile.plotSeriesPerSystem = chart_options_form.cleaned_data[
                "plotSeriesPerSystem"
            ]
            user_profile.plotHistograms = chart_options_form.cleaned_data[
                "plotHistograms"
            ]
            user_profile.plotRFInitialSortingChoice = chart_options_form.cleaned_data[
                "plotRFInitialSortingChoice"
            ]
            user_profile.plotInitialSortingDirection = chart_options_form.cleaned_data[
                "plotInitialSortingDirection"
            ]

            if "mean" in chart_options_form.cleaned_data["plotAverageChoice"]:
                user_profile.plotMean = True
            else:
                user_profile.plotMean = False

            if "median" in chart_options_form.cleaned_data["plotAverageChoice"]:
                user_profile.plotMedian = True
            else:
                user_profile.plotMedian = False

            if "boxplot" in chart_options_form.cleaned_data["plotAverageChoice"]:
                user_profile.plotBoxplots = True
            else:
                user_profile.plotBoxplots = False

            user_profile.save()

        else:
            average_choices = []
            if user_profile.plotMean:
                average_choices.append("mean")
            if user_profile.plotMedian:
                average_choices.append("median")
            if user_profile.plotBoxplots:
                average_choices.append("boxplot")

            form_data = {
                "plotCharts": user_profile.plotCharts,
                "plotRFStudyPerDayAndHour": user_profile.plotRFStudyPerDayAndHour,
                "plotRFStudyFreq": user_profile.plotRFStudyFreq,
                "plotRFStudyDAP": user_profile.plotRFStudyDAP,
                "plotRFStudyDAPOverTime": user_profile.plotRFStudyDAPOverTime,
                "plotRFRequestFreq": user_profile.plotRFRequestFreq,
                "plotRFRequestDAP": user_profile.plotRFRequestDAP,
                "plotRFRequestDAPOverTime": user_profile.plotRFRequestDAPOverTime,
                "plotRFOverTimePeriod": user_profile.plotRFOverTimePeriod,
                "plotRFSplitByPhysician": user_profile.plotRFSplitByPhysician,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotRFInitialSortingChoice": user_profile.plotRFInitialSortingChoice,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices,
            }
            chart_options_form = RFChartOptionsForm(form_data)
    return chart_options_form
