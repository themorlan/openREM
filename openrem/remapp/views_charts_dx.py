from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from remapp.models import create_user_profile
import logging

logger = logging.getLogger(__name__)


def generate_required_dx_charts_list(profile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    if (
        profile.plotDXAcquisitionMeanDAPOverTime
        or profile.plotDXAcquisitionMeanmAsOverTime
        or profile.plotDXAcquisitionMeankVpOverTime
        or profile.plotDXAcquisitionMeankVpOverTime
    ):
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

    if profile.plotDXStudyPerDayAndHour:
        required_charts.append(
            {
                "title": "Chart of study description workload",
                "var_name": "studyWorkload",
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

    if profile.plotDXAcquisitionDAPvsMass:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol DAP vs patient mass",
                "var_name": "acquisitionDAPvsMass",
            }
        )
    if profile.plotDXStudyDAPvsMass:
        required_charts.append(
            {
                "title": "Chart of study description DAP vs patient mass",
                "var_name": "studyDAPvsMass",
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
    from remapp.interface.mod_filters import dx_acq_filter
    from openremproject import settings
    from django.http import JsonResponse
    from datetime import datetime

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
    """Calculations for radiographic charts"""
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
        construct_scatter_chart,
        construct_over_time_charts,
    )

    if (
        user_profile.plotDXAcquisitionMeanDAPOverTime
        or user_profile.plotDXAcquisitionMeankVpOverTime
        or user_profile.plotDXAcquisitionMeanmAsOverTime
    ):
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
    if (
        user_profile.plotDXAcquisitionMeanDAP
        or user_profile.plotDXAcquisitionFreq
        or user_profile.plotDXAcquisitionMeankVp
        or user_profile.plotDXAcquisitionMeanmAs
        or user_profile.plotDXAcquisitionMeankVpOverTime
        or user_profile.plotDXAcquisitionMeanmAsOverTime
        or user_profile.plotDXAcquisitionMeanDAPOverTime
        or user_profile.plotDXAcquisitionDAPvsMass
    ):

        name_fields = [
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
        ]

        value_fields = []
        value_multipliers = []
        if (
            user_profile.plotDXAcquisitionMeanDAP
            or user_profile.plotDXAcquisitionMeanDAPOverTime
            or user_profile.plotDXAcquisitionDAPvsMass
        ):
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

        date_fields = []
        if (
            user_profile.plotDXAcquisitionMeanDAPOverTime
            or user_profile.plotDXAcquisitionMeankVpOverTime
            or user_profile.plotDXAcquisitionMeanmAsOverTime
        ):
            date_fields.append("study_date")

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
            system_name_field=system_field,
            data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
            data_point_value_multipliers=value_multipliers,
            uid="projectionxrayradiationdose__irradeventxraydata__pk",
        )
        #######################################################################
        sorted_acquisition_dap_categories = None
        if user_profile.plotDXAcquisitionMeanDAP:
            sorted_acquisition_dap_categories = create_sorted_category_list(
                df,
                "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["acquisitionMeanDAPData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol DAP mean",
                        sorted_category_list=sorted_acquisition_dap_categories,
                        average_choice="mean",
                        return_as_dict=return_as_dict,
                    )

                if user_profile.plotMedian:
                    return_structure["acquisitionMedianDAPData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol DAP median",
                        sorted_category_list=sorted_acquisition_dap_categories,
                        average_choice="median",
                        return_as_dict=return_as_dict,
                    )

            if user_profile.plotBoxplots:
                return_structure["acquisitionBoxplotDAPData"] = plotly_boxplot(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Acquisition protocol",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol DAP boxplot",
                    sorted_category_list=sorted_acquisition_dap_categories,
                    return_as_dict=return_as_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_dap_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_dap_categories.values())[0]

                return_structure[
                    "acquisitionHistogramDAPData"
                ] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol DAP histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                    return_as_dict=return_as_dict,
                )

        sorted_acquisition_kvp_categories = None
        if user_profile.plotDXAcquisitionMeankVp:
            sorted_acquisition_kvp_categories = create_sorted_category_list(
                df,
                "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["acquisitionMeankVpData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Mean kVp",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol kVp mean",
                        sorted_category_list=sorted_acquisition_kvp_categories,
                        average_choice="mean",
                        return_as_dict=return_as_dict,
                    )

                if user_profile.plotMedian:
                    return_structure["acquisitionMediankVpData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Median kVp",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol kVp median",
                        sorted_category_list=sorted_acquisition_kvp_categories,
                        average_choice="median",
                        return_as_dict=return_as_dict,
                    )

            if user_profile.plotBoxplots:
                return_structure["acquisitionBoxplotkVpData"] = plotly_boxplot(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                    value_axis_title="kVp",
                    name_axis_title="Acquisition protocol",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol kVp boxplot",
                    sorted_category_list=sorted_acquisition_kvp_categories,
                    return_as_dict=return_as_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_kvp_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_kvp_categories.values())[0]

                return_structure[
                    "acquisitionHistogramkVpData"
                ] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                    value_axis_title="kVp",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol kVp histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                    return_as_dict=return_as_dict,
                )

        sorted_acquisition_mas_categories = None
        if user_profile.plotDXAcquisitionMeanmAs:
            sorted_acquisition_mas_categories = create_sorted_category_list(
                df,
                "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["acquisitionMeanmAsData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Mean mAs",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol mAs mean",
                        sorted_category_list=sorted_acquisition_mas_categories,
                        average_choice="mean",
                        return_as_dict=return_as_dict,
                    )

                if user_profile.plotMedian:
                    return_structure["acquisitionMedianmAsData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Median mAs",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol mAs median",
                        sorted_category_list=sorted_acquisition_mas_categories,
                        average_choice="median",
                        return_as_dict=return_as_dict,
                    )

            if user_profile.plotBoxplots:
                return_structure["acquisitionBoxplotmAsData"] = plotly_boxplot(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                    value_axis_title="mAs",
                    name_axis_title="Acquisition protocol",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol mAs boxplot",
                    sorted_category_list=sorted_acquisition_mas_categories,
                    return_as_dict=return_as_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_mas_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_mas_categories.values())[0]

                return_structure[
                    "acquisitionHistogrammAsData"
                ] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                    value_axis_title="mAs",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol mAs histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                    return_as_dict=return_as_dict,
                )

        if user_profile.plotDXAcquisitionFreq:
            sorted_categories = None
            if user_profile.plotDXAcquisitionMeanDAP:
                sorted_categories = sorted_acquisition_dap_categories
            elif user_profile.plotDXAcquisitionMeankVp:
                sorted_categories = sorted_acquisition_kvp_categories
            elif user_profile.plotDXAcquisitionMeanmAs:
                sorted_categories = sorted_acquisition_mas_categories

            return_structure["acquisitionFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                sorting_choice=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                legend_title="Acquisition protocol",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM DX acquisition protocol frequency",
                sorted_categories=sorted_categories,
                return_as_dict=return_as_dict,
            )

        if user_profile.plotDXAcquisitionMeanDAPOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_value_col="projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                df_date_col="study_date",
                name_title="Acquisition protocol",
                value_title="DAP (cGy.cm<sup>2</sup>)",
                date_title="Study date",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX acquisition protocol DAP over time",
                return_as_dict=return_as_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanDAPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianDAPOverTime"] = result["median"]

        if user_profile.plotDXAcquisitionMeankVpOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_value_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                df_date_col="study_date",
                name_title="Acquisition protocol",
                value_title="kVp",
                date_title="Study date",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX acquisition protocol kVp over time",
                return_as_dict=return_as_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeankVpOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMediankVpOverTime"] = result["median"]

        if user_profile.plotDXAcquisitionMeanmAsOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_value_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                df_date_col="study_date",
                name_title="Acquisition protocol",
                value_title="mAs",
                date_title="Study date",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX acquisition protocol mAs over time",
                return_as_dict=return_as_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanmAsOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianmAsOverTime"] = result["median"]

        if user_profile.plotDXAcquisitionDAPvsMass:
            return_structure["acquisitionDAPvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                x_axis_title="Patient mass (kg)",
                y_axis_title="DAP (mGy.cm<sup>2</sub>)",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Acquisition protocol",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX acquisition protocol DAP vs patient mass",
                return_as_dict=return_as_dict,
            )

    #######################################################################
    # Prepare study- and request-level Pandas DataFrame to use for charts
    if (
        user_profile.plotDXStudyMeanDAP
        or user_profile.plotDXStudyFreq
        or user_profile.plotDXStudyPerDayAndHour
        or user_profile.plotDXStudyDAPvsMass
        or user_profile.plotDXRequestMeanDAP
        or user_profile.plotDXRequestFreq
        or user_profile.plotDXRequestDAPvsMass
    ):

        name_fields = []
        if (
            user_profile.plotDXStudyMeanDAP
            or user_profile.plotDXStudyFreq
            or user_profile.plotDXStudyPerDayAndHour
            or user_profile.plotDXStudyDAPvsMass
        ):
            name_fields.append("study_description")
        if (
            user_profile.plotDXRequestMeanDAP
            or user_profile.plotDXRequestFreq
            or user_profile.plotDXRequestDAPvsMass
        ):
            name_fields.append("requested_procedure_code_meaning")

        value_fields = []
        value_multipliers = []
        if (
            user_profile.plotDXStudyMeanDAP
            or user_profile.plotDXRequestMeanDAP
            or user_profile.plotDXStudyDAPvsMass
            or user_profile.plotDXRequestDAPvsMass
        ):
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

        sorted_study_dap_categories = None
        if user_profile.plotDXStudyMeanDAP:
            sorted_study_dap_categories = create_sorted_category_list(
                df,
                "study_description",
                "total_dap",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "study_description",
                    "total_dap",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["studyMeanDAPData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX Study description DAP mean",
                        sorted_category_list=sorted_study_dap_categories,
                        average_choice="mean",
                        return_as_dict=return_as_dict,
                    )

                if user_profile.plotMedian:
                    return_structure["studyMedianDAPData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX Study description DAP median",
                        sorted_category_list=sorted_study_dap_categories,
                        average_choice="median",
                        return_as_dict=return_as_dict,
                    )

            if user_profile.plotBoxplots:
                return_structure["studyBoxplotDAPData"] = plotly_boxplot(
                    df,
                    "study_description",
                    "total_dap",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Study description",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX study description DAP boxplot",
                    sorted_category_list=sorted_study_dap_categories,
                    return_as_dict=return_as_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = "study_description"
                group_by_col = "x_ray_system_name"
                legend_title = "Study description"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_study_dap_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "study_description"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_study_dap_categories.values())[0]

                return_structure["studyHistogramDAPData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "total_dap",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX study description DAP histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                    return_as_dict=return_as_dict,
                )

        if user_profile.plotDXStudyFreq:
            return_structure["studyFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="study_description",
                sorting_choice=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                legend_title="Study description",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM DX study description frequency",
                sorted_categories=sorted_study_dap_categories,
                return_as_dict=return_as_dict,
            )

        sorted_request_dap_categories = None
        if user_profile.plotDXRequestMeanDAP:
            sorted_request_dap_categories = create_sorted_category_list(
                df,
                "requested_procedure_code_meaning",
                "total_dap",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dap",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["requestMeanDAPData"] = plotly_barchart(
                        df_aggregated,
                        "requested_procedure_code_meaning",
                        value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Requested procedure",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX requested procedure DAP mean",
                        sorted_category_list=sorted_request_dap_categories,
                        average_choice="mean",
                        return_as_dict=return_as_dict,
                    )

                if user_profile.plotMedian:
                    return_structure["requestMedianDAPData"] = plotly_barchart(
                        df_aggregated,
                        "requested_procedure_code_meaning",
                        value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Requested procedure",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX requested procedure DAP median",
                        sorted_category_list=sorted_request_dap_categories,
                        average_choice="median",
                        return_as_dict=return_as_dict,
                    )

            if user_profile.plotBoxplots:
                return_structure["requestBoxplotDAPData"] = plotly_boxplot(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dap",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Requested procedure",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX requested procedure DAP boxplot",
                    sorted_category_list=sorted_request_dap_categories,
                    return_as_dict=return_as_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = "requested_procedure_code_meaning"
                group_by_col = "x_ray_system_name"
                legend_title = "Requested procedure"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_request_dap_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "requested_procedure_code_meaning"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_request_dap_categories.values())[0]

                return_structure["requestHistogramDAPData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "total_dap",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX requested procedure DAP histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                    return_as_dict=return_as_dict,
                )

        if user_profile.plotDXRequestFreq:
            return_structure["requestFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="requested_procedure_code_meaning",
                sorting_choice=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                legend_title="Requested procedure",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM DX requested procedure frequency",
                sorted_categories=sorted_request_dap_categories,
                return_as_dict=return_as_dict,
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
                return_as_dict=return_as_dict,
            )

        if user_profile.plotDXStudyDAPvsMass:
            return_structure["studyDAPvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="study_description",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="total_dap",
                x_axis_title="Patient mass (kg)",
                y_axis_title="DAP (mGy.cm<sup>2</sub>)",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Study description",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX study description DAP vs patient mass",
                return_as_dict=return_as_dict,
            )

        if user_profile.plotDXRequestDAPvsMass:
            return_structure["requestDAPvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="requested_procedure_code_meaning",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="total_dap",
                x_axis_title="Patient mass (kg)",
                y_axis_title="DAP (mGy.cm<sup>2</sub>)",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotDXInitialSortingChoice,
                ],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Requested procedure",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX requested procedure DAP vs patient mass",
                return_as_dict=return_as_dict,
            )

    return return_structure


def dx_chart_form_processing(request, user_profile):
    from remapp.forms import DXChartOptionsForm

    # Obtain the chart options from the request
    chart_options_form = DXChartOptionsForm(request.GET)
    # check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.plotCharts = chart_options_form.cleaned_data["plotCharts"]
            user_profile.plotDXAcquisitionMeanDAP = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeanDAP"
            ]
            user_profile.plotDXAcquisitionFreq = chart_options_form.cleaned_data[
                "plotDXAcquisitionFreq"
            ]
            user_profile.plotDXStudyMeanDAP = chart_options_form.cleaned_data[
                "plotDXStudyMeanDAP"
            ]
            user_profile.plotDXStudyFreq = chart_options_form.cleaned_data[
                "plotDXStudyFreq"
            ]
            user_profile.plotDXRequestMeanDAP = chart_options_form.cleaned_data[
                "plotDXRequestMeanDAP"
            ]
            user_profile.plotDXRequestFreq = chart_options_form.cleaned_data[
                "plotDXRequestFreq"
            ]
            user_profile.plotDXAcquisitionMeankVp = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeankVp"
            ]
            user_profile.plotDXAcquisitionMeanmAs = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeanmAs"
            ]
            user_profile.plotDXStudyPerDayAndHour = chart_options_form.cleaned_data[
                "plotDXStudyPerDayAndHour"
            ]
            user_profile.plotDXAcquisitionMeankVpOverTime = (
                chart_options_form.cleaned_data["plotDXAcquisitionMeankVpOverTime"]
            )
            user_profile.plotDXAcquisitionMeanmAsOverTime = (
                chart_options_form.cleaned_data["plotDXAcquisitionMeanmAsOverTime"]
            )
            user_profile.plotDXAcquisitionMeanDAPOverTime = (
                chart_options_form.cleaned_data["plotDXAcquisitionMeanDAPOverTime"]
            )
            user_profile.plotDXAcquisitionMeanDAPOverTimePeriod = (
                chart_options_form.cleaned_data[
                    "plotDXAcquisitionMeanDAPOverTimePeriod"
                ]
            )
            user_profile.plotDXAcquisitionDAPvsMass = chart_options_form.cleaned_data[
                "plotDXAcquisitionDAPvsMass"
            ]
            user_profile.plotDXStudyDAPvsMass = chart_options_form.cleaned_data[
                "plotDXStudyDAPvsMass"
            ]
            user_profile.plotDXRequestDAPvsMass = chart_options_form.cleaned_data[
                "plotDXRequestDAPvsMass"
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
            user_profile.plotDXInitialSortingChoice = chart_options_form.cleaned_data[
                "plotDXInitialSortingChoice"
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

        # If submit was not clicked then use the settings already stored in the user's profile
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
                "plotDXAcquisitionMeanDAP": user_profile.plotDXAcquisitionMeanDAP,
                "plotDXAcquisitionFreq": user_profile.plotDXAcquisitionFreq,
                "plotDXStudyMeanDAP": user_profile.plotDXStudyMeanDAP,
                "plotDXStudyFreq": user_profile.plotDXStudyFreq,
                "plotDXRequestMeanDAP": user_profile.plotDXRequestMeanDAP,
                "plotDXRequestFreq": user_profile.plotDXRequestFreq,
                "plotDXAcquisitionMeankVp": user_profile.plotDXAcquisitionMeankVp,
                "plotDXAcquisitionMeanmAs": user_profile.plotDXAcquisitionMeanmAs,
                "plotDXStudyPerDayAndHour": user_profile.plotDXStudyPerDayAndHour,
                "plotDXAcquisitionMeankVpOverTime": user_profile.plotDXAcquisitionMeankVpOverTime,
                "plotDXAcquisitionMeanmAsOverTime": user_profile.plotDXAcquisitionMeanmAsOverTime,
                "plotDXAcquisitionMeanDAPOverTime": user_profile.plotDXAcquisitionMeanDAPOverTime,
                "plotDXAcquisitionMeanDAPOverTimePeriod": user_profile.plotDXAcquisitionMeanDAPOverTimePeriod,
                "plotDXAcquisitionDAPvsMass": user_profile.plotDXAcquisitionDAPvsMass,
                "plotDXStudyDAPvsMass": user_profile.plotDXStudyDAPvsMass,
                "plotDXRequestDAPvsMass": user_profile.plotDXRequestDAPvsMass,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotDXInitialSortingChoice": user_profile.plotDXInitialSortingChoice,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices,
            }
            chart_options_form = DXChartOptionsForm(form_data)
    return chart_options_form
