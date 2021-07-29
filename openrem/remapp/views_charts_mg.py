# pylint: disable=too-many-lines
import logging
from datetime import datetime
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import JsonResponse
from remapp.forms import MGChartOptionsForm
from remapp.interface.mod_filters import MGSummaryListFilter, MGFilterPlusPid
from remapp.models import GeneralStudyModuleAttr, create_user_profile
from remapp.views_admin import (
    required_average_choices,
    initialise_mg_form_data,
    set_average_chart_options,
    set_mg_chart_options,
    set_common_chart_options,
)
from .interface.chart_functions import (
    create_dataframe,
    create_dataframe_weekdays,
    plotly_barchart_weekdays,
    plotly_binned_statistic_barchart,
    plotly_boxplot,
    plotly_barchart,
    plotly_histogram_barchart,
    plotly_scatter,
    plotly_frequency_barchart,
    construct_over_time_charts,
    plotly_set_default_theme,
    create_dataframe_aggregates,
)

logger = logging.getLogger(__name__)


def generate_required_mg_charts_list(profile):
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    if profile.plotMGAcquisitionAGDOverTime:
        keys = list(dict(profile.TIME_PERIOD).keys())
        values = list(dict(profile.TIME_PERIOD).values())
        time_period = (values[keys.index(profile.plotMGOverTimePeriod)]).lower()

    if profile.plotMGacquisitionFreq:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol frequency",
                "var_name": "acquisitionFrequency",
            }
        )

    if profile.plotMGaverageAGD:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol mean AGD",
                    "var_name": "acquisitionMeanAGD",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol median AGD",
                    "var_name": "acquisitionMedianAGD",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of acquisition protocol AGD",
                    "var_name": "acquisitionBoxplotAGD",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of acquisition protocol AGD",
                    "var_name": "acquisitionHistogramAGD",
                }
            )

    if profile.plotMGAGDvsThickness:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol AGD vs compressed breast thickness",
                "var_name": "acquisitionScatterAGDvsThick",
            }
        )

    if profile.plotMGaverageAGDvsThickness:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol mean AGD vs compressed breast thickness",
                    "var_name": "acquisitionMeanAGDvsThick",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol median AGD vs compressed breast thickness",
                    "var_name": "acquisitionMedianAGDvsThick",
                }
            )

    if profile.plotMGAcquisitionAGDOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol mean AGD over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMeanAGDOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol median AGD over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMedianAGDOverTime",
                }
            )

    if profile.plotMGkVpvsThickness:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol kVp vs compressed breast thickness",
                "var_name": "acquisitionScatterkVpvsThick",
            }
        )

    if profile.plotMGmAsvsThickness:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol mAs vs compressed breast thickness",
                "var_name": "acquisitionScattermAsvsThick",
            }
        )

    if profile.plotMGStudyPerDayAndHour:
        required_charts.append(
            {
                "title": "Chart of study description workload",
                "var_name": "studyWorkload",
            }
        )

    return required_charts


@login_required
def mg_summary_chart_data(request):
    """Obtain data for mammography chart data Ajax view"""
    if request.user.groups.filter(name="pidgroup"):
        f = MGFilterPlusPid(
            request.GET,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="MG")
            .order_by()
            .distinct(),
        )
    else:
        f = MGSummaryListFilter(
            request.GET,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="MG")
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
        start_time = datetime.now()

    return_structure = mg_plot_calculations(f, user_profile)

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def mg_plot_calculations(f, user_profile, return_as_dict=False):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """Calculations for mammography charts"""
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

    sorting_choice = user_profile.plotMGInitialSortingChoice.lower()
    ascending_order = True
    if user_profile.plotInitialSortingDirection == 0:
        ascending_order = False

    if user_profile.plotMGAcquisitionAGDOverTime:
        plot_timeunit_period = user_profile.plotMGOverTimePeriod

    #######################################################################
    # Prepare acquisition-level Pandas DataFrame to use for charts
    charts_of_interest = [
        user_profile.plotMGAGDvsThickness,
        user_profile.plotMGkVpvsThickness,
        user_profile.plotMGmAsvsThickness,
        user_profile.plotMGaverageAGDvsThickness,
        user_profile.plotMGaverageAGD,
        user_profile.plotMGacquisitionFreq,
        user_profile.plotMGAcquisitionAGDOverTime,
    ]
    if any(charts_of_interest):  # pylint: disable=too-many-nested-blocks

        name_fields = [
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
        ]

        value_fields = []
        value_multipliers = []
        charts_of_interest = [
            user_profile.plotMGAGDvsThickness,
            user_profile.plotMGaverageAGDvsThickness,
            user_profile.plotMGaverageAGD,
            user_profile.plotMGAcquisitionAGDOverTime,
        ]
        if any(charts_of_interest):
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose"
            )
            value_multipliers.append(1)
        if user_profile.plotMGkVpvsThickness:
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp"
            )
            value_multipliers.append(1)
        if user_profile.plotMGmAsvsThickness:
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure"
            )
            value_multipliers.append(0.001)
        charts_of_interest = [
            user_profile.plotMGAGDvsThickness,
            user_profile.plotMGkVpvsThickness,
            user_profile.plotMGmAsvsThickness,
            user_profile.plotMGaverageAGDvsThickness,
        ]
        if any(charts_of_interest):
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness"
            )
            value_multipliers.append(1)

        date_fields = []
        if user_profile.plotMGAcquisitionAGDOverTime:
            date_fields.append("study_date")

        time_fields = []

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

        sorted_acquisition_agd_categories = None
        if user_profile.plotMGaverageAGDvsThickness or user_profile.plotMGaverageAGD:

            if user_profile.plotBoxplots and "median" not in average_choices:
                average_choices = average_choices + ["median"]

            name_field = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
            value_field = "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose"  # pylint: disable=line-too-long

            df_aggregated = create_dataframe_aggregates(  # pylint: disable=line-too-long
                df,
                [name_field],
                value_field,  # pylint: disable=line-too-long
                stats_to_use=average_choices + ["count"],
            )

            if sorting_choice == "name":
                sorted_acquisition_agd_categories = {name_field: (df_aggregated.sort_values(by=name_field, ascending=ascending_order)[name_field]).unique().tolist()}  # pylint: disable=line-too-long
            elif sorting_choice == "frequency":
                sorted_acquisition_agd_categories = {name_field: (df_aggregated.sort_values(by="count", ascending=ascending_order)[name_field]).unique().tolist()}  # pylint: disable=line-too-long

            if user_profile.plotMGaverageAGDvsThickness:
                category_names_col = name_field
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = name_field
                    legend_title = "System"

                parameter_dict = {  # pylint: disable=line-too-long
                    "df_x_value_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",  # pylint: disable=line-too-long
                    "df_y_value_col": value_field,  # pylint: disable=line-too-long
                    "x_axis_title": "Compressed breast thickness (mm)",
                    "y_axis_title": "AGD (mGy)",
                    "df_category_col": category_names_col,
                    "df_facet_col": group_by_col,
                    "facet_title": legend_title,
                    "user_bins": [20, 30, 40, 50, 60, 70, 80, 90],
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol average AGD vs thickness",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                if user_profile.plotMean:
                    if sorting_choice == "value":
                        sorted_acquisition_agd_categories = {name_field: (df_aggregated.sort_values(by="mean", ascending=ascending_order)[name_field]).unique().tolist()}  # pylint: disable=line-too-long

                    if user_profile.plotGroupingChoice == "series":
                        parameter_dict["df_category_name_list"] = list(df[category_names_col].unique())
                        parameter_dict["df_facet_category_list"] = list(sorted_acquisition_agd_categories.values())[0]
                    else:
                        parameter_dict["df_category_name_list"] = list(sorted_acquisition_agd_categories.values())[0]
                        parameter_dict["df_facet_category_list"] = list(df[group_by_col].unique())

                    parameter_dict["stat_name"] = "mean"
                    return_structure[
                        "meanAGDvsThickness"
                    ] = plotly_binned_statistic_barchart(
                        df,
                        parameter_dict,
                    )

                if user_profile.plotMedian:
                    if sorting_choice == "value":
                        sorted_acquisition_agd_categories = {name_field: (df_aggregated.sort_values(by="median", ascending=ascending_order)[name_field]).unique().tolist()}  # pylint: disable=line-too-long

                    if user_profile.plotGroupingChoice == "series":
                        parameter_dict["df_category_name_list"] = list(df[category_names_col].unique())
                        parameter_dict["df_facet_category_list"] = list(sorted_acquisition_agd_categories.values())[0]
                    else:
                        parameter_dict["df_category_name_list"] = list(sorted_acquisition_agd_categories.values())[0]
                        parameter_dict["df_facet_category_list"] = list(df[group_by_col].unique())

                    parameter_dict["stat_name"] = "median"
                    return_structure[
                        "medianAGDvsThickness"
                    ] = plotly_binned_statistic_barchart(
                        df,
                        parameter_dict,
                    )

            if user_profile.plotMGaverageAGD:
                if user_profile.plotMean or user_profile.plotMedian:

                    parameter_dict = {
                        "df_name_col": name_field,
                        "name_axis_title": "Acquisition protocol",
                        "colourmap": user_profile.plotColourMapChoice,
                        "facet_col": None,
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "return_as_dict": return_as_dict,
                    }
                    if user_profile.plotMean:
                        if sorting_choice == "value":
                            sorted_acquisition_agd_categories = {name_field: (df_aggregated.sort_values(by="mean", ascending=ascending_order)[name_field]).unique().tolist()}  # pylint: disable=line-too-long

                        parameter_dict["sorted_category_list"] = sorted_acquisition_agd_categories
                        parameter_dict["value_axis_title"] = "Mean AGD (mGy)"
                        parameter_dict[
                            "filename"
                        ] = "OpenREM MG acquisition protocol AGD mean"
                        parameter_dict["average_choice"] = "mean"
                        (
                            return_structure["acquisitionMeanAGDData"],
                            return_structure["acquisitionMeanAGDDataCSV"],
                        ) = plotly_barchart(  # pylint: disable=line-too-long
                            df_aggregated,
                            parameter_dict,
                            "acquisitionMeanAGDData.csv",
                        )

                    if user_profile.plotMedian:
                        if sorting_choice == "value":
                            sorted_acquisition_agd_categories = {name_field: (df_aggregated.sort_values(by="median", ascending=ascending_order)[name_field]).unique().tolist()}  # pylint: disable=line-too-long

                        parameter_dict["sorted_category_list"] = sorted_acquisition_agd_categories
                        parameter_dict["value_axis_title"] = "Median AGD (mGy)"
                        parameter_dict[
                            "filename"
                        ] = "OpenREM MG acquisition protocol AGD median"
                        parameter_dict["average_choice"] = "median"
                        (
                            return_structure["acquisitionMedianAGDData"],
                            return_structure["acquisitionMedianAGDDataCSV"],
                        ) = plotly_barchart(  # pylint: disable=line-too-long
                            df_aggregated,
                            parameter_dict,
                            "acquisitionMedianAGDData.csv",
                        )

                if user_profile.plotBoxplots:
                    parameter_dict = {  # pylint: disable=line-too-long
                        "df_name_col": name_field,
                        "df_value_col": value_field,  # pylint: disable=line-too-long
                        "value_axis_title": "AGD (mGy)",
                        "name_axis_title": "Acquisition protocol",
                        "colourmap": user_profile.plotColourMapChoice,
                        "filename": "OpenREM MG acquisition protocol AGD boxplot",
                        "facet_col": None,
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "return_as_dict": return_as_dict,
                    }

                    if sorting_choice == "value":
                        sorted_acquisition_agd_categories = {name_field: (df_aggregated.sort_values(by="median", ascending=ascending_order)[name_field]).unique().tolist()}  # pylint: disable=line-too-long

                    parameter_dict["sorted_category_list"] = sorted_acquisition_agd_categories

                    return_structure["acquisitionBoxplotAGDData"] = plotly_boxplot(
                        df,
                        parameter_dict,
                    )

                if user_profile.plotHistograms:
                    category_names_col = name_field
                    group_by_col = "x_ray_system_name"
                    legend_title = "Acquisition protocol"
                    facet_names = list(df[group_by_col].unique())
                    category_names = list(sorted_acquisition_agd_categories.values())[0]

                    if user_profile.plotGroupingChoice == "series":
                        category_names_col = "x_ray_system_name"
                        group_by_col = name_field
                        legend_title = "System"
                        category_names = facet_names
                        facet_names = list(sorted_acquisition_agd_categories.values())[0]

                    parameter_dict = {  # pylint: disable=line-too-long
                        "df_facet_col": group_by_col,
                        "df_category_col": category_names_col,
                        "df_value_col": value_field,  # pylint: disable=line-too-long
                        "value_axis_title": "AGD (mGy)",
                        "legend_title": legend_title,
                        "n_bins": user_profile.plotHistogramBins,
                        "colourmap": user_profile.plotColourMapChoice,
                        "filename": "OpenREM MG acquisition protocol AGD histogram",
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "df_facet_category_list": facet_names,
                        "df_category_name_list": category_names,
                        "global_max_min": user_profile.plotHistogramGlobalBins,
                        "return_as_dict": return_as_dict,
                    }
                    return_structure[
                        "acquisitionHistogramAGDData"
                    ] = plotly_histogram_barchart(
                        df,
                        parameter_dict,
                    )

        if user_profile.plotMGAGDvsThickness:
            parameter_dict = {  # pylint: disable=line-too-long
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "df_x_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",  # pylint: disable=line-too-long
                "df_y_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",  # pylint: disable=line-too-long
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                "grouping_choice": user_profile.plotGroupingChoice,
                "legend_title": "Acquisition protocol",
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "x_axis_title": "Compressed breast thickness (mm)",
                "y_axis_title": "AGD (mGy)",
                "filename": "OpenREM CT acquisition protocol AGD vs thickness",
                "return_as_dict": return_as_dict,
            }
            return_structure[
                "AGDvsThickness"
            ] = plotly_scatter(  # pylint: disable=line-too-long
                df,
                parameter_dict,
            )

        if user_profile.plotMGkVpvsThickness:
            parameter_dict = {  # pylint: disable=line-too-long
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "df_x_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",  # pylint: disable=line-too-long
                "df_y_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",  # pylint: disable=line-too-long
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                "grouping_choice": user_profile.plotGroupingChoice,
                "legend_title": "Acquisition protocol",
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "x_axis_title": "Compressed breast thickness (mm)",
                "y_axis_title": "kVp",
                "filename": "OpenREM CT acquisition protocol kVp vs thickness",
                "return_as_dict": return_as_dict,
            }
            return_structure[
                "kVpvsThickness"
            ] = plotly_scatter(  # pylint: disable=line-too-long
                df,
                parameter_dict,
            )

        if user_profile.plotMGmAsvsThickness:
            parameter_dict = {  # pylint: disable=line-too-long
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "df_x_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",  # pylint: disable=line-too-long
                "df_y_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",  # pylint: disable=line-too-long
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                "grouping_choice": user_profile.plotGroupingChoice,
                "legend_title": "Acquisition protocol",
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "x_axis_title": "Compressed breast thickness (mm)",
                "y_axis_title": "mAs",
                "filename": "OpenREM CT acquisition protocol mAs vs thickness",
                "return_as_dict": return_as_dict,
            }
            return_structure[
                "mAsvsThickness"
            ] = plotly_scatter(  # pylint: disable=line-too-long
                df,
                parameter_dict,
            )

        if user_profile.plotMGacquisitionFreq:
            parameter_dict = {
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                "legend_title": "Acquisition",
                "df_x_axis_col": "x_ray_system_name",
                "x_axis_title": "System",
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "filename": "OpenREM MG acquisition protocol frequency",
                "groupby_cols": None,
                "facet_col": None,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "return_as_dict": return_as_dict,
            }
            (
                return_structure["acquisitionFrequencyData"],
                return_structure["acquisitionFrequencyDataCSV"],
            ) = plotly_frequency_barchart(  # pylint: disable=line-too-long
                df, parameter_dict, csv_name="acquisitionFrequencyData.csv"
            )

        if user_profile.plotMGAcquisitionAGDOverTime:
            facet_title = "System"

            if user_profile.plotGroupingChoice == "series":
                facet_title = "Acquisition protocol"

            parameter_dict = {  # pylint: disable=line-too-long
                "df_name_col": "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "df_value_col": "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",  # pylint: disable=line-too-long
                "df_date_col": "study_date",
                "name_title": "Acquisition protocol",
                "value_title": "AGD (mGy)",
                "date_title": "Study date",
                "facet_title": facet_title,
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                "time_period": plot_timeunit_period,
                "average_choices": average_choices + ["count"],
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "filename": "OpenREM MG acquisition protocol AGD over time",
                "return_as_dict": return_as_dict,
            }
            result = construct_over_time_charts(
                df,
                parameter_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanAGDOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianAGDOverTime"] = result["median"]

    #######################################################################
    # Prepare study- and request-level Pandas DataFrame to use for charts
    if user_profile.plotMGStudyPerDayAndHour:

        name_fields = ["study_description"]

        value_fields = []

        date_fields = []
        time_fields = []
        if user_profile.plotMGStudyPerDayAndHour:
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
            uid="pk",
        )

        if user_profile.plotMGStudyPerDayAndHour:
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
                filename="OpenREM CT study description workload",
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                return_as_dict=return_as_dict,
            )

    return return_structure


def mg_chart_form_processing(request, user_profile):
    # pylint: disable=too-many-statements
    # Obtain the chart options from the request
    chart_options_form = MGChartOptionsForm(request.GET)
    # Check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required

            set_common_chart_options(chart_options_form, user_profile)

            set_average_chart_options(chart_options_form, user_profile)

            set_mg_chart_options(chart_options_form, user_profile)

            user_profile.save()

        else:
            average_choices = required_average_choices(user_profile)

            mg_form_data = initialise_mg_form_data(user_profile)

            form_data = {
                "plotCharts": user_profile.plotCharts,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices,
            }

            form_data = {**form_data, **mg_form_data}

            chart_options_form = MGChartOptionsForm(form_data)
    return chart_options_form
