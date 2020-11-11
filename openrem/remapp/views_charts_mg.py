import logging
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from remapp.models import GeneralStudyModuleAttr, create_user_profile
from remapp.interface.mod_filters import MGSummaryListFilter, MGFilterPlusPid
from openremproject import settings
from django.http import JsonResponse
from remapp.forms import MGChartOptionsForm
if settings.DEBUG:
    from datetime import datetime
from .interface.chart_functions import (
    create_dataframe,
    create_dataframe_weekdays,
    plotly_barchart_weekdays,
    plotly_binned_statistic_barchart,
    plotly_boxplot,
    plotly_barchart,
    plotly_histogram_barchart,
    construct_scatter_chart,
    construct_frequency_chart,
    construct_over_time_charts,
    plotly_set_default_theme,
    create_sorted_category_list,
    create_dataframe_aggregates,
)

logger = logging.getLogger(__name__)


def generate_required_mg_charts_list(profile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    if profile.plotMGAcquisitionAGDOverTime:
        keys = list(dict(profile.TIME_PERIOD).keys())
        values = list(dict(profile.TIME_PERIOD).values())
        time_period = (values[keys.index(profile.plotMGOverTimePeriod)]).lower()

    if profile.plotMGAGDvsThickness:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol AGD vs compressed breast thickness",
                "var_name": "acquisitionScatterAGDvsThick",
            }
        )

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

    if user_profile.plotMGAcquisitionAGDOverTime:
        plot_timeunit_period = user_profile.plotMGOverTimePeriod

    #######################################################################
    # Prepare acquisition-level Pandas DataFrame to use for charts
    if (
        user_profile.plotMGAGDvsThickness
        or user_profile.plotMGkVpvsThickness
        or user_profile.plotMGmAsvsThickness
        or user_profile.plotMGaverageAGDvsThickness
        or user_profile.plotMGaverageAGD
        or user_profile.plotMGacquisitionFreq
        or user_profile.plotMGAcquisitionAGDOverTime
    ):

        name_fields = [
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
        ]

        value_fields = []
        if (
            user_profile.plotMGAGDvsThickness
            or user_profile.plotMGaverageAGDvsThickness
            or user_profile.plotMGaverageAGD
            or user_profile.plotMGAcquisitionAGDOverTime
        ):
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose"
            )
        if user_profile.plotMGkVpvsThickness:
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp"
            )
        if user_profile.plotMGmAsvsThickness:
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure"
            )
        if (
            user_profile.plotMGAGDvsThickness
            or user_profile.plotMGkVpvsThickness
            or user_profile.plotMGmAsvsThickness
            or user_profile.plotMGaverageAGDvsThickness
        ):
            value_fields.append(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness"
            )

        date_fields = []
        if user_profile.plotMGAcquisitionAGDOverTime:
            date_fields.append("study_date")

        time_fields = []

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
            uid="projectionxrayradiationdose__irradeventxraydata__pk",
        )

        sorted_acquisition_agd_categories = None
        if user_profile.plotMGaverageAGDvsThickness or user_profile.plotMGaverageAGD:
            sorted_acquisition_agd_categories = create_sorted_category_list(
                df,
                "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
            )

            if user_profile.plotMGaverageAGDvsThickness:
                category_names_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_agd_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_agd_categories.values())[0]

                if user_profile.plotMean:
                    return_structure[
                        "meanAGDvsThickness"
                    ] = plotly_binned_statistic_barchart(
                        df,
                        "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
                        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
                        x_axis_title="Compressed breast thickness (mm)",
                        y_axis_title="AGD (mGy)",
                        df_category_col=category_names_col,
                        df_facet_col=group_by_col,
                        facet_title=legend_title,
                        user_bins=[20, 30, 40, 50, 60, 70, 80, 90],
                        colour_map=user_profile.plotColourMapChoice,
                        file_name="OpenREM CT acquisition protocol AGD vs thickness",
                        facet_col_wrap=user_profile.plotFacetColWrapVal,
                        df_facet_category_list=facet_names,
                        df_category_name_list=category_names,
                        stat_name="mean",
                        return_as_dict=return_as_dict,
                    )

                if user_profile.plotMedian:
                    return_structure[
                        "medianAGDvsThickness"
                    ] = plotly_binned_statistic_barchart(
                        df,
                        "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
                        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
                        x_axis_title="Compressed breast thickness (mm)",
                        y_axis_title="AGD (mGy)",
                        df_category_col=category_names_col,
                        df_facet_col=group_by_col,
                        facet_title=legend_title,
                        user_bins=[20, 30, 40, 50, 60, 70, 80, 90],
                        colour_map=user_profile.plotColourMapChoice,
                        file_name="OpenREM CT acquisition protocol AGD vs thickness",
                        facet_col_wrap=user_profile.plotFacetColWrapVal,
                        df_facet_category_list=facet_names,
                        df_category_name_list=category_names,
                        stat_name="median",
                        return_as_dict=return_as_dict,
                    )

            if user_profile.plotMGaverageAGD:
                if user_profile.plotMean or user_profile.plotMedian:
                    df_aggregated = create_dataframe_aggregates(
                        df,
                        [
                            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                        ],
                        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
                        stats=average_choices + ["count"],
                    )

                    if user_profile.plotMean:
                        return_structure["acquisitionMeanAGDData"] = plotly_barchart(
                            df_aggregated,
                            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                            value_axis_title="Mean AGD (mGy)",
                            name_axis_title="Acquisition protocol",
                            colourmap=user_profile.plotColourMapChoice,
                            filename="OpenREM MG acquisition protocol AGD mean",
                            sorted_category_list=sorted_acquisition_agd_categories,
                            average_choice="mean",
                            return_as_dict=return_as_dict,
                        )

                    if user_profile.plotMedian:
                        return_structure["acquisitionMedianAGDData"] = plotly_barchart(
                            df_aggregated,
                            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                            value_axis_title="Median AGD (mGy)",
                            name_axis_title="Acquisition protocol",
                            colourmap=user_profile.plotColourMapChoice,
                            filename="OpenREM MG acquisition protocol AGD median",
                            sorted_category_list=sorted_acquisition_agd_categories,
                            average_choice="median",
                            return_as_dict=return_as_dict,
                        )

                if user_profile.plotBoxplots:
                    return_structure["acquisitionBoxplotAGDData"] = plotly_boxplot(
                        df,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
                        value_axis_title="AGD (mGy)",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM MG acquisition protocol AGD boxplot",
                        sorted_category_list=sorted_acquisition_agd_categories,
                        return_as_dict=return_as_dict,
                    )

                if user_profile.plotHistograms:
                    category_names_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                    group_by_col = "x_ray_system_name"
                    legend_title = "Acquisition protocol"
                    facet_names = list(df[group_by_col].unique())
                    category_names = list(sorted_acquisition_agd_categories.values())[0]

                    if user_profile.plotGroupingChoice == "series":
                        category_names_col = "x_ray_system_name"
                        group_by_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                        legend_title = "System"
                        category_names = facet_names
                        facet_names = list(sorted_acquisition_agd_categories.values())[
                            0
                        ]

                    return_structure[
                        "acquisitionHistogramAGDData"
                    ] = plotly_histogram_barchart(
                        df,
                        group_by_col,
                        category_names_col,
                        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
                        value_axis_title="AGD (mGy)",
                        legend_title=legend_title,
                        n_bins=user_profile.plotHistogramBins,
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM MG acquisition protocol AGD histogram",
                        facet_col_wrap=user_profile.plotFacetColWrapVal,
                        df_facet_category_list=facet_names,
                        df_category_name_list=category_names,
                        global_max_min=user_profile.plotHistogramGlobalBins,
                        return_as_dict=return_as_dict,
                    )

        if user_profile.plotMGAGDvsThickness:
            return_structure["AGDvsThickness"] = construct_scatter_chart(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_x_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
                df_y_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
                x_axis_title="Compressed breast thickness (mm)",
                y_axis_title="AGD (mGy)",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Acquisition protocol",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol AGD vs thickness",
                return_as_dict=return_as_dict,
            )

        if user_profile.plotMGkVpvsThickness:
            return_structure["kVpvsThickness"] = construct_scatter_chart(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_x_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
                df_y_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                x_axis_title="Compressed breast thickness (mm)",
                y_axis_title="kVp",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Acquisition protocol",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol kVp vs thickness",
                return_as_dict=return_as_dict,
            )

        if user_profile.plotMGmAsvsThickness:
            return_structure["mAsvsThickness"] = construct_scatter_chart(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_x_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
                df_y_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                x_axis_title="Compressed breast thickness (mm)",
                y_axis_title="mAs",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Acquisition protocol",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol mAs vs thickness",
                return_as_dict=return_as_dict,
            )

        if user_profile.plotMGacquisitionFreq:
            sorted_categories = None
            if user_profile.plotMGaverageAGD:
                sorted_categories = sorted_acquisition_agd_categories

            return_structure["acquisitionFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                sorting_choice=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                legend_title="Acquisition protocol",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM MG acquisition protocol frequency",
                sorted_categories=sorted_categories,
                return_as_dict=return_as_dict,
            )

        if user_profile.plotMGAcquisitionAGDOverTime:
            facet_title = "System"

            if user_profile.plotGroupingChoice == "series":
                facet_title = "Acquisition protocol"

            result = construct_over_time_charts(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_value_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
                df_date_col="study_date",
                name_title="Acquisition protocol",
                value_title="AGD (mGy)",
                date_title="Study date",
                facet_title=facet_title,
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotMGInitialSortingChoice,
                ],
                time_period=plot_timeunit_period,
                average_choices=average_choices + ["count"],
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM MG acquisition protocol AGD over time",
                return_as_dict=return_as_dict,
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
    # Obtain the chart options from the request
    chart_options_form = MGChartOptionsForm(request.GET)
    # Check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.plotCharts = chart_options_form.cleaned_data["plotCharts"]
            user_profile.plotMGStudyPerDayAndHour = chart_options_form.cleaned_data[
                "plotMGStudyPerDayAndHour"
            ]
            user_profile.plotMGaverageAGD = chart_options_form.cleaned_data[
                "plotMGaverageAGD"
            ]
            user_profile.plotMGacquisitionFreq = chart_options_form.cleaned_data[
                "plotMGacquisitionFreq"
            ]
            user_profile.plotMGAGDvsThickness = chart_options_form.cleaned_data[
                "plotMGAGDvsThickness"
            ]
            user_profile.plotMGaverageAGDvsThickness = chart_options_form.cleaned_data[
                "plotMGaverageAGDvsThickness"
            ]
            user_profile.plotMGkVpvsThickness = chart_options_form.cleaned_data[
                "plotMGkVpvsThickness"
            ]
            user_profile.plotMGmAsvsThickness = chart_options_form.cleaned_data[
                "plotMGmAsvsThickness"
            ]
            user_profile.plotMGAcquisitionAGDOverTime = chart_options_form.cleaned_data[
                "plotMGAcquisitionAGDOverTime"
            ]
            user_profile.plotMGOverTimePeriod = chart_options_form.cleaned_data[
                "plotMGOverTimePeriod"
            ]
            user_profile.plotSeriesPerSystem = chart_options_form.cleaned_data[
                "plotSeriesPerSystem"
            ]
            user_profile.plotHistograms = chart_options_form.cleaned_data[
                "plotHistograms"
            ]
            user_profile.plotGroupingChoice = chart_options_form.cleaned_data[
                "plotGrouping"
            ]
            user_profile.plotMGInitialSortingChoice = chart_options_form.cleaned_data[
                "plotMGInitialSortingChoice"
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
                "plotMGStudyPerDayAndHour": user_profile.plotMGStudyPerDayAndHour,
                "plotMGacquisitionFreq": user_profile.plotMGacquisitionFreq,
                "plotMGaverageAGD": user_profile.plotMGaverageAGD,
                "plotMGAGDvsThickness": user_profile.plotMGAGDvsThickness,
                "plotMGkVpvsThickness": user_profile.plotMGkVpvsThickness,
                "plotMGmAsvsThickness": user_profile.plotMGmAsvsThickness,
                "plotMGaverageAGDvsThickness": user_profile.plotMGaverageAGDvsThickness,
                "plotMGAcquisitionAGDOverTime": user_profile.plotMGAcquisitionAGDOverTime,
                "plotMGOverTimePeriod": user_profile.plotMGOverTimePeriod,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotMGInitialSortingChoice": user_profile.plotMGInitialSortingChoice,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices,
            }
            chart_options_form = MGChartOptionsForm(form_data)
    return chart_options_form
