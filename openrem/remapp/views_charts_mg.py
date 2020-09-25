from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from remapp.models import GeneralStudyModuleAttr, create_user_profile
import logging

logger = logging.getLogger(__name__)


def generate_required_mg_charts_list(profile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    if profile.plotMGAGDvsThickness:
        required_charts.append(
            {
                "title": "Chart of AGD vs compressed breast thickness for each acquisition protocol",
                "var_name": "acquisitionScatterAGDvsThick",
            }
        )

    if profile.plotMGaverageAGDvsThickness:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean AGD vs compressed breast thickness for each acquisition protocol",
                    "var_name": "acquisitionMeanAGDvsThick",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median AGD vs compressed breast thickness for each acquisition protocol",
                    "var_name": "acquisitionMedianAGDvsThick",
                }
            )

    if profile.plotMGkVpvsThickness:
        required_charts.append(
            {
                "title": "Chart of kVp vs compressed breast thickness for each acquisition protocol",
                "var_name": "acquisitionScatterkVpvsThick",
            }
        )

    if profile.plotMGmAsvsThickness:
        required_charts.append(
            {
                "title": "Chart of mAs vs compressed breast thickness for each acquisition protocol",
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
    from remapp.interface.mod_filters import MGSummaryListFilter, MGFilterPlusPid
    from openremproject import settings
    from django.http import JsonResponse

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
        from datetime import datetime

        start_time = datetime.now()

    return_structure = mg_plot_calculations(f, user_profile)

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def mg_plot_calculations(f, user_profile):
    """Calculations for mammography charts"""
    from .interface.chart_functions import (
        create_dataframe,
        create_dataframe_weekdays,
        plotly_barchart_weekdays,
        plotly_binned_statistic_barchart,
        construct_scatter_chart,
        plotly_set_default_theme,
        create_sorted_category_list,
    )

    # Set the Plotly chart theme
    plotly_set_default_theme(user_profile.plotThemeChoice)

    return_structure = {}

    #######################################################################
    # Create a data frame to use for charts
    name_fields = []
    if user_profile.plotMGStudyPerDayAndHour:
        name_fields.append("study_description")
    if (
        user_profile.plotMGAGDvsThickness
        or user_profile.plotMGkVpvsThickness
        or user_profile.plotMGmAsvsThickness
        or user_profile.plotMGaverageAGDvsThickness
    ):
        name_fields.append(
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
        )

    value_fields = []
    if user_profile.plotMGAGDvsThickness or user_profile.plotMGaverageAGDvsThickness:
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
        uid="projectionxrayradiationdose__irradeventxraydata__pk",
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
        )

    if user_profile.plotMGaverageAGDvsThickness:
        sorted_acquisition_agd_categories = create_sorted_category_list(
            df,
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
            [
                user_profile.plotInitialSortingDirection,
                user_profile.plotMGInitialSortingChoice,
            ],
        )

        category_names_col = (
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
        )
        group_by_col = "x_ray_system_name"
        legend_title = "Acquisition protocol"
        facet_names = list(df[group_by_col].unique())
        category_names = list(sorted_acquisition_agd_categories.values())[0]

        if user_profile.plotGroupingChoice == "series":
            category_names_col = "x_ray_system_name"
            group_by_col = (
                "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
            )
            legend_title = "System"
            category_names = facet_names
            facet_names = list(sorted_acquisition_agd_categories.values())[0]

        if user_profile.plotMean:
            return_structure["meanAGDvsThickness"] = plotly_binned_statistic_barchart(
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
            )

        if user_profile.plotMedian:
            return_structure["medianAGDvsThickness"] = plotly_binned_statistic_barchart(
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
        )

    return return_structure


def mg_chart_form_processing(request, user_profile):
    from remapp.forms import MGChartOptionsForm

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
            user_profile.plotSeriesPerSystem = chart_options_form.cleaned_data[
                "plotSeriesPerSystem"
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

            user_profile.save()

        else:
            average_choices = []
            if user_profile.plotMean:
                average_choices.append("mean")
            if user_profile.plotMedian:
                average_choices.append("median")

            form_data = {
                "plotCharts": user_profile.plotCharts,
                "plotMGStudyPerDayAndHour": user_profile.plotMGStudyPerDayAndHour,
                "plotMGAGDvsThickness": user_profile.plotMGAGDvsThickness,
                "plotMGkVpvsThickness": user_profile.plotMGkVpvsThickness,
                "plotMGmAsvsThickness": user_profile.plotMGmAsvsThickness,
                "plotMGaverageAGDvsThickness": user_profile.plotMGaverageAGDvsThickness,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotMGInitialSortingChoice": user_profile.plotMGInitialSortingChoice,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices,
            }
            chart_options_form = MGChartOptionsForm(form_data)
    return chart_options_form
