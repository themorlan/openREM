from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from remapp.models import create_user_profile
import logging

logger = logging.getLogger(__name__)


def generate_required_ct_charts_list(profile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    from django.utils.safestring import mark_safe

    required_charts = []

    if profile.plotCTAcquisitionMeanDLP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DLP for each acquisition protocol",
                    "var_name": "acquisitionMeanDLP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DLP for each acquisition protocol",
                    "var_name": "acquisitionMedianDLP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of DLP for each acquisition protocol",
                    "var_name": "acquisitionBoxplotDLP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of DLP for each acquisition protocol",
                    "var_name": "acquisitionHistogramDLP",
                }
            )

    if profile.plotCTAcquisitionMeanCTDI:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Chart of mean CTDI<sub>vol</sub> for each acquisition protocol"
                    ),
                    "var_name": "acquisitionMeanCTDI",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Chart of median CTDI<sub>vol</sub> for each acquisition protocol"
                    ),
                    "var_name": "acquisitionMedianCTDI",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Boxplot of CTDI<sub>vol</sub> for each acquisition protocol"
                    ),
                    "var_name": "acquisitionBoxplotCTDI",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Histogram of CTDI<sub>vol</sub> for each acquisition protocol"
                    ),
                    "var_name": "acquisitionHistogramCTDI",
                }
            )

    if profile.plotCTAcquisitionFreq:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol frequency",
                "var_name": "acquisitionFrequency",
            }
        )

    if profile.plotCTAcquisitionCTDIvsMass:
        required_charts.append(
            {
                "title": mark_safe(
                    "Chart of CTDI<sub>vol</sub> vs patient mass for each acquisition protocol"
                ),
                "var_name": "acquisitionScatterCTDIvsMass",
            }
        )

    if profile.plotCTAcquisitionDLPvsMass:
        required_charts.append(
            {
                "title": "Chart of DLP vs patient mass for each acquisition protocol",
                "var_name": "acquisitionScatterDLPvsMass",
            }
        )

    if profile.plotCTAcquisitionCTDIOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Chart of mean CTDI<sub>vol</sub> per acquisition protocol over time ("
                        + profile.plotCTOverTimePeriod
                        + ")"
                    ),
                    "var_name": "acquisitionMeanCTDIOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Chart of median CTDI<sub>vol</sub> per acquisition protocol over time ("
                        + profile.plotCTOverTimePeriod
                        + ")"
                    ),
                    "var_name": "acquisitionMedianCTDIOverTime",
                }
            )

    if profile.plotCTAcquisitionDLPOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DLP per acquisition protocol over time ("
                    + profile.plotCTOverTimePeriod
                    + ")",
                    "var_name": "acquisitionMeanDLPOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DLP per acquisition protocol over time ("
                    + profile.plotCTOverTimePeriod
                    + ")",
                    "var_name": "acquisitionMedianDLPOverTime",
                }
            )

    if profile.plotCTStudyMeanDLP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DLP for each study description",
                    "var_name": "studyMeanDLP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DLP for each study description",
                    "var_name": "studyMedianDLP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of DLP for each study description",
                    "var_name": "studyBoxplotDLP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of DLP for each study description",
                    "var_name": "studyHistogramDLP",
                }
            )

    if profile.plotCTStudyMeanCTDI:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Chart of mean CTDI<sub>vol</sub> for each study description"
                    ),
                    "var_name": "studyMeanCTDI",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Chart of median CTDI<sub>vol</sub> for each study description"
                    ),
                    "var_name": "studyMedianCTDI",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Boxplot of CTDI<sub>vol</sub> for each study description"
                    ),
                    "var_name": "studyBoxplotCTDI",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": mark_safe(
                        "Histogram of CTDI<sub>vol</sub> for each study description"
                    ),
                    "var_name": "studyHistogramCTDI",
                }
            )

    if profile.plotCTStudyFreq:
        required_charts.append(
            {
                "title": "Chart of study description frequency",
                "var_name": "studyFrequency",
            }
        )

    if profile.plotCTStudyNumEvents:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean number of events for each study description",
                    "var_name": "studyMeanNumEvents",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median number of events for each study description",
                    "var_name": "studyMedianNumEvents",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of number of events for each study description",
                    "var_name": "studyBoxplotNumEvents",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of number of events for each study description",
                    "var_name": "studyHistogramNumEvents",
                }
            )

    if profile.plotCTRequestMeanDLP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DLP for each requested procedure",
                    "var_name": "requestMeanDLP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DLP for each requested procedure",
                    "var_name": "requestMedianDLP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of DLP for each requested procedure",
                    "var_name": "requestBoxplotDLP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of DLP for each requested procedure",
                    "var_name": "requestHistogramDLP",
                }
            )

    if profile.plotCTRequestFreq:
        required_charts.append(
            {
                "title": "Chart of requested procedure frequency",
                "var_name": "requestFrequency",
            }
        )

    if profile.plotCTRequestNumEvents:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean number of events for each requested procedure",
                    "var_name": "requestMeanNumEvents",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median number of events for each requested procedure",
                    "var_name": "requestMedianNumEvents",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of number of events for each requested procedure",
                    "var_name": "requestBoxplotNumEvents",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of number of events for each requested procedure",
                    "var_name": "requestHistogramNumEvents",
                }
            )

    if profile.plotCTRequestDLPOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DLP per requested procedure over time ("
                    + profile.plotCTOverTimePeriod
                    + ")",
                    "var_name": "requestMeanDLPOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DLP per requested procedure over time ("
                    + profile.plotCTOverTimePeriod
                    + ")",
                    "var_name": "requestMedianDLPOverTime",
                }
            )

    if profile.plotCTStudyPerDayAndHour:
        required_charts.append(
            {
                "title": "Chart of study description workload",
                "var_name": "studyWorkload",
            }
        )

    if profile.plotCTStudyMeanDLPOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of mean DLP per study description over time ("
                    + profile.plotCTOverTimePeriod
                    + ")",
                    "var_name": "studyMeanDLPOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of median DLP per study description over time ("
                    + profile.plotCTOverTimePeriod
                    + ")",
                    "var_name": "studyMedianDLPOverTime",
                }
            )

    return required_charts


@login_required
def ct_summary_chart_data(request):
    """Obtain data for CT charts Ajax call"""
    from remapp.interface.mod_filters import ct_acq_filter
    from openremproject import settings
    from django.http import JsonResponse

    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = ct_acq_filter(request.GET, pid=pid)

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

    return_structure = ct_plot_calculations(f, user_profile)

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def ct_plot_calculations(f, user_profile):
    """CT chart data calculations"""
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

    # Set the Plotly chart theme
    plotly_set_default_theme(user_profile.plotThemeChoice)

    return_structure = {}

    average_choices = []
    if user_profile.plotMean:
        average_choices.append("mean")
    if user_profile.plotMedian:
        average_choices.append("median")

    if (
        user_profile.plotCTAcquisitionDLPOverTime
        or user_profile.plotCTAcquisitionCTDIOverTime
        or user_profile.plotCTStudyMeanDLPOverTime
        or user_profile.plotCTRequestDLPOverTime
    ):
        # Obtain the key name in the TIME_PERIOD tuple from the user time period choice (the key value)
        keys = list(dict(user_profile.TIME_PERIOD).keys())
        values = list(dict(user_profile.TIME_PERIOD).values())
        plot_timeunit_period = keys[
            [tp.lower() for tp in values].index(user_profile.plotCTOverTimePeriod)
        ]

    #######################################################################
    # Prepare acquisition-level Pandas DataFrame to use for charts
    if (
        user_profile.plotCTAcquisitionFreq
        or user_profile.plotCTAcquisitionMeanCTDI
        or user_profile.plotCTAcquisitionMeanDLP
        or user_profile.plotCTAcquisitionCTDIvsMass
        or user_profile.plotCTAcquisitionDLPvsMass
        or user_profile.plotCTAcquisitionCTDIOverTime
        or user_profile.plotCTAcquisitionDLPOverTime
    ):

        name_fields = ["ctradiationdose__ctirradiationeventdata__acquisition_protocol"]

        value_fields = []
        if (
            user_profile.plotCTAcquisitionMeanDLP
            or user_profile.plotCTAcquisitionDLPvsMass
            or user_profile.plotCTAcquisitionDLPOverTime
        ):
            value_fields.append("ctradiationdose__ctirradiationeventdata__dlp")
        if (
            user_profile.plotCTAcquisitionMeanCTDI
            or user_profile.plotCTAcquisitionCTDIvsMass
            or user_profile.plotCTAcquisitionCTDIOverTime
        ):
            value_fields.append("ctradiationdose__ctirradiationeventdata__mean_ctdivol")
        if (
            user_profile.plotCTAcquisitionCTDIvsMass
            or user_profile.plotCTAcquisitionDLPvsMass
        ):
            value_fields.append("patientstudymoduleattr__patient_weight")

        date_fields = []
        if (
            user_profile.plotCTAcquisitionCTDIOverTime
            or user_profile.plotCTAcquisitionDLPOverTime
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
            uid="ctradiationdose__ctirradiationeventdata__pk",
        )
        #######################################################################

        #######################################################################
        # Create the required acquisition-level charts
        sorted_acquisition_dlp_categories = None
        if user_profile.plotCTAcquisitionMeanDLP:
            sorted_acquisition_dlp_categories = create_sorted_category_list(
                df,
                "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                "ctradiationdose__ctirradiationeventdata__dlp",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__dlp",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["acquisitionMeanDLPData"] = plotly_barchart(
                        df_aggregated,
                        "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                        value_axis_title="Mean DLP (mGy.cm)",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT acquisition protocol DLP mean",
                        sorted_category_list=sorted_acquisition_dlp_categories,
                        average_choice="mean",
                    )

                if user_profile.plotMedian:
                    return_structure["acquisitionMedianDLPData"] = plotly_barchart(
                        df_aggregated,
                        "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                        value_axis_title="Median DLP (mGy.cm)",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT acquisition protocol DLP median",
                        sorted_category_list=sorted_acquisition_dlp_categories,
                        average_choice="median",
                    )

            if user_profile.plotBoxplots:
                return_structure["acquisitionBoxplotDLPData"] = plotly_boxplot(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Acquisition protocol",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT acquisition protocol DLP boxplot",
                    sorted_category_list=sorted_acquisition_dlp_categories,
                )

            if user_profile.plotHistograms:
                category_names_col = (
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
                )
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_dlp_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = (
                        "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
                    )
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_dlp_categories.values())[0]

                return_structure[
                    "acquisitionHistogramDLPData"
                ] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "ctradiationdose__ctirradiationeventdata__dlp",
                    value_axis_title="DLP (mGy.cm)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT acquisition protocol DLP histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        sorted_acquisition_ctdi_categories = None
        if user_profile.plotCTAcquisitionMeanCTDI:
            sorted_acquisition_ctdi_categories = create_sorted_category_list(
                df,
                "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["acquisitionMeanCTDIData"] = plotly_barchart(
                        df_aggregated,
                        "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                        value_axis_title="Mean CTDI<sub>vol</sub> (mGy)",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT acquisition protocol CTDI mean",
                        sorted_category_list=sorted_acquisition_ctdi_categories,
                        average_choice="mean",
                    )

                if user_profile.plotMedian:
                    return_structure["acquisitionMedianCTDIData"] = plotly_barchart(
                        df_aggregated,
                        "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                        value_axis_title="Median CTDI<sub>vol</sub> (mGy)",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT acquisition protocol CTDI median",
                        sorted_category_list=sorted_acquisition_ctdi_categories,
                        average_choice="median",
                    )

            if user_profile.plotBoxplots:
                return_structure["acquisitionBoxplotCTDIData"] = plotly_boxplot(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI<sub>vol</sub> (mGy)",
                    name_axis_title="Acquisition protocol",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT acquisition protocol CTDI boxplot",
                    sorted_category_list=sorted_acquisition_ctdi_categories,
                )

            if user_profile.plotHistograms:
                category_names_col = (
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
                )
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_ctdi_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = (
                        "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
                    )
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_ctdi_categories.values())[0]

                return_structure[
                    "acquisitionHistogramCTDIData"
                ] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI<sub>vol</sub> (mGy)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT acquisition protocol CTDI histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        if user_profile.plotCTAcquisitionFreq:
            sorted_categories = None
            if sorted_acquisition_dlp_categories:
                sorted_categories = sorted_acquisition_dlp_categories
            elif sorted_acquisition_ctdi_categories:
                sorted_categories = sorted_acquisition_ctdi_categories

            return_structure["acquisitionFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                sorting_choice=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                legend_title="Acquisition protocol",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM CT acquisition protocol frequency",
                sorted_categories=sorted_categories,
            )

        if user_profile.plotCTAcquisitionCTDIvsMass:
            return_structure["acquisitionScatterCTDIvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                x_axis_title="Patient mass (kg)",
                y_axis_title="CTDI<sub>vol</sub> (mGy)",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Acquisition protocol",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol CTDI vs patient mass",
            )

        if user_profile.plotCTAcquisitionDLPvsMass:
            return_structure["acquisitionScatterDLPvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="ctradiationdose__ctirradiationeventdata__dlp",
                x_axis_title="Patient mass (kg)",
                y_axis_title="DLP (mGy.cm)",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Acquisition protocol",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol DLP vs patient mass",
            )

        if user_profile.plotCTAcquisitionCTDIOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                df_value_col="ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                df_date_col="study_date",
                name_title="Acquisition protocol",
                value_title="CTDI<sub>vol</sub> (mGy)",
                date_title="Study date",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol CTDI over time",
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanCTDIOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianCTDIOverTime"] = result["median"]

        if user_profile.plotCTAcquisitionDLPOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                df_value_col="ctradiationdose__ctirradiationeventdata__dlp",
                df_date_col="study_date",
                name_title="Acquisition protocol",
                value_title="DLP (mGy.cm)",
                date_title="Study date",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol DLP over time",
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanDLPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianDLPOverTime"] = result["median"]

    #######################################################################
    # Prepare study- and request-level Pandas DataFrame to use for charts
    if (
        user_profile.plotCTRequestFreq
        or user_profile.plotCTRequestMeanDLP
        or user_profile.plotCTRequestNumEvents
        or user_profile.plotCTRequestDLPOverTime
        or user_profile.plotCTStudyFreq
        or user_profile.plotCTStudyMeanDLP
        or user_profile.plotCTStudyMeanCTDI
        or user_profile.plotCTStudyNumEvents
        or user_profile.plotCTStudyMeanDLPOverTime
        or user_profile.plotCTStudyPerDayAndHour
    ):

        name_fields = []
        if (
            user_profile.plotCTStudyMeanDLP
            or user_profile.plotCTStudyFreq
            or user_profile.plotCTStudyMeanDLPOverTime
            or user_profile.plotCTStudyPerDayAndHour
            or user_profile.plotCTStudyNumEvents
            or user_profile.plotCTStudyMeanCTDI
        ):
            name_fields.append("study_description")
        if (
            user_profile.plotCTRequestMeanDLP
            or user_profile.plotCTRequestFreq
            or user_profile.plotCTRequestNumEvents
            or user_profile.plotCTRequestDLPOverTime
        ):
            name_fields.append("requested_procedure_code_meaning")

        value_fields = []
        if (
            user_profile.plotCTStudyMeanDLP
            or user_profile.plotCTStudyMeanDLPOverTime
            or user_profile.plotCTRequestMeanDLP
            or user_profile.plotCTRequestDLPOverTime
        ):
            value_fields.append("total_dlp")
        if user_profile.plotCTStudyMeanCTDI:
            value_fields.append("ctradiationdose__ctirradiationeventdata__mean_ctdivol")
        if user_profile.plotCTStudyNumEvents or user_profile.plotCTRequestNumEvents:
            value_fields.append("number_of_events")

        date_fields = []
        time_fields = []
        if (
            user_profile.plotCTStudyMeanDLPOverTime
            or user_profile.plotCTStudyPerDayAndHour
            or user_profile.plotCTRequestDLPOverTime
        ):
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
        #######################################################################

        #######################################################################
        # Create the required study- and request-level charts
        sorted_study_dlp_categories = None
        if user_profile.plotCTStudyMeanDLP:
            sorted_study_dlp_categories = create_sorted_category_list(
                df,
                "study_description",
                "total_dlp",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "study_description",
                    "total_dlp",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["studyMeanDLPData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Mean DLP (mGy.cm)",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT study description DLP mean",
                        sorted_category_list=sorted_study_dlp_categories,
                        average_choice="mean",
                    )

                if user_profile.plotMedian:
                    return_structure["studyMedianDLPData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Median DLP (mGy.cm)",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT study description DLP median",
                        sorted_category_list=sorted_study_dlp_categories,
                        average_choice="median",
                    )

            if user_profile.plotBoxplots:
                return_structure["studyBoxplotDLPData"] = plotly_boxplot(
                    df,
                    "study_description",
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Study description",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT study description DLP boxplot",
                    sorted_category_list=sorted_study_dlp_categories,
                )

            if user_profile.plotHistograms:
                category_names_col = "study_description"
                group_by_col = "x_ray_system_name"
                legend_title = "Study description"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_study_dlp_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "study_description"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_study_dlp_categories.values())[0]

                return_structure["studyHistogramDLPData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT study description DLP histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        sorted_study_ctdi_categories = None
        if user_profile.plotCTStudyMeanCTDI:
            sorted_study_ctdi_categories = create_sorted_category_list(
                df,
                "study_description",
                "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "study_description",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["studyMeanCTDIData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Mean CTDI<sub>vol</sub> (mGy)",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT study description CTDI mean",
                        sorted_category_list=sorted_study_ctdi_categories,
                        average_choice="mean",
                    )

                if user_profile.plotMedian:
                    return_structure["studyMedianCTDIData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Median CTDI<sub>vol</sub> (mGy)",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT study description CTDI median",
                        sorted_category_list=sorted_study_ctdi_categories,
                        average_choice="median",
                    )

            if user_profile.plotBoxplots:
                return_structure["studyBoxplotCTDIData"] = plotly_boxplot(
                    df,
                    "study_description",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI<sub>vol</sub> (mGy)",
                    name_axis_title="Study description",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT study description CTDI boxplot",
                    sorted_category_list=sorted_study_ctdi_categories,
                )

            if user_profile.plotHistograms:
                category_names_col = "study_description"
                group_by_col = "x_ray_system_name"
                legend_title = "Study description"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_study_ctdi_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "study_description"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_study_ctdi_categories.values())[0]

                return_structure["studyHistogramCTDIData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI<sub>vol</sub> (mGy)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT study description CTDI histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        sorted_study_events_categories = None
        if user_profile.plotCTStudyNumEvents:
            sorted_study_events_categories = create_sorted_category_list(
                df,
                "study_description",
                "number_of_events",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "study_description",
                    "number_of_events",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["studyMeanNumEventsData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Mean events",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT study description events mean",
                        sorted_category_list=sorted_study_events_categories,
                        average_choice="mean",
                    )

                if user_profile.plotMedian:
                    return_structure["studyMedianNumEventsData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Median events",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT study description events median",
                        sorted_category_list=sorted_study_events_categories,
                        average_choice="median",
                    )

            if user_profile.plotBoxplots:
                return_structure["studyBoxplotNumEventsData"] = plotly_boxplot(
                    df,
                    "study_description",
                    "number_of_events",
                    value_axis_title="Events",
                    name_axis_title="Study description",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT study description events boxplot",
                    sorted_category_list=sorted_study_events_categories,
                )

            if user_profile.plotHistograms:
                category_names_col = "study_description"
                group_by_col = "x_ray_system_name"
                legend_title = "Study description"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_study_events_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "study_description"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_study_events_categories.values())[0]

                return_structure[
                    "studyHistogramNumEventsData"
                ] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "number_of_events",
                    value_axis_title="Events",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT study description events histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        if user_profile.plotCTStudyFreq:
            sorted_categories = None
            if sorted_study_dlp_categories:
                sorted_categories = sorted_study_dlp_categories
            elif sorted_study_ctdi_categories:
                sorted_categories = sorted_study_ctdi_categories
            elif sorted_study_events_categories:
                sorted_categories = sorted_study_events_categories

            return_structure["studyFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="study_description",
                sorting_choice=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                legend_title="Study description",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM CT study description frequency",
                sorted_categories=sorted_categories,
            )

        sorted_request_dlp_categories = None
        if user_profile.plotCTRequestMeanDLP:
            sorted_request_dlp_categories = create_sorted_category_list(
                df,
                "requested_procedure_code_meaning",
                "total_dlp",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dlp",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["requestMeanData"] = plotly_barchart(
                        df_aggregated,
                        "requested_procedure_code_meaning",
                        value_axis_title="Mean DLP (mGy.cm)",
                        name_axis_title="Requested procedure",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT requested procedure DLP mean",
                        sorted_category_list=sorted_request_dlp_categories,
                        average_choice="mean",
                    )

                if user_profile.plotMedian:
                    return_structure["requestMedianData"] = plotly_barchart(
                        df_aggregated,
                        "requested_procedure_code_meaning",
                        value_axis_title="Mean DLP (mGy.cm)",
                        name_axis_title="Requested procedure",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT requested procedure DLP median",
                        sorted_category_list=sorted_request_dlp_categories,
                        average_choice="median",
                    )

            if user_profile.plotBoxplots:
                return_structure["requestBoxplotData"] = plotly_boxplot(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Requested procedure",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT requested procedure DLP boxplot",
                    sorted_category_list=sorted_request_dlp_categories,
                )

            if user_profile.plotHistograms:
                category_names_col = "requested_procedure_code_meaning"
                group_by_col = "x_ray_system_name"
                legend_title = "Requested procedure"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_request_dlp_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "requested_procedure_code_meaning"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_request_dlp_categories.values())[0]

                return_structure["requestHistogramData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT requested procedure DLP histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        sorted_request_events_categories = None
        if user_profile.plotCTRequestNumEvents:
            sorted_request_events_categories = create_sorted_category_list(
                df,
                "requested_procedure_code_meaning",
                "number_of_events",
                [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
            )

            if user_profile.plotMean:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "requested_procedure_code_meaning",
                    "number_of_events",
                    stats=average_choices + ["count"],
                )

                if user_profile.plotMean:
                    return_structure["requestMeanNumEventsData"] = plotly_barchart(
                        df_aggregated,
                        "requested_procedure_code_meaning",
                        value_axis_title="Mean events",
                        name_axis_title="Requested procedure",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT requested procedure events mean",
                        sorted_category_list=sorted_request_events_categories,
                        average_choice="mean",
                    )

                if user_profile.plotMedian:
                    return_structure["requestMedianNumEventsData"] = plotly_barchart(
                        df_aggregated,
                        "requested_procedure_code_meaning",
                        value_axis_title="Median events",
                        name_axis_title="Requested procedure",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT requested procedure events median",
                        sorted_category_list=sorted_request_events_categories,
                        average_choice="median",
                    )

            if user_profile.plotMedian:
                return_structure["requestBoxplotNumEventsData"] = plotly_boxplot(
                    df,
                    "requested_procedure_code_meaning",
                    "number_of_events",
                    value_axis_title="Events",
                    name_axis_title="Requested procedure",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT requested procedure events boxplot",
                    sorted_category_list=sorted_request_events_categories,
                )

            if user_profile.plotHistograms:
                category_names_col = "requested_procedure_code_meaning"
                group_by_col = "x_ray_system_name"
                legend_title = "Requested procedure"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_request_events_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "requested_procedure_code_meaning"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_request_events_categories.values())[0]

                return_structure[
                    "requestHistogramNumEventsData"
                ] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "number_of_events",
                    value_axis_title="Events",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM CT requested procedure events histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        if user_profile.plotCTRequestFreq:
            sorted_categories = None
            if sorted_request_dlp_categories:
                sorted_categories = sorted_request_dlp_categories
            elif sorted_request_events_categories:
                sorted_categories = sorted_request_events_categories

            return_structure["requestFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="requested_procedure_code_meaning",
                sorting_choice=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                legend_title="Requested procedure",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM CT requested procedure frequency",
                sorted_categories=sorted_categories,
            )

        if user_profile.plotCTStudyMeanDLPOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="study_description",
                df_value_col="total_dlp",
                df_date_col="study_date",
                name_title="Study description",
                value_title="DLP (mGy.cm)",
                date_title="Study date",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT study description DLP over time",
            )

            if user_profile.plotMean:
                return_structure["studyMeanDLPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["studyMedianDLPOverTime"] = result["median"]

        if user_profile.plotCTRequestDLPOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="requested_procedure_code_meaning",
                df_value_col="total_dlp",
                df_date_col="study_date",
                name_title="Requested procedure",
                value_title="DLP (mGy.cm)",
                date_title="Study date",
                sorting=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT requested procedure DLP over time",
            )

            if user_profile.plotMean:
                return_structure["requestMeanDLPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["requestMedianDLPOverTime"] = result["median"]

        if user_profile.plotCTStudyPerDayAndHour:
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
        #######################################################################

    return return_structure


def ct_chart_form_processing(request, user_profile):
    from remapp.forms import CTChartOptionsForm

    # Obtain the chart options from the request
    chart_options_form = CTChartOptionsForm(request.GET)
    # Check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.plotCharts = chart_options_form.cleaned_data["plotCharts"]
            user_profile.plotCTAcquisitionMeanDLP = chart_options_form.cleaned_data[
                "plotCTAcquisitionMeanDLP"
            ]
            user_profile.plotCTAcquisitionMeanCTDI = chart_options_form.cleaned_data[
                "plotCTAcquisitionMeanCTDI"
            ]
            user_profile.plotCTAcquisitionFreq = chart_options_form.cleaned_data[
                "plotCTAcquisitionFreq"
            ]
            user_profile.plotCTAcquisitionCTDIvsMass = chart_options_form.cleaned_data[
                "plotCTAcquisitionCTDIvsMass"
            ]
            user_profile.plotCTAcquisitionDLPvsMass = chart_options_form.cleaned_data[
                "plotCTAcquisitionDLPvsMass"
            ]
            user_profile.plotCTAcquisitionCTDIOverTime = (
                chart_options_form.cleaned_data["plotCTAcquisitionCTDIOverTime"]
            )
            user_profile.plotCTAcquisitionDLPOverTime = chart_options_form.cleaned_data[
                "plotCTAcquisitionDLPOverTime"
            ]
            user_profile.plotCTStudyMeanDLP = chart_options_form.cleaned_data[
                "plotCTStudyMeanDLP"
            ]
            user_profile.plotCTStudyMeanCTDI = chart_options_form.cleaned_data[
                "plotCTStudyMeanCTDI"
            ]
            user_profile.plotCTStudyFreq = chart_options_form.cleaned_data[
                "plotCTStudyFreq"
            ]
            user_profile.plotCTStudyNumEvents = chart_options_form.cleaned_data[
                "plotCTStudyNumEvents"
            ]
            user_profile.plotCTStudyPerDayAndHour = chart_options_form.cleaned_data[
                "plotCTStudyPerDayAndHour"
            ]
            user_profile.plotCTStudyMeanDLPOverTime = chart_options_form.cleaned_data[
                "plotCTStudyMeanDLPOverTime"
            ]
            user_profile.plotCTRequestMeanDLP = chart_options_form.cleaned_data[
                "plotCTRequestMeanDLP"
            ]
            user_profile.plotCTRequestFreq = chart_options_form.cleaned_data[
                "plotCTRequestFreq"
            ]
            user_profile.plotCTRequestNumEvents = chart_options_form.cleaned_data[
                "plotCTRequestNumEvents"
            ]
            user_profile.plotCTRequestDLPOverTime = chart_options_form.cleaned_data[
                "plotCTRequestDLPOverTime"
            ]
            user_profile.plotCTOverTimePeriod = chart_options_form.cleaned_data[
                "plotCTOverTimePeriod"
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
            user_profile.plotCTInitialSortingChoice = chart_options_form.cleaned_data[
                "plotCTInitialSortingChoice"
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
                "plotCTAcquisitionMeanDLP": user_profile.plotCTAcquisitionMeanDLP,
                "plotCTAcquisitionMeanCTDI": user_profile.plotCTAcquisitionMeanCTDI,
                "plotCTAcquisitionFreq": user_profile.plotCTAcquisitionFreq,
                "plotCTAcquisitionCTDIvsMass": user_profile.plotCTAcquisitionCTDIvsMass,
                "plotCTAcquisitionDLPvsMass": user_profile.plotCTAcquisitionDLPvsMass,
                "plotCTAcquisitionCTDIOverTime": user_profile.plotCTAcquisitionCTDIOverTime,
                "plotCTAcquisitionDLPOverTime": user_profile.plotCTAcquisitionDLPOverTime,
                "plotCTStudyMeanDLP": user_profile.plotCTStudyMeanDLP,
                "plotCTStudyMeanCTDI": user_profile.plotCTStudyMeanCTDI,
                "plotCTStudyFreq": user_profile.plotCTStudyFreq,
                "plotCTStudyNumEvents": user_profile.plotCTStudyNumEvents,
                "plotCTRequestMeanDLP": user_profile.plotCTRequestMeanDLP,
                "plotCTRequestFreq": user_profile.plotCTRequestFreq,
                "plotCTRequestNumEvents": user_profile.plotCTRequestNumEvents,
                "plotCTRequestDLPOverTime": user_profile.plotCTRequestDLPOverTime,
                "plotCTStudyPerDayAndHour": user_profile.plotCTStudyPerDayAndHour,
                "plotCTStudyMeanDLPOverTime": user_profile.plotCTStudyMeanDLPOverTime,
                "plotCTOverTimePeriod": user_profile.plotCTOverTimePeriod,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotCTInitialSortingChoice": user_profile.plotCTInitialSortingChoice,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices,
            }
            chart_options_form = CTChartOptionsForm(form_data)
    return chart_options_form
