# pylint: disable=too-many-lines
import logging
from datetime import datetime
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import JsonResponse
from django.utils.safestring import mark_safe
from openremproject import settings
from remapp.forms import CTChartOptionsForm
from remapp.interface.mod_filters import ct_acq_filter
from remapp.models import create_user_profile
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
    plotly_frequency_barchart,
    plotly_scatter,
    construct_over_time_charts,
)

logger = logging.getLogger(__name__)


def generate_required_ct_charts_list(profile):
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    charts_of_interest = [
        profile.plotCTAcquisitionDLPOverTime, profile.plotCTAcquisitionCTDIOverTime,
        profile.plotCTStudyMeanDLPOverTime, profile.plotCTRequestDLPOverTime,
    ]
    if any(charts_of_interest):
        keys = list(dict(profile.TIME_PERIOD).keys())
        values = list(dict(profile.TIME_PERIOD).values())
        time_period = (values[keys.index(profile.plotCTOverTimePeriod)]).lower()

    if profile.plotCTAcquisitionMeanDLP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol mean DLP",
                    "var_name": "acquisitionMeanDLP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol median DLP",
                    "var_name": "acquisitionMedianDLP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of acquisition protocol DLP",
                    "var_name": "acquisitionBoxplotDLP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of acquisition protocol DLP",
                    "var_name": "acquisitionHistogramDLP",
                }
            )

    if profile.plotCTAcquisitionMeanCTDI:
        if profile.plotMean:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Chart of acquisition protocol mean CTDI<sub>vol</sub>"
                    ),
                    "var_name": "acquisitionMeanCTDI",
                }
            )
        if profile.plotMedian:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Chart of acquisition protocol median CTDI<sub>vol</sub>"
                    ),
                    "var_name": "acquisitionMedianCTDI",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Boxplot of acquisition protocol CTDI<sub>vol</sub>"
                    ),
                    "var_name": "acquisitionBoxplotCTDI",
                }
            )
        if profile.plotHistograms:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Histogram of acquisition protocol CTDI<sub>vol</sub>"
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
        required_charts.append(  # nosec
            {
                "title": mark_safe(
                    "Chart of acquisition protocol CTDI<sub>vol</sub> vs patient mass"
                ),
                "var_name": "acquisitionScatterCTDIvsMass",
            }
        )

    if profile.plotCTAcquisitionDLPvsMass:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol DLP vs patient mass",
                "var_name": "acquisitionScatterDLPvsMass",
            }
        )

    if profile.plotCTAcquisitionCTDIOverTime:
        if profile.plotMean:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Chart of acquisition protocol mean CTDI<sub>vol</sub> over time ("
                        + time_period
                        + ")"
                    ),
                    "var_name": "acquisitionMeanCTDIOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Chart of acquisition protocol median CTDI<sub>vol</sub> over time ("
                        + time_period
                        + ")"
                    ),
                    "var_name": "acquisitionMedianCTDIOverTime",
                }
            )

    if profile.plotCTAcquisitionDLPOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol mean DLP over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMeanDLPOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of acquisition protocol median DLP over time ("
                    + time_period
                    + ")",
                    "var_name": "acquisitionMedianDLPOverTime",
                }
            )

    if profile.plotCTStudyMeanDLP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of study description mean DLP",
                    "var_name": "studyMeanDLP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of study description median DLP",
                    "var_name": "studyMedianDLP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of study description DLP",
                    "var_name": "studyBoxplotDLP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of study description DLP",
                    "var_name": "studyHistogramDLP",
                }
            )

    if profile.plotCTStudyMeanCTDI:
        if profile.plotMean:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Chart of study description mean CTDI<sub>vol</sub>"
                    ),
                    "var_name": "studyMeanCTDI",
                }
            )
        if profile.plotMedian:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Chart of study description median CTDI<sub>vol</sub>"
                    ),
                    "var_name": "studyMedianCTDI",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Boxplot of study description CTDI<sub>vol</sub>"
                    ),
                    "var_name": "studyBoxplotCTDI",
                }
            )
        if profile.plotHistograms:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Histogram of study description CTDI<sub>vol</sub>"
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
                    "title": "Chart of study description mean number of events",
                    "var_name": "studyMeanNumEvents",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of study description median number of events",
                    "var_name": "studyMedianNumEvents",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of study description number of events",
                    "var_name": "studyBoxplotNumEvents",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of study description number of events",
                    "var_name": "studyHistogramNumEvents",
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
                    "title": "Chart of study description mean DLP over time ("
                    + time_period
                    + ")",
                    "var_name": "studyMeanDLPOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of study description median DLP over time ("
                    + time_period
                    + ")",
                    "var_name": "studyMedianDLPOverTime",
                }
            )

    if profile.plotCTRequestMeanDLP:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of requested procedure mean DLP",
                    "var_name": "requestMeanDLP",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of requested procedure median DLP",
                    "var_name": "requestMedianDLP",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of requested procedure DLP",
                    "var_name": "requestBoxplotDLP",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of requested procedure DLP",
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
                    "title": "Chart of requested procedure mean number of events",
                    "var_name": "requestMeanNumEvents",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of requested procedure median number of events",
                    "var_name": "requestMedianNumEvents",
                }
            )
        if profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of requested procedure number of events",
                    "var_name": "requestBoxplotNumEvents",
                }
            )
        if profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of requested procedure number of events",
                    "var_name": "requestHistogramNumEvents",
                }
            )

    if profile.plotCTRequestDLPOverTime:
        if profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of requested procedure mean DLP over time ("
                    + time_period
                    + ")",
                    "var_name": "requestMeanDLPOverTime",
                }
            )
        if profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of requested procedure median DLP over time ("
                    + time_period
                    + ")",
                    "var_name": "requestMedianDLPOverTime",
                }
            )

    return required_charts


@login_required
def ct_summary_chart_data(request):
    """Obtain data for CT charts Ajax call"""
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
        start_time = datetime.now()

    return_structure = ct_plot_calculations(f, user_profile)

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def ct_plot_calculations(f, user_profile, return_as_dict=False):
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """CT chart data calculations"""

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

    charts_of_interest = [
        user_profile.plotCTAcquisitionDLPOverTime, user_profile.plotCTAcquisitionCTDIOverTime,
        user_profile.plotCTStudyMeanDLPOverTime, user_profile.plotCTRequestDLPOverTime,
    ]
    if any(charts_of_interest):
        plot_timeunit_period = user_profile.plotCTOverTimePeriod

    #######################################################################
    # Prepare acquisition-level Pandas DataFrame to use for charts
    charts_of_interest = [
        user_profile.plotCTAcquisitionFreq, user_profile.plotCTAcquisitionMeanCTDI,
        user_profile.plotCTAcquisitionMeanDLP, user_profile.plotCTAcquisitionCTDIvsMass,
        user_profile.plotCTAcquisitionDLPvsMass, user_profile.plotCTAcquisitionCTDIOverTime,
        user_profile.plotCTAcquisitionDLPOverTime,
    ]
    if any(charts_of_interest):

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

        time_fields = []
        date_fields = []
        if (
            user_profile.plotCTAcquisitionCTDIOverTime
            or user_profile.plotCTAcquisitionDLPOverTime
        ):
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
            data_point_name_remove_trailing_whitespace=user_profile.plotRemoveCategoryTrailingWhitespace,
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
                    ["ctradiationdose__ctirradiationeventdata__acquisition_protocol"],
                    "ctradiationdose__ctirradiationeventdata__dlp",
                    stats_to_use=average_choices + ["count"],
                )

                parameter_dict = {
                    "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "name_axis_title": "Acquisition protocol",
                    "colourmap": user_profile.plotColourMapChoice,
                    "sorted_category_list": sorted_acquisition_dlp_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean DLP (mGy.cm)"
                    parameter_dict["filename"] = "OpenREM CT acquisition protocol DLP mean"
                    parameter_dict["average_choice"] = "mean"
                    return_structure["acquisitionMeanDLPData"], return_structure["acquisitionMeanDLPDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="acquisitionMeanDLPData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median DLP (mGy.cm)"
                    parameter_dict["filename"] = "OpenREM CT acquisition protocol DLP median"
                    parameter_dict["average_choice"] = "median"
                    return_structure["acquisitionMedianDLPData"], return_structure["acquisitionMedianDLPDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="acquisitionMedianDLPData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "df_value_col": "ctradiationdose__ctirradiationeventdata__dlp",
                    "value_axis_title": "DLP (mGy.cm)",
                    "name_axis_title": "Acquisition protocol",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol DLP boxplot",
                    "sorted_category_list": sorted_acquisition_dlp_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                return_structure["acquisitionBoxplotDLPData"] = plotly_boxplot(
                    df,
                    parameter_dict,
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

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": "ctradiationdose__ctirradiationeventdata__dlp",
                    "value_axis_title": "DLP (mGy.cm)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol DLP histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "df_facet_category_list": facet_names,
                    "df_category_name_list": category_names,
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["acquisitionHistogramDLPData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
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
                    ["ctradiationdose__ctirradiationeventdata__acquisition_protocol"],
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    stats_to_use=average_choices + ["count"],
                )

                parameter_dict = {
                    "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "name_axis_title": "Acquisition protocol",
                    "colourmap": user_profile.plotColourMapChoice,
                    "sorted_category_list": sorted_acquisition_ctdi_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean CTDI<sub>vol</sub> (mGy)"
                    parameter_dict["filename"] = "OpenREM CT acquisition protocol CTDI mean"
                    parameter_dict["average_choice"] = "mean"
                    return_structure["acquisitionMeanCTDIData"], return_structure["acquisitionMeanCTDIDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="acquisitionMeanCTDIData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median CTDI<sub>vol</sub> (mGy)"
                    parameter_dict["filename"] = "OpenREM CT acquisition protocol CTDI median"
                    parameter_dict["average_choice"] = "median"
                    return_structure["acquisitionMedianCTDIData"], return_structure["acquisitionMedianCTDIDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="acquisitionMedianCTDIData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "df_value_col": "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    "value_axis_title": "CTDI<sub>vol</sub> (mGy)",
                    "name_axis_title": "Acquisition protocol",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol CTDI boxplot",
                    "sorted_category_list": sorted_acquisition_ctdi_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                return_structure["acquisitionBoxplotCTDIData"] = plotly_boxplot(
                    df,
                    parameter_dict,
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

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    "value_axis_title": "CTDI<sub>vol</sub> (mGy)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol CTDI histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "df_facet_category_list": facet_names,
                    "df_category_name_list": category_names,
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["acquisitionHistogramCTDIData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if user_profile.plotCTAcquisitionFreq:
            sorted_categories = None
            if sorted_acquisition_dlp_categories:
                sorted_categories = sorted_acquisition_dlp_categories
            elif sorted_acquisition_ctdi_categories:
                sorted_categories = sorted_acquisition_ctdi_categories

            parameter_dict = {
                "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                "legend_title": "Acquisition protocol",
                "df_x_axis_col": "x_ray_system_name",
                "x_axis_title": "System",
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "filename": "OpenREM CT acquisition protocol frequency",
                "sorted_categories": sorted_categories,
                "groupby_cols": None,
                "facet_col": None,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "return_as_dict": return_as_dict,
            }
            return_structure["acquisitionFrequencyData"], return_structure["acquisitionFrequencyDataCSV"] = plotly_frequency_barchart(  # pylint: disable=line-too-long
                df,
                parameter_dict,
                csv_name="acquisitionFrequencyData.csv",
            )

        if user_profile.plotCTAcquisitionCTDIvsMass:
            parameter_dict = {
                "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                "df_x_col": "patientstudymoduleattr__patient_weight",
                "df_y_col": "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                "grouping_choice": user_profile.plotGroupingChoice,
                "legend_title": "Acquisition protocol",
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "x_axis_title": "Patient mass (kg)",
                "y_axis_title": "CTDI<sub>vol</sub> (mGy)",
                "filename": "OpenREM CT acquisition protocol CTDI vs patient mass",
                "return_as_dict": return_as_dict,
            }
            return_structure["acquisitionScatterCTDIvsMass"] = plotly_scatter(
                df,
                parameter_dict,
            )

        if user_profile.plotCTAcquisitionDLPvsMass:
            parameter_dict = {
                "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                "df_x_col": "patientstudymoduleattr__patient_weight",
                "df_y_col": "ctradiationdose__ctirradiationeventdata__dlp",
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                "grouping_choice": user_profile.plotGroupingChoice,
                "legend_title": "Acquisition protocol",
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "x_axis_title": "Patient mass (kg)",
                "y_axis_title": "DLP (mGy.cm)",
                "filename": "OpenREM CT acquisition protocol DLP vs patient mass",
                "return_as_dict": return_as_dict,
            }
            return_structure["acquisitionScatterDLPvsMass"] = plotly_scatter(
                df,
                parameter_dict,
            )

        if user_profile.plotCTAcquisitionCTDIOverTime:
            facet_title = "System"

            if user_profile.plotGroupingChoice == "series":
                facet_title = "Acquisition protocol"

            parameter_dict = {
                "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                "df_value_col": "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                "df_date_col": "study_date",
                "name_title": "Acquisition protocol",
                "value_title": "CTDI<sub>vol</sub> (mGy)",
                "date_title": "Study date",
                "facet_title": facet_title,
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                "time_period": plot_timeunit_period,
                "average_choices": average_choices + ["count"],
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "filename": "OpenREM CT acquisition protocol CTDI over time",
                "return_as_dict": return_as_dict,
            }
            result = construct_over_time_charts(
                df,
                parameter_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanCTDIOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianCTDIOverTime"] = result["median"]

        if user_profile.plotCTAcquisitionDLPOverTime:
            facet_title = "System"

            if user_profile.plotGroupingChoice == "series":
                facet_title = "Acquisition protocol"

            parameter_dict = {
                "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                "df_value_col": "ctradiationdose__ctirradiationeventdata__dlp",
                "df_date_col": "study_date",
                "name_title": "Acquisition protocol",
                "value_title": "DLP (mGy.cm)",
                "date_title": "Study date",
                "facet_title": facet_title,
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                "time_period": plot_timeunit_period,
                "average_choices": average_choices + ["count"],
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "filename": "OpenREM CT acquisition protocol DLP over time",
                "return_as_dict": return_as_dict,
            }
            result = construct_over_time_charts(
                df,
                parameter_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanDLPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianDLPOverTime"] = result["median"]

    #######################################################################
    # Prepare study- and request-level Pandas DataFrame to use for charts
    charts_of_interest = [
        user_profile.plotCTRequestFreq, user_profile.plotCTRequestMeanDLP,
        user_profile.plotCTRequestNumEvents, user_profile.plotCTRequestDLPOverTime,
        user_profile.plotCTStudyFreq, user_profile.plotCTStudyMeanDLP,
        user_profile.plotCTStudyMeanCTDI, user_profile.plotCTStudyNumEvents,
        user_profile.plotCTStudyMeanDLPOverTime, user_profile.plotCTStudyPerDayAndHour,
    ]
    if any(charts_of_interest):

        name_fields = []
        charts_of_interest = [
            user_profile.plotCTStudyMeanDLP, user_profile.plotCTStudyFreq,
            user_profile.plotCTStudyMeanDLPOverTime, user_profile.plotCTStudyPerDayAndHour,
            user_profile.plotCTStudyNumEvents, user_profile.plotCTStudyMeanCTDI,
        ]
        if any(charts_of_interest):
            name_fields.append("study_description")

        charts_of_interest = [
            user_profile.plotCTRequestMeanDLP, user_profile.plotCTRequestFreq,
            user_profile.plotCTRequestNumEvents, user_profile.plotCTRequestDLPOverTime,
        ]
        if any(charts_of_interest):
            name_fields.append("requested_procedure_code_meaning")

        value_fields = []
        charts_of_interest = [
            user_profile.plotCTStudyMeanDLP, user_profile.plotCTStudyMeanDLPOverTime,
            user_profile.plotCTRequestMeanDLP, user_profile.plotCTRequestDLPOverTime,
        ]
        if any(charts_of_interest):
            value_fields.append("total_dlp")
        if user_profile.plotCTStudyMeanCTDI:
            value_fields.append("ctradiationdose__ctirradiationeventdata__mean_ctdivol")
        if user_profile.plotCTStudyNumEvents or user_profile.plotCTRequestNumEvents:
            value_fields.append("number_of_events")

        date_fields = []
        time_fields = []
        charts_of_interest = [
            user_profile.plotCTStudyMeanDLPOverTime, user_profile.plotCTStudyPerDayAndHour,
            user_profile.plotCTRequestDLPOverTime,
        ]
        if any(charts_of_interest):
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
            data_point_name_remove_trailing_whitespace=user_profile.plotRemoveCategoryTrailingWhitespace,
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
                    ["study_description"],
                    "total_dlp",
                    stats_to_use=average_choices + ["count"],
                )

                parameter_dict = {
                    "df_name_col": "study_description",
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "sorted_category_list": sorted_study_dlp_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean DLP (mGy.cm)"
                    parameter_dict["filename"] = "OpenREM CT study description DLP mean"
                    parameter_dict["average_choice"] = "mean"
                    return_structure["studyMeanDLPData"], return_structure["studyMeanDLPDataCSV"] = plotly_barchart(
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMeanDLPData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median DLP (mGy.cm)"
                    parameter_dict["filename"] = "OpenREM CT study description DLP median"
                    parameter_dict["average_choice"] = "median"
                    return_structure["studyMedianDLPData"], return_structure["studyMedianDLPDataCSV"] = plotly_barchart(
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMedianDLPData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": "study_description",
                    "df_value_col": "total_dlp",
                    "value_axis_title": "DLP (mGy.cm)",
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description DLP boxplot",
                    "sorted_category_list": sorted_study_dlp_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                return_structure["studyBoxplotDLPData"] = plotly_boxplot(
                    df,
                    parameter_dict,
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

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": "total_dlp",
                    "value_axis_title": "DLP (mGy.cm)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description DLP histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "df_facet_category_list": facet_names,
                    "df_category_name_list": category_names,
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["studyHistogramDLPData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
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
                    ["study_description"],
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    stats_to_use=average_choices + ["count"],
                )

                parameter_dict = {
                    "df_name_col": "study_description",
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "sorted_category_list": sorted_study_ctdi_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean CTDI<sub>vol</sub> (mGy)"
                    parameter_dict["filename"] = "OpenREM CT study description CTDI mean"
                    parameter_dict["average_choice"] = "mean"
                    return_structure["studyMeanCTDIData"], return_structure["studyMeanCTDIDataCSV"] = plotly_barchart(
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMeanCTDIData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median CTDI<sub>vol</sub> (mGy)"
                    parameter_dict["filename"] = "OpenREM CT study description CTDI median"
                    parameter_dict["average_choice"] = "median"
                    return_structure["studyMedianCTDIData"], return_structure["studyMedianCTDIDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMedianCTDIData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": "study_description",
                    "df_value_col": "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    "value_axis_title": "CTDI<sub>vol</sub> (mGy)",
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description DLP boxplot",
                    "sorted_category_list": sorted_study_ctdi_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                return_structure["studyBoxplotCTDIData"] = plotly_boxplot(
                    df,
                    parameter_dict,
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

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    "value_axis_title": "CTDI<sub>vol</sub> (mGy)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description CTDI histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "df_facet_category_list": facet_names,
                    "df_category_name_list": category_names,
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["studyHistogramCTDIData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
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
                    ["study_description"],
                    "number_of_events",
                    stats_to_use=average_choices + ["count"],
                )

                parameter_dict = {
                    "df_name_col": "study_description",
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "sorted_category_list": sorted_study_events_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean events"
                    parameter_dict["filename"] = "OpenREM CT study description events mean"
                    parameter_dict["average_choice"] = "mean"
                    return_structure["studyMeanNumEventsData"], return_structure["studyMeanNumEventsDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMeanNumEventsData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median events"
                    parameter_dict["filename"] = "OpenREM CT study description events median"
                    parameter_dict["average_choice"] = "median"
                    return_structure["studyMedianNumEventsData"], return_structure["studyMedianNumEventsDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMedianNumEventsData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": "study_description",
                    "df_value_col": "number_of_events",
                    "value_axis_title": "Events",
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description events boxplot",
                    "sorted_category_list": sorted_study_events_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                return_structure["studyBoxplotNumEventsData"] = plotly_boxplot(
                    df,
                    parameter_dict,
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

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": "number_of_events",
                    "value_axis_title": "Events",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol events histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "df_facet_category_list": facet_names,
                    "df_category_name_list": category_names,
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["studyHistogramNumEventsData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if user_profile.plotCTStudyFreq:
            sorted_categories = None
            if sorted_study_dlp_categories:
                sorted_categories = sorted_study_dlp_categories
            elif sorted_study_ctdi_categories:
                sorted_categories = sorted_study_ctdi_categories
            elif sorted_study_events_categories:
                sorted_categories = sorted_study_events_categories

            parameter_dict = {
                "df_name_col": "study_description",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                "legend_title": "Study description",
                "df_x_axis_col": "x_ray_system_name",
                "x_axis_title": "System",
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "filename": "OpenREM CT study description frequency",
                "sorted_categories": sorted_categories,
                "groupby_cols": None,
                "facet_col": None,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "return_as_dict": return_as_dict,
            }
            return_structure["studyFrequencyData"], return_structure["studyFrequencyDataCSV"] = plotly_frequency_barchart(  # pylint: disable=line-too-long
                df,
                parameter_dict,
                csv_name="studyFrequencyData.csv",
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
                    ["requested_procedure_code_meaning"],
                    "total_dlp",
                    stats_to_use=average_choices + ["count"],
                )

                parameter_dict = {
                    "df_name_col": "requested_procedure_code_meaning",
                    "name_axis_title": "Requested procedure",
                    "colourmap": user_profile.plotColourMapChoice,
                    "sorted_category_list": sorted_request_dlp_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean DLP (mGy.cm)"
                    parameter_dict["filename"] = "OpenREM CT requested procedure DLP mean"
                    parameter_dict["average_choice"] = "mean"
                    return_structure["requestMeanDLPData"], return_structure["requestMeanDLPDataCSV"] = plotly_barchart(
                        df_aggregated,
                        parameter_dict,
                        csv_name="requestMeanDLPData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median DLP (mGy.cm)"
                    parameter_dict["filename"] = "OpenREM CT requested procedure DLP median"
                    parameter_dict["average_choice"] = "median"
                    return_structure["requestMedianDLPData"], return_structure["requestMedianDLPDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="requestMedianDLPData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": "requested_procedure_code_meaning",
                    "df_value_col": "total_dlp",
                    "value_axis_title": "DLP (mGy.cm)",
                    "name_axis_title": "Requested procedure",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT requested procedure DLP boxplot",
                    "sorted_category_list": sorted_request_dlp_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                return_structure["requestBoxplotDLPData"] = plotly_boxplot(
                    df,
                    parameter_dict,
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

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": "total_dlp",
                    "value_axis_title": "DLP (mGy.cm)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT requested procedure DLP histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "df_facet_category_list": facet_names,
                    "df_category_name_list": category_names,
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["requestHistogramDLPData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
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

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    ["requested_procedure_code_meaning"],
                    "number_of_events",
                    stats_to_use=average_choices + ["count"],
                )

                parameter_dict = {
                    "df_name_col": "requested_procedure_code_meaning",
                    "name_axis_title": "Requested procedure",
                    "colourmap": user_profile.plotColourMapChoice,
                    "sorted_category_list": sorted_request_events_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean events"
                    parameter_dict["filename"] = "OpenREM CT requested procedure events mean"
                    parameter_dict["average_choice"] = "mean"
                    return_structure["requestMeanNumEventsData"], return_structure["requestMeanNumEventsDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="requestMeanNumEventsData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median events"
                    parameter_dict["filename"] = "OpenREM CT requested procedure events median"
                    parameter_dict["average_choice"] = "median"
                    return_structure["requestMedianNumEventsData"], return_structure["requestMedianNumEventsDataCSV"] = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="requestMedianNumEventsData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": "requested_procedure_code_meaning",
                    "df_value_col": "number_of_events",
                    "value_axis_title": "Events",
                    "name_axis_title": "Requested procedure",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT requested procedure events boxplot",
                    "sorted_category_list": sorted_request_events_categories,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }
                return_structure["requestBoxplotNumEventsData"] = plotly_boxplot(
                    df,
                    parameter_dict,
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

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": "number_of_events",
                    "value_axis_title": "Events",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT requested procedure events histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "df_facet_category_list": facet_names,
                    "df_category_name_list": category_names,
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["requestHistogramNumEventsData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if user_profile.plotCTRequestFreq:
            sorted_categories = None
            if sorted_request_dlp_categories:
                sorted_categories = sorted_request_dlp_categories
            elif sorted_request_events_categories:
                sorted_categories = sorted_request_events_categories

            parameter_dict = {
                "df_name_col": "requested_procedure_code_meaning",
                "sorting_choice": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                "legend_title": "Requested procedure",
                "df_x_axis_col": "x_ray_system_name",
                "x_axis_title": "System",
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "filename": "OpenREM CT requested procedure frequency",
                "sorted_categories": sorted_categories,
                "groupby_cols": None,
                "facet_col": None,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "return_as_dict": return_as_dict,
            }
            return_structure["requestFrequencyData"], return_structure["requestFrequencyDataCSV"] = plotly_frequency_barchart(  # pylint: disable=line-too-long
                df,
                parameter_dict,
                csv_name="requestFrequencyData.csv"
            )

        if user_profile.plotCTStudyMeanDLPOverTime:
            facet_title = "System"

            if user_profile.plotGroupingChoice == "series":
                facet_title = "Study description"

            parameter_dict = {
                "df_name_col": "study_description",
                "df_value_col": "total_dlp",
                "df_date_col": "study_date",
                "name_title": "Study description",
                "value_title": "DLP (mGy.cm)",
                "date_title": "Study date",
                "facet_title": facet_title,
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                "time_period": plot_timeunit_period,
                "average_choices": average_choices + ["count"],
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "filename": "OpenREM CT study description DLP over time",
                "return_as_dict": return_as_dict,
            }
            result = construct_over_time_charts(
                df,
                parameter_dict,
            )

            if user_profile.plotMean:
                return_structure["studyMeanDLPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["studyMedianDLPOverTime"] = result["median"]

        if user_profile.plotCTRequestDLPOverTime:
            facet_title = "System"

            if user_profile.plotGroupingChoice == "series":
                facet_title = "Requested procedure"

            parameter_dict = {
                "df_name_col": "requested_procedure_code_meaning",
                "df_value_col": "total_dlp",
                "df_date_col": "study_date",
                "name_title": "Requested procedure",
                "value_title": "DLP (mGy.cm)",
                "date_title": "Study date",
                "facet_title": facet_title,
                "sorting": [
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                "time_period": plot_timeunit_period,
                "average_choices": average_choices + ["count"],
                "grouping_choice": user_profile.plotGroupingChoice,
                "colourmap": user_profile.plotColourMapChoice,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "filename": "OpenREM CT requested procedure DLP over time",
                "return_as_dict": return_as_dict,
            }
            result = construct_over_time_charts(
                df,
                parameter_dict,
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
                return_as_dict=return_as_dict,
            )
        #######################################################################

    return return_structure


def ct_chart_form_processing(request, user_profile):
    # pylint: disable=too-many-statements
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
