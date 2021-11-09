# pylint: disable=too-many-lines
import logging
from datetime import datetime
from django.db.models import (
    Subquery,
    OuterRef,
)
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import JsonResponse
from django.utils.safestring import mark_safe
from django.utils.translation import gettext as _
from remapp.forms import (
    CTChartOptionsForm,
    CTChartOptionsFormIncStandard,
)
from remapp.interface.mod_filters import ct_acq_filter
from remapp.models import (
    create_user_profile,
    CommonVariables,
    StandardNames,
    StandardNameSettings,
)
from remapp.views_admin import (
    required_average_choices,
    required_ct_acquisition_types,
    initialise_ct_form_data,
    set_ct_chart_options,
    set_average_chart_options,
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
    create_standard_study_df,
)

logger = logging.getLogger(__name__)


def generate_required_ct_charts_list(profile):
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list("enable_standard_names", flat=True)[0]

    required_charts = []

    charts_of_interest = [
        profile.plotCTAcquisitionDLPOverTime,
        profile.plotCTAcquisitionCTDIOverTime,
        profile.plotCTStudyMeanDLPOverTime,
        profile.plotCTRequestDLPOverTime,
    ]
    if enable_standard_names:
        charts_of_interest.append(profile.plotCTStandardStudyMeanDLPOverTime)

    if any(charts_of_interest):
        keys = list(dict(profile.TIME_PERIOD).keys())
        values = list(dict(profile.TIME_PERIOD).values())
        time_period = (values[keys.index(profile.plotCTOverTimePeriod)]).lower()

    if profile.plotCTAcquisitionFreq:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol frequency",
                "var_name": "acquisitionFrequency",
            }
        )

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

    if profile.plotCTAcquisitionDLPvsMass:
        required_charts.append(
            {
                "title": "Chart of acquisition protocol DLP vs patient mass",
                "var_name": "acquisitionScatterDLPvsMass",
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

    if enable_standard_names:
        if profile.plotCTStandardAcquisitionMeanDLP:
            if profile.plotMean:
                required_charts.append(
                    {
                        "title": "Chart of standard acquisition name mean DLP",
                        "var_name": "standardAcquisitionMeanDLP",
                    }
                )
            if profile.plotMedian:
                required_charts.append(
                    {
                        "title": "Chart of standard acquisition name median DLP",
                        "var_name": "standardAcquisitionMedianDLP",
                    }
                )
            if profile.plotBoxplots:
                required_charts.append(
                    {
                        "title": "Boxplot of standard acquisition name DLP",
                        "var_name": "standardAcquisitionBoxplotDLP",
                    }
                )
            if profile.plotHistograms:
                required_charts.append(
                    {
                        "title": "Histogram of standard acquisition name DLP",
                        "var_name": "standardAcquisitionHistogramDLP",
                    }
                )

    if profile.plotCTStudyFreq:
        required_charts.append(
            {
                "title": "Chart of study description frequency",
                "var_name": "studyFrequency",
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

    if profile.plotCTStudyPerDayAndHour:
        required_charts.append(
            {
                "title": "Chart of study description workload",
                "var_name": "studyWorkload",
            }
        )

    if enable_standard_names:
        if profile.plotCTStandardStudyFreq:
            required_charts.append(
                {
                    "title": "Chart of standard study name frequency",
                    "var_name": "standardStudyFrequency",
                }
            )

        if profile.plotCTStandardStudyMeanDLP:
            if profile.plotMean:
                required_charts.append(
                    {
                        "title": "Chart of standard study name mean DLP",
                        "var_name": "standardStudyMeanDLP",
                    }
                )
            if profile.plotMedian:
                required_charts.append(
                    {
                        "title": "Chart of standard study name median DLP",
                        "var_name": "standardStudyMedianDLP",
                    }
                )
            if profile.plotBoxplots:
                required_charts.append(
                    {
                        "title": "Boxplot of standard study name DLP",
                        "var_name": "standardStudyBoxplotDLP",
                    }
                )
            if profile.plotHistograms:
                required_charts.append(
                    {
                        "title": "Histogram of standard study name DLP",
                        "var_name": "standardStudyHistogramDLP",
                    }
                )

        if profile.plotCTStandardStudyNumEvents:
            if profile.plotMean:
                required_charts.append(
                    {
                        "title": "Chart of standard study name mean number of events",
                        "var_name": "standardStudyMeanNumEvents",
                    }
                )
            if profile.plotMedian:
                required_charts.append(
                    {
                        "title": "Chart of standard study name median number of events",
                        "var_name": "standardStudyMedianNumEvents",
                    }
                )
            if profile.plotBoxplots:
                required_charts.append(
                    {
                        "title": "Boxplot of standard study name number of events",
                        "var_name": "standardStudyBoxplotNumEvents",
                    }
                )
            if profile.plotHistograms:
                required_charts.append(
                    {
                        "title": "Histogram of standard study name number of events",
                        "var_name": "standardStudyHistogramNumEvents",
                    }
                )

        if profile.plotCTStandardStudyMeanDLPOverTime:
            if profile.plotMean:
                required_charts.append(
                    {
                        "title": "Chart of standard study name mean DLP over time ("
                        + time_period
                        + ")",
                        "var_name": "standardStudyMeanDLPOverTime",
                    }
                )
            if profile.plotMedian:
                required_charts.append(
                    {
                        "title": "Chart of standard study name median DLP over time ("
                        + time_period
                        + ")",
                        "var_name": "standardStudyMedianDLPOverTime",
                    }
                )

        if profile.plotCTStandardStudyPerDayAndHour:
            required_charts.append(
                {
                    "title": "Chart of standard study name workload",
                    "var_name": "standardStudyWorkload",
                }
            )

    if profile.plotCTRequestFreq:
        required_charts.append(
            {
                "title": "Chart of requested procedure frequency",
                "var_name": "requestFrequency",
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

    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list("enable_standard_names", flat=True)[0]

    # Set the Plotly chart theme
    plotly_set_default_theme(user_profile.plotThemeChoice)

    return_structure = {}

    average_choices = []
    if user_profile.plotMean:
        average_choices.append(CommonVariables.MEAN)
    if user_profile.plotMedian:
        average_choices.append(CommonVariables.MEDIAN)

    charts_of_interest = [
        user_profile.plotCTAcquisitionDLPOverTime,
        user_profile.plotCTAcquisitionCTDIOverTime,
        user_profile.plotCTStudyMeanDLPOverTime,
        user_profile.plotCTRequestDLPOverTime,
    ]
    if enable_standard_names:
        charts_of_interest.append(user_profile.plotCTStandardStudyMeanDLPOverTime)
    if any(charts_of_interest):
        plot_timeunit_period = user_profile.plotCTOverTimePeriod

    #######################################################################
    # Prepare acquisition-level Pandas DataFrame to use for charts
    charts_of_interest = [
        user_profile.plotCTAcquisitionFreq,
        user_profile.plotCTAcquisitionMeanCTDI,
        user_profile.plotCTAcquisitionMeanDLP,
        user_profile.plotCTAcquisitionCTDIvsMass,
        user_profile.plotCTAcquisitionDLPvsMass,
        user_profile.plotCTAcquisitionCTDIOverTime,
        user_profile.plotCTAcquisitionDLPOverTime,
    ]
    if enable_standard_names:
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionMeanDLP)

    if any(charts_of_interest):

        name_fields = ["ctradiationdose__ctirradiationeventdata__acquisition_protocol"]
        name_fields.append(
            "ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning"
        )
        name_fields.append(
            "ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_value"
        )
        if enable_standard_names:
            name_fields.append("standard_acquisition_name")

        value_fields = []
        charts_of_interest = [
            user_profile.plotCTAcquisitionMeanDLP,
            user_profile.plotCTAcquisitionDLPvsMass,
            user_profile.plotCTAcquisitionDLPOverTime,
        ]
        if enable_standard_names:
            charts_of_interest.append(user_profile.plotCTStandardAcquisitionMeanDLP)
        if any(charts_of_interest):
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

        query_set = f.qs
        if enable_standard_names:
            standard_names = StandardNames.objects.filter(modality__iexact="CT")
            query_set = f.qs.annotate(
                standard_acquisition_name=Subquery(
                    standard_names.filter(
                        acquisition_protocol=OuterRef("ctradiationdose__ctirradiationeventdata__acquisition_protocol")
                    ).values("standard_name")
                )
            )

        df = create_dataframe(
            query_set,
            fields,
            data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
            data_point_name_remove_whitespace_padding=user_profile.plotRemoveCategoryWhitespacePadding,
            char_wrap=user_profile.plotLabelCharWrap,
            uid="ctradiationdose__ctirradiationeventdata__pk",
        )

        # Only keep the required acquisition type code meanings and values
        code_meanings_to_keep = required_ct_acquisition_types(user_profile)

        code_values_to_keep = [
            CommonVariables.CT_ACQUISITION_TYPE_CODES[k.title()]
            for k in code_meanings_to_keep
        ]
        code_values_to_keep = [j for sub in code_values_to_keep for j in sub]

        if not code_values_to_keep and not code_meanings_to_keep:
            chart_message = _(
                "<br/>This may be because there are no acquisition types selected in the chart options. "
                "Try selecting at least one acquisition type."
            )
        else:
            chart_message = ""

        # Make the code meanings and values lower case if the user has selected case-insensitive categories. The
        # create_dataframe method will make all the code meaning and value categories in the queryset lower case,
        # so the meanings and values to keep also have to be made lower case otherwise the df.isin command on the
        # next line does not work as expected.
        if user_profile.plotCaseInsensitiveCategories:
            code_meanings_to_keep = [
                each_string.lower() for each_string in code_meanings_to_keep
            ]
            code_values_to_keep = [
                each_string.lower() for each_string in code_values_to_keep
            ]

        df = df[
            df.isin(
                {
                    "ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_value": code_values_to_keep,
                    "ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning": code_meanings_to_keep,
                }
            ).any(axis=1)
        ]
        #######################################################################

        #######################################################################
        # Create the required acquisition-level charts
        if user_profile.plotCTAcquisitionMeanDLP:

            if user_profile.plotBoxplots and "median" not in average_choices:
                average_choices = average_choices + ["median"]

            name_field = "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
            value_field = "ctradiationdose__ctirradiationeventdata__dlp"

            df_aggregated = create_dataframe_aggregates(
                df,
                [name_field],
                value_field,
                stats_to_use=average_choices + ["count"],
            )

            if user_profile.plotMean or user_profile.plotMedian:

                parameter_dict = {
                    "df_name_col": name_field,
                    "name_axis_title": "Acquisition protocol",
                    "colourmap": user_profile.plotColourMapChoice,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "custom_msg_line": chart_message,
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean DLP (mGy.cm)"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT acquisition protocol DLP mean"
                    parameter_dict["average_choice"] = "mean"
                    (
                        return_structure["acquisitionMeanDLPData"],
                        return_structure["acquisitionMeanDLPDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="acquisitionMeanDLPData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median DLP (mGy.cm)"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT acquisition protocol DLP median"
                    parameter_dict["average_choice"] = "median"
                    (
                        return_structure["acquisitionMedianDLPData"],
                        return_structure["acquisitionMedianDLPDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="acquisitionMedianDLPData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": name_field,
                    "df_value_col": value_field,
                    "value_axis_title": "DLP (mGy.cm)",
                    "name_axis_title": "Acquisition protocol",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol DLP boxplot",
                    "facet_col": None,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                    "custom_msg_line": chart_message,
                }

                return_structure["acquisitionBoxplotDLPData"] = plotly_boxplot(
                    df,
                    parameter_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = name_field
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = name_field
                    legend_title = "System"

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": value_field,
                    "value_axis_title": "DLP (mGy.cm)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol DLP histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                    "custom_msg_line": chart_message,
                }
                return_structure[
                    "acquisitionHistogramDLPData"
                ] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if user_profile.plotCTAcquisitionMeanCTDI:

            if user_profile.plotBoxplots and "median" not in average_choices:
                average_choices = average_choices + ["median"]

            name_field = "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
            value_field = "ctradiationdose__ctirradiationeventdata__mean_ctdivol"

            df_aggregated = create_dataframe_aggregates(
                df,
                [name_field],
                value_field,
                stats_to_use=average_choices + ["count"],
            )

            if user_profile.plotMean or user_profile.plotMedian:

                parameter_dict = {
                    "df_name_col": name_field,
                    "name_axis_title": "Acquisition protocol",
                    "colourmap": user_profile.plotColourMapChoice,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                    "custom_msg_line": chart_message,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean CTDI<sub>vol</sub> (mGy)"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT acquisition protocol CTDI mean"
                    parameter_dict["average_choice"] = "mean"
                    (
                        return_structure["acquisitionMeanCTDIData"],
                        return_structure["acquisitionMeanCTDIDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="acquisitionMeanCTDIData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict[
                        "value_axis_title"
                    ] = "Median CTDI<sub>vol</sub> (mGy)"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT acquisition protocol CTDI median"
                    parameter_dict["average_choice"] = "median"
                    (
                        return_structure["acquisitionMedianCTDIData"],
                        return_structure["acquisitionMedianCTDIDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="acquisitionMedianCTDIData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": name_field,
                    "df_value_col": value_field,
                    "value_axis_title": "CTDI<sub>vol</sub> (mGy)",
                    "name_axis_title": "Acquisition protocol",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol CTDI boxplot",
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "return_as_dict": return_as_dict,
                    "custom_msg_line": chart_message,
                }

                return_structure["acquisitionBoxplotCTDIData"] = plotly_boxplot(
                    df,
                    parameter_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = name_field
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = name_field
                    legend_title = "System"

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": value_field,
                    "value_axis_title": "CTDI<sub>vol</sub> (mGy)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT acquisition protocol CTDI histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                    "custom_msg_line": chart_message,
                }
                return_structure[
                    "acquisitionHistogramCTDIData"
                ] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if user_profile.plotCTAcquisitionFreq:
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
                "groupby_cols": None,
                "facet_col": None,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "return_as_dict": return_as_dict,
                "custom_msg_line": chart_message,
            }
            (
                return_structure["acquisitionFrequencyData"],
                return_structure["acquisitionFrequencyDataCSV"],
            ) = plotly_frequency_barchart(  # pylint: disable=line-too-long
                df,
                parameter_dict,
                csv_name="acquisitionFrequencyData.csv",
            )

        if user_profile.plotCTAcquisitionCTDIvsMass:
            parameter_dict = {
                "df_name_col": "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                "df_x_col": "patientstudymoduleattr__patient_weight",
                "df_y_col": "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                "sorting_choice": [
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
                "custom_msg_line": chart_message,
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
                "sorting_choice": [
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
                "custom_msg_line": chart_message,
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
                "sorting_choice": [
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
                "custom_msg_line": chart_message,
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
                "sorting_choice": [
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
                "custom_msg_line": chart_message,
            }
            result = construct_over_time_charts(
                df,
                parameter_dict,
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanDLPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianDLPOverTime"] = result["median"]

        if enable_standard_names:
            df_without_blanks = df[(df["standard_acquisition_name"] != "blank") & (df["standard_acquisition_name"] != "Blank")]

            if user_profile.plotCTStandardAcquisitionMeanDLP:

                if user_profile.plotBoxplots and "median" not in average_choices:
                    average_choices = average_choices + ["median"]

                name_field = "standard_acquisition_name"
                value_field = "ctradiationdose__ctirradiationeventdata__dlp"

                # Calculate an aggregated dataframe
                df_aggregated = create_dataframe_aggregates(
                    df_without_blanks,
                    [name_field],
                    value_field,
                    stats_to_use=average_choices + ["count"],
                )

                # Drop blank values - I don't know why these appear
                df_aggregated = df_aggregated[(df_aggregated["standard_acquisition_name"] != "blank") & (df_aggregated["standard_acquisition_name"] != "Blank")]

                if user_profile.plotMean or user_profile.plotMedian:

                    parameter_dict = {
                        "df_name_col": name_field,
                        "name_axis_title": "Standard acquisition name",
                        "colourmap": user_profile.plotColourMapChoice,
                        "facet_col": None,
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "return_as_dict": return_as_dict,
                        "sorting_choice": [
                            user_profile.plotInitialSortingDirection,
                            user_profile.plotCTInitialSortingChoice,
                        ],
                        "custom_msg_line": chart_message,
                    }
                    if user_profile.plotMean:
                        parameter_dict["value_axis_title"] = "Mean DLP (mGy.cm)"
                        parameter_dict[
                            "filename"
                        ] = "OpenREM CT standard acquisition name DLP mean"
                        parameter_dict["average_choice"] = "mean"
                        (
                            return_structure["standardAcquisitionMeanDLPData"],
                            return_structure["standardAcquisitionMeanDLPDataCSV"],
                        ) = plotly_barchart(  # pylint: disable=line-too-long
                            df_aggregated,
                            parameter_dict,
                            csv_name="standardAcquisitionMeanDLPData.csv",
                        )

                    if user_profile.plotMedian:
                        parameter_dict["value_axis_title"] = "Median DLP (mGy.cm)"
                        parameter_dict[
                            "filename"
                        ] = "OpenREM CT standard acquisition name DLP median"
                        parameter_dict["average_choice"] = "median"
                        (
                            return_structure["standardAcquisitionMedianDLPData"],
                            return_structure["standardAcquisitionMedianDLPDataCSV"],
                        ) = plotly_barchart(  # pylint: disable=line-too-long
                            df_aggregated,
                            parameter_dict,
                            csv_name="standardAcquisitionMedianDLPData.csv",
                        )

                if user_profile.plotBoxplots:
                    parameter_dict = {
                        "df_name_col": name_field,
                        "df_value_col": value_field,
                        "value_axis_title": "DLP (mGy.cm)",
                        "name_axis_title": "Standard acquisition name",
                        "colourmap": user_profile.plotColourMapChoice,
                        "filename": "OpenREM CT standard acquisition name DLP boxplot",
                        "facet_col": None,
                        "sorting_choice": [
                            user_profile.plotInitialSortingDirection,
                            user_profile.plotCTInitialSortingChoice,
                        ],
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "return_as_dict": return_as_dict,
                        "custom_msg_line": chart_message,
                    }

                    return_structure["standardAcquisitionBoxplotDLPData"] = plotly_boxplot(
                        df_without_blanks,
                        parameter_dict,
                    )

                if user_profile.plotHistograms:
                    category_names_col = name_field
                    group_by_col = "x_ray_system_name"
                    legend_title = "Standard acquisition name"

                    if user_profile.plotGroupingChoice == "series":
                        category_names_col = "x_ray_system_name"
                        group_by_col = name_field
                        legend_title = "System"

                    parameter_dict = {
                        "df_facet_col": group_by_col,
                        "df_category_col": category_names_col,
                        "df_value_col": value_field,
                        "value_axis_title": "DLP (mGy.cm)",
                        "legend_title": legend_title,
                        "n_bins": user_profile.plotHistogramBins,
                        "colourmap": user_profile.plotColourMapChoice,
                        "filename": "OpenREM CT standard acquisition name DLP histogram",
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "sorting_choice": [
                            user_profile.plotInitialSortingDirection,
                            user_profile.plotCTInitialSortingChoice,
                        ],
                        "global_max_min": user_profile.plotHistogramGlobalBins,
                        "return_as_dict": return_as_dict,
                        "custom_msg_line": chart_message,
                    }
                    return_structure[
                        "standardAcquisitionHistogramDLPData"
                    ] = plotly_histogram_barchart(
                        df_without_blanks,
                        parameter_dict,
                    )

    #######################################################################
    # Prepare study- and request-level Pandas DataFrame to use for charts
    charts_of_interest = [
        user_profile.plotCTRequestFreq,
        user_profile.plotCTRequestMeanDLP,
        user_profile.plotCTRequestNumEvents,
        user_profile.plotCTRequestDLPOverTime,
        user_profile.plotCTStudyFreq,
        user_profile.plotCTStudyMeanDLP,
        user_profile.plotCTStudyMeanCTDI,
        user_profile.plotCTStudyNumEvents,
        user_profile.plotCTStudyMeanDLPOverTime,
        user_profile.plotCTStudyPerDayAndHour,
    ]
    if enable_standard_names:
        charts_of_interest.append(user_profile.plotCTStandardStudyMeanDLP)
        charts_of_interest.append(user_profile.plotCTStandardStudyNumEvents)
        charts_of_interest.append(user_profile.plotCTStandardStudyFreq)
        charts_of_interest.append(user_profile.plotCTStandardStudyPerDayAndHour)
        charts_of_interest.append(user_profile.plotCTStandardStudyMeanDLPOverTime)

    if any(charts_of_interest):
        name_fields = []
        charts_of_interest = [
            user_profile.plotCTStudyMeanDLP,
            user_profile.plotCTStudyFreq,
            user_profile.plotCTStudyMeanDLPOverTime,
            user_profile.plotCTStudyPerDayAndHour,
            user_profile.plotCTStudyNumEvents,
            user_profile.plotCTStudyMeanCTDI,
        ]
        if any(charts_of_interest):
            name_fields.append("study_description")

        charts_of_interest = [
            user_profile.plotCTRequestMeanDLP,
            user_profile.plotCTRequestFreq,
            user_profile.plotCTRequestNumEvents,
            user_profile.plotCTRequestDLPOverTime,
        ]
        if any(charts_of_interest):
            name_fields.append("requested_procedure_code_meaning")

        if enable_standard_names:
            charts_of_interest = [
                user_profile.plotCTStandardStudyMeanDLP,
                user_profile.plotCTStandardStudyNumEvents,
                user_profile.plotCTStandardStudyFreq,
                user_profile.plotCTStandardStudyPerDayAndHour,
                user_profile.plotCTStandardStudyMeanDLPOverTime,
            ]
            if any(charts_of_interest):
                name_fields.append("standard_study_name")
                name_fields.append("standard_request_name")
                name_fields.append("standard_procedure_name")

        value_fields = []
        charts_of_interest = [
            user_profile.plotCTStudyMeanDLP,
            user_profile.plotCTStudyMeanDLPOverTime,
            user_profile.plotCTRequestMeanDLP,
            user_profile.plotCTRequestDLPOverTime,
        ]
        if enable_standard_names:
            charts_of_interest.append(user_profile.plotCTStandardStudyMeanDLP)
            charts_of_interest.append(user_profile.plotCTStandardStudyMeanDLPOverTime)

        if any(charts_of_interest):
            value_fields.append("total_dlp")

        if user_profile.plotCTStudyMeanCTDI:
            value_fields.append("ctradiationdose__ctirradiationeventdata__mean_ctdivol")

        charts_of_interest = [
            user_profile.plotCTStudyNumEvents,
            user_profile.plotCTRequestNumEvents
        ]
        if enable_standard_names:
            charts_of_interest.append(user_profile.plotCTStandardStudyNumEvents)

        if any(charts_of_interest):
            value_fields.append("number_of_events")

        date_fields = []
        time_fields = []
        charts_of_interest = [
            user_profile.plotCTStudyMeanDLPOverTime,
            user_profile.plotCTStudyPerDayAndHour,
            user_profile.plotCTRequestDLPOverTime,
        ]
        if enable_standard_names:
            charts_of_interest.append(user_profile.plotCTStandardStudyPerDayAndHour)
            charts_of_interest.append(user_profile.plotCTStandardStudyMeanDLPOverTime)
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
            data_point_name_remove_whitespace_padding=user_profile.plotRemoveCategoryWhitespacePadding,
            char_wrap=user_profile.plotLabelCharWrap,
            uid="pk",
        )
        #######################################################################

        #######################################################################
        # Create the required study- and request-level charts
        if user_profile.plotCTStudyMeanDLP:

            if user_profile.plotBoxplots and "median" not in average_choices:
                average_choices = average_choices + ["median"]

            name_field = "study_description"
            value_field = "total_dlp"

            df_aggregated = create_dataframe_aggregates(
                df,
                [name_field],
                value_field,
                stats_to_use=average_choices + ["count"],
            )

            if user_profile.plotMean or user_profile.plotMedian:

                parameter_dict = {
                    "df_name_col": name_field,
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean DLP (mGy.cm)"
                    parameter_dict["filename"] = "OpenREM CT study description DLP mean"
                    parameter_dict["average_choice"] = "mean"
                    (
                        return_structure["studyMeanDLPData"],
                        return_structure["studyMeanDLPDataCSV"],
                    ) = plotly_barchart(
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMeanDLPData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median DLP (mGy.cm)"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT study description DLP median"
                    parameter_dict["average_choice"] = "median"
                    (
                        return_structure["studyMedianDLPData"],
                        return_structure["studyMedianDLPDataCSV"],
                    ) = plotly_barchart(
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMedianDLPData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": name_field,
                    "df_value_col": value_field,
                    "value_axis_title": "DLP (mGy.cm)",
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description DLP boxplot",
                    "facet_col": None,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }

                return_structure["studyBoxplotDLPData"] = plotly_boxplot(
                    df,
                    parameter_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = name_field
                group_by_col = "x_ray_system_name"
                legend_title = "Study description"

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = name_field
                    legend_title = "System"

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": value_field,
                    "value_axis_title": "DLP (mGy.cm)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description DLP histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["studyHistogramDLPData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if enable_standard_names:
            if (
                user_profile.plotCTStandardStudyMeanDLP or
                user_profile.plotCTStandardStudyFreq or
                user_profile.plotCTStandardStudyPerDayAndHour or
                user_profile.plotCTStandardStudyMeanDLPOverTime or
                user_profile.plotCTStandardStudyNumEvents
            ):

                # Create a standard name data frame to be used by the study-level charts
                std_field_name = "standard_study"
                value_fields = []

                if user_profile.plotCTStandardStudyPerDayAndHour or user_profile.plotCTStandardStudyMeanDLPOverTime:
                    value_fields.extend(["study_date", "study_time"])

                if user_profile.plotCTStandardStudyMeanDLP or user_profile.plotCTStandardStudyMeanDLPOverTime:
                    value_fields.append("total_dlp")

                if user_profile.plotCTStandardStudyNumEvents:
                    value_fields.append("number_of_events")

                standard_name_df = create_standard_study_df(df, std_name=std_field_name, df_agg_cols=value_fields)

                if user_profile.plotCTStandardStudyMeanDLP:

                    if user_profile.plotBoxplots and "median" not in average_choices:
                        average_choices = average_choices + ["median"]

                    df_aggregated = create_dataframe_aggregates(
                        standard_name_df,
                        [std_field_name],
                        "total_dlp",
                        stats_to_use=average_choices + ["count"],
                    )

                    if user_profile.plotMean or user_profile.plotMedian:

                        parameter_dict = {
                            "df_name_col": std_field_name,
                            "name_axis_title": "Standard study name",
                            "colourmap": user_profile.plotColourMapChoice,
                            "facet_col": None,
                            "facet_col_wrap": user_profile.plotFacetColWrapVal,
                            "return_as_dict": return_as_dict,
                            "sorting_choice": [
                                user_profile.plotInitialSortingDirection,
                                user_profile.plotCTInitialSortingChoice,
                            ],
                        }
                        if user_profile.plotMean:
                            parameter_dict["value_axis_title"] = "Mean DLP (mGy.cm)"
                            parameter_dict["filename"] = "OpenREM CT standard study name DLP mean"
                            parameter_dict["average_choice"] = "mean"
                            (
                                return_structure["standardStudyMeanDLPData"],
                                return_structure["standardStudyMeanDLPDataCSV"],
                            ) = plotly_barchart(
                                df_aggregated,
                                parameter_dict,
                                csv_name="standardStudyMeanDLPData.csv",
                            )

                        if user_profile.plotMedian:
                            parameter_dict["value_axis_title"] = "Median DLP (mGy.cm)"
                            parameter_dict[
                                "filename"
                            ] = "OpenREM CT standard study name DLP median"
                            parameter_dict["average_choice"] = "median"
                            (
                                return_structure["standardStudyMedianDLPData"],
                                return_structure["standardStudyMedianDLPDataCSV"],
                            ) = plotly_barchart(
                                df_aggregated,
                                parameter_dict,
                                csv_name="standardStudyMedianDLPData.csv",
                            )

                    if user_profile.plotBoxplots:
                        parameter_dict = {
                            "df_name_col": std_field_name,
                            "df_value_col": "total_dlp",
                            "value_axis_title": "DLP (mGy.cm)",
                            "name_axis_title": "Standard study name",
                            "colourmap": user_profile.plotColourMapChoice,
                            "filename": "OpenREM CT standard study name DLP boxplot",
                            "facet_col": None,
                            "sorting_choice": [
                                user_profile.plotInitialSortingDirection,
                                user_profile.plotCTInitialSortingChoice,
                            ],
                            "facet_col_wrap": user_profile.plotFacetColWrapVal,
                            "return_as_dict": return_as_dict,
                        }

                        return_structure["standardStudyBoxplotDLPData"] = plotly_boxplot(
                            standard_name_df,
                            parameter_dict,
                        )

                    if user_profile.plotHistograms:
                        category_names_col = std_field_name
                        group_by_col = "x_ray_system_name"
                        legend_title = "Standard study name"

                        if user_profile.plotGroupingChoice == "series":
                            category_names_col = "x_ray_system_name"
                            group_by_col = std_field_name
                            legend_title = "System"

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
                            "sorting_choice": [
                                user_profile.plotInitialSortingDirection,
                                user_profile.plotCTInitialSortingChoice,
                            ],
                            "global_max_min": user_profile.plotHistogramGlobalBins,
                            "return_as_dict": return_as_dict,
                        }
                        return_structure["standardStudyHistogramDLPData"] = plotly_histogram_barchart(
                            standard_name_df,
                            parameter_dict,
                        )

                if user_profile.plotCTStandardStudyNumEvents:

                    if user_profile.plotBoxplots and "median" not in average_choices:
                        average_choices = average_choices + ["median"]

                    df_aggregated = create_dataframe_aggregates(
                        standard_name_df,
                        [std_field_name],
                        "number_of_events",
                        stats_to_use=average_choices + ["count"],
                    )

                    if user_profile.plotMean or user_profile.plotMedian:

                        parameter_dict = {
                            "df_name_col": std_field_name,
                            "name_axis_title": "Standard study name",
                            "colourmap": user_profile.plotColourMapChoice,
                            "facet_col": None,
                            "facet_col_wrap": user_profile.plotFacetColWrapVal,
                            "return_as_dict": return_as_dict,
                            "sorting_choice": [
                                user_profile.plotInitialSortingDirection,
                                user_profile.plotCTInitialSortingChoice,
                            ],
                        }
                        if user_profile.plotMean:
                            parameter_dict["value_axis_title"] = "Mean events"
                            parameter_dict[
                                "filename"
                            ] = "OpenREM CT standard study name events mean"
                            parameter_dict["average_choice"] = "mean"
                            (
                                return_structure["standardStudyMeanNumEventsData"],
                                return_structure["standardStudyMeanNumEventsDataCSV"],
                            ) = plotly_barchart(  # pylint: disable=line-too-long
                                df_aggregated,
                                parameter_dict,
                                csv_name="standardStudyMeanNumEventsData.csv",
                            )

                        if user_profile.plotMedian:
                            parameter_dict["value_axis_title"] = "Median events"
                            parameter_dict[
                                "filename"
                            ] = "OpenREM CT standard study name events median"
                            parameter_dict["average_choice"] = "median"
                            (
                                return_structure["standardStudyMedianNumEventsData"],
                                return_structure["standardStudyMedianNumEventsDataCSV"],
                            ) = plotly_barchart(  # pylint: disable=line-too-long
                                df_aggregated,
                                parameter_dict,
                                csv_name="standardStudyMedianNumEventsData.csv",
                            )

                    if user_profile.plotBoxplots:
                        parameter_dict = {
                            "df_name_col": std_field_name,
                            "df_value_col": "number_of_events",
                            "value_axis_title": "Events",
                            "name_axis_title": "Standard study name",
                            "colourmap": user_profile.plotColourMapChoice,
                            "filename": "OpenREM CT standard study name events boxplot",
                            "facet_col": None,
                            "sorting_choice": [
                                user_profile.plotInitialSortingDirection,
                                user_profile.plotCTInitialSortingChoice,
                            ],
                            "facet_col_wrap": user_profile.plotFacetColWrapVal,
                            "return_as_dict": return_as_dict,
                        }

                        return_structure["standardStudyBoxplotNumEventsData"] = plotly_boxplot(
                            standard_name_df,
                            parameter_dict,
                        )

                    if user_profile.plotHistograms:
                        category_names_col = std_field_name
                        group_by_col = "x_ray_system_name"
                        legend_title = "Standard study name"

                        if user_profile.plotGroupingChoice == "series":
                            category_names_col = "x_ray_system_name"
                            group_by_col = std_field_name
                            legend_title = "System"

                        parameter_dict = {
                            "df_facet_col": group_by_col,
                            "df_category_col": category_names_col,
                            "df_value_col": "number_of_events",
                            "value_axis_title": "Events",
                            "legend_title": legend_title,
                            "n_bins": user_profile.plotHistogramBins,
                            "colourmap": user_profile.plotColourMapChoice,
                            "filename": "OpenREM CT standard study name events histogram",
                            "facet_col_wrap": user_profile.plotFacetColWrapVal,
                            "sorting_choice": [
                                user_profile.plotInitialSortingDirection,
                                user_profile.plotCTInitialSortingChoice,
                            ],
                            "global_max_min": user_profile.plotHistogramGlobalBins,
                            "return_as_dict": return_as_dict,
                        }
                        return_structure[
                            "standardStudyHistogramNumEventsData"
                        ] = plotly_histogram_barchart(
                            standard_name_df,
                            parameter_dict,
                        )

                if user_profile.plotCTStandardStudyFreq:
                    parameter_dict = {
                        "df_name_col": std_field_name,
                        "sorting_choice": [
                            user_profile.plotInitialSortingDirection,
                            user_profile.plotCTInitialSortingChoice,
                        ],
                        "legend_title": "Standard study name",
                        "df_x_axis_col": "x_ray_system_name",
                        "x_axis_title": "System",
                        "grouping_choice": user_profile.plotGroupingChoice,
                        "colourmap": user_profile.plotColourMapChoice,
                        "filename": "OpenREM CT standard study name frequency",
                        "groupby_cols": None,
                        "facet_col": None,
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "return_as_dict": return_as_dict,
                    }
                    (
                        return_structure["standardStudyFrequencyData"],
                        return_structure["standardStudyFrequencyDataCSV"],
                    ) = plotly_frequency_barchart(  # pylint: disable=line-too-long
                        standard_name_df,
                        parameter_dict,
                        csv_name="standardStudyFrequencyData.csv",
                    )

                if user_profile.plotCTStandardStudyPerDayAndHour:
                    df_time_series_per_weekday = create_dataframe_weekdays(
                        standard_name_df, std_field_name, df_date_col="study_date"
                    )

                    return_structure["standardStudyWorkloadData"] = plotly_barchart_weekdays(
                        df_time_series_per_weekday,
                        "weekday",
                        std_field_name,
                        name_axis_title="Weekday",
                        value_axis_title="Frequency",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM CT standard study name workload",
                        facet_col_wrap=user_profile.plotFacetColWrapVal,
                        sorting_choice=[
                            user_profile.plotInitialSortingDirection,
                            user_profile.plotCTInitialSortingChoice,
                        ],
                        return_as_dict=return_as_dict,
                    )

                if user_profile.plotCTStandardStudyMeanDLPOverTime:
                    facet_title = "System"

                    if user_profile.plotGroupingChoice == "series":
                        facet_title = "Standard study name"

                    parameter_dict = {
                        "df_name_col": std_field_name,
                        "df_value_col": "total_dlp",
                        "df_date_col": "study_date",
                        "name_title": "Standard study name",
                        "value_title": "DLP (mGy.cm)",
                        "date_title": "Study date",
                        "facet_title": facet_title,
                        "sorting_choice": [
                            user_profile.plotInitialSortingDirection,
                            user_profile.plotCTInitialSortingChoice,
                        ],
                        "time_period": plot_timeunit_period,
                        "average_choices": average_choices + ["count"],
                        "grouping_choice": user_profile.plotGroupingChoice,
                        "colourmap": user_profile.plotColourMapChoice,
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "filename": "OpenREM CT standard study name DLP over time",
                        "return_as_dict": return_as_dict,
                    }
                    result = construct_over_time_charts(
                        standard_name_df,
                        parameter_dict,
                    )

                    if user_profile.plotMean:
                        return_structure["standardStudyMeanDLPOverTime"] = result["mean"]
                    if user_profile.plotMedian:
                        return_structure["standardStudyMedianDLPOverTime"] = result["median"]

        if user_profile.plotCTStudyMeanCTDI:

            if user_profile.plotBoxplots and "median" not in average_choices:
                average_choices = average_choices + ["median"]

            name_field = "study_description"
            value_field = "ctradiationdose__ctirradiationeventdata__mean_ctdivol"

            df_aggregated = create_dataframe_aggregates(
                df,
                [name_field],
                value_field,
                stats_to_use=average_choices + ["count"],
            )

            if user_profile.plotMean or user_profile.plotMedian:

                parameter_dict = {
                    "df_name_col": name_field,
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean CTDI<sub>vol</sub> (mGy)"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT study description CTDI mean"
                    parameter_dict["average_choice"] = "mean"
                    (
                        return_structure["studyMeanCTDIData"],
                        return_structure["studyMeanCTDIDataCSV"],
                    ) = plotly_barchart(
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMeanCTDIData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict[
                        "value_axis_title"
                    ] = "Median CTDI<sub>vol</sub> (mGy)"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT study description CTDI median"
                    parameter_dict["average_choice"] = "median"
                    (
                        return_structure["studyMedianCTDIData"],
                        return_structure["studyMedianCTDIDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMedianCTDIData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": name_field,
                    "df_value_col": value_field,
                    "value_axis_title": "CTDI<sub>vol</sub> (mGy)",
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description DLP boxplot",
                    "facet_col": None,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }

                return_structure["studyBoxplotCTDIData"] = plotly_boxplot(
                    df,
                    parameter_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = name_field
                group_by_col = "x_ray_system_name"
                legend_title = "Study description"

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = name_field
                    legend_title = "System"

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": value_field,
                    "value_axis_title": "CTDI<sub>vol</sub> (mGy)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description CTDI histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["studyHistogramCTDIData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if user_profile.plotCTStudyNumEvents:

            if user_profile.plotBoxplots and "median" not in average_choices:
                average_choices = average_choices + ["median"]

            name_field = "study_description"
            value_field = "number_of_events"

            df_aggregated = create_dataframe_aggregates(
                df,
                [name_field],
                value_field,
                stats_to_use=average_choices + ["count"],
            )

            if user_profile.plotMean or user_profile.plotMedian:

                parameter_dict = {
                    "df_name_col": name_field,
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean events"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT study description events mean"
                    parameter_dict["average_choice"] = "mean"
                    (
                        return_structure["studyMeanNumEventsData"],
                        return_structure["studyMeanNumEventsDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMeanNumEventsData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median events"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT study description events median"
                    parameter_dict["average_choice"] = "median"
                    (
                        return_structure["studyMedianNumEventsData"],
                        return_structure["studyMedianNumEventsDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="studyMedianNumEventsData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": name_field,
                    "df_value_col": value_field,
                    "value_axis_title": "Events",
                    "name_axis_title": "Study description",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description events boxplot",
                    "facet_col": None,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }

                return_structure["studyBoxplotNumEventsData"] = plotly_boxplot(
                    df,
                    parameter_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = name_field
                group_by_col = "x_ray_system_name"
                legend_title = "Study description"

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = name_field
                    legend_title = "System"

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": value_field,
                    "value_axis_title": "Events",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT study description events histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure[
                    "studyHistogramNumEventsData"
                ] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if user_profile.plotCTStudyFreq:
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

        if user_profile.plotCTRequestMeanDLP:

            if user_profile.plotBoxplots and "median" not in average_choices:
                average_choices = average_choices + ["median"]

            name_field = "requested_procedure_code_meaning"
            value_field = "total_dlp"

            df_aggregated = create_dataframe_aggregates(
                df,
                [name_field],
                value_field,
                stats_to_use=average_choices + ["count"],
            )

            if user_profile.plotMean or user_profile.plotMedian:

                parameter_dict = {
                    "df_name_col": name_field,
                    "name_axis_title": "Requested procedure",
                    "colourmap": user_profile.plotColourMapChoice,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean DLP (mGy.cm)"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT requested procedure DLP mean"
                    parameter_dict["average_choice"] = "mean"
                    (
                        return_structure["requestMeanDLPData"],
                        return_structure["requestMeanDLPDataCSV"],
                    ) = plotly_barchart(
                        df_aggregated,
                        parameter_dict,
                        csv_name="requestMeanDLPData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median DLP (mGy.cm)"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT requested procedure DLP median"
                    parameter_dict["average_choice"] = "median"
                    (
                        return_structure["requestMedianDLPData"],
                        return_structure["requestMedianDLPDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="requestMedianDLPData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": name_field,
                    "df_value_col": value_field,
                    "value_axis_title": "DLP (mGy.cm)",
                    "name_axis_title": "Requested procedure",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT requested procedure DLP boxplot",
                    "facet_col": None,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }

                return_structure["requestBoxplotDLPData"] = plotly_boxplot(
                    df,
                    parameter_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = name_field
                group_by_col = "x_ray_system_name"
                legend_title = "Requested procedure"

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = name_field
                    legend_title = "System"

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": value_field,
                    "value_axis_title": "DLP (mGy.cm)",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT requested procedure DLP histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure["requestHistogramDLPData"] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if user_profile.plotCTRequestNumEvents:

            if user_profile.plotBoxplots and "median" not in average_choices:
                average_choices = average_choices + ["median"]

            name_field = "requested_procedure_code_meaning"
            value_field = "number_of_events"

            df_aggregated = create_dataframe_aggregates(
                df,
                [name_field],
                value_field,
                stats_to_use=average_choices + ["count"],
            )

            if user_profile.plotMean or user_profile.plotMedian:

                parameter_dict = {
                    "df_name_col": name_field,
                    "name_axis_title": "Requested procedure",
                    "colourmap": user_profile.plotColourMapChoice,
                    "facet_col": None,
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                }
                if user_profile.plotMean:
                    parameter_dict["value_axis_title"] = "Mean events"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT requested procedure events mean"
                    parameter_dict["average_choice"] = "mean"
                    (
                        return_structure["requestMeanNumEventsData"],
                        return_structure["requestMeanNumEventsDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="requestMeanNumEventsData.csv",
                    )

                if user_profile.plotMedian:
                    parameter_dict["value_axis_title"] = "Median events"
                    parameter_dict[
                        "filename"
                    ] = "OpenREM CT requested procedure events median"
                    parameter_dict["average_choice"] = "median"
                    (
                        return_structure["requestMedianNumEventsData"],
                        return_structure["requestMedianNumEventsDataCSV"],
                    ) = plotly_barchart(  # pylint: disable=line-too-long
                        df_aggregated,
                        parameter_dict,
                        csv_name="requestMedianNumEventsData.csv",
                    )

            if user_profile.plotBoxplots:
                parameter_dict = {
                    "df_name_col": name_field,
                    "df_value_col": value_field,
                    "value_axis_title": "Events",
                    "name_axis_title": "Requested procedure",
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT requested procedure events boxplot",
                    "facet_col": None,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "return_as_dict": return_as_dict,
                }

                return_structure["requestBoxplotNumEventsData"] = plotly_boxplot(
                    df,
                    parameter_dict,
                )

            if user_profile.plotHistograms:
                category_names_col = name_field
                group_by_col = "x_ray_system_name"
                legend_title = "Requested procedure"

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = name_field
                    legend_title = "System"

                parameter_dict = {
                    "df_facet_col": group_by_col,
                    "df_category_col": category_names_col,
                    "df_value_col": value_field,
                    "value_axis_title": "Events",
                    "legend_title": legend_title,
                    "n_bins": user_profile.plotHistogramBins,
                    "colourmap": user_profile.plotColourMapChoice,
                    "filename": "OpenREM CT requested procedure events histogram",
                    "facet_col_wrap": user_profile.plotFacetColWrapVal,
                    "sorting_choice": [
                        user_profile.plotInitialSortingDirection,
                        user_profile.plotCTInitialSortingChoice,
                    ],
                    "global_max_min": user_profile.plotHistogramGlobalBins,
                    "return_as_dict": return_as_dict,
                }
                return_structure[
                    "requestHistogramNumEventsData"
                ] = plotly_histogram_barchart(
                    df,
                    parameter_dict,
                )

        if user_profile.plotCTRequestFreq:
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
                "groupby_cols": None,
                "facet_col": None,
                "facet_col_wrap": user_profile.plotFacetColWrapVal,
                "return_as_dict": return_as_dict,
            }
            (
                return_structure["requestFrequencyData"],
                return_structure["requestFrequencyDataCSV"],
            ) = plotly_frequency_barchart(  # pylint: disable=line-too-long
                df, parameter_dict, csv_name="requestFrequencyData.csv"
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
                "sorting_choice": [
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
                "sorting_choice": [
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
                sorting_choice=[
                    user_profile.plotInitialSortingDirection,
                    user_profile.plotCTInitialSortingChoice,
                ],
                return_as_dict=return_as_dict,
            )
        #######################################################################

    return return_structure


def ct_chart_form_processing(request, user_profile):
    # pylint: disable=too-many-statements

    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list("enable_standard_names", flat=True)[0]

    # Obtain the chart options from the request
    chart_options_form = None
    if enable_standard_names:
        chart_options_form = CTChartOptionsFormIncStandard(request.GET)
    else:
        chart_options_form = CTChartOptionsForm(request.GET)

    # Check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required

            set_common_chart_options(chart_options_form, user_profile)

            set_average_chart_options(chart_options_form, user_profile)

            set_ct_chart_options(chart_options_form, user_profile)

            user_profile.save()

        else:
            average_choices = required_average_choices(user_profile)

            ct_acquisition_types = required_ct_acquisition_types(user_profile)

            ct_form_data = initialise_ct_form_data(ct_acquisition_types, user_profile)

            form_data = {
                "plotCharts": user_profile.plotCharts,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices,
            }

            form_data = {**form_data, **ct_form_data}

            chart_options_form = None
            if enable_standard_names:
                chart_options_form = CTChartOptionsFormIncStandard(form_data)
            else:
                chart_options_form = CTChartOptionsForm(form_data)

    return chart_options_form
