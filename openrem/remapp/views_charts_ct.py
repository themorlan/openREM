# pylint: disable=too-many-lines
import logging
from datetime import datetime
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
    generate_average_chart_group,
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
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

    required_charts = []

    charts_of_interest = [
        profile.plotCTAcquisitionDLPOverTime,
        profile.plotCTAcquisitionCTDIOverTime,
        profile.plotCTStudyMeanDLPOverTime,
        profile.plotCTRequestDLPOverTime,
    ]
    if enable_standard_names:
        charts_of_interest.append(profile.plotCTStandardStudyMeanDLPOverTime)
        charts_of_interest.append(profile.plotCTStandardAcquisitionDLPOverTime)
        charts_of_interest.append(profile.plotCTStandardAcquisitionCTDIOverTime)

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
        if profile.plotCTStandardAcquisitionFreq:
            required_charts.append(
                {
                    "title": "Chart of standard acquisition name frequency",
                    "var_name": "standardAcquisitionFrequency",
                }
            )

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

        if profile.plotCTStandardAcquisitionMeanCTDI:
            if profile.plotMean:
                required_charts.append(  # nosec
                    {
                        "title": mark_safe(
                            "Chart of standard acquisition name mean CTDI<sub>vol</sub>"
                        ),
                        "var_name": "standardAcquisitionMeanCTDI",
                    }
                )
            if profile.plotMedian:
                required_charts.append(  # nosec
                    {
                        "title": mark_safe(
                            "Chart of standard acquisition name median CTDI<sub>vol</sub>"
                        ),
                        "var_name": "standardAcquisitionMedianCTDI",
                    }
                )
            if profile.plotBoxplots:
                required_charts.append(  # nosec
                    {
                        "title": mark_safe(
                            "Boxplot of standard acquisition name CTDI<sub>vol</sub>"
                        ),
                        "var_name": "standardAcquisitionBoxplotCTDI",
                    }
                )
            if profile.plotHistograms:
                required_charts.append(  # nosec
                    {
                        "title": mark_safe(
                            "Histogram of standard acquisition name CTDI<sub>vol</sub>"
                        ),
                        "var_name": "standardAcquisitionHistogramCTDI",
                    }
                )

        if profile.plotCTStandardAcquisitionDLPOverTime:
            if profile.plotMean:
                required_charts.append(
                    {
                        "title": "Chart of standard acquisition name mean DLP over time ("
                        + time_period
                        + ")",
                        "var_name": "standardAcquisitionMeanDLPOverTime",
                    }
                )
            if profile.plotMedian:
                required_charts.append(
                    {
                        "title": "Chart of standard acquisition name median DLP over time ("
                        + time_period
                        + ")",
                        "var_name": "standardAcquisitionMedianDLPOverTime",
                    }
                )

        if profile.plotCTStandardAcquisitionCTDIOverTime:
            if profile.plotMean:
                required_charts.append(  # nosec
                    {
                        "title": mark_safe(
                            "Chart of standard acquisition name mean CTDI<sub>vol</sub> over time ("
                            + time_period
                            + ")"
                        ),
                        "var_name": "standardAcquisitionMeanCTDIOverTime",
                    }
                )
            if profile.plotMedian:
                required_charts.append(  # nosec
                    {
                        "title": mark_safe(
                            "Chart of standard acquisition name median CTDI<sub>vol</sub> over time ("
                            + time_period
                            + ")"
                        ),
                        "var_name": "standardAcquisitionMedianCTDIOverTime",
                    }
                )

        if profile.plotCTStandardAcquisitionDLPvsMass:
            required_charts.append(
                {
                    "title": "Chart of standard acquisition name DLP vs patient mass",
                    "var_name": "standardAcquisitionScatterDLPvsMass",
                }
            )

        if profile.plotCTStandardAcquisitionCTDIvsMass:
            required_charts.append(  # nosec
                {
                    "title": mark_safe(
                        "Chart of standard acquisition name CTDI<sub>vol</sub> vs patient mass"
                    ),
                    "var_name": "standardAcquisitionScatterCTDIvsMass",
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
    """Obtain data for CT charts Ajax call."""
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
    if not f.qs.exists():
        return {}

    # Obtain the system-level enable_standard_names setting
    try:
        StandardNameSettings.objects.get()
    except ObjectDoesNotExist:
        StandardNameSettings.objects.create()
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

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
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionDLPOverTime)
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionCTDIOverTime)
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
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionFreq)
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionMeanDLP)
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionMeanCTDI)
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionDLPOverTime)
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionCTDIOverTime)
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionDLPvsMass)
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionCTDIvsMass)

    if any(charts_of_interest):

        name_fields = [
            "ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning",
            "ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_value",
        ]

        charts_of_interest = [
            user_profile.plotCTAcquisitionFreq,
            user_profile.plotCTAcquisitionMeanCTDI,
            user_profile.plotCTAcquisitionMeanDLP,
            user_profile.plotCTAcquisitionCTDIvsMass,
            user_profile.plotCTAcquisitionDLPvsMass,
            user_profile.plotCTAcquisitionCTDIOverTime,
            user_profile.plotCTAcquisitionDLPOverTime,
        ]
        if any(charts_of_interest):
            name_fields.append(
                "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
            )

        if enable_standard_names:
            charts_of_interest = [
                user_profile.plotCTStandardAcquisitionFreq,
                user_profile.plotCTStandardAcquisitionMeanDLP,
                user_profile.plotCTStandardAcquisitionMeanCTDI,
                user_profile.plotCTStandardAcquisitionDLPOverTime,
                user_profile.plotCTStandardAcquisitionCTDIOverTime,
                user_profile.plotCTStandardAcquisitionDLPvsMass,
                user_profile.plotCTStandardAcquisitionCTDIvsMass,
            ]
            if any(charts_of_interest):
                name_fields.append(
                    "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name"
                )

        value_fields = []
        charts_of_interest = [
            user_profile.plotCTAcquisitionMeanDLP,
            user_profile.plotCTAcquisitionDLPvsMass,
            user_profile.plotCTAcquisitionDLPOverTime,
        ]
        if enable_standard_names:
            charts_of_interest.append(user_profile.plotCTStandardAcquisitionMeanDLP)
            charts_of_interest.append(user_profile.plotCTStandardAcquisitionDLPOverTime)
            charts_of_interest.append(user_profile.plotCTStandardAcquisitionDLPvsMass)
        if any(charts_of_interest):
            value_fields.append("ctradiationdose__ctirradiationeventdata__dlp")

        charts_of_interest = [
            user_profile.plotCTAcquisitionMeanCTDI,
            user_profile.plotCTAcquisitionCTDIvsMass,
            user_profile.plotCTAcquisitionCTDIOverTime,
        ]
        if enable_standard_names:
            charts_of_interest.append(user_profile.plotCTStandardAcquisitionMeanCTDI)
            charts_of_interest.append(
                user_profile.plotCTStandardAcquisitionCTDIOverTime
            )
        charts_of_interest.append(user_profile.plotCTStandardAcquisitionCTDIvsMass)
        if any(charts_of_interest):
            value_fields.append("ctradiationdose__ctirradiationeventdata__mean_ctdivol")

        charts_of_interest = [
            user_profile.plotCTAcquisitionCTDIvsMass,
            user_profile.plotCTAcquisitionDLPvsMass,
        ]
        if enable_standard_names:
            charts_of_interest.append(user_profile.plotCTStandardAcquisitionDLPvsMass)
            charts_of_interest.append(user_profile.plotCTStandardAcquisitionCTDIvsMass)
        if any(charts_of_interest):
            value_fields.append("patientstudymoduleattr__patient_weight")

        time_fields = []
        date_fields = []

        charts_of_interest = [
            user_profile.plotCTAcquisitionCTDIOverTime,
            user_profile.plotCTAcquisitionDLPOverTime,
        ]
        if enable_standard_names:
            charts_of_interest.append(user_profile.plotCTStandardAcquisitionDLPOverTime)
            charts_of_interest.append(
                user_profile.plotCTStandardAcquisitionCTDIOverTime
            )
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

            name_field = "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
            value_field = "ctradiationdose__ctirradiationeventdata__dlp"
            value_text = "DLP"
            units_text = "(mGy.cm)"
            name_text = "Acquisition protocol"
            variable_name_start = "acquisition"
            variable_value_name = "DLP"
            modality_text = "CT"

            new_charts = generate_average_chart_group(
                average_choices,
                chart_message,
                df,
                modality_text,
                name_field,
                name_text,
                return_as_dict,
                return_structure,
                units_text,
                user_profile,
                value_field,
                value_text,
                variable_name_start,
                variable_value_name,
                user_profile.plotCTInitialSortingChoice,
            )

            return_structure = {**return_structure, **new_charts}

        if user_profile.plotCTAcquisitionMeanCTDI:

            name_field = "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
            value_field = "ctradiationdose__ctirradiationeventdata__mean_ctdivol"
            value_text = "CTDI <sub>vol</sub>"
            units_text = "(mGy)"
            name_text = "Acquisition protocol"
            variable_name_start = "acquisition"
            modality_text = "CT"
            variable_value_name = "CTDI"

            new_charts = generate_average_chart_group(
                average_choices,
                chart_message,
                df,
                modality_text,
                name_field,
                name_text,
                return_as_dict,
                return_structure,
                units_text,
                user_profile,
                value_field,
                value_text,
                variable_name_start,
                variable_value_name,
                user_profile.plotCTInitialSortingChoice,
            )

            return_structure = {**return_structure, **new_charts}

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
            charts_of_interest = [
                user_profile.plotCTStandardAcquisitionFreq,
                user_profile.plotCTStandardAcquisitionMeanDLP,
                user_profile.plotCTStandardAcquisitionMeanCTDI,
                user_profile.plotCTStandardAcquisitionDLPOverTime,
                user_profile.plotCTStandardAcquisitionCTDIOverTime,
                user_profile.plotCTStandardAcquisitionDLPvsMass,
                user_profile.plotCTStandardAcquisitionCTDIvsMass,
            ]
            if any(charts_of_interest):
                # Exclude "Blank" and "blank" standard_acqusition_name data
                df_without_blanks = df[
                    (
                        df[
                            "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name"
                        ]
                        != "blank"
                    )
                    & (
                        df[
                            "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name"
                        ]
                        != "Blank"
                    )
                ].copy()
                # Remove any unused categories (this will include "Blank" or "blank")
                df_without_blanks[
                    "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name"
                ] = df_without_blanks[
                    "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name"
                ].cat.remove_unused_categories()

                if user_profile.plotCTStandardAcquisitionFreq:
                    parameter_dict = {
                        "df_name_col": "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name",
                        "sorting_choice": [
                            user_profile.plotInitialSortingDirection,
                            user_profile.plotCTInitialSortingChoice,
                        ],
                        "legend_title": "Standard acquisition name",
                        "df_x_axis_col": "x_ray_system_name",
                        "x_axis_title": "System",
                        "grouping_choice": user_profile.plotGroupingChoice,
                        "colourmap": user_profile.plotColourMapChoice,
                        "filename": "OpenREM CT standard acquisition name frequency",
                        "groupby_cols": None,
                        "facet_col": None,
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "return_as_dict": return_as_dict,
                        "custom_msg_line": chart_message,
                    }
                    (
                        return_structure["standardAcquisitionFrequencyData"],
                        return_structure["standardAcquisitionFrequencyDataCSV"],
                    ) = plotly_frequency_barchart(  # pylint: disable=line-too-long
                        df_without_blanks,
                        parameter_dict,
                        csv_name="standardAcquisitionFrequencyData.csv",
                    )

                if user_profile.plotCTStandardAcquisitionMeanDLP:
                    name_field = "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name"
                    value_field = "ctradiationdose__ctirradiationeventdata__dlp"
                    value_text = "DLP"
                    units_text = "(mGy.cm)"
                    name_text = "Standard acquisition name"
                    variable_name_start = "standardAcquisition"
                    variable_value_name = "DLP"
                    modality_text = "CT"
                    chart_message = ""

                    new_charts = generate_average_chart_group(
                        average_choices,
                        chart_message,
                        df_without_blanks,
                        modality_text,
                        name_field,
                        name_text,
                        return_as_dict,
                        return_structure,
                        units_text,
                        user_profile,
                        value_field,
                        value_text,
                        variable_name_start,
                        variable_value_name,
                        user_profile.plotCTInitialSortingChoice,
                    )

                    return_structure = {**return_structure, **new_charts}

                if user_profile.plotCTStandardAcquisitionMeanCTDI:
                    name_field = "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name"
                    value_field = (
                        "ctradiationdose__ctirradiationeventdata__mean_ctdivol"
                    )
                    value_text = "CTDI<sub>vol</sub>"
                    units_text = "(mGy)"
                    name_text = "Standard acquisition name"
                    variable_name_start = "standardAcquisition"
                    variable_value_name = "CTDI"
                    modality_text = "CT"
                    chart_message = ""

                    new_charts = generate_average_chart_group(
                        average_choices,
                        chart_message,
                        df_without_blanks,
                        modality_text,
                        name_field,
                        name_text,
                        return_as_dict,
                        return_structure,
                        units_text,
                        user_profile,
                        value_field,
                        value_text,
                        variable_name_start,
                        variable_value_name,
                        user_profile.plotCTInitialSortingChoice,
                    )

                    return_structure = {**return_structure, **new_charts}

                if user_profile.plotCTStandardAcquisitionCTDIOverTime:
                    facet_title = "System"

                    if user_profile.plotGroupingChoice == "series":
                        facet_title = "Standard acquisition name"

                    parameter_dict = {
                        "df_name_col": "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name",
                        "df_value_col": "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                        "df_date_col": "study_date",
                        "name_title": "Standard acquisition name",
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
                        "filename": "OpenREM CT standard acquisition name CTDI over time",
                        "return_as_dict": return_as_dict,
                        "custom_msg_line": chart_message,
                    }
                    result = construct_over_time_charts(
                        df_without_blanks,
                        parameter_dict,
                    )

                    if user_profile.plotMean:
                        return_structure[
                            "standardAcquisitionMeanCTDIOverTime"
                        ] = result["mean"]
                    if user_profile.plotMedian:
                        return_structure[
                            "standardAcquisitionMedianCTDIOverTime"
                        ] = result["median"]

                if user_profile.plotCTStandardAcquisitionDLPOverTime:
                    facet_title = "System"

                    if user_profile.plotGroupingChoice == "series":
                        facet_title = "Standard acquisition name"

                    parameter_dict = {
                        "df_name_col": "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name",
                        "df_value_col": "ctradiationdose__ctirradiationeventdata__dlp",
                        "df_date_col": "study_date",
                        "name_title": "Standard acquisition name",
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
                        "filename": "OpenREM CT standard acquisition name DLP over time",
                        "return_as_dict": return_as_dict,
                        "custom_msg_line": chart_message,
                    }
                    result = construct_over_time_charts(
                        df_without_blanks,
                        parameter_dict,
                    )

                    if user_profile.plotMean:
                        return_structure["standardAcquisitionMeanDLPOverTime"] = result[
                            "mean"
                        ]
                    if user_profile.plotMedian:
                        return_structure[
                            "standardAcquisitionMedianDLPOverTime"
                        ] = result["median"]

                if user_profile.plotCTStandardAcquisitionCTDIvsMass:
                    parameter_dict = {
                        "df_name_col": "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name",
                        "df_x_col": "patientstudymoduleattr__patient_weight",
                        "df_y_col": "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                        "sorting_choice": [
                            user_profile.plotInitialSortingDirection,
                            user_profile.plotCTInitialSortingChoice,
                        ],
                        "grouping_choice": user_profile.plotGroupingChoice,
                        "legend_title": "Standard acquisition name",
                        "colourmap": user_profile.plotColourMapChoice,
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "x_axis_title": "Patient mass (kg)",
                        "y_axis_title": "CTDI<sub>vol</sub> (mGy)",
                        "filename": "OpenREM CT standard acquisition name CTDI vs patient mass",
                        "return_as_dict": return_as_dict,
                        "custom_msg_line": chart_message,
                    }
                    return_structure[
                        "standardAcquisitionScatterCTDIvsMass"
                    ] = plotly_scatter(
                        df_without_blanks,
                        parameter_dict,
                    )

                if user_profile.plotCTStandardAcquisitionDLPvsMass:
                    parameter_dict = {
                        "df_name_col": "ctradiationdose__ctirradiationeventdata__standard_protocols__standard_name",
                        "df_x_col": "patientstudymoduleattr__patient_weight",
                        "df_y_col": "ctradiationdose__ctirradiationeventdata__dlp",
                        "sorting_choice": [
                            user_profile.plotInitialSortingDirection,
                            user_profile.plotCTInitialSortingChoice,
                        ],
                        "grouping_choice": user_profile.plotGroupingChoice,
                        "legend_title": "Standard acquisition name",
                        "colourmap": user_profile.plotColourMapChoice,
                        "facet_col_wrap": user_profile.plotFacetColWrapVal,
                        "x_axis_title": "Patient mass (kg)",
                        "y_axis_title": "DLP (mGy.cm)",
                        "filename": "OpenREM CT standard acquisition name DLP vs patient mass",
                        "return_as_dict": return_as_dict,
                        "custom_msg_line": chart_message,
                    }
                    return_structure[
                        "standardAcquisitionScatterDLPvsMass"
                    ] = plotly_scatter(
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
                name_fields.append("standard_names__standard_name")

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
            user_profile.plotCTRequestNumEvents,
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

        # If only standard_names__standard_name is required then exclude all entries where these are None as these are
        # not required for standard name charts.
        queryset = f.qs
        if name_fields == ["standard_names__standard_name"]:
            queryset = queryset.exclude(standard_names__standard_name__isnull=True)

        df = create_dataframe(
            queryset,
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

            name_field = "study_description"
            value_field = "total_dlp"
            value_text = "DLP"
            units_text = "(mGy.cm)"
            name_text = "Study description"
            variable_name_start = "study"
            variable_value_name = "DLP"
            modality_text = "CT"
            chart_message = ""

            new_charts = generate_average_chart_group(
                average_choices,
                chart_message,
                df,
                modality_text,
                name_field,
                name_text,
                return_as_dict,
                return_structure,
                units_text,
                user_profile,
                value_field,
                value_text,
                variable_name_start,
                variable_value_name,
                user_profile.plotCTInitialSortingChoice,
            )

            return_structure = {**return_structure, **new_charts}

        if enable_standard_names:
            charts_of_interest = [
                user_profile.plotCTStandardStudyMeanDLP,
                user_profile.plotCTStandardStudyNumEvents,
                user_profile.plotCTStandardStudyFreq,
                user_profile.plotCTStandardStudyPerDayAndHour,
                user_profile.plotCTStandardStudyMeanDLPOverTime,
            ]
            if any(charts_of_interest):

                # Create a standard name data frame - remove any blank standard names
                standard_name_df = df[
                    (df["standard_names__standard_name"] != "blank")
                    & (df["standard_names__standard_name"] != "Blank")
                ].copy()
                # Remove any unused categories (this will include "Blank" or "blank")
                standard_name_df["standard_names__standard_name"] = standard_name_df[
                    "standard_names__standard_name"
                ].cat.remove_unused_categories()

                if user_profile.plotCTStandardStudyMeanDLP:

                    name_field = "standard_names__standard_name"
                    value_field = "total_dlp"
                    value_text = "DLP"
                    units_text = "(mGy.cm)"
                    name_text = "Standard study name"
                    variable_name_start = "standardStudy"
                    variable_value_name = "DLP"
                    modality_text = "CT"
                    chart_message = ""

                    new_charts = generate_average_chart_group(
                        average_choices,
                        chart_message,
                        standard_name_df,
                        modality_text,
                        name_field,
                        name_text,
                        return_as_dict,
                        return_structure,
                        units_text,
                        user_profile,
                        value_field,
                        value_text,
                        variable_name_start,
                        variable_value_name,
                        user_profile.plotCTInitialSortingChoice,
                    )

                    return_structure = {**return_structure, **new_charts}

                if user_profile.plotCTStandardStudyNumEvents:

                    name_field = "standard_names__standard_name"
                    value_field = "number_of_events"
                    value_text = "Number of events"
                    units_text = ""
                    name_text = "Standard study name"
                    variable_name_start = "standardStudy"
                    variable_value_name = "NumEvents"
                    modality_text = "CT"
                    chart_message = ""

                    new_charts = generate_average_chart_group(
                        average_choices,
                        chart_message,
                        standard_name_df,
                        modality_text,
                        name_field,
                        name_text,
                        return_as_dict,
                        return_structure,
                        units_text,
                        user_profile,
                        value_field,
                        value_text,
                        variable_name_start,
                        variable_value_name,
                        user_profile.plotCTInitialSortingChoice,
                    )

                    return_structure = {**return_structure, **new_charts}

                if user_profile.plotCTStandardStudyFreq:
                    parameter_dict = {
                        "df_name_col": "standard_names__standard_name",
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
                        standard_name_df,
                        "standard_names__standard_name",
                        df_date_col="study_date",
                    )

                    return_structure[
                        "standardStudyWorkloadData"
                    ] = plotly_barchart_weekdays(
                        df_time_series_per_weekday,
                        "weekday",
                        "standard_names__standard_name",
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
                        "df_name_col": "standard_names__standard_name",
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
                        return_structure["standardStudyMeanDLPOverTime"] = result[
                            "mean"
                        ]
                    if user_profile.plotMedian:
                        return_structure["standardStudyMedianDLPOverTime"] = result[
                            "median"
                        ]

        if user_profile.plotCTStudyMeanCTDI:

            name_field = "study_description"
            value_field = "ctradiationdose__ctirradiationeventdata__mean_ctdivol"
            value_text = "CTDI<sub>vol</sub>"
            units_text = "(mGy)"
            name_text = "Study description"
            variable_name_start = "study"
            variable_value_name = "CTDI"
            modality_text = "CT"
            chart_message = ""

            new_charts = generate_average_chart_group(
                average_choices,
                chart_message,
                df,
                modality_text,
                name_field,
                name_text,
                return_as_dict,
                return_structure,
                units_text,
                user_profile,
                value_field,
                value_text,
                variable_name_start,
                variable_value_name,
                user_profile.plotCTInitialSortingChoice,
            )

            return_structure = {**return_structure, **new_charts}

        if user_profile.plotCTStudyNumEvents:

            name_field = "study_description"
            value_field = "number_of_events"
            value_text = "Number of events"
            units_text = ""
            name_text = "Study description"
            variable_name_start = "study"
            variable_value_name = "NumEvents"
            modality_text = "CT"
            chart_message = ""

            new_charts = generate_average_chart_group(
                average_choices,
                chart_message,
                df,
                modality_text,
                name_field,
                name_text,
                return_as_dict,
                return_structure,
                units_text,
                user_profile,
                value_field,
                value_text,
                variable_name_start,
                variable_value_name,
                user_profile.plotCTInitialSortingChoice,
            )

            return_structure = {**return_structure, **new_charts}

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

            name_field = "requested_procedure_code_meaning"
            value_field = "total_dlp"
            value_text = "DLP"
            units_text = "(mGy.cm)"
            name_text = "Requested procedure"
            variable_name_start = "request"
            variable_value_name = "DLP"
            modality_text = "CT"
            chart_message = ""

            new_charts = generate_average_chart_group(
                average_choices,
                chart_message,
                df,
                modality_text,
                name_field,
                name_text,
                return_as_dict,
                return_structure,
                units_text,
                user_profile,
                value_field,
                value_text,
                variable_name_start,
                variable_value_name,
                user_profile.plotCTInitialSortingChoice,
            )

            return_structure = {**return_structure, **new_charts}

        if user_profile.plotCTRequestNumEvents:

            name_field = "requested_procedure_code_meaning"
            value_field = "number_of_events"
            value_text = "Number of events"
            units_text = ""
            name_text = "Requested procedure"
            variable_name_start = "request"
            variable_value_name = "NumEvents"
            modality_text = "CT"
            chart_message = ""

            new_charts = generate_average_chart_group(
                average_choices,
                chart_message,
                df,
                modality_text,
                name_field,
                name_text,
                return_as_dict,
                return_structure,
                units_text,
                user_profile,
                value_field,
                value_text,
                variable_name_start,
                variable_value_name,
                user_profile.plotCTInitialSortingChoice,
            )

            return_structure = {**return_structure, **new_charts}

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
    enable_standard_names = StandardNameSettings.objects.values_list(
        "enable_standard_names", flat=True
    )[0]

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
