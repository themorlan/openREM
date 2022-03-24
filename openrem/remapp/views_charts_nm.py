# pylint: disable=too-many-lines
import logging
from datetime import datetime
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import JsonResponse
from django.utils.safestring import mark_safe
from django.utils.translation import gettext as _
from remapp.forms import CTChartOptionsForm, NMChartOptionsForm
from remapp.interface.mod_filters import ct_acq_filter, nm_acq_filter
from remapp.models import (
    UserProfile,
    create_user_profile,
    CommonVariables,
)
from remapp.views_admin import (
    initialise_nm_form_data,
    required_average_choices,
    required_ct_acquisition_types,
    initialise_ct_form_data,
    set_ct_chart_options,
    set_average_chart_options,
    set_common_chart_options,
    set_nm_chart_options,
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
)

logger = logging.getLogger(__name__)

def nm_chart_form_processing(request, user_profile):
    # pylint: disable=too-many-statements
    # Obtain the chart options from the request
    chart_options_form = NMChartOptionsForm(request.GET)
    # Check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required

            set_common_chart_options(chart_options_form, user_profile)

            set_average_chart_options(chart_options_form, user_profile)

            set_nm_chart_options(chart_options_form, user_profile)

            user_profile.save()

        else:
            average_choices = required_average_choices(user_profile)

            nm_form_data = initialise_nm_form_data(user_profile)

            form_data = {
                "plotCharts": user_profile.plotCharts,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices,
            }

            form_data = {**form_data, **nm_form_data}

            chart_options_form = NMChartOptionsForm(form_data)
    return chart_options_form

def generate_required_nm_charts_list(user_profile : UserProfile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    charts_requiring_time = [
        user_profile.plotNMInjectedDoseOverTime
    ]
    if any(charts_requiring_time):
        keys = list(dict(user_profile.TIME_PERIOD).keys())
        values = list(dict(user_profile.TIME_PERIOD).values())
        time_period = (values[keys.index(user_profile.plotNMOverTimePeriod)]).lower()
    if user_profile.plotNMStudyFreq:
        required_charts.append(
            {
                "title": "Chart of study description frequency",
                "var_name": "studyFrequency",
            }
        )
    if user_profile.plotNMStudyPerDayAndHour:
        required_charts.append(
            {
                "title": "Chart of study description workload",
                "var_name": "studyWorkload",
            }
        )
    if user_profile.plotNMInjectedDosePerStudy:
        if user_profile.plotMean:
            required_charts.append(
                {
                    "title": "Chart of injected dose per study mean",
                    "var_name": "studyInjectedDoseMean"
                }
            )
        if user_profile.plotMedian:
            required_charts.append(
                {
                    "title": "Chart of injected dose per study median",
                    "var_name": "studyInjectedDoseMedian"
                }
            )
        if user_profile.plotBoxplots:
            required_charts.append(
                {
                    "title": "Boxplot of injected dose per study",
                    "var_name": "studyInjectedDoseBoxplot"
                }
            )
        if user_profile.plotHistograms:
            required_charts.append(
                {
                    "title": "Histogram of injected dose per study",
                    "var_name": "studyInjectedDoseHistogram"
                }
            )
    if user_profile.plotNMInjectedDoseOverTime:
        if user_profile.plotMean:
            required_charts.append(
                {
                    "title": f"Chart of injected dose mean over time ({time_period})",
                    "var_name": "studyInjectedDoseOverTimeMean"
                }
            )
        if user_profile.plotMedian:
            required_charts.append(
                {
                    "title": f"Chart of injected dose median over time ({time_period})",
                    "var_name": "studyInjectedDoseOverTimeMedian"
                }
            )
        if user_profile.plotBoxplots:
            required_charts.append(
                {
                    "title": f"Boxplot of injected dose over time ({time_period})",
                    "var_name": "studyInjectedDoseOverTimeBoxplot"
                }
            )
    if user_profile.plotNMInjectedDoseOverWeight:
        required_charts.append(
            {
                "title": "Chart of injected dose versus patient weight",
                "var_name": "studyInjectedDoseOverWeight"
            }
        )

    return required_charts

@login_required
def nm_summary_chart_data(request):
    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = nm_acq_filter(request.GET, pid=pid)

    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile

    if settings.DEBUG:
        start_time = datetime.now()

    return_structure = nm_plot_calculations(f, user_profile)

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def nm_plot_calculations(f, user_profile: UserProfile, return_as_dict=False):
    return_structure = {}

    # Depending on the selected charts, define all the fields we care about and load them into a Dataframe
    name_fields = []
    date_fields = []
    time_fields = []
    value_fields = []
    system_field = []
    chart_of_interest = [
        user_profile.plotNMStudyFreq, 
        user_profile.plotNMStudyPerDayAndHour,
        user_profile.plotNMInjectedDoseOverWeight,
    ]
    if any(chart_of_interest):
        name_fields.append("study_description")

    charts_of_interest = [
        user_profile.plotNMStudyPerDayAndHour
    ]
    if any(charts_of_interest):
        date_fields.append("study_date")
        time_fields.append("study_time")

    charts_of_interest = [
        user_profile.plotNMInjectedDoseOverWeight,
        user_profile.plotNMInjectedDoseOverTime,
        user_profile.plotNMInjectedDosePerStudy,
    ]
    if any(charts_of_interest):
        value_fields.append("radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity")

    charts_of_interest = [
        user_profile.plotNMInjectedDoseOverWeight,
    ]
    if any(charts_of_interest):
        value_fields.append("patientstudymoduleattr__patient_weight")

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

    # Based on df create all the Charts that are wanted
    if user_profile.plotNMStudyFreq:
        parameter_dict = {
            "df_name_col": "study_description",
            "sorting_choice": [
                user_profile.plotInitialSortingDirection,
                user_profile.plotNMInitialSortingChoice
            ],
            "legend_title": "Study description",
            "df_x_axis_col": "x_ray_system_name",
            "x_axis_title": "System",
            "grouping_choice": user_profile.plotGroupingChoice,
            "colourmap": user_profile.plotColourMapChoice,
            "filename": "OpenREM NM study description frequency",
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
    if user_profile.plotNMStudyPerDayAndHour:
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
            filename="OpenREM NM study description workload",
            facet_col_wrap=user_profile.plotFacetColWrapVal,
            sorting_choice=[
                user_profile.plotInitialSortingDirection,
                user_profile.plotNMInitialSortingChoice,
            ],
            return_as_dict=return_as_dict,
        )
    if user_profile.plotNMInjectedDoseOverWeight:
        parameter_dict = {
            "df_name_col": "study_description",
            "df_x_col": "patientstudymoduleattr__patient_weight",
            "df_y_col": "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity",
            "sorting_choice": [
                user_profile.plotInitialSortingDirection,
                user_profile.plotNMInitialSortingChoice,
            ],
            "grouping_choice": user_profile.plotGroupingChoice,
            "legend_title": "Study description",
            "colourmap": user_profile.plotColourMapChoice,
            "facet_col_wrap": user_profile.plotFacetColWrapVal,
            "x_axis_title": "Patient mass (kg)",
            "y_axis_title": "Administed Activity (MBq)",
            "filename": "OpenREM Nuclear Medicine Dose vs patient mass",
            "return_as_dict": return_as_dict,
        }
        return_structure["studyInjectedDoseOverWeightData"] = plotly_scatter(
            df,
            parameter_dict,
        )

    return return_structure