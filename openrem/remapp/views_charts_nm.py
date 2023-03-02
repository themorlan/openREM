# pylint: disable=too-many-lines
import logging
from datetime import datetime
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist
from django.http import JsonResponse
from django.utils.translation import gettext as _
from remapp.forms import NMChartOptionsForm
from remapp.interface.mod_filters import nm_filter
from remapp.models import (
    UserProfile,
    create_user_profile,
    CommonVariables,
)
from remapp.views_admin import (
    initialise_nm_form_data,
    required_average_choices,
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


def generate_required_nm_charts_list(user_profile: UserProfile):
    """Get a list of the title string and base variable name for each required chart."""
    required_charts = []

    charts_requiring_time = [user_profile.plotNMInjectedDoseOverTime]
    if any(charts_requiring_time):
        keys = list(dict(user_profile.TIME_PERIOD).keys())
        values = list(dict(user_profile.TIME_PERIOD).values())
        time_period = (values[keys.index(user_profile.plotNMOverTimePeriod)]).lower()
    if user_profile.plotNMStudyFreq:
        required_charts.append(
            {
                "title": _("Chart of study description frequency"),
                "var_name": "studyFrequency",
            }
        )
    if user_profile.plotNMStudyPerDayAndHour:
        required_charts.append(
            {
                "title": _("Chart of study description workload"),
                "var_name": "studyWorkload",
            }
        )
    if user_profile.plotNMInjectedDosePerStudy:
        if user_profile.plotMean:
            required_charts.append(
                {
                    "title": _("Chart of injected dose per study mean"),
                    "var_name": "studyInjectedDoseMean",
                }
            )
        if user_profile.plotMedian:
            required_charts.append(
                {
                    "title": _("Chart of injected dose per study median"),
                    "var_name": "studyInjectedDoseMedian",
                }
            )
        if user_profile.plotBoxplots:
            required_charts.append(
                {
                    "title": _("Boxplot of injected dose per study"),
                    "var_name": "studyInjectedDoseBoxplot",
                }
            )
        if user_profile.plotHistograms:
            required_charts.append(
                {
                    "title": _("Histogram of injected dose per study"),
                    "var_name": "studyInjectedDoseHistogram",
                }
            )
    if user_profile.plotNMInjectedDoseOverTime:
        if user_profile.plotMean:
            required_charts.append(
                {
                    "title": _(
                        "Chart of injected dose mean over time ({time_period})"
                    ).format(time_period=time_period),
                    "var_name": "studyInjectedDoseOverTimeMean",
                }
            )
        if user_profile.plotMedian:
            required_charts.append(
                {
                    "title": _(
                        "Chart of injected dose median over time ({time_period})"
                    ).format(time_period=time_period),
                    "var_name": "studyInjectedDoseOverTimeMedian",
                }
            )
        if user_profile.plotBoxplots:
            required_charts.append(
                {
                    "title": f"Boxplot of injected dose over time ({time_period})",
                    "var_name": "studyInjectedDoseOverTimeBoxplot",
                }
            )
    if user_profile.plotNMInjectedDoseOverWeight:
        required_charts.append(
            {
                "title": _("Chart of injected dose versus patient weight"),
                "var_name": "studyInjectedDoseOverWeight",
            }
        )

    return required_charts


@login_required
def nm_summary_chart_data(request):
    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = nm_filter(request.GET, pid=pid)

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
        user_profile.plotNMInjectedDoseOverTime,
        user_profile.plotNMInjectedDosePerStudy,
    ]
    if any(chart_of_interest):
        name_fields.append("study_description")

    charts_of_interest = [
        user_profile.plotNMStudyPerDayAndHour,
        user_profile.plotNMInjectedDoseOverTime,
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
        value_fields.append(
            "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity"
        )

    charts_of_interest = [
        user_profile.plotNMInjectedDoseOverWeight,
    ]
    if any(charts_of_interest):
        value_fields.append("patientstudymoduleattr__patient_weight")

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

    # Prepare additional settings for the plots
    average_choices = []
    if user_profile.plotMean:
        average_choices.append(CommonVariables.MEAN)
    if user_profile.plotMedian:
        average_choices.append(CommonVariables.MEDIAN)
    if user_profile.plotBoxplots:
        average_choices.append(CommonVariables.BOXPLOT)

    # Based on df create all the Charts that are wanted

    if user_profile.plotNMStudyFreq:
        return_structure.update(
            _generate_nm_study_freq(user_profile, return_as_dict, df)
        )
    if user_profile.plotNMStudyPerDayAndHour:
        return_structure.update(
            _generate_nm_study_workload(user_profile, return_as_dict, df)
        )
    if user_profile.plotNMInjectedDoseOverWeight:
        return_structure.update(
            _generate_nm_dose_over_patient_weight(user_profile, return_as_dict, df)
        )
    if user_profile.plotNMInjectedDosePerStudy:
        return_structure.update(
            _generate_nm_dose_per_study(
                user_profile, return_as_dict, df, average_choices
            )
        )
    if user_profile.plotNMInjectedDoseOverTime:
        return_structure.update(
            _generate_nm_dose_over_time(
                user_profile, return_as_dict, df, average_choices
            )
        )

    return return_structure


def _generate_nm_study_freq(user_profile, return_as_dict, df):
    return_structure = {}
    parameter_dict = {
        "df_name_col": "study_description",
        "sorting_choice": [
            user_profile.plotInitialSortingDirection,
            user_profile.plotNMInitialSortingChoice,
        ],
        "legend_title": _("Study description"),
        "df_x_axis_col": "x_ray_system_name",
        "x_axis_title": _("System"),
        "grouping_choice": user_profile.plotGroupingChoice,
        "colourmap": user_profile.plotColourMapChoice,
        "filename": _("OpenREM NM study description frequency"),
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
    return return_structure


def _generate_nm_study_workload(user_profile, return_as_dict, df):
    return_structure = {}
    df_time_series_per_weekday = create_dataframe_weekdays(
        df, "study_description", df_date_col="study_date"
    )

    return_structure["studyWorkloadData"] = plotly_barchart_weekdays(
        df_time_series_per_weekday,
        "weekday",
        "study_description",
        name_axis_title=_("Weekday"),
        value_axis_title=_("Frequency"),
        colourmap=user_profile.plotColourMapChoice,
        filename=_("OpenREM NM study description workload"),
        facet_col_wrap=user_profile.plotFacetColWrapVal,
        sorting_choice=[
            user_profile.plotInitialSortingDirection,
            user_profile.plotNMInitialSortingChoice,
        ],
        return_as_dict=return_as_dict,
    )
    return return_structure


def _generate_nm_dose_over_patient_weight(user_profile, return_as_dict, df):
    return_structure = {}
    parameter_dict = {
        "df_name_col": "study_description",
        "df_x_col": "patientstudymoduleattr__patient_weight",
        "df_y_col": "radiopharmaceuticalradiationdose__"
        "radiopharmaceuticaladministrationeventdata__"
        "administered_activity",
        "sorting_choice": [
            user_profile.plotInitialSortingDirection,
            user_profile.plotNMInitialSortingChoice,
        ],
        "grouping_choice": user_profile.plotGroupingChoice,
        "legend_title": _("Study description"),
        "colourmap": user_profile.plotColourMapChoice,
        "facet_col_wrap": user_profile.plotFacetColWrapVal,
        "x_axis_title": _("Patient mass (kg)"),
        "y_axis_title": _("Administed Activity (MBq)"),
        "filename": _("OpenREM Nuclear Medicine Dose vs patient mass"),
        "return_as_dict": return_as_dict,
    }
    return_structure["studyInjectedDoseOverWeightData"] = plotly_scatter(
        df,
        parameter_dict,
    )
    return return_structure


def _generate_nm_dose_per_study(user_profile, return_as_dict, df, average_choices):
    return_structure = {}
    name_field = "study_description"
    value_field = "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity"

    if user_profile.plotMean or user_profile.plotMedian:
        t = list(average_choices)
        if "boxplot" in t:
            t.remove("boxplot")
        df_aggregated = create_dataframe_aggregates(
            df,
            [name_field],
            value_field,
            stats_to_use=t + ["count"],
        )

        parameter_dict = {
            "df_name_col": "study_description",
            "name_axis_title": _("Study description"),
            "colourmap": user_profile.plotColourMapChoice,
            "facet_col": None,
            "facet_col_wrap": user_profile.plotFacetColWrapVal,
            "return_as_dict": return_as_dict,
            "sorting_choice": [
                user_profile.plotInitialSortingDirection,
                user_profile.plotNMInitialSortingChoice,
            ],
        }
        if user_profile.plotMean:
            parameter_dict["value_axis_title"] = _("Mean Injected Dose (MBq)")
            parameter_dict["filename"] = _(
                "OpenREM nuclear medicine study injected dose mean"
            )
            parameter_dict["average_choice"] = "mean"
            (
                return_structure["studyMeanInjectedDoseData"],
                return_structure["studyMeanInjectedDoseDataCSV"],
            ) = plotly_barchart(
                df_aggregated,
                parameter_dict,
                csv_name="studyMeanInjectedDoseData.csv",
            )

        if user_profile.plotMedian:
            parameter_dict["value_axis_title"] = _("Median Injected Dose (MBq)")
            parameter_dict["filename"] = _(
                "OpenREM nuclear medicine study injected dose median"
            )
            parameter_dict["average_choice"] = "median"
            (
                return_structure["studyMedianInjectedDoseData"],
                return_structure["studyMedianInjectedDoseDataCSV"],
            ) = plotly_barchart(
                df_aggregated,
                parameter_dict,
                csv_name="studyMedianInjectedDoseData.csv",
            )

    if user_profile.plotBoxplots:
        parameter_dict = {
            "df_name_col": name_field,
            "df_value_col": value_field,
            "value_axis_title": _("Injected Dose (MBq)"),
            "name_axis_title": _("Study description"),
            "colourmap": user_profile.plotColourMapChoice,
            "filename": _("OpenREM nuclear medicine study injected dose boxplot"),
            "facet_col": None,
            "sorting_choice": [
                user_profile.plotInitialSortingDirection,
                user_profile.plotNMInitialSortingChoice,
            ],
            "facet_col_wrap": user_profile.plotFacetColWrapVal,
            "return_as_dict": return_as_dict,
        }

        return_structure["studyBoxplotInjectedDoseData"] = plotly_boxplot(
            df,
            parameter_dict,
        )

    if user_profile.plotHistograms:
        category_names_col = name_field
        group_by_col = "x_ray_system_name"
        legend_title = _("Study description")

        if user_profile.plotGroupingChoice == "series":
            category_names_col = "x_ray_system_name"
            group_by_col = name_field
            legend_title = _("System")

        parameter_dict = {
            "df_facet_col": group_by_col,
            "df_category_col": category_names_col,
            "df_value_col": value_field,
            "value_axis_title": _("Injected Dose (MBq)"),
            "legend_title": legend_title,
            "n_bins": user_profile.plotHistogramBins,
            "colourmap": user_profile.plotColourMapChoice,
            "filename": _("OpenREM nuclear medicine study injected dose histogram"),
            "facet_col_wrap": user_profile.plotFacetColWrapVal,
            "sorting_choice": [
                user_profile.plotInitialSortingDirection,
                user_profile.plotNMInitialSortingChoice,
            ],
            "global_max_min": user_profile.plotHistogramGlobalBins,
            "return_as_dict": return_as_dict,
        }
        return_structure["studyHistogramInjectedDoseData"] = plotly_histogram_barchart(
            df,
            parameter_dict,
        )
    return return_structure


def _generate_nm_dose_over_time(
    user_profile: UserProfile, return_as_dict, df, average_choices
):
    return_structure = {}

    if user_profile.plotGroupingChoice == "series":
        facet_title = _("Study description")
    else:
        facet_title = _("System")
    name_field = "study_description"
    value_field = "radiopharmaceuticalradiationdose__radiopharmaceuticaladministrationeventdata__administered_activity"

    if user_profile.plotMean or user_profile.plotMedian:
        parameter_dict = {
            "df_name_col": name_field,
            "df_value_col": value_field,
            "df_date_col": "study_date",
            "name_title": _("Study description"),
            "value_title": _("Administered Dose (MBq)"),
            "date_title": _("Study date"),
            "facet_title": facet_title,
            "sorting_choice": [
                user_profile.plotInitialSortingDirection,
                user_profile.plotNMInitialSortingChoice,
            ],
            "time_period": user_profile.plotNMOverTimePeriod,
            "average_choices": average_choices + ["count"],
            "grouping_choice": user_profile.plotGroupingChoice,
            "colourmap": user_profile.plotColourMapChoice,
            "facet_col_wrap": user_profile.plotFacetColWrapVal,
            "filename": _("OpenREM Nuclear medicine injected dose over time"),
            "return_as_dict": return_as_dict,
        }
        result = construct_over_time_charts(
            df,
            parameter_dict,
        )

        if user_profile.plotMean:
            return_structure["studyInjectedDoseOverTimeMeanData"] = result["mean"]
        if user_profile.plotMedian:
            return_structure["studyInjectedDoseOverTimeMedianData"] = result["median"]
        if user_profile.plotBoxplots:
            return_structure["studyInjectedDoseOverTimeBoxplotData"] = result["boxplot"]

    return return_structure
