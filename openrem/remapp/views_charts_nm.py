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
from remapp.interface.mod_filters import ct_acq_filter
from remapp.models import (
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

def generate_required_nm_charts_list(user_profile):
    return []