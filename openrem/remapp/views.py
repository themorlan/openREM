#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2012,2013  The Royal Marsden NHS Foundation Trust
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    Additional permission under section 7 of GPLv3:
#    You shall not make any use of the name of The Royal Marsden NHS
#    Foundation trust in connection with this Program in any press or
#    other public announcement without the prior written consent of
#    The Royal Marsden NHS Foundation Trust.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    8/10/2014: DJP added new DX section and added DX to home page.
#    9/10/2014: DJP changed DX to CR
#
"""
..  module:: views.
    :synopsis: Module to render appropriate content according to request.

..  moduleauthor:: Ed McDonagh

"""
from __future__ import absolute_import

# Following two lines added so that sphinx autodocumentation works.
from future import standard_library

standard_library.install_aliases()
from builtins import map  # pylint: disable=redefined-builtin
import os

os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"

from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import Group
from django.core.exceptions import ObjectDoesNotExist
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.urls import reverse_lazy
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
import json
import logging
import remapp
from openrem.openremproject.settings import MEDIA_ROOT
from .models import GeneralStudyModuleAttr, create_user_profile

try:
    from numpy import *

    plotting = 1
except ImportError:
    plotting = 0


from django.template.defaultfilters import register


logger = logging.getLogger(__name__)


@register.filter
def multiply(value, arg):
    """
    Return multiplication within Django templates

    :param value: the value to multiply
    :param arg: the second value to multiply
    :return: the multiplication
    """
    try:
        value = float(value)
        arg = float(arg)
        return value * arg
    except ValueError:
        return None


def logout_page(request):
    """
    Log users out and re-direct them to the main page.
    """
    logout(request)
    return HttpResponseRedirect(reverse_lazy("home"))


@login_required
def dx_summary_list_filter(request):
    """Obtain data for radiographic summary view
    """
    from remapp.interface.mod_filters import dx_acq_filter
    from remapp.forms import DXChartOptionsForm, itemsPerPageForm
    from openremproject import settings

    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = dx_acq_filter(request.GET, pid=pid)

    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile

    # Obtain the chart options from the request
    chart_options_form = DXChartOptionsForm(request.GET)
    # check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.plotCharts = chart_options_form.cleaned_data["plotCharts"]
            user_profile.plotDXAcquisitionMeanDAP = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeanDAP"
            ]
            user_profile.plotDXAcquisitionFreq = chart_options_form.cleaned_data[
                "plotDXAcquisitionFreq"
            ]
            user_profile.plotDXStudyMeanDAP = chart_options_form.cleaned_data[
                "plotDXStudyMeanDAP"
            ]
            user_profile.plotDXStudyFreq = chart_options_form.cleaned_data[
                "plotDXStudyFreq"
            ]
            user_profile.plotDXRequestMeanDAP = chart_options_form.cleaned_data[
                "plotDXRequestMeanDAP"
            ]
            user_profile.plotDXRequestFreq = chart_options_form.cleaned_data[
                "plotDXRequestFreq"
            ]
            user_profile.plotDXAcquisitionMeankVp = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeankVp"
            ]
            user_profile.plotDXAcquisitionMeanmAs = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeanmAs"
            ]
            user_profile.plotDXStudyPerDayAndHour = chart_options_form.cleaned_data[
                "plotDXStudyPerDayAndHour"
            ]
            user_profile.plotDXAcquisitionMeankVpOverTime = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeankVpOverTime"
            ]
            user_profile.plotDXAcquisitionMeanmAsOverTime = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeanmAsOverTime"
            ]
            user_profile.plotDXAcquisitionMeanDAPOverTime = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeanDAPOverTime"
            ]
            user_profile.plotDXAcquisitionMeanDAPOverTimePeriod = chart_options_form.cleaned_data[
                "plotDXAcquisitionMeanDAPOverTimePeriod"
            ]
            user_profile.plotDXAcquisitionDAPvsMass = chart_options_form.cleaned_data[
                "plotDXAcquisitionDAPvsMass"
            ]
            user_profile.plotDXStudyDAPvsMass = chart_options_form.cleaned_data[
                "plotDXStudyDAPvsMass"
            ]
            user_profile.plotDXRequestDAPvsMass = chart_options_form.cleaned_data[
                "plotDXRequestDAPvsMass"
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
            user_profile.plotDXInitialSortingChoice = chart_options_form.cleaned_data[
                "plotDXInitialSortingChoice"
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

        # If submit was not clicked then use the settings already stored in the user's profile
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
                "plotDXAcquisitionMeanDAP": user_profile.plotDXAcquisitionMeanDAP,
                "plotDXAcquisitionFreq": user_profile.plotDXAcquisitionFreq,
                "plotDXStudyMeanDAP": user_profile.plotDXStudyMeanDAP,
                "plotDXStudyFreq": user_profile.plotDXStudyFreq,
                "plotDXRequestMeanDAP": user_profile.plotDXRequestMeanDAP,
                "plotDXRequestFreq": user_profile.plotDXRequestFreq,
                "plotDXAcquisitionMeankVp": user_profile.plotDXAcquisitionMeankVp,
                "plotDXAcquisitionMeanmAs": user_profile.plotDXAcquisitionMeanmAs,
                "plotDXStudyPerDayAndHour": user_profile.plotDXStudyPerDayAndHour,
                "plotDXAcquisitionMeankVpOverTime": user_profile.plotDXAcquisitionMeankVpOverTime,
                "plotDXAcquisitionMeanmAsOverTime": user_profile.plotDXAcquisitionMeanmAsOverTime,
                "plotDXAcquisitionMeanDAPOverTime": user_profile.plotDXAcquisitionMeanDAPOverTime,
                "plotDXAcquisitionMeanDAPOverTimePeriod": user_profile.plotDXAcquisitionMeanDAPOverTimePeriod,
                "plotDXAcquisitionDAPvsMass": user_profile.plotDXAcquisitionDAPvsMass,
                "plotDXStudyDAPvsMass": user_profile.plotDXStudyDAPvsMass,
                "plotDXRequestDAPvsMass": user_profile.plotDXRequestDAPvsMass,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotDXInitialSortingChoice": user_profile.plotDXInitialSortingChoice,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices
            }
            chart_options_form = DXChartOptionsForm(form_data)

    # Obtain the number of items per page from the request
    items_per_page_form = itemsPerPageForm(request.GET)
    # check whether the form data is valid
    if items_per_page_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.itemsPerPage = items_per_page_form.cleaned_data["itemsPerPage"]
            user_profile.save()

        # If submit was not clicked then use the settings already stored in the user's profile
        else:
            form_data = {"itemsPerPage": user_profile.itemsPerPage}
            items_per_page_form = itemsPerPageForm(form_data)

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    paginator = Paginator(f.qs, user_profile.itemsPerPage)
    page = request.GET.get("page")
    try:
        study_list = paginator.page(page)
    except PageNotAnInteger:
        study_list = paginator.page(1)
    except EmptyPage:
        study_list = paginator.page(paginator.num_pages)

    return_structure = {
        "filter": f,
        "study_list": study_list,
        "admin": admin,
        "chartOptionsForm": chart_options_form,
        "itemsPerPageForm": items_per_page_form,
    }

    if user_profile.plotCharts:
        return_structure["required_charts"] = generate_required_dx_charts_list(user_profile)

    return render(request, "remapp/dxfiltered.html", return_structure,)


def generate_required_dx_charts_list(profile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    if profile.plotDXAcquisitionMeanDAP:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DAP for each acquisition protocol",
                                    "var_name": "acquisitionMeanDAP"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DAP for each acquisition protocol",
                                    "var_name": "acquisitionMedianDAP"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of DAP for each acquisition protocol",
                                    "var_name": "acquisitionBoxplotDAP"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DAP for each acquisition protocol",
                                    "var_name": "acquisitionHistogramDAP"})

    if profile.plotDXAcquisitionFreq:
        required_charts.append({"title": "Chart of acquisition protocol frequency",
                                "var_name": "acquisitionFrequency"})

    if profile.plotDXStudyMeanDAP:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DAP for each study description",
                                    "var_name": "studyMeanDAP"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DAP for each study description",
                                    "var_name": "studyMedianDAP"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of DAP for each study description",
                                    "var_name": "studyBoxplotDAP"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DAP for each study description",
                                    "var_name": "studyHistogramDAP"})

    if profile.plotDXStudyFreq:
        required_charts.append({"title": "Chart of study description frequency",
                                "var_name": "studyFrequency"})

    if profile.plotDXRequestMeanDAP:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DAP for each requested procedure",
                                    "var_name": "requestMeanDAP"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DAP for each requested procedure",
                                    "var_name": "requestMedianDAP"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of DAP for each requested procedure",
                                    "var_name": "requestBoxplotDAP"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DAP for each requested procedure",
                                    "var_name": "requestHistogramDAP"})

    if profile.plotDXRequestFreq:
        required_charts.append({"title": "Chart of requested procedure frequency",
                                "var_name": "requestFrequency"})

    if profile.plotDXAcquisitionMeankVp:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean kVp for each acquisition protocol",
                                    "var_name": "acquisitionMeankVp"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median kVp for each acquisition protocol",
                                    "var_name": "acquisitionMediankVp"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of kVp for each acquisition protocol",
                                    "var_name": "acquisitionBoxplotkVp"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DAP for each acquisition protocol",
                                    "var_name": "acquisitionHistogramkVp"})

    if profile.plotDXAcquisitionMeanmAs:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean mAs for each acquisition protocol",
                                    "var_name": "acquisitionMeanmAs"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median mAs for each acquisition protocol",
                                    "var_name": "acquisitionMedianmAs"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of mAs for each acquisition protocol",
                                    "var_name": "acquisitionBoxplotmAs"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DAP for each acquisition protocol",
                                    "var_name": "acquisitionHistogrammAs"})

    if profile.plotDXStudyPerDayAndHour:
        required_charts.append({"title": "Chart of study description workload",
                                "var_name": "studyWorkload"})

    if profile.plotDXAcquisitionMeankVpOverTime:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean kVp per acquisition protocol over time (" + profile.plotDXAcquisitionMeanDAPOverTimePeriod + ")",
                                    "var_name": "acquisitionMeankVpOverTime"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median kVp per acquisition protocol over time (" + profile.plotDXAcquisitionMeanDAPOverTimePeriod + ")",
                                    "var_name": "acquisitionMediankVpOverTime"})

    if profile.plotDXAcquisitionMeanmAsOverTime:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean mAs per acquisition protocol over time (" + profile.plotDXAcquisitionMeanDAPOverTimePeriod + ")",
                                    "var_name": "acquisitionMeanmAsOverTime"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median mAs per acquisition protocol over time (" + profile.plotDXAcquisitionMeanDAPOverTimePeriod + ")",
                                    "var_name": "acquisitionMedianmAsOverTime"})

    if profile.plotDXAcquisitionMeanDAPOverTime:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DAP per acquisition protocol over time (" + profile.plotDXAcquisitionMeanDAPOverTimePeriod + ")",
                                    "var_name": "acquisitionMeanDAPOverTime"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DAP per acquisition protocol over time (" + profile.plotDXAcquisitionMeanDAPOverTimePeriod + ")",
                                    "var_name": "acquisitionMedianDAPOverTime"})

    if profile.plotDXAcquisitionDAPvsMass:
        required_charts.append({"title": "Chart of acquisition protocol DAP vs patient mass",
                                "var_name": "acquisitionDAPvsMass"})
    if profile.plotDXStudyDAPvsMass:
        required_charts.append({"title": "Chart of study description DAP vs patient mass",
                                "var_name": "studyDAPvsMass"})
    if profile.plotDXRequestDAPvsMass:
        required_charts.append({"title": "Chart of requested procedure DAP vs patient mass",
                                "var_name": "requestDAPvsMass"})

    return required_charts


@login_required
def dx_summary_chart_data(request):
    """Obtain data for Ajax chart call
    """
    from remapp.interface.mod_filters import dx_acq_filter
    from openremproject import settings
    from django.http import JsonResponse

    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = dx_acq_filter(request.GET, pid=pid)

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


    return_structure = dx_plot_calculations(
        f,
        user_profile
    )

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def dx_plot_calculations(
    f,
    user_profile
):
    """Calculations for radiographic charts
    """
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
        construct_over_time_charts
    )

    if (
        user_profile.plotDXAcquisitionMeanDAPOverTime
        or user_profile.plotDXAcquisitionMeankVpOverTime
        or user_profile.plotDXAcquisitionMeanmAsOverTime
    ):
        # Obtain the key name in the TIME_PERIOD tuple from the user time period choice (the key value)
        keys = list(dict(user_profile.TIME_PERIOD).keys())
        values = list(dict(user_profile.TIME_PERIOD).values())
        plot_timeunit_period = keys[[tp.lower() for tp in values].index(user_profile.plotDXAcquisitionMeanDAPOverTimePeriod)]

    # Set the Plotly chart theme
    plotly_set_default_theme(user_profile.plotThemeChoice)

    return_structure = {}

    average_choices = []
    if user_profile.plotMean:
        average_choices.append("mean")
    if user_profile.plotMedian:
        average_choices.append("median")

    #######################################################################
    # Prepare acquisition-level Pandas DataFrame to use for charts
    if (
        user_profile.plotDXAcquisitionMeanDAP
        or user_profile.plotDXAcquisitionFreq
        or user_profile.plotDXAcquisitionMeankVp
        or user_profile.plotDXAcquisitionMeanmAs
        or user_profile.plotDXAcquisitionMeankVpOverTime
        or user_profile.plotDXAcquisitionMeanmAsOverTime
        or user_profile.plotDXAcquisitionMeanDAPOverTime
        or user_profile.plotDXAcquisitionDAPvsMass
    ):

        name_fields = ["projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"]

        value_fields = []
        value_multipliers = []
        if (
            user_profile.plotDXAcquisitionMeanDAP
            or user_profile.plotDXAcquisitionMeanDAPOverTime
            or user_profile.plotDXAcquisitionDAPvsMass
        ):
            value_fields.append("projectionxrayradiationdose__irradeventxraydata__dose_area_product")
            value_multipliers.append(1000000)
        if (
            user_profile.plotDXAcquisitionMeankVp
            or user_profile.plotDXAcquisitionMeankVpOverTime
        ):
            value_fields.append("projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp")
            value_multipliers.append(1)
        if (
            user_profile.plotDXAcquisitionMeanmAs
            or user_profile.plotDXAcquisitionMeanmAsOverTime
        ):
            value_fields.append("projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure")
            value_multipliers.append(0.001)
        if (
            user_profile.plotDXAcquisitionDAPvsMass
        ):
            value_fields.append("patientstudymoduleattr__patient_weight")
            value_multipliers.append(1)

        date_fields = []
        if (
            user_profile.plotDXAcquisitionMeanDAPOverTime
            or user_profile.plotDXAcquisitionMeankVpOverTime
            or user_profile.plotDXAcquisitionMeanmAsOverTime
        ):
            date_fields.append("study_date")

        system_field = None
        if user_profile.plotSeriesPerSystem:
            system_field = "generalequipmentmoduleattr__unique_equipment_name_id__display_name"

        df = create_dataframe(
            f.qs,
            data_point_name_fields=name_fields,
            data_point_value_fields=value_fields,
            data_point_date_fields=date_fields,
            system_name_field=system_field,
            data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
            data_point_value_multipliers=value_multipliers,
            uid="projectionxrayradiationdose__irradeventxraydata__pk"
        )
        #######################################################################
        sorted_acquisition_dap_categories = None
        if user_profile.plotDXAcquisitionMeanDAP:
            sorted_acquisition_dap_categories = create_sorted_category_list(
                df,
                "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                [user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                    stats=average_choices + ["count"]
                )

                if user_profile.plotMean:
                    return_structure["acquisitionMeanDAPData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol DAP mean",
                        sorted_category_list=sorted_acquisition_dap_categories,
                        average_choice="mean"
                    )

                if user_profile.plotMedian:
                    return_structure["acquisitionMedianDAPData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol DAP median",
                        sorted_category_list=sorted_acquisition_dap_categories,
                        average_choice="median"
                    )

            if user_profile.plotBoxplots:
                return_structure["acquisitionBoxplotDAPData"] = plotly_boxplot(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Acquisition protocol",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol DAP boxplot",
                    sorted_category_list=sorted_acquisition_dap_categories
                )

            if user_profile.plotHistograms:
                category_names_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_dap_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_dap_categories.values())[0]

                return_structure["acquisitionHistogramDAPData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol DAP histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        sorted_acquisition_kvp_categories = None
        if user_profile.plotDXAcquisitionMeankVp:
            sorted_acquisition_kvp_categories = create_sorted_category_list(
                df,
                "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                [user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                    stats=average_choices + ["count"]
                )

                if user_profile.plotMean:
                    return_structure["acquisitionMeankVpData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Mean kVp",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol kVp mean",
                        sorted_category_list=sorted_acquisition_kvp_categories,
                        average_choice="mean"
                    )

                if user_profile.plotMedian:
                    return_structure["acquisitionMediankVpData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Median kVp",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol kVp median",
                        sorted_category_list=sorted_acquisition_kvp_categories,
                        average_choice="median"
                    )

            if user_profile.plotBoxplots:
                return_structure["acquisitionBoxplotkVpData"] = plotly_boxplot(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                    value_axis_title="kVp",
                    name_axis_title="Acquisition protocol",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol kVp boxplot",
                    sorted_category_list=sorted_acquisition_kvp_categories
                )

            if user_profile.plotHistograms:
                category_names_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_kvp_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_kvp_categories.values())[0]

                return_structure["acquisitionHistogramDAPData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                    value_axis_title="kVp",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol kVp histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        sorted_acquisition_mas_categories = None
        if user_profile.plotDXAcquisitionMeanmAs:
            sorted_acquisition_mas_categories = create_sorted_category_list(
                df,
                "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                [user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                    stats=average_choices + ["count"]
                )

                if user_profile.plotMean:
                    return_structure["acquisitionMeanmAsData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Mean mAs",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol mAs mean",
                        sorted_category_list=sorted_acquisition_mas_categories,
                        average_choice="mean"
                    )

                if user_profile.plotMedian:
                    return_structure["acquisitionMedianmAsData"] = plotly_barchart(
                        df_aggregated,
                        "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                        value_axis_title="Median mAs",
                        name_axis_title="Acquisition protocol",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX acquisition protocol mAs median",
                        sorted_category_list=sorted_acquisition_mas_categories,
                        average_choice="median"
                    )

            if user_profile.plotBoxplots:
                return_structure["acquisitionBoxplotmAsData"] = plotly_boxplot(
                    df,
                    "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                    value_axis_title="mAs",
                    name_axis_title="Acquisition protocol",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol mAs boxplot",
                    sorted_category_list=sorted_acquisition_mas_categories
                )

            if user_profile.plotHistograms:
                category_names_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_mas_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_mas_categories.values())[0]

                return_structure["acquisitionHistogramDAPData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                    value_axis_title="mAs",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX acquisition protocol mAs histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        if user_profile.plotDXAcquisitionFreq:
            sorted_categories = None
            if user_profile.plotDXAcquisitionMeanDAP:
                sorted_categories = sorted_acquisition_dap_categories
            elif user_profile.plotDXAcquisitionMeankVp:
                sorted_categories = sorted_acquisition_kvp_categories
            elif user_profile.plotDXAcquisitionMeanmAs:
                sorted_categories = sorted_acquisition_mas_categories

            return_structure["acquisitionFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                sorting_choice=user_profile.plotDXInitialSortingChoice,
                legend_title="Acquisition protocol",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM DX acquisition protocol frequency",
                sorted_categories=sorted_categories
            )

        if user_profile.plotDXAcquisitionMeanDAPOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_value_col="projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                df_date_col="study_date",
                name_title="Acquisition protocol",
                value_title="DAP (cGy.cm<sup>2</sup>)",
                date_title="Study date",
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX acquisition protocol DAP over time"
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanDAPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianDAPOverTime"] = result["median"]

        if user_profile.plotDXAcquisitionMeankVpOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_value_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
                df_date_col="study_date",
                name_title="Acquisition protocol",
                value_title="kVp",
                date_title="Study date",
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX acquisition protocol kVp over time"
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeankVpOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMediankVpOverTime"] = result["median"]

        if user_profile.plotDXAcquisitionMeanmAsOverTime:
            result = construct_over_time_charts(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_value_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
                df_date_col="study_date",
                name_title="Acquisition protocol",
                value_title="mAs",
                date_title="Study date",
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX acquisition protocol mAs over time"
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanmAsOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianmAsOverTime"] = result["median"]

        if user_profile.plotDXAcquisitionDAPvsMass:
            return_structure["acquisitionDAPvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="projectionxrayradiationdose__irradeventxraydata__dose_area_product",
                x_axis_title="Patient mass (kg)",
                y_axis_title="DAP (mGy.cm<sup>2</sub>)",
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Acquisition protocol",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX acquisition protocol DAP vs patient mass"
            )

    #######################################################################
    # Prepare study- and request-level Pandas DataFrame to use for charts
    if (
        user_profile.plotDXStudyMeanDAP,
        user_profile.plotDXStudyFreq,
        user_profile.plotDXStudyPerDayAndHour,
        user_profile.plotDXStudyDAPvsMass,
        user_profile.plotDXRequestMeanDAP,
        user_profile.plotDXRequestFreq,
        user_profile.plotDXRequestDAPvsMass
    ):

        name_fields = []
        if (
            user_profile.plotDXStudyMeanDAP
            or user_profile.plotDXStudyFreq
            or user_profile.plotDXStudyPerDayAndHour
            or user_profile.plotDXStudyDAPvsMass
        ):
            name_fields.append("study_description")
        if (
            user_profile.plotDXRequestMeanDAP
            or user_profile.plotDXRequestFreq
            or user_profile.plotDXRequestDAPvsMass
        ):
            name_fields.append("requested_procedure_code_meaning")

        value_fields = []
        value_multipliers = []
        if (
            user_profile.plotDXStudyMeanDAP
            or user_profile.plotDXRequestMeanDAP
            or user_profile.plotDXStudyDAPvsMass
            or user_profile.plotDXRequestDAPvsMass
        ):
            value_fields.append("total_dap")
            value_multipliers.append(1000000)
        if (
            user_profile.plotDXStudyDAPvsMass
            or user_profile.plotDXRequestDAPvsMass
        ):
            value_fields.append("patientstudymoduleattr__patient_weight")
            value_multipliers.append(1)

        date_fields = []
        time_fields = []
        if (
            user_profile.plotDXStudyPerDayAndHour
        ):
            date_fields.append("study_date")
            time_fields.append("study_time")

        system_field = None
        if user_profile.plotSeriesPerSystem:
            system_field = "generalequipmentmoduleattr__unique_equipment_name_id__display_name"

        df = create_dataframe(
            f.qs,
            data_point_name_fields=name_fields,
            data_point_value_fields=value_fields,
            data_point_date_fields=date_fields,
            data_point_time_fields=time_fields,
            system_name_field=system_field,
            data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
            data_point_value_multipliers=value_multipliers,
            uid="pk"
        )
        #######################################################################

        sorted_study_dap_categories = None
        if user_profile.plotDXStudyMeanDAP:
            sorted_study_dap_categories = create_sorted_category_list(
                df,
                "study_description",
                "total_dap",
                [user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "study_description",
                    "total_dap",
                    stats=average_choices + ["count"]
                )

                if user_profile.plotMean:
                    return_structure["studyMeanDAPData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX Study description DAP mean",
                        sorted_category_list=sorted_study_dap_categories,
                        average_choice="mean"
                    )
                    
                if user_profile.plotMedian:
                    return_structure["studyMedianDAPData"] = plotly_barchart(
                        df_aggregated,
                        "study_description",
                        value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Study description",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX Study description DAP median",
                        sorted_category_list=sorted_study_dap_categories,
                        average_choice="median"
                    )

            if user_profile.plotBoxplots:
                return_structure["studyBoxplotDAPData"] = plotly_boxplot(
                    df,
                    "study_description",
                    "total_dap",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Study description",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX study description DAP boxplot",
                    sorted_category_list=sorted_study_dap_categories
                )

            if user_profile.plotHistograms:
                category_names_col = "study_description"
                group_by_col = "x_ray_system_name"
                legend_title = "Study description"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_study_dap_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "study_description"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_study_dap_categories.values())[0]

                return_structure["studyHistogramDAPData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "total_dap",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX study description DAP histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        if user_profile.plotDXStudyFreq:
            return_structure["studyFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="study_description",
                sorting_choice=user_profile.plotDXInitialSortingChoice,
                legend_title="Study description",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM DX study description frequency",
                sorted_categories = sorted_study_dap_categories
            )

        sorted_request_dap_categories = None
        if user_profile.plotDXRequestMeanDAP:
            sorted_request_dap_categories = create_sorted_category_list(
                df,
                "requested_procedure_code_meaning",
                "total_dap",
                [user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dap",
                    stats=average_choices + ["count"]
                )

                if user_profile.plotMean:
                    return_structure["requestMeanDAPData"] = plotly_barchart(
                        df_aggregated,
                        "requested_procedure_code_meaning",
                        value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Requested procedure",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX requested procedure DAP mean",
                        sorted_category_list=sorted_request_dap_categories,
                        average_choice="mean"
                    )

                if user_profile.plotMedian:
                    return_structure["requestMedianDAPData"] = plotly_barchart(
                        df_aggregated,
                        "requested_procedure_code_meaning",
                        value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                        name_axis_title="Requested procedure",
                        colourmap=user_profile.plotColourMapChoice,
                        filename="OpenREM DX requested procedure DAP median",
                        sorted_category_list=sorted_request_dap_categories,
                        average_choice="median"
                    )

            if user_profile.plotBoxplots:
                return_structure["requestBoxplotDAPData"] = plotly_boxplot(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dap",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Requested procedure",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX requested procedure DAP boxplot",
                    sorted_category_list=sorted_request_dap_categories
                )

            if user_profile.plotHistograms:
                category_names_col = "requested_procedure_code_meaning"
                group_by_col = "x_ray_system_name"
                legend_title = "Requested procedure"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_request_dap_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "requested_procedure_code_meaning"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_request_dap_categories.values())[0]

                return_structure["requestHistogramDAPData"] = plotly_histogram_barchart(
                    df,
                    group_by_col,
                    category_names_col,
                    "total_dap",
                    value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                    legend_title=legend_title,
                    n_bins=user_profile.plotHistogramBins,
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM DX requested procedure DAP histogram",
                    facet_col_wrap=user_profile.plotFacetColWrapVal,
                    df_facet_category_list=facet_names,
                    df_category_name_list=category_names,
                )

        if user_profile.plotDXRequestFreq:
            return_structure["requestFrequencyData"] = construct_frequency_chart(
                df=df,
                df_name_col="requested_procedure_code_meaning",
                sorting_choice=user_profile.plotDXInitialSortingChoice,
                legend_title="Requested procedure",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM DX requested procedure frequency",
                sorted_categories=sorted_request_dap_categories
            )

        if user_profile.plotDXStudyPerDayAndHour:
            df_time_series_per_weekday = create_dataframe_weekdays(
                df,
                "study_description",
                df_date_col="study_date"
            )

            return_structure["studyWorkloadData"] = plotly_barchart_weekdays(
                df_time_series_per_weekday,
                "weekday",
                "study_description",
                name_axis_title="Weekday",
                value_axis_title="Frequency",
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM DX study description workload",
                facet_col_wrap=user_profile.plotFacetColWrapVal
            )

        if user_profile.plotDXStudyDAPvsMass:
            return_structure["studyDAPvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="study_description",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="total_dap",
                x_axis_title="Patient mass (kg)",
                y_axis_title="DAP (mGy.cm<sup>2</sub>)",
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Study description",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX study description DAP vs patient mass"
            )

        if user_profile.plotDXRequestDAPvsMass:
            return_structure["requestDAPvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="requested_procedure_code_meaning",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="total_dap",
                x_axis_title="Patient mass (kg)",
                y_axis_title="DAP (mGy.cm<sup>2</sub>)",
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotDXInitialSortingChoice],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Requested procedure",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM DX requested procedure DAP vs patient mass"
            )

    return return_structure


@login_required
def dx_detail_view(request, pk=None):
    """Detail view for a DX study
    """

    try:
        study = GeneralStudyModuleAttr.objects.get(pk=pk)
    except:
        messages.error(request, "That study was not found")
        return redirect(reverse_lazy("dx_summary_list_filter"))

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    projection_set = study.projectionxrayradiationdose_set.get()
    events_all = projection_set.irradeventxraydata_set.select_related(
        "anatomical_structure",
        "laterality",
        "target_region",
        "image_view",
        "patient_orientation_modifier_cid",
        "acquisition_plane",
    ).all()
    accum_set = projection_set.accumxraydose_set.all()
    # accum_integrated = projection_set.accumxraydose_set.get().accumintegratedprojradiogdose_set.get()

    return render(
        request,
        "remapp/dxdetail.html",
        {
            "generalstudymoduleattr": study,
            "admin": admin,
            "projection_set": projection_set,
            "events_all": events_all,
            "accum_set": accum_set,
        },
    )


@login_required
def rf_summary_list_filter(request):
    """Obtain data for radiographic summary view
    """
    from remapp.interface.mod_filters import RFSummaryListFilter, RFFilterPlusPid
    from openremproject import settings
    from remapp.forms import RFChartOptionsForm, itemsPerPageForm
    from remapp.models import HighDoseMetricAlertSettings

    if request.user.groups.filter(name="pidgroup"):
        f = RFFilterPlusPid(
            request.GET,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="RF")
            .order_by("-study_date", "-study_time")
            .distinct(),
        )
    else:
        f = RFSummaryListFilter(
            request.GET,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="RF")
            .order_by("-study_date", "-study_time")
            .distinct(),
        )

    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile

    # Obtain the chart options from the request
    chart_options_form = RFChartOptionsForm(request.GET)
    # Check whether the form data is valid
    if chart_options_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.plotCharts = chart_options_form.cleaned_data["plotCharts"]
            user_profile.plotRFStudyPerDayAndHour = chart_options_form.cleaned_data[
                "plotRFStudyPerDayAndHour"
            ]
            user_profile.plotRFStudyFreq = chart_options_form.cleaned_data[
                "plotRFStudyFreq"
            ]
            user_profile.plotRFStudyDAP = chart_options_form.cleaned_data[
                "plotRFStudyDAP"
            ]
            user_profile.plotRFRequestFreq = chart_options_form.cleaned_data[
                "plotRFRequestFreq"
            ]
            user_profile.plotRFRequestDAP = chart_options_form.cleaned_data[
                "plotRFRequestDAP"
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
            user_profile.plotRFInitialSortingChoice = chart_options_form.cleaned_data[
                "plotRFInitialSortingChoice"
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
                "plotRFStudyPerDayAndHour": user_profile.plotRFStudyPerDayAndHour,
                "plotRFStudyFreq": user_profile.plotRFStudyFreq,
                "plotRFStudyDAP": user_profile.plotRFStudyDAP,
                "plotRFRequestFreq": user_profile.plotRFRequestFreq,
                "plotRFRequestDAP": user_profile.plotRFRequestDAP,
                "plotGrouping": user_profile.plotGroupingChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
                "plotRFInitialSortingChoice": user_profile.plotRFInitialSortingChoice,
                "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
                "plotAverageChoice": average_choices
            }
            chart_options_form = RFChartOptionsForm(form_data)

    # Obtain the number of items per page from the request
    items_per_page_form = itemsPerPageForm(request.GET)
    # check whether the form data is valid
    if items_per_page_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.itemsPerPage = items_per_page_form.cleaned_data["itemsPerPage"]
            user_profile.save()

        # If submit was not clicked then use the settings already stored in the user's profile
        else:
            form_data = {"itemsPerPage": user_profile.itemsPerPage}
            items_per_page_form = itemsPerPageForm(form_data)

    # Import total DAP and total dose at reference point alert levels. Create with default values if not found.
    try:
        HighDoseMetricAlertSettings.objects.get()
    except ObjectDoesNotExist:
        HighDoseMetricAlertSettings.objects.create()
    alert_levels = HighDoseMetricAlertSettings.objects.values(
        "show_accum_dose_over_delta_weeks",
        "alert_total_dap_rf",
        "alert_total_rp_dose_rf",
        "accum_dose_delta_weeks",
    )[0]

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
    }

    # # Calculate skin dose map for all objects in the database
    # import cPickle as pickle
    # import gzip
    # num_studies = f.count()
    # current_study = 0
    # for study in f:
    #     current_study += 1
    #     print "working on " + str(study.pk) + " (" + str(current_study) + " of " + str(num_studies) + ")"
    #     # Check to see if there is already a skin map pickle with the same study ID.
    #     try:
    #         study_date = study.study_date
    #         if study_date:
    #             skin_map_path = os.path.join(MEDIA_ROOT, 'skin_maps', "{0:0>4}".format(study_date.year), "{0:0>2}".format(study_date.month), "{0:0>2}".format(study_date.day), 'skin_map_'+str(study.pk)+'.p')
    #         else:
    #             skin_map_path = os.path.join(MEDIA_ROOT, 'skin_maps', 'skin_map_' + str(study.pk) + '.p')
    #     except:
    #         skin_map_path = os.path.join(MEDIA_ROOT, 'skin_maps', 'skin_map_'+str(study.pk)+'.p')
    #
    #     from remapp.version import __skin_map_version__
    #     loaded_existing_data = False
    #     if os.path.exists(skin_map_path):
    #         with gzip.open(skin_map_path, 'rb') as pickle_file:
    #             existing_skin_map_data = pickle.load(pickle_file)
    #         try:
    #             if existing_skin_map_data['skin_map_version'] == __skin_map_version__:
    #                 loaded_existing_data = True
    #                 print str(study.pk) + " already calculated"
    #         except KeyError:
    #             pass
    #
    #     if not loaded_existing_data:
    #         from remapp.tools.make_skin_map import make_skin_map
    #         make_skin_map(study.pk)
    #         print str(study.pk) + " done"

    for group in request.user.groups.all():
        admin[group.name] = True

    paginator = Paginator(f.qs, user_profile.itemsPerPage)
    page = request.GET.get("page")
    try:
        study_list = paginator.page(page)
    except PageNotAnInteger:
        study_list = paginator.page(1)
    except EmptyPage:
        study_list = paginator.page(paginator.num_pages)

    return_structure = {
        "filter": f,
        "study_list": study_list,
        "admin": admin,
        "chartOptionsForm": chart_options_form,
        "itemsPerPageForm": items_per_page_form,
        "alertLevels": alert_levels,
    }

    if user_profile.plotCharts:
        return_structure["required_charts"] = generate_required_rf_charts_list(user_profile)

    return render(request, "remapp/rffiltered.html", return_structure)


def generate_required_rf_charts_list(profile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    if profile.plotRFStudyDAP:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DAP for each study description",
                                    "var_name": "studyMeanDAP"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DAP for each study description",
                                    "var_name": "studyMedianDAP"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of DAP for each study description",
                                    "var_name": "studyBoxplotDAP"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DAP for each study description",
                                    "var_name": "studyHistogramDAP"})

    if profile.plotRFRequestDAP:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DAP for each requested procedure",
                                    "var_name": "requestMeanDAP"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DAP for each requested procedure",
                                    "var_name": "requestMedianDAP"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of DAP for each requested procedure",
                                    "var_name": "requestBoxplotDAP"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DAP for each requested procedure",
                                    "var_name": "requestHistogramDAP"})

    if profile.plotRFStudyFreq:
        required_charts.append({"title": "Chart of study description frequency",
                                "var_name": "studyFrequency"})

    if profile.plotRFRequestFreq:
        required_charts.append({"title": "Chart of requested procedure frequency",
                                "var_name": "requestFrequency"})

    if profile.plotRFStudyPerDayAndHour:
        required_charts.append({"title": "Chart of study description workload",
                                "var_name": "studyWorkload"})

    return required_charts


@login_required
def rf_summary_chart_data(request):
    """Obtain data for Ajax chart call
    """
    from remapp.interface.mod_filters import RFSummaryListFilter, RFFilterPlusPid
    from openremproject import settings
    from django.http import JsonResponse

    if request.user.groups.filter(name="pidgroup"):
        f = RFFilterPlusPid(
            request.GET,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="RF")
            .order_by()
            .distinct(),
        )
    else:
        f = RFSummaryListFilter(
            request.GET,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="RF")
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

    return_structure = rf_plot_calculations(
        f,
        user_profile
    )

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def rf_plot_calculations(
    f,
    user_profile
):
    """Calculations for fluoroscopy charts
    """
    from .interface.chart_functions import (
        create_dataframe,
        create_dataframe_weekdays,
        create_dataframe_aggregates,
        create_sorted_category_list,
        plotly_boxplot,
        plotly_barchart,
        plotly_barchart_mean_median,
        plotly_histogram_barchart,
        plotly_barchart_weekdays,
        plotly_set_default_theme,
        construct_frequency_chart,
        construct_scatter_chart,
        construct_over_time_charts
    )

    # Set the Plotly chart theme
    plotly_set_default_theme(user_profile.plotThemeChoice)

    return_structure = {}

    average_choices = []
    if user_profile.plotMean:
        average_choices.append("mean")
    if user_profile.plotMedian:
        average_choices.append("median")

    sorted_categories = None

    #######################################################################
    # Prepare Pandas DataFrame to use for charts
    name_fields = []
    if user_profile.plotRFStudyFreq or user_profile.plotRFStudyDAP or user_profile.plotRFStudyPerDayAndHour:
        name_fields.append("study_description")
    if user_profile.plotRFRequestFreq or user_profile.plotRFRequestDAP:
        name_fields.append("requested_procedure_code_meaning")

    value_fields = []
    value_multipliers = []
    if user_profile.plotRFStudyDAP or user_profile.plotRFRequestDAP:
        value_fields.append("total_dap")
        value_multipliers.append(1000000)

    date_fields = []
    time_fields = []
    if user_profile.plotRFStudyPerDayAndHour:
        date_fields.append("study_date")
        time_fields.append("study_time")

    system_field = None
    if user_profile.plotSeriesPerSystem:
        system_field = "generalequipmentmoduleattr__unique_equipment_name_id__display_name"

    df = create_dataframe(
        f.qs,
        data_point_name_fields=name_fields,
        data_point_value_fields=value_fields,
        data_point_date_fields=date_fields,
        data_point_time_fields=time_fields,
        system_name_field=system_field,
        data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
        data_point_value_multipliers=value_multipliers,
        uid="pk"
    )
    #######################################################################

    if user_profile.plotRFStudyPerDayAndHour:
        df_time_series_per_weekday = create_dataframe_weekdays(
            df,
            "study_description",
            df_date_col="study_date"
        )

        return_structure["studyWorkloadData"] = plotly_barchart_weekdays(
            df_time_series_per_weekday,
            "weekday",
            "study_description",
            name_axis_title="Weekday",
            value_axis_title="Frequency",
            colourmap=user_profile.plotColourMapChoice,
            filename="OpenREM RF study description workload",
            facet_col_wrap=user_profile.plotFacetColWrapVal
        )

    stats_to_include = ["count"]
    if user_profile.plotMean:
        stats_to_include.append("mean")
    if user_profile.plotMedian:
        stats_to_include.append("median")

    sorted_study_categories = None
    if user_profile.plotRFStudyDAP:
        sorted_study_categories = create_sorted_category_list(
            df,
            "study_description",
            "total_dap",
            [user_profile.plotInitialSortingDirection, user_profile.plotRFInitialSortingChoice]
        )

        if user_profile.plotMean or user_profile.plotMedian:
            df_aggregated = create_dataframe_aggregates(
                df,
                "study_description",
                "total_dap",
                stats=stats_to_include
            )

            if user_profile.plotMean:
                return_structure["studyMeanData"] = plotly_barchart(
                    df_aggregated,
                    "study_description",
                    value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Study description",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM RF study description DAP mean",
                    sorted_category_list=sorted_study_categories,
                    average_choice="mean"
                )

            if user_profile.plotMedian:
                return_structure["studyMedianData"] = plotly_barchart(
                    df_aggregated,
                    "study_description",
                    value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                    name_axis_title="Study description",
                    colourmap=user_profile.plotColourMapChoice,
                    filename="OpenREM RF study description DAP median",
                    sorted_category_list=sorted_study_categories,
                    average_choice="median"
                )

        if user_profile.plotBoxplots:
            return_structure["studyBoxplotData"] = plotly_boxplot(
                df,
                "study_description",
                "total_dap",
                value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                name_axis_title="Study description",
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF study description DAP boxplot",
                sorted_category_list=sorted_study_categories
            )

        if user_profile.plotHistograms:
            category_names_col = "study_description"
            group_by_col = "x_ray_system_name"
            legend_title = "Study description"
            facet_names = list(df[group_by_col].unique())
            category_names = list(sorted_study_categories.values())[0]

            if user_profile.plotGroupingChoice == "series":
                category_names_col = "x_ray_system_name"
                group_by_col = "study_description"
                legend_title = "System"
                category_names = facet_names
                facet_names = list(sorted_study_categories.values())[0]

            return_structure["studyHistogramData"] = plotly_histogram_barchart(
                df,
                group_by_col,
                category_names_col,
                "total_dap",
                value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                legend_title=legend_title,
                n_bins=user_profile.plotHistogramBins,
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF study description DAP histogram",
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                df_facet_category_list=facet_names,
                df_category_name_list=category_names,
            )

    if user_profile.plotRFStudyFreq:
        return_structure["studyFrequencyData"] = construct_frequency_chart(
            df=df,
            df_name_col="study_description",
            sorting_choice=[user_profile.plotInitialSortingDirection, user_profile.plotRFInitialSortingChoice],
            legend_title="Study description",
            df_x_axis_col="x_ray_system_name",
            x_axis_title="System",
            grouping_choice=user_profile.plotGroupingChoice,
            colour_map=user_profile.plotColourMapChoice,
            file_name="OpenREM RF study description frequency",
            sorted_categories=sorted_study_categories
        )

    sorted_request_categories = None
    if user_profile.plotRFRequestDAP:
        sorted_request_categories = create_sorted_category_list(
            df,
            "requested_procedure_code_meaning",
            "total_dap",
            [user_profile.plotInitialSortingDirection, user_profile.plotRFInitialSortingChoice]
        )

        df_aggregated = create_dataframe_aggregates(
            df,
            "requested_procedure_code_meaning",
            "total_dap",
            stats=stats_to_include
        )

        if user_profile.plotMean:
            return_structure["requestMeanData"] = plotly_barchart(
                df_aggregated,
                "requested_procedure_code_meaning",
                value_axis_title="Mean DAP (cGy.cm<sup>2</sup>)",
                name_axis_title="Requested procedure",
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF requested procedure DAP mean",
                sorted_category_list=sorted_request_categories,
                average_choice="mean"
            )

        if user_profile.plotMedian:
            return_structure["requestMedianData"] = plotly_barchart(
                df_aggregated,
                "requested_procedure_code_meaning",
                value_axis_title="Median DAP (cGy.cm<sup>2</sup>)",
                name_axis_title="Requested procedure",
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF requested procedure DAP median",
                sorted_category_list=sorted_request_categories,
                average_choice="median"
            )

        if user_profile.plotBoxplots:
            return_structure["requestBoxplotData"] = plotly_boxplot(
                df,
                "requested_procedure_code_meaning",
                "total_dap",
                value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                name_axis_title="Requested procedure",
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF requested procedure DAP boxplot",
                sorted_category_list=sorted_request_categories
            )

        if user_profile.plotHistograms:
            category_names_col = "requested_procedure_code_meaning"
            group_by_col = "x_ray_system_name"
            legend_title = "Requested procedure"
            facet_names = list(df[group_by_col].unique())
            category_names = list(sorted_request_categories.values())[0]

            if user_profile.plotGroupingChoice == "series":
                category_names_col = "x_ray_system_name"
                group_by_col = "requested_procedure_code_meaning"
                legend_title = "System"
                category_names = facet_names
                facet_names = list(sorted_request_categories.values())[0]

            return_structure["requestHistogramData"] = plotly_histogram_barchart(
                df,
                group_by_col,
                category_names_col,
                "total_dap",
                value_axis_title="DAP (cGy.cm<sup>2</sup>)",
                legend_title=legend_title,
                n_bins=user_profile.plotHistogramBins,
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM RF requested procedure DAP histogram",
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                df_facet_category_list=facet_names,
                df_category_name_list=category_names,
            )

    if user_profile.plotRFRequestFreq:
        return_structure["requestFrequencyData"] = construct_frequency_chart(
            df=df,
            df_name_col="requested_procedure_code_meaning",
            sorting_choice=[user_profile.plotInitialSortingDirection, user_profile.plotRFInitialSortingChoice],
            legend_title="Requested procedure",
            df_x_axis_col="x_ray_system_name",
            x_axis_title="System",
            grouping_choice=user_profile.plotGroupingChoice,
            colour_map=user_profile.plotColourMapChoice,
            file_name="OpenREM RF requested procedure frequency",
            sorted_categories=sorted_request_categories
        )

    return return_structure


@login_required
def rf_detail_view(request, pk=None):
    """Detail view for an RF study
    """
    from decimal import Decimal
    from django.db.models import Sum
    import numpy as np
    from remapp.models import HighDoseMetricAlertSettings, SkinDoseMapCalcSettings
    from django.core.exceptions import ObjectDoesNotExist
    from datetime import timedelta

    try:
        study = GeneralStudyModuleAttr.objects.get(pk=pk)
    except ObjectDoesNotExist:
        messages.error(request, "That study was not found")
        return redirect(reverse_lazy("rf_summary_list_filter"))

    # get the totals
    irradiation_types = [("Fluoroscopy",), ("Acquisition",)]
    fluoro_dap_total = Decimal(0)
    fluoro_rp_total = Decimal(0)
    acq_dap_total = Decimal(0)
    acq_rp_total = Decimal(0)
    stu_dose_totals = [(0, 0), (0, 0)]
    stu_time_totals = [0, 0]
    total_dap = 0
    total_dose = 0
    # Iterate over the planes (for bi-plane systems, for single plane systems there is only one)
    projection_xray_dose_set = study.projectionxrayradiationdose_set.get()
    accumxraydose_set_all_planes = projection_xray_dose_set.accumxraydose_set.select_related(
        "acquisition_plane"
    ).all()
    events_all = projection_xray_dose_set.irradeventxraydata_set.select_related(
        "irradiation_event_type",
        "patient_table_relationship_cid",
        "patient_orientation_cid",
        "patient_orientation_modifier_cid",
        "acquisition_plane",
    ).all()
    for dose_ds in accumxraydose_set_all_planes:
        accum_dose_ds = dose_ds.accumprojxraydose_set.get()
        try:
            fluoro_dap_total += accum_dose_ds.fluoro_gym2_to_cgycm2()
        except TypeError:
            pass
        try:
            fluoro_rp_total += accum_dose_ds.fluoro_dose_rp_total
        except TypeError:
            pass
        try:
            acq_dap_total += accum_dose_ds.acq_gym2_to_cgycm2()
        except TypeError:
            pass
        try:
            acq_rp_total += accum_dose_ds.acquisition_dose_rp_total
        except TypeError:
            pass
        stu_dose_totals[0] = (fluoro_dap_total, fluoro_rp_total)
        stu_dose_totals[1] = (acq_dap_total, acq_rp_total)
        stu_time_totals[0] = stu_time_totals[0] + accum_dose_ds.total_fluoro_time
        stu_time_totals[1] = stu_time_totals[1] + accum_dose_ds.total_acquisition_time
        total_dap = total_dap + accum_dose_ds.dose_area_product_total
        total_dose = total_dose + accum_dose_ds.dose_rp_total

    # get info for different Acquisition Types
    stu_inc_totals = (
        GeneralStudyModuleAttr.objects.filter(
            pk=pk,
            projectionxrayradiationdose__irradeventxraydata__irradiation_event_type__code_meaning__contains="Acquisition",
        )
        .annotate(
            sum_dap=Sum(
                "projectionxrayradiationdose__irradeventxraydata__dose_area_product"
            )
            * 1000000,
            sum_dose_rp=Sum(
                "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__dose_rp"
            ),
        )
        .order_by(
            "projectionxrayradiationdose__irradeventxraydata__irradiation_event_type"
        )
    )
    stu_dose_totals.extend(
        stu_inc_totals.values_list("sum_dap", "sum_dose_rp").order_by(
            "projectionxrayradiationdose__irradeventxraydata__irradiation_event_type"
        )
    )
    acq_irr_types = (
        stu_inc_totals.values_list(
            "projectionxrayradiationdose__irradeventxraydata__irradiation_event_type__code_meaning"
        )
        .order_by(
            "projectionxrayradiationdose__irradeventxraydata__irradiation_event_type"
        )
        .distinct()
    )
    # stu_time_totals = [None] * len(stu_irr_types)
    for _, irr_type in enumerate(acq_irr_types):
        stu_time_totals.append(
            list(
                GeneralStudyModuleAttr.objects.filter(
                    pk=pk,
                    projectionxrayradiationdose__irradeventxraydata__irradiation_event_type__code_meaning=irr_type[
                        0
                    ],
                )
                .aggregate(
                    Sum(
                        "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__irradiation_duration"
                    )
                )
                .values()
            )[0]
        )
    irradiation_types.extend([("- " + acq_type[0],) for acq_type in acq_irr_types])

    # Add the study totals
    irradiation_types.append(("Total",))
    stu_dose_totals.append((total_dap * 1000000, total_dose))
    # does total duration (summed over fluoroscopy and acquisitions) means something?
    stu_time_totals.append(stu_time_totals[0] + stu_time_totals[1])

    study_totals = np.column_stack(
        (irradiation_types, stu_dose_totals, stu_time_totals)
    ).tolist()

    try:
        SkinDoseMapCalcSettings.objects.get()
    except ObjectDoesNotExist:
        SkinDoseMapCalcSettings.objects.create()

    # Import total DAP and total dose at reference point alert levels. Create with default values if not found.
    try:
        HighDoseMetricAlertSettings.objects.get()
    except ObjectDoesNotExist:
        HighDoseMetricAlertSettings.objects.create()
    alert_levels = HighDoseMetricAlertSettings.objects.values(
        "show_accum_dose_over_delta_weeks",
        "alert_total_dap_rf",
        "alert_total_rp_dose_rf",
        "accum_dose_delta_weeks",
    )[0]

    # Obtain the studies that are within delta weeks if needed
    if alert_levels["show_accum_dose_over_delta_weeks"]:
        patient_id = study.patientmoduleattr_set.values_list("patient_id", flat=True)[0]
        if patient_id:
            study_date = study.study_date
            week_delta = HighDoseMetricAlertSettings.objects.values_list(
                "accum_dose_delta_weeks", flat=True
            )[0]
            oldest_date = study_date - timedelta(weeks=week_delta)
            included_studies = GeneralStudyModuleAttr.objects.filter(
                modality_type__exact="RF",
                patientmoduleattr__patient_id__exact=patient_id,
                study_date__range=[oldest_date, study_date],
            )
        else:
            included_studies = None
    else:
        included_studies = None

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
        "enable_skin_dose_maps": SkinDoseMapCalcSettings.objects.values_list(
            "enable_skin_dose_maps", flat=True
        )[0],
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    return render(
        request,
        "remapp/rfdetail.html",
        {
            "generalstudymoduleattr": study,
            "admin": admin,
            "study_totals": study_totals,
            "projection_xray_dose_set": projection_xray_dose_set,
            "accumxraydose_set_all_planes": accumxraydose_set_all_planes,
            "events_all": events_all,
            "alert_levels": alert_levels,
            "studies_in_week_delta": included_studies,
        },
    )


@login_required
def rf_detail_view_skin_map(request, pk=None):
    """View to calculate a skin dose map. Currently just a copy of rf_detail_view
    """
    from django.contrib import messages
    from remapp.models import GeneralStudyModuleAttr
    from django.http import JsonResponse
    import pickle as pickle
    import gzip

    from django.core.exceptions import ObjectDoesNotExist

    try:
        GeneralStudyModuleAttr.objects.get(pk=pk)
    except ObjectDoesNotExist:
        messages.error(request, "That study was not found")
        return redirect(reverse_lazy("rf_summary_list_filter"))

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    # Check to see if there is already a skin map pickle with the same study ID.
    try:
        study_date = GeneralStudyModuleAttr.objects.get(pk=pk).study_date
        if study_date:
            skin_map_path = os.path.join(
                MEDIA_ROOT,
                "skin_maps",
                "{0:0>4}".format(study_date.year),
                "{0:0>2}".format(study_date.month),
                "{0:0>2}".format(study_date.day),
                "skin_map_" + str(pk) + ".p",
            )
        else:
            skin_map_path = os.path.join(
                MEDIA_ROOT, "skin_maps", "skin_map_" + str(pk) + ".p"
            )
    except:
        skin_map_path = os.path.join(
            MEDIA_ROOT, "skin_maps", "skin_map_" + str(pk) + ".p"
        )

    from remapp.version import __skin_map_version__

    # If patient weight is missing from the database then db_pat_mass will be undefined
    try:
        db_pat_mass = float(
            GeneralStudyModuleAttr.objects.get(pk=pk)
            .patientstudymoduleattr_set.get()
            .patient_weight
        )
    except (ValueError, TypeError):
        db_pat_mass = 73.2
    if not db_pat_mass:
        db_pat_mass = 73.2

    # If patient weight is missing from the database then db_pat_mass will be undefined
    try:
        db_pat_height = (
            float(
                GeneralStudyModuleAttr.objects.get(pk=pk)
                .patientstudymoduleattr_set.get()
                .patient_size
            )
            * 100
        )
    except (ValueError, TypeError):
        db_pat_height = 178.6
    if not db_pat_height:
        db_pat_height = 178.6

    loaded_existing_data = False
    pat_mass_unchanged = False
    pat_height_unchanged = False
    if os.path.exists(skin_map_path):
        with gzip.open(skin_map_path, "rb") as f:
            existing_skin_map_data = pickle.load(f)
        try:
            if existing_skin_map_data["skin_map_version"] == __skin_map_version__:
                # Round the float values to 1 decimal place and convert to string before comparing
                if str(round(existing_skin_map_data["patient_height"], 1)) == str(
                    round(db_pat_height, 1)
                ):
                    pat_height_unchanged = True

                # Round the float values to 1 decimal place and convert to string before comparing
                if str(round(existing_skin_map_data["patient_mass"], 1)) == str(
                    round(db_pat_mass, 1)
                ):
                    pat_mass_unchanged = True

                if pat_height_unchanged and pat_mass_unchanged:
                    return_structure = existing_skin_map_data
                    loaded_existing_data = True
        except KeyError:
            pass

    if not loaded_existing_data:
        from remapp.tools.make_skin_map import make_skin_map

        make_skin_map(pk)
        with gzip.open(skin_map_path, "rb") as f:
            return_structure = pickle.load(f)

    return_structure["primary_key"] = pk
    return JsonResponse(return_structure, safe=False)


@login_required
def ct_summary_list_filter(request):
    """Obtain data for CT summary view
    """
    from remapp.interface.mod_filters import ct_acq_filter
    from remapp.forms import CTChartOptionsForm, itemsPerPageForm
    from openremproject import settings

    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = ct_acq_filter(request.GET, pid=pid)

    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile

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
            user_profile.plotCTAcquisitionCTDIOverTime = chart_options_form.cleaned_data[
                "plotCTAcquisitionCTDIOverTime"
            ]
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
                "plotAverageChoice": average_choices
            }
            chart_options_form = CTChartOptionsForm(form_data)

    # Obtain the number of items per page from the request
    items_per_page_form = itemsPerPageForm(request.GET)
    # check whether the form data is valid
    if items_per_page_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.itemsPerPage = items_per_page_form.cleaned_data["itemsPerPage"]
            user_profile.save()

        # If submit was not clicked then use the settings already stored in the user's profile
        else:
            form_data = {"itemsPerPage": user_profile.itemsPerPage}
            items_per_page_form = itemsPerPageForm(form_data)

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    paginator = Paginator(f.qs, user_profile.itemsPerPage)
    page = request.GET.get("page")
    try:
        study_list = paginator.page(page)
    except PageNotAnInteger:
        study_list = paginator.page(1)
    except EmptyPage:
        study_list = paginator.page(paginator.num_pages)

    return_structure = {
        "filter": f,
        "study_list": study_list,
        "admin": admin,
        "chartOptionsForm": chart_options_form,
        "itemsPerPageForm": items_per_page_form,
    }

    if user_profile.plotCharts:
        return_structure["required_charts"] = generate_required_ct_charts_list(user_profile)

    return render(request, "remapp/ctfiltered.html", return_structure,)


def generate_required_ct_charts_list(profile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    from django.utils.safestring import mark_safe

    required_charts = []

    if profile.plotCTAcquisitionMeanDLP:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DLP for each acquisition protocol",
                                    "var_name": "acquisitionMeanDLP"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DLP for each acquisition protocol",
                                    "var_name": "acquisitionMedianDLP"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of DLP for each acquisition protocol",
                                    "var_name": "acquisitionBoxplotDLP"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DLP for each acquisition protocol",
                                    "var_name": "acquisitionHistogramDLP"})

    if profile.plotCTAcquisitionMeanCTDI:
        if profile.plotMean:
            required_charts.append({"title": mark_safe("Chart of mean CTDI<sub>vol</sub> for each acquisition protocol"),
                                    "var_name": "acquisitionMeanCTDI"})
        if profile.plotMedian:
            required_charts.append({"title": mark_safe("Chart of median CTDI<sub>vol</sub> for each acquisition protocol"),
                                    "var_name": "acquisitionMedianCTDI"})
        if profile.plotBoxplots:
            required_charts.append({"title": mark_safe("Boxplot of CTDI<sub>vol</sub> for each acquisition protocol"),
                                    "var_name": "acquisitionBoxplotCTDI"})
        if profile.plotHistograms:
            required_charts.append({"title": mark_safe("Histogram of CTDI<sub>vol</sub> for each acquisition protocol"),
                                    "var_name": "acquisitionHistogramCTDI"})

    if profile.plotCTAcquisitionFreq:
        required_charts.append({"title": "Chart of acquisition protocol frequency",
                                "var_name": "acquisitionFrequency"})

    if profile.plotCTAcquisitionCTDIvsMass:
        required_charts.append({"title": mark_safe("Chart of CTDI<sub>vol</sub> vs patient mass for each acquisition protocol"),
                                "var_name": "acquisitionScatterCTDIvsMass"})

    if profile.plotCTAcquisitionDLPvsMass:
        required_charts.append({"title": "Chart of DLP vs patient mass for each acquisition protocol",
                                "var_name": "acquisitionScatterDLPvsMass"})

    if profile.plotCTAcquisitionCTDIOverTime:
        if profile.plotMean:
            required_charts.append({"title": mark_safe("Chart of mean CTDI<sub>vol</sub> per acquisition protocol over time (" + profile.plotCTOverTimePeriod + ")"),
                                    "var_name": "acquisitionMeanCTDIOverTime"})
        if profile.plotMedian:
            required_charts.append({"title": mark_safe("Chart of median CTDI<sub>vol</sub> per acquisition protocol over time (" + profile.plotCTOverTimePeriod + ")"),
                                    "var_name": "acquisitionMedianCTDIOverTime"})

    if profile.plotCTAcquisitionDLPOverTime:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DLP per acquisition protocol over time (" + profile.plotCTOverTimePeriod + ")",
                                    "var_name": "acquisitionMeanDLPOverTime"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DLP per acquisition protocol over time (" + profile.plotCTOverTimePeriod + ")",
                                    "var_name": "acquisitionMedianDLPOverTime"})

    if profile.plotCTStudyMeanDLP:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DLP for each study description",
                                    "var_name": "studyMeanDLP"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DLP for each study description",
                                    "var_name": "studyMedianDLP"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of DLP for each study description",
                                    "var_name": "studyBoxplotDLP"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DLP for each study description",
                                    "var_name": "studyHistogramDLP"})

    if profile.plotCTStudyMeanCTDI:
        if profile.plotMean:
            required_charts.append({"title": mark_safe("Chart of mean CTDI<sub>vol</sub> for each study description"),
                                    "var_name": "studyMeanCTDI"})
        if profile.plotMedian:
            required_charts.append({"title": mark_safe("Chart of median CTDI<sub>vol</sub> for each study description"),
                                    "var_name": "studyMedianCTDI"})
        if profile.plotBoxplots:
            required_charts.append({"title": mark_safe("Boxplot of CTDI<sub>vol</sub> for each study description"),
                                    "var_name": "studyBoxplotCTDI"})
        if profile.plotHistograms:
            required_charts.append({"title": mark_safe("Histogram of CTDI<sub>vol</sub> for each study description"),
                                    "var_name": "studyHistogramCTDI"})

    if profile.plotCTStudyFreq:
        required_charts.append({"title": "Chart of study description frequency",
                                "var_name": "studyFrequency"})

    if profile.plotCTStudyNumEvents:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean number of events for each study description",
                                    "var_name": "studyMeanNumEvents"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median number of events for each study description",
                                    "var_name": "studyMedianNumEvents"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of number of events for each study description",
                                    "var_name": "studyBoxplotNumEvents"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of number of events for each study description",
                                    "var_name": "studyHistogramNumEvents"})

    if profile.plotCTRequestMeanDLP:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean DLP for each requested procedure",
                                    "var_name": "requestMeanDLP"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median DLP for each requested procedure",
                                    "var_name": "requestMedianDLP"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of DLP for each requested procedure",
                                    "var_name": "requestBoxplotDLP"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of DLP for each requested procedure",
                                    "var_name": "requestHistogramDLP"})

    if profile.plotCTRequestFreq:
        required_charts.append({"title": "Chart of requested procedure frequency",
                                "var_name": "requestFrequency"})

    if profile.plotCTRequestNumEvents:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean number of events for each requested procedure",
                                    "var_name": "requestMeanNumEvents"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median number of events for each requested procedure",
                                    "var_name": "requestMedianNumEvents"})
        if profile.plotBoxplots:
            required_charts.append({"title": "Boxplot of number of events for each requested procedure",
                                    "var_name": "requestBoxplotNumEvents"})
        if profile.plotHistograms:
            required_charts.append({"title": "Histogram of number of events for each requested procedure",
                                    "var_name": "requestHistogramNumEvents"})

    if profile.plotCTRequestDLPOverTime:
        if profile.plotMean:
            required_charts.append({
                                       "title": "Chart of mean DLP per requested procedure over time (" + profile.plotCTOverTimePeriod + ")",
                                       "var_name": "requestMeanDLPOverTime"})
        if profile.plotMedian:
            required_charts.append({
                                       "title": "Chart of median DLP per requested procedure over time (" + profile.plotCTOverTimePeriod + ")",
                                       "var_name": "requestMedianDLPOverTime"})

    if profile.plotCTStudyPerDayAndHour:
        required_charts.append({"title": "Chart of study description workload",
                                "var_name": "studyWorkload"})

    if profile.plotCTStudyMeanDLPOverTime:
        if profile.plotMean:
            required_charts.append({
                                       "title": "Chart of mean DLP per study description over time (" + profile.plotCTOverTimePeriod + ")",
                                       "var_name": "studyMeanDLPOverTime"})
        if profile.plotMedian:
            required_charts.append({
                                       "title": "Chart of median DLP per study description over time (" + profile.plotCTOverTimePeriod + ")",
                                       "var_name": "studyMedianDLPOverTime"})

    return required_charts


@login_required
def ct_summary_chart_data(request):
    """Obtain data for CT charts Ajax call
    """
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

    return_structure = ct_plot_calculations(
        f,
        user_profile
    )

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def ct_plot_calculations(
    f,
    user_profile
):
    """CT chart data calculations
    """
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
        construct_over_time_charts
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
        plot_timeunit_period = keys[[tp.lower() for tp in values].index(user_profile.plotCTOverTimePeriod)]

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
        if user_profile.plotCTAcquisitionMeanDLP or user_profile.plotCTAcquisitionDLPvsMass or user_profile.plotCTAcquisitionDLPOverTime:
            value_fields.append("ctradiationdose__ctirradiationeventdata__dlp")
        if user_profile.plotCTAcquisitionMeanCTDI or user_profile.plotCTAcquisitionCTDIvsMass or user_profile.plotCTAcquisitionCTDIOverTime:
            value_fields.append("ctradiationdose__ctirradiationeventdata__mean_ctdivol")
        if user_profile.plotCTAcquisitionCTDIvsMass or user_profile.plotCTAcquisitionDLPvsMass:
            value_fields.append("patientstudymoduleattr__patient_weight")

        date_fields = []
        if user_profile.plotCTAcquisitionCTDIOverTime or user_profile.plotCTAcquisitionDLPOverTime:
            date_fields.append("study_date")

        system_field = None
        if user_profile.plotSeriesPerSystem:
            system_field = "generalequipmentmoduleattr__unique_equipment_name_id__display_name"

        df = create_dataframe(
            f.qs,
            data_point_name_fields=name_fields,
            data_point_value_fields=value_fields,
            data_point_date_fields=date_fields,
            system_name_field=system_field,
            data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
            uid="ctradiationdose__ctirradiationeventdata__pk"
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
                [user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__dlp",
                    stats=average_choices + ["count"]
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
                        average_choice="mean"
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
                        average_choice="median"
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
                    sorted_category_list=sorted_acquisition_dlp_categories
                )

            if user_profile.plotHistograms:
                category_names_col = "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_dlp_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_dlp_categories.values())[0]

                return_structure["acquisitionHistogramDLPData"] = plotly_histogram_barchart(
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
                [user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    stats=average_choices + ["count"]
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
                        average_choice="mean"
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
                        average_choice="median"
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
                    sorted_category_list=sorted_acquisition_ctdi_categories
                )

            if user_profile.plotHistograms:
                category_names_col = "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
                group_by_col = "x_ray_system_name"
                legend_title = "Acquisition protocol"
                facet_names = list(df[group_by_col].unique())
                category_names = list(sorted_acquisition_ctdi_categories.values())[0]

                if user_profile.plotGroupingChoice == "series":
                    category_names_col = "x_ray_system_name"
                    group_by_col = "ctradiationdose__ctirradiationeventdata__acquisition_protocol"
                    legend_title = "System"
                    category_names = facet_names
                    facet_names = list(sorted_acquisition_ctdi_categories.values())[0]

                return_structure["acquisitionHistogramCTDIData"] = plotly_histogram_barchart(
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
                sorting_choice=[user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice],
                legend_title="Acquisition protocol",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM CT acquisition protocol frequency",
                sorted_categories=sorted_categories
            )

        if user_profile.plotCTAcquisitionCTDIvsMass:
            return_structure["acquisitionScatterCTDIvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                x_axis_title="Patient mass (kg)",
                y_axis_title="CTDI<sub>vol</sub> (mGy)",
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Acquisition protocol",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol CTDI vs patient mass"
            )

        if user_profile.plotCTAcquisitionDLPvsMass:
            return_structure["acquisitionScatterDLPvsMass"] = construct_scatter_chart(
                df=df,
                df_name_col="ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                df_x_col="patientstudymoduleattr__patient_weight",
                df_y_col="ctradiationdose__ctirradiationeventdata__dlp",
                x_axis_title = "Patient mass (kg)",
                y_axis_title = "DLP (mGy.cm)",
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice],
                grouping_choice=user_profile.plotGroupingChoice,
                legend_title="Acquisition protocol",
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol DLP vs patient mass"
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
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol CTDI over time"
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
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT acquisition protocol DLP over time"
            )

            if user_profile.plotMean:
                return_structure["acquisitionMeanDLPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["acquisitionMedianDLPOverTime"] = result["median"]

    #######################################################################
    # Prepare study- and request-level Pandas DataFrame to use for charts
    if (user_profile.plotCTRequestFreq
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
            system_field = "generalequipmentmoduleattr__unique_equipment_name_id__display_name"

        df = create_dataframe(
            f.qs,
            data_point_name_fields=name_fields,
            data_point_value_fields=value_fields,
            data_point_date_fields=date_fields,
            data_point_time_fields=time_fields,
            system_name_field=system_field,
            data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
            uid="pk"
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
                [user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "study_description",
                    "total_dlp",
                    stats=average_choices + ["count"]
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
                        average_choice="mean"
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
                        average_choice="median"
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
                    sorted_category_list=sorted_study_dlp_categories
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
                [user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "study_description",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    stats=average_choices + ["count"]
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
                        average_choice="mean"
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
                        average_choice="median"
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
                    sorted_category_list=sorted_study_ctdi_categories
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
                [user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "study_description",
                    "number_of_events",
                    stats=average_choices + ["count"]
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
                        average_choice="mean"
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
                        average_choice="median"
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
                    sorted_category_list=sorted_study_events_categories
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

                return_structure["studyHistogramNumEventsData"] = plotly_histogram_barchart(
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
                sorting_choice=[user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice],
                legend_title="Study description",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM CT study description frequency",
                sorted_categories=sorted_categories
            )

        sorted_request_dlp_categories = None
        if user_profile.plotCTRequestMeanDLP:
            sorted_request_dlp_categories = create_sorted_category_list(
                df,
                "requested_procedure_code_meaning",
                "total_dlp",
                [user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice]
            )

            if user_profile.plotMean or user_profile.plotMedian:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dlp",
                    stats=average_choices + ["count"]
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
                        average_choice="mean"
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
                        average_choice="median"
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
                    sorted_category_list=sorted_request_dlp_categories
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
                [user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice]
            )

            if user_profile.plotMean:
                df_aggregated = create_dataframe_aggregates(
                    df,
                    "requested_procedure_code_meaning",
                    "number_of_events",
                    stats=average_choices + ["count"]
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
                        average_choice="mean"
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
                        average_choice="median"
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
                    sorted_category_list=sorted_request_events_categories
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

                return_structure["requestHistogramNumEventsData"] = plotly_histogram_barchart(
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
                sorting_choice=[user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice],
                legend_title="Requested procedure",
                df_x_axis_col="x_ray_system_name",
                x_axis_title="System",
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                file_name="OpenREM CT requested procedure frequency",
                sorted_categories=sorted_categories
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
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT study description DLP over time"
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
                sorting=[user_profile.plotInitialSortingDirection, user_profile.plotCTInitialSortingChoice],
                time_period=plot_timeunit_period,
                average_choices=average_choices,
                grouping_choice=user_profile.plotGroupingChoice,
                colour_map=user_profile.plotColourMapChoice,
                facet_col_wrap=user_profile.plotFacetColWrapVal,
                file_name="OpenREM CT requested procedure DLP over time"
            )

            if user_profile.plotMean:
                return_structure["requestMeanDLPOverTime"] = result["mean"]
            if user_profile.plotMedian:
                return_structure["requestMedianDLPOverTime"] = result["median"]

        if user_profile.plotCTStudyPerDayAndHour:
            df_time_series_per_weekday = create_dataframe_weekdays(
                df,
                "study_description",
                df_date_col="study_date"
            )

            return_structure["studyWorkloadData"] = plotly_barchart_weekdays(
                df_time_series_per_weekday,
                "weekday",
                "study_description",
                name_axis_title="Weekday",
                value_axis_title="Frequency",
                colourmap=user_profile.plotColourMapChoice,
                filename="OpenREM CT study description workload",
                facet_col_wrap=user_profile.plotFacetColWrapVal
            )
        #######################################################################

    return return_structure


@login_required
def ct_detail_view(request, pk=None):
    """Detail view for a CT study
    """
    from django.contrib import messages
    from remapp.models import GeneralStudyModuleAttr

    try:
        study = GeneralStudyModuleAttr.objects.get(pk=pk)
    except ObjectDoesNotExist:
        messages.error(request, "That study was not found")
        return redirect(reverse_lazy("ct_summary_list_filter"))

    events_all = (
        study.ctradiationdose_set.get()
        .ctirradiationeventdata_set.select_related(
            "ct_acquisition_type", "ctdiw_phantom_type"
        )
        .order_by("pk")
    )

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    return render(
        request,
        "remapp/ctdetail.html",
        {"generalstudymoduleattr": study, "admin": admin, "events_all": events_all},
    )


@login_required
def mg_summary_list_filter(request):
    """Mammography data for summary view
    """
    from remapp.interface.mod_filters import MGSummaryListFilter, MGFilterPlusPid
    from openremproject import settings
    from remapp.forms import MGChartOptionsForm, itemsPerPageForm

    filter_data = request.GET.copy()
    if "page" in filter_data:
        del filter_data["page"]

    if request.user.groups.filter(name="pidgroup"):
        f = MGFilterPlusPid(
            filter_data,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="MG")
            .order_by("-study_date", "-study_time")
            .distinct(),
        )
    else:
        f = MGSummaryListFilter(
            filter_data,
            queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="MG")
            .order_by("-study_date", "-study_time")
            .distinct(),
        )

    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile

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
                "plotAverageChoice": average_choices
            }
            chart_options_form = MGChartOptionsForm(form_data)

    # Obtain the number of items per page from the request
    items_per_page_form = itemsPerPageForm(request.GET)
    # check whether the form data is valid
    if items_per_page_form.is_valid():
        # Use the form data if the user clicked on the submit button
        if "submit" in request.GET:
            # process the data in form.cleaned_data as required
            user_profile.itemsPerPage = items_per_page_form.cleaned_data["itemsPerPage"]
            user_profile.save()

        # If submit was not clicked then use the settings already stored in the user's profile
        else:
            form_data = {"itemsPerPage": user_profile.itemsPerPage}
            items_per_page_form = itemsPerPageForm(form_data)

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    paginator = Paginator(f.qs, user_profile.itemsPerPage)
    page = request.GET.get("page")
    try:
        study_list = paginator.page(page)
    except PageNotAnInteger:
        study_list = paginator.page(1)
    except EmptyPage:
        study_list = paginator.page(paginator.num_pages)

    return_structure = {
        "filter": f,
        "study_list": study_list,
        "admin": admin,
        "chartOptionsForm": chart_options_form,
        "itemsPerPageForm": items_per_page_form,
    }

    if user_profile.plotCharts:
        return_structure["required_charts"] = generate_required_mg_charts_list(user_profile)

    return render(request, "remapp/mgfiltered.html", return_structure,)


def generate_required_mg_charts_list(profile):
    """Obtain a list of dictionaries containing the title string and base
    variable name for each required chart"""
    required_charts = []

    if profile.plotMGAGDvsThickness:
        required_charts.append({"title": "Chart of AGD vs compressed breast thickness for each acquisition protocol",
                                "var_name": "acquisitionScatterAGDvsThick"})

    if profile.plotMGaverageAGDvsThickness:
        if profile.plotMean:
            required_charts.append({"title": "Chart of mean AGD vs compressed breast thickness for each acquisition protocol",
                                    "var_name": "acquisitionMeanAGDvsThick"})
        if profile.plotMedian:
            required_charts.append({"title": "Chart of median AGD vs compressed breast thickness for each acquisition protocol",
                                    "var_name": "acquisitionMedianAGDvsThick"})

    if profile.plotMGkVpvsThickness:
        required_charts.append({"title": "Chart of kVp vs compressed breast thickness for each acquisition protocol",
                                "var_name": "acquisitionScatterkVpvsThick"})

    if profile.plotMGmAsvsThickness:
        required_charts.append({"title": "Chart of mAs vs compressed breast thickness for each acquisition protocol",
                                "var_name": "acquisitionScattermAsvsThick"})

    if profile.plotMGStudyPerDayAndHour:
        required_charts.append({"title": "Chart of study description workload",
                                "var_name": "studyWorkload"})

    return required_charts


@login_required
def mg_summary_chart_data(request):
    """Obtain data for mammography chart data Ajax view
    """
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

    return_structure = mg_plot_calculations(
        f,
        user_profile
    )

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def mg_plot_calculations(
    f,
    user_profile
):
    """Calculations for mammography charts
    """
    from .interface.chart_functions import (
        create_dataframe,
        create_dataframe_weekdays,
        plotly_barchart_weekdays,
        plotly_binned_statistic_barchart,
        construct_scatter_chart,
        plotly_set_default_theme,
        create_sorted_category_list
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
        name_fields.append("projectionxrayradiationdose__irradeventxraydata__acquisition_protocol")

    value_fields = []
    if user_profile.plotMGAGDvsThickness or user_profile.plotMGaverageAGDvsThickness:
        value_fields.append("projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose")
    if user_profile.plotMGkVpvsThickness:
        value_fields.append("projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp")
    if user_profile.plotMGmAsvsThickness:
        value_fields.append("projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure")
    if (
        user_profile.plotMGAGDvsThickness
        or user_profile.plotMGkVpvsThickness
        or user_profile.plotMGmAsvsThickness
        or user_profile.plotMGaverageAGDvsThickness
    ):
        value_fields.append("projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness")

    date_fields = []
    time_fields = []
    if user_profile.plotMGStudyPerDayAndHour:
        date_fields.append("study_date")
        time_fields.append("study_time")

    system_field = None
    if user_profile.plotSeriesPerSystem:
        system_field = "generalequipmentmoduleattr__unique_equipment_name_id__display_name"

    df = create_dataframe(
        f.qs,
        data_point_name_fields=name_fields,
        data_point_value_fields=value_fields,
        data_point_date_fields=date_fields,
        data_point_time_fields=time_fields,
        system_name_field=system_field,
        data_point_name_lowercase=user_profile.plotCaseInsensitiveCategories,
        uid="projectionxrayradiationdose__irradeventxraydata__pk"
    )

    if user_profile.plotMGStudyPerDayAndHour:
        df_time_series_per_weekday = create_dataframe_weekdays(
            df,
            "study_description",
            df_date_col="study_date"
        )

        return_structure["studyWorkloadData"] = plotly_barchart_weekdays(
            df_time_series_per_weekday,
            "weekday",
            "study_description",
            name_axis_title="Weekday",
            value_axis_title="Frequency",
            colourmap=user_profile.plotColourMapChoice,
            filename="OpenREM CT study description workload",
            facet_col_wrap=user_profile.plotFacetColWrapVal
        )

    if user_profile.plotMGaverageAGDvsThickness:
        sorted_acquisition_agd_categories = create_sorted_category_list(
            df,
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
            [user_profile.plotInitialSortingDirection, user_profile.plotMGInitialSortingChoice]
        )

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
                stat_name="mean"
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
                stat_name="median"
            )

    if user_profile.plotMGAGDvsThickness:
        return_structure["AGDvsThickness"] = construct_scatter_chart(
            df=df,
            df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            df_x_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
            df_y_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
            x_axis_title="Compressed breast thickness (mm)",
            y_axis_title="AGD (mGy)",
            sorting=[user_profile.plotInitialSortingDirection, user_profile.plotMGInitialSortingChoice],
            grouping_choice=user_profile.plotGroupingChoice,
            legend_title="Acquisition protocol",
            colour_map=user_profile.plotColourMapChoice,
            facet_col_wrap=user_profile.plotFacetColWrapVal,
            file_name="OpenREM CT acquisition protocol AGD vs thickness"
        )

    if user_profile.plotMGkVpvsThickness:
        return_structure["kVpvsThickness"] = construct_scatter_chart(
            df=df,
            df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            df_x_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
            df_y_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
            x_axis_title="Compressed breast thickness (mm)",
            y_axis_title="kVp",
            sorting=[user_profile.plotInitialSortingDirection, user_profile.plotMGInitialSortingChoice],
            grouping_choice=user_profile.plotGroupingChoice,
            legend_title="Acquisition protocol",
            colour_map=user_profile.plotColourMapChoice,
            facet_col_wrap=user_profile.plotFacetColWrapVal,
            file_name="OpenREM CT acquisition protocol kVp vs thickness"
        )

    if user_profile.plotMGmAsvsThickness:
        return_structure["mAsvsThickness"] = construct_scatter_chart(
            df=df,
            df_name_col="projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            df_x_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
            df_y_col="projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
            x_axis_title="Compressed breast thickness (mm)",
            y_axis_title="mAs",
            sorting=[user_profile.plotInitialSortingDirection, user_profile.plotMGInitialSortingChoice],
            grouping_choice=user_profile.plotGroupingChoice,
            legend_title="Acquisition protocol",
            colour_map=user_profile.plotColourMapChoice,
            facet_col_wrap=user_profile.plotFacetColWrapVal,
            file_name="OpenREM CT acquisition protocol mAs vs thickness"
        )

    return return_structure


@login_required
def mg_detail_view(request, pk=None):
    """Detail view for a CT study
    """
    from django.contrib import messages
    from remapp.models import GeneralStudyModuleAttr

    try:
        study = GeneralStudyModuleAttr.objects.get(pk=pk)
    except:
        messages.error(request, "That study was not found")
        return redirect(reverse_lazy("mg_summary_list_filter"))

    admin = {
        "openremversion": remapp.__version__,
        "docsversion": remapp.__docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    projection_xray_dose_set = study.projectionxrayradiationdose_set.get()
    accum_mammo_set = (
        projection_xray_dose_set.accumxraydose_set.get()
        .accummammographyxraydose_set.select_related("laterality")
        .all()
    )
    events_all = projection_xray_dose_set.irradeventxraydata_set.select_related(
        "laterality", "image_view"
    ).all()

    return render(
        request,
        "remapp/mgdetail.html",
        {
            "generalstudymoduleattr": study,
            "admin": admin,
            "projection_xray_dose_set": projection_xray_dose_set,
            "accum_mammo_set": accum_mammo_set,
            "events_all": events_all,
        },
    )


def openrem_home(request):
    from remapp.models import (
        PatientIDSettings,
        DicomDeleteSettings,
        AdminTaskQuestions,
        HomePageAdminSettings,
        UpgradeStatus,
    )
    from django.db.models import Q  # For the Q "OR" query used for DX and CR
    from collections import OrderedDict

    try:
        HomePageAdminSettings.objects.get()
    except ObjectDoesNotExist:
        HomePageAdminSettings.objects.create()

    test_dicom_store_settings = DicomDeleteSettings.objects.all()
    if not test_dicom_store_settings:
        DicomDeleteSettings.objects.create()

    if not Group.objects.filter(name="viewgroup"):
        vg = Group(name="viewgroup")
        vg.save()
    if not Group.objects.filter(name="exportgroup"):
        eg = Group(name="exportgroup")
        eg.save()
    if not Group.objects.filter(name="admingroup"):
        ag = Group(name="admingroup")
        ag.save()
    if not Group.objects.filter(name="pidgroup"):
        pg = Group(name="pidgroup")
        pg.save()
    if not Group.objects.filter(name="importsizegroup"):
        sg = Group(name="importsizegroup")
        sg.save()
    if not Group.objects.filter(name="importqrgroup"):
        qg = Group(name="importqrgroup")
        qg.save()

    id_settings = PatientIDSettings.objects.all()
    if not id_settings:
        PatientIDSettings.objects.create()

    users_in_groups = {"any": False, "admin": False}
    for g in Group.objects.all():
        if Group.objects.get(name=g).user_set.all():
            users_in_groups["any"] = True
            if g.name == "admingroup":
                users_in_groups["admin"] = True

    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except (ObjectDoesNotExist, AttributeError):
        # Attribute error needed for AnonymousUser, who doesn't have a userprofile attribute
        if request.user.is_authenticated:
            # Create a default userprofile for the user if one doesn't exist
            create_user_profile(
                sender=request.user, instance=request.user, created=True
            )
            user_profile = request.user.userprofile

    allstudies = GeneralStudyModuleAttr.objects.all()
    modalities = OrderedDict()
    modalities["CT"] = {
        "name": "CT",
        "count": allstudies.filter(modality_type__exact="CT").count(),
    }
    modalities["RF"] = {
        "name": "Fluoroscopy",
        "count": allstudies.filter(modality_type__exact="RF").count(),
    }
    modalities["MG"] = {
        "name": "Mammography",
        "count": allstudies.filter(modality_type__exact="MG").count(),
    }
    modalities["DX"] = {
        "name": "Radiography",
        "count": allstudies.filter(
            Q(modality_type__exact="DX") | Q(modality_type__exact="CR")
        ).count(),
    }

    mods_to_delete = []
    for modality in modalities:
        if not modalities[modality]["count"]:
            mods_to_delete += [
                modality,
            ]
            if request.user.is_authenticated:
                setattr(user_profile, "display{0}".format(modality), False)
        else:
            if request.user.is_authenticated:
                setattr(user_profile, "display{0}".format(modality), True)
    if request.user.is_authenticated:
        user_profile.save()

    for modality in mods_to_delete:
        del modalities[modality]

    homedata = {
        "total": allstudies.count(),
    }

    # Determine whether to calculate workload settings
    display_workload_stats = HomePageAdminSettings.objects.values_list(
        "enable_workload_stats", flat=True
    )[0]
    home_config = {"display_workload_stats": display_workload_stats}
    if display_workload_stats:
        if request.user.is_authenticated:
            home_config["day_delta_a"] = user_profile.summaryWorkloadDaysA
            home_config["day_delta_b"] = user_profile.summaryWorkloadDaysB
        else:
            home_config["day_delta_a"] = 7
            home_config["day_delta_b"] = 28

    admin = dict(openremversion=remapp.__version__, docsversion=remapp.__docs_version__)

    for group in request.user.groups.all():
        admin[group.name] = True

    admin_questions = {}
    admin_questions_true = False
    if request.user.groups.filter(name="admingroup"):
        not_patient_indicator_question = (
            AdminTaskQuestions.get_solo().ask_revert_to_074_question
        )
        admin_questions[
            "not_patient_indicator_question"
        ] = not_patient_indicator_question
        # if any(value for value in admin_questions.itervalues()):
        #     admin_questions_true = True  # Don't know why this doesn't work
        if not_patient_indicator_question:
            admin_questions_true = True  # Doing this instead

    upgrade_status = UpgradeStatus.get_solo()
    migration_complete = upgrade_status.from_0_9_1_summary_fields
    if not migration_complete and homedata["total"] == 0:
        upgrade_status.from_0_9_1_summary_fields = True
        upgrade_status.save()
        migration_complete = True

    # from remapp.tools.send_high_dose_alert_emails import send_rf_high_dose_alert_email
    # send_rf_high_dose_alert_email(417637)
    # send_rf_high_dose_alert_email(417973)
    # # Send a test e-mail
    # from django.core.mail import send_mail
    # from openremproject import settings
    # from remapp.models import HighDoseMetricAlertSettings
    # from django.contrib.auth.models import User
    #
    # try:
    #     HighDoseMetricAlertSettings.objects.get()
    # except ObjectDoesNotExist:
    #     HighDoseMetricAlertSettings.objects.create()
    #
    # send_alert_emails = HighDoseMetricAlertSettings.objects.values_list('send_high_dose_metric_alert_emails', flat=True)[0]
    # if send_alert_emails:
    #     recipients = User.objects.filter(highdosemetricalertrecipients__receive_high_dose_metric_alerts__exact=True).values_list('email', flat=True)
    #     send_mail('OpenREM high dose alert test',
    #               'This is a test for high dose alert e-mails from OpenREM',
    #               settings.EMAIL_DOSE_ALERT_SENDER,
    #               recipients,
    #               fail_silently=False)
    # # End of sending a test e-mail

    return render(
        request,
        "remapp/home.html",
        {
            "homedata": homedata,
            "admin": admin,
            "users_in_groups": users_in_groups,
            "admin_questions": admin_questions,
            "admin_questions_true": admin_questions_true,
            "modalities": modalities,
            "home_config": home_config,
            "migration_complete": migration_complete,
        },
    )


@csrf_exempt
def update_modality_totals(request):
    """AJAX function to update study numbers automatically

    :param request: request object
    :return: dictionary of totals
    """
    from django.db.models import Q

    if request.is_ajax():
        allstudies = GeneralStudyModuleAttr.objects.all()
        resp = {
            "total": allstudies.count(),
            "total_mg": allstudies.filter(modality_type__exact="MG").count(),
            "total_ct": allstudies.filter(modality_type__exact="CT").count(),
            "total_rf": allstudies.filter(modality_type__contains="RF").count(),
            "total_dx": allstudies.filter(
                Q(modality_type__exact="DX") | Q(modality_type__exact="CR")
            ).count(),
        }

        return HttpResponse(json.dumps(resp), content_type="application/json")


@csrf_exempt
def update_latest_studies(request):
    """AJAX function to calculate the latest studies for each display name for a particular modality.

    :param request: Request object
    :return: HTML table of modalities
    """
    from django.db.models import Q, Min
    from datetime import datetime, timedelta
    from collections import OrderedDict
    from remapp.models import HomePageAdminSettings

    if request.is_ajax():
        data = request.POST
        modality = data.get("modality")
        if modality == "DX":
            studies = GeneralStudyModuleAttr.objects.filter(
                Q(modality_type__exact="DX") | Q(modality_type__exact="CR")
            ).all()
        else:
            studies = GeneralStudyModuleAttr.objects.filter(
                modality_type__exact=modality
            ).all()

        display_names = (
            studies.values_list(
                "generalequipmentmoduleattr__unique_equipment_name__display_name"
            )
            .distinct()
            .annotate(
                pk_value=Min("generalequipmentmoduleattr__unique_equipment_name__pk")
            )
        )

        modalitydata = {}

        if request.user.is_authenticated:
            day_delta_a = request.user.userprofile.summaryWorkloadDaysA
            day_delta_b = request.user.userprofile.summaryWorkloadDaysB
        else:
            day_delta_a = 7
            day_delta_b = 28

        for display_name, pk in display_names:
            display_name_studies = studies.filter(
                generalequipmentmoduleattr__unique_equipment_name__display_name__exact=display_name
            )
            latestdate = display_name_studies.latest("study_date").study_date
            latestuid = display_name_studies.filter(
                study_date__exact=latestdate
            ).latest("study_time")
            latestdatetime = datetime.combine(
                latestuid.study_date, latestuid.study_time
            )
            deltaseconds = int((datetime.now() - latestdatetime).total_seconds())

            modalitydata[display_name] = {
                "total": display_name_studies.count(),
                "latest": latestdatetime,
                "deltaseconds": deltaseconds,
                "displayname": display_name,
                "displayname_pk": modality.lower() + str(pk),
            }
        ordereddata = OrderedDict(
            sorted(
                list(modalitydata.items()), key=lambda t: t[1]["latest"], reverse=True
            )
        )

        admin = {}
        for group in request.user.groups.all():
            admin[group.name] = True

        template = "remapp/home-list-modalities.html"
        data = ordereddata

        display_workload_stats = HomePageAdminSettings.objects.values_list(
            "enable_workload_stats", flat=True
        )[0]
        today = datetime.now()
        date_a = today - timedelta(days=day_delta_a)
        date_b = today - timedelta(days=day_delta_b)
        home_config = {
            "display_workload_stats": display_workload_stats,
            "day_delta_a": day_delta_a,
            "day_delta_b": day_delta_b,
            "date_a": datetime.strftime(date_a, "%Y-%m-%d"),
            "date_b": datetime.strftime(date_b, "%Y-%m-%d"),
        }

        return render(
            request,
            template,
            {
                "data": data,
                "modality": modality.lower(),
                "home_config": home_config,
                "admin": admin,
            },
        )


@csrf_exempt
def update_study_workload(request):
    """AJAX function to calculate the number of studies in two user-defined time periods for a particular modality.

    :param request: Request object
    :return: HTML table of modalities
    """
    from django.db.models import Q, Min
    from datetime import datetime, timedelta
    from collections import OrderedDict

    if request.is_ajax():
        data = request.POST
        modality = data.get("modality")
        if modality == "DX":
            studies = GeneralStudyModuleAttr.objects.filter(
                Q(modality_type__exact="DX") | Q(modality_type__exact="CR")
            ).all()
        else:
            studies = GeneralStudyModuleAttr.objects.filter(
                modality_type__exact=modality
            ).all()

        display_names = (
            studies.values_list(
                "generalequipmentmoduleattr__unique_equipment_name__display_name"
            )
            .distinct()
            .annotate(
                pk_value=Min("generalequipmentmoduleattr__unique_equipment_name__pk")
            )
        )

        modalitydata = {}

        if request.user.is_authenticated:
            day_delta_a = request.user.userprofile.summaryWorkloadDaysA
            day_delta_b = request.user.userprofile.summaryWorkloadDaysB
        else:
            day_delta_a = 7
            day_delta_b = 28

        today = datetime.now()
        date_a = today - timedelta(days=day_delta_a)
        date_b = today - timedelta(days=day_delta_b)

        for display_name, pk in display_names:
            display_name_studies = studies.filter(
                generalequipmentmoduleattr__unique_equipment_name__display_name__exact=display_name
            )

            try:
                displayname = display_name.encode("utf-8")
            except AttributeError:
                displayname = "Unexpected display name non-ASCII issue"

            modalitydata[display_name] = {
                "studies_in_past_days_a": display_name_studies.filter(
                    study_date__range=[date_a, today]
                ).count(),
                "studies_in_past_days_b": display_name_studies.filter(
                    study_date__range=[date_b, today]
                ).count(),
                "displayname": displayname,
                "displayname_pk": modality.lower() + str(pk),
            }
        data = OrderedDict(
            sorted(
                list(modalitydata.items()),
                key=lambda t: t[1]["displayname_pk"],
                reverse=True,
            )
        )

        template = "remapp/home-modality-workload.html"

        return render(request, template, {"data": data, "modality": modality.lower()})
