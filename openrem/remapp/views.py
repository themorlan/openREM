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

    if (
        user_profile.median_available
        and "postgresql" in settings.DATABASES["default"]["ENGINE"]
    ):
        median_available = True
    elif "postgresql" in settings.DATABASES["default"]["ENGINE"]:
        user_profile.median_available = True
        user_profile.save()
        median_available = True
    else:
        user_profile.median_available = False
        user_profile.save()
        median_available = False

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
            if median_available:
                user_profile.plotAverageChoice = chart_options_form.cleaned_data[
                    "plotMeanMedianOrBoth"
                ]
            user_profile.plotSeriesPerSystem = chart_options_form.cleaned_data[
                "plotSeriesPerSystem"
            ]
            user_profile.plotHistograms = chart_options_form.cleaned_data[
                "plotHistograms"
            ]
            user_profile.save()

        # If submit was not clicked then use the settings already stored in the user's profile
        else:
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
                "plotMeanMedianOrBoth": user_profile.plotAverageChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
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

    return render(request, "remapp/dxfiltered.html", return_structure,)


@login_required
def dx_summary_chart_data(request):
    """Obtain data for Ajax chart call
    """
    from remapp.interface.mod_filters import DXSummaryListFilter
    from django.db.models import Q
    from openremproject import settings
    from django.http import JsonResponse

    f = DXSummaryListFilter(
        request.GET,
        queryset=GeneralStudyModuleAttr.objects.filter(
            Q(modality_type__exact="DX") | Q(modality_type__exact="CR")
        )
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

    if (
        user_profile.median_available
        and "postgresql" in settings.DATABASES["default"]["ENGINE"]
    ):
        median_available = True
    elif "postgresql" in settings.DATABASES["default"]["ENGINE"]:
        user_profile.median_available = True
        user_profile.save()
        median_available = True
    else:
        user_profile.median_available = False
        user_profile.save()
        median_available = False

    if settings.DEBUG:
        from datetime import datetime

        start_time = datetime.now()

    return_structure = dx_plot_calculations(
        f,
        user_profile.plotDXAcquisitionMeanDAP,
        user_profile.plotDXAcquisitionFreq,
        user_profile.plotDXStudyMeanDAP,
        user_profile.plotDXStudyFreq,
        user_profile.plotDXRequestMeanDAP,
        user_profile.plotDXRequestFreq,
        user_profile.plotDXAcquisitionMeankVpOverTime,
        user_profile.plotDXAcquisitionMeanmAsOverTime,
        user_profile.plotDXAcquisitionMeanDAPOverTime,
        user_profile.plotDXAcquisitionMeanDAPOverTimePeriod,
        user_profile.plotDXAcquisitionMeankVp,
        user_profile.plotDXAcquisitionMeanmAs,
        user_profile.plotDXStudyPerDayAndHour,
        median_available,
        user_profile.plotAverageChoice,
        user_profile.plotSeriesPerSystem,
        user_profile.plotHistogramBins,
        user_profile.plotHistograms,
        user_profile.plotCaseInsensitiveCategories,
    )

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def dx_plot_calculations(
    f,
    plot_acquisition_mean_dap,
    plot_acquisition_freq,
    plot_study_mean_dap,
    plot_study_freq,
    plot_request_mean_dap,
    plot_request_freq,
    plot_acquisition_mean_kvp_over_time,
    plot_acquisition_mean_mas_over_time,
    plot_acquisition_mean_dap_over_time,
    plot_acquisition_mean_dap_over_time_period,
    plot_acquisition_mean_kvp,
    plot_acquisition_mean_mas,
    plot_study_per_day_and_hour,
    median_available,
    plot_average_choice,
    plot_series_per_systems,
    plot_histogram_bins,
    plot_histograms,
    plot_case_insensitive_categories,
):
    """Calculations for radiographic charts
    """
    from .interface.chart_functions import (
        average_chart_inc_histogram_data,
        average_chart_over_time_data,
        workload_chart_data,
    )
    from django.utils.datastructures import MultiValueDictKeyError

    return_structure = {}

    if (
        plot_study_mean_dap
        or plot_study_freq
        or plot_study_per_day_and_hour
        or plot_request_mean_dap
        or plot_request_freq
    ):
        try:
            if f.form.data["acquisition_protocol"]:
                exp_include = f.qs.values_list("study_instance_uid")
        except MultiValueDictKeyError:
            pass
        except KeyError:
            pass

    if plot_study_mean_dap or plot_study_freq or plot_study_per_day_and_hour:
        try:
            if f.form.data["acquisition_protocol"]:
                # The user has filtered on acquisition_protocol, so need to use the slow method of querying the database
                # to avoid studies being duplicated when there is more than one of a particular acquisition type in a
                # study.
                study_events = GeneralStudyModuleAttr.objects.exclude(
                    total_dap__isnull=True
                ).filter(study_instance_uid__in=exp_include)
            else:
                # The user hasn't filtered on acquisition, so we can use the faster database querying.
                study_events = f.qs
        except MultiValueDictKeyError:
            study_events = f.qs
        except KeyError:
            study_events = f.qs

    if plot_request_mean_dap or plot_request_freq:
        try:
            if f.form.data["acquisition_protocol"]:
                # The user has filtered on acquisition_protocol, so need to use the slow method of querying the database
                # to avoid studies being duplicated when there is more than one of a particular acquisition type in a
                # study.
                request_events = GeneralStudyModuleAttr.objects.exclude(
                    total_dap__isnull=True
                ).filter(study_instance_uid__in=exp_include)
            else:
                # The user hasn't filtered on acquisition, so we can use the faster database querying.
                request_events = f.qs
        except MultiValueDictKeyError:
            request_events = f.qs
        except KeyError:
            request_events = f.qs

    if plot_acquisition_mean_dap or plot_acquisition_freq:
        result = average_chart_inc_histogram_data(
            f.qs,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
            1000000,
            plot_acquisition_mean_dap,
            plot_acquisition_freq,
            plot_series_per_systems,
            plot_average_choice,
            median_available,
            plot_histogram_bins,
            calculate_histograms=plot_histograms,
            case_insensitive_categories=plot_case_insensitive_categories,
        )

        return_structure["acquisitionSystemList"] = result["system_list"]
        return_structure["acquisition_names"] = result["series_names"]
        return_structure["acquisitionSummary"] = result["summary"]
        if plot_acquisition_mean_dap and plot_histograms:
            return_structure["acquisitionHistogramData"] = result["histogram_data"]

    if plot_request_mean_dap or plot_request_freq:
        result = average_chart_inc_histogram_data(
            request_events,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
            "requested_procedure_code_meaning",
            "total_dap",
            1000000,
            plot_request_mean_dap,
            plot_request_freq,
            plot_series_per_systems,
            plot_average_choice,
            median_available,
            plot_histogram_bins,
            calculate_histograms=plot_histograms,
            case_insensitive_categories=plot_case_insensitive_categories,
        )

        return_structure["requestSystemList"] = result["system_list"]
        return_structure["request_names"] = result["series_names"]
        return_structure["requestSummary"] = result["summary"]
        if plot_request_mean_dap and plot_histograms:
            return_structure["requestHistogramData"] = result["histogram_data"]

    if plot_study_mean_dap or plot_study_freq:
        result = average_chart_inc_histogram_data(
            study_events,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
            "study_description",
            "total_dap",
            1000000,
            plot_study_mean_dap,
            plot_study_freq,
            plot_series_per_systems,
            plot_average_choice,
            median_available,
            plot_histogram_bins,
            calculate_histograms=plot_histograms,
            case_insensitive_categories=plot_case_insensitive_categories,
        )

        return_structure["studySystemList"] = result["system_list"]
        return_structure["study_names"] = result["series_names"]
        return_structure["studySummary"] = result["summary"]
        if plot_study_mean_dap and plot_histograms:
            return_structure["studyHistogramData"] = result["histogram_data"]

    if plot_acquisition_mean_kvp:
        result = average_chart_inc_histogram_data(
            f.qs,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
            1,
            plot_acquisition_mean_kvp,
            0,
            plot_series_per_systems,
            plot_average_choice,
            median_available,
            plot_histogram_bins,
            calculate_histograms=plot_histograms,
            case_insensitive_categories=plot_case_insensitive_categories,
        )

        return_structure["acquisitionkVpSystemList"] = result["system_list"]
        return_structure["acquisition_kvp_names"] = result["series_names"]
        return_structure["acquisitionkVpSummary"] = result["summary"]
        if plot_histograms:
            return_structure["acquisitionHistogramkVpData"] = result["histogram_data"]

    if plot_acquisition_mean_mas:
        result = average_chart_inc_histogram_data(
            f.qs,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
            0.001,
            plot_acquisition_mean_mas,
            0,
            plot_series_per_systems,
            plot_average_choice,
            median_available,
            plot_histogram_bins,
            calculate_histograms=plot_histograms,
            case_insensitive_categories=plot_case_insensitive_categories,
        )

        return_structure["acquisitionmAsSystemList"] = result["system_list"]
        return_structure["acquisition_mas_names"] = result["series_names"]
        return_structure["acquisitionmAsSummary"] = result["summary"]
        if plot_histograms:
            return_structure["acquisitionHistogrammAsData"] = result["histogram_data"]

    if plot_acquisition_mean_dap_over_time:
        result = average_chart_over_time_data(
            f.qs,
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "projectionxrayradiationdose__irradeventxraydata__dose_area_product",
            "study_date",
            "projectionxrayradiationdose__irradeventxraydata__date_time_started",
            median_available,
            plot_average_choice,
            1000000,
            plot_acquisition_mean_dap_over_time_period,
            case_insensitive_categories=plot_case_insensitive_categories,
        )
        if median_available and (
            plot_average_choice == "median" or plot_average_choice == "both"
        ):
            return_structure["acquisitionMedianDAPoverTime"] = result[
                "median_over_time"
            ]
        if plot_average_choice == "mean" or plot_average_choice == "both":
            return_structure["acquisitionMeanDAPoverTime"] = result["mean_over_time"]
        if not plot_acquisition_mean_dap and not plot_acquisition_freq:
            return_structure["acquisition_names"] = result["series_names"]

    if plot_acquisition_mean_kvp_over_time:
        result = average_chart_over_time_data(
            f.qs,
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
            "study_date",
            "projectionxrayradiationdose__irradeventxraydata__date_time_started",
            median_available,
            plot_average_choice,
            1,
            plot_acquisition_mean_dap_over_time_period,
            case_insensitive_categories=plot_case_insensitive_categories,
        )
        if median_available and (
            plot_average_choice == "median" or plot_average_choice == "both"
        ):
            return_structure["acquisitionMediankVpoverTime"] = result[
                "median_over_time"
            ]
        if plot_average_choice == "mean" or plot_average_choice == "both":
            return_structure["acquisitionMeankVpoverTime"] = result["mean_over_time"]
        return_structure["acquisition_kvp_names"] = result["series_names"]

    if plot_acquisition_mean_mas_over_time:
        result = average_chart_over_time_data(
            f.qs,
            "projectionxrayradiationdose__irradeventxraydata__acquisition_protocol",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
            "study_date",
            "projectionxrayradiationdose__irradeventxraydata__date_time_started",
            median_available,
            plot_average_choice,
            0.001,
            plot_acquisition_mean_dap_over_time_period,
            case_insensitive_categories=plot_case_insensitive_categories,
        )
        if median_available and (
            plot_average_choice == "median" or plot_average_choice == "both"
        ):
            return_structure["acquisitionMedianmAsoverTime"] = result[
                "median_over_time"
            ]
        if plot_average_choice == "mean" or plot_average_choice == "both":
            return_structure["acquisitionMeanmAsoverTime"] = result["mean_over_time"]
        return_structure["acquisition_mas_names"] = result["series_names"]

    if plot_study_per_day_and_hour:
        result = workload_chart_data(study_events)
        return_structure["studiesPerHourInWeekdays"] = result["workload"]

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

    if (
        user_profile.median_available
        and "postgresql" in settings.DATABASES["default"]["ENGINE"]
    ):
        median_available = True
    elif "postgresql" in settings.DATABASES["default"]["ENGINE"]:
        user_profile.median_available = True
        user_profile.save()
        median_available = True
    else:
        user_profile.median_available = False
        user_profile.save()
        median_available = False

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
            if median_available:
                user_profile.plotAverageChoice = chart_options_form.cleaned_data[
                    "plotMeanMedianOrBoth"
                ]
            user_profile.plotSeriesPerSystem = chart_options_form.cleaned_data[
                "plotSeriesPerSystem"
            ]
            user_profile.plotHistograms = chart_options_form.cleaned_data[
                "plotHistograms"
            ]
            user_profile.save()

        else:
            form_data = {
                "plotCharts": user_profile.plotCharts,
                "plotRFStudyPerDayAndHour": user_profile.plotRFStudyPerDayAndHour,
                "plotRFStudyFreq": user_profile.plotRFStudyFreq,
                "plotRFStudyDAP": user_profile.plotRFStudyDAP,
                "plotRFRequestFreq": user_profile.plotRFRequestFreq,
                "plotRFRequestDAP": user_profile.plotRFRequestDAP,
                "plotMeanMedianOrBoth": user_profile.plotAverageChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
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

    return render(request, "remapp/rffiltered.html", return_structure)


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

    if (
        user_profile.median_available
        and "postgresql" in settings.DATABASES["default"]["ENGINE"]
    ):
        median_available = True
    elif "postgresql" in settings.DATABASES["default"]["ENGINE"]:
        user_profile.median_available = True
        user_profile.save()
        median_available = True
    else:
        user_profile.median_available = False
        user_profile.save()
        median_available = False

    if settings.DEBUG:
        from datetime import datetime

        start_time = datetime.now()

    return_structure = rf_plot_calculations(
        f,
        median_available,
        user_profile.plotAverageChoice,
        user_profile.plotSeriesPerSystem,
        user_profile.plotHistogramBins,
        user_profile.plotRFStudyPerDayAndHour,
        user_profile.plotRFStudyFreq,
        user_profile.plotRFStudyDAP,
        user_profile.plotRFRequestFreq,
        user_profile.plotRFRequestDAP,
        user_profile.plotHistograms,
        user_profile.plotCaseInsensitiveCategories,
    )

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def rf_plot_calculations(
    f,
    median_available,
    plot_average_choice,
    plot_series_per_systems,
    plot_histogram_bins,
    plot_study_per_day_and_hour,
    plot_study_freq,
    plot_study_dap,
    plot_request_freq,
    plot_request_dap,
    plot_histograms,
    plot_case_insensitive_categories,
):
    """Calculations for fluoroscopy charts
    """
    from .interface.chart_functions import (
        average_chart_inc_histogram_data,
        workload_chart_data,
    )

    return_structure = {}

    if (
        plot_study_per_day_and_hour
        or plot_study_freq
        or plot_study_dap
        or plot_request_freq
        or plot_request_dap
    ):
        # No acquisition-level filters, so can use f.qs for all charts at the moment.
        # exp_include = f.qs.values_list('study_instance_uid')
        # study_events = GeneralStudyModuleAttr.objects.filter(study_instance_uid__in=exp_include)
        study_and_request_events = f.qs

    if plot_study_per_day_and_hour:
        result = workload_chart_data(study_and_request_events)
        return_structure["studiesPerHourInWeekdays"] = result["workload"]

    if plot_study_freq or plot_study_dap:
        result = average_chart_inc_histogram_data(
            study_and_request_events,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
            "study_description",
            "total_dap",
            1000000,
            plot_study_dap,
            plot_study_freq,
            plot_series_per_systems,
            plot_average_choice,
            median_available,
            plot_histogram_bins,
            calculate_histograms=plot_histograms,
            case_insensitive_categories=plot_case_insensitive_categories,
        )

        return_structure["studySystemList"] = result["system_list"]
        return_structure["studyNameList"] = result["series_names"]
        return_structure["studySummary"] = result["summary"]
        if plot_study_dap and plot_histograms:
            return_structure["studyHistogramData"] = result["histogram_data"]

    if plot_request_freq or plot_request_dap:
        result = average_chart_inc_histogram_data(
            study_and_request_events,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
            "requested_procedure_code_meaning",
            "total_dap",
            1000000,
            plot_request_dap,
            plot_request_freq,
            plot_series_per_systems,
            plot_average_choice,
            median_available,
            plot_histogram_bins,
            calculate_histograms=plot_histograms,
            case_insensitive_categories=plot_case_insensitive_categories,
        )

        return_structure["requestSystemList"] = result["system_list"]
        return_structure["requestNameList"] = result["series_names"]
        return_structure["requestSummary"] = result["summary"]
        if plot_request_dap and plot_histograms:
            return_structure["requestHistogramData"] = result["histogram_data"]

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
            user_profile.plotCTRequestMeanDLP = chart_options_form.cleaned_data[
                "plotCTRequestMeanDLP"
            ]
            user_profile.plotCTRequestFreq = chart_options_form.cleaned_data[
                "plotCTRequestFreq"
            ]
            user_profile.plotCTRequestNumEvents = chart_options_form.cleaned_data[
                "plotCTRequestNumEvents"
            ]
            user_profile.plotCTStudyPerDayAndHour = chart_options_form.cleaned_data[
                "plotCTStudyPerDayAndHour"
            ]
            user_profile.plotCTStudyMeanDLPOverTime = chart_options_form.cleaned_data[
                "plotCTStudyMeanDLPOverTime"
            ]
            user_profile.plotCTStudyMeanDLPOverTimePeriod = chart_options_form.cleaned_data[
                "plotCTStudyMeanDLPOverTimePeriod"
            ]
            user_profile.plotAverageChoice = chart_options_form.cleaned_data[
                "plotMeanMedianOrBoth"
            ]
            user_profile.plotSeriesPerSystem = chart_options_form.cleaned_data[
                "plotSeriesPerSystem"
            ]
            user_profile.plotHistograms = chart_options_form.cleaned_data[
                "plotHistograms"
            ]
            user_profile.save()

        else:
            form_data = {
                "plotCharts": user_profile.plotCharts,
                "plotCTAcquisitionMeanDLP": user_profile.plotCTAcquisitionMeanDLP,
                "plotCTAcquisitionMeanCTDI": user_profile.plotCTAcquisitionMeanCTDI,
                "plotCTAcquisitionFreq": user_profile.plotCTAcquisitionFreq,
                "plotCTStudyMeanDLP": user_profile.plotCTStudyMeanDLP,
                "plotCTStudyMeanCTDI": user_profile.plotCTStudyMeanCTDI,
                "plotCTStudyFreq": user_profile.plotCTStudyFreq,
                "plotCTStudyNumEvents": user_profile.plotCTStudyNumEvents,
                "plotCTRequestMeanDLP": user_profile.plotCTRequestMeanDLP,
                "plotCTRequestFreq": user_profile.plotCTRequestFreq,
                "plotCTRequestNumEvents": user_profile.plotCTRequestNumEvents,
                "plotCTStudyPerDayAndHour": user_profile.plotCTStudyPerDayAndHour,
                "plotCTStudyMeanDLPOverTime": user_profile.plotCTStudyMeanDLPOverTime,
                "plotCTStudyMeanDLPOverTimePeriod": user_profile.plotCTStudyMeanDLPOverTimePeriod,
                "plotMeanMedianOrBoth": user_profile.plotAverageChoice,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
                "plotHistograms": user_profile.plotHistograms,
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

    return render(request, "remapp/ctfiltered.html", return_structure,)


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

    # Obtain the key name in the TIME_PERIOD tuple from the user time period choice (the key value)
    keys = list(dict(user_profile.TIME_PERIOD).keys())
    values = list(dict(user_profile.TIME_PERIOD).values())
    altair_timeunit = keys[[x.lower() for x in values].index(user_profile.plotCTStudyMeanDLPOverTimePeriod)]

    return_structure = ct_plot_calculations(
        f,
        user_profile.plotCTAcquisitionFreq,
        user_profile.plotCTAcquisitionMeanCTDI,
        user_profile.plotCTAcquisitionMeanDLP,
        user_profile.plotCTRequestFreq,
        user_profile.plotCTRequestMeanDLP,
        user_profile.plotCTRequestNumEvents,
        user_profile.plotCTStudyFreq,
        user_profile.plotCTStudyMeanDLP,
        user_profile.plotCTStudyMeanCTDI,
        user_profile.plotCTStudyNumEvents,
        user_profile.plotCTStudyMeanDLPOverTime,
        altair_timeunit,
        user_profile.plotCTStudyPerDayAndHour,
        user_profile.plotAverageChoice,
        user_profile.plotSeriesPerSystem,
        user_profile.plotHistogramBins,
        user_profile.plotHistograms,
        user_profile.plotCaseInsensitiveCategories
    )

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def ct_plot_calculations(
    f,
    plot_acquisition_freq,
    plot_acquisition_mean_ctdi,
    plot_acquisition_mean_dlp,
    plot_request_freq,
    plot_request_mean_dlp,
    plot_request_num_events,
    plot_study_freq,
    plot_study_mean_dlp,
    plot_study_mean_ctdi,
    plot_study_num_events,
    plot_study_mean_dlp_over_time,
    plot_study_mean_dlp_over_time_period,
    plot_study_per_day_and_hour,
    plot_average_choice,
    plot_series_per_systems,
    plot_histogram_bins,
    plot_histograms,
    plot_case_insensitive_categories
):
    """CT chart data calculations
    """
    from .interface.chart_functions import (
        create_dataframe,
        create_dataframe_time_series,
        create_dataframe_weekdays,
        altair_barchart_average,
        altair_barchart_frequency,
        altair_linechart_average,
        altair_barchart_workload,
        altair_barchart_histogram,
        plotly_boxplot,
        plotly_barchart,
        plotly_histogram,
        plotly_stacked_histogram,
        plotly_timeseries_linechart,
        plotly_barchart_weekdays,
        average_chart_inc_histogram_data,
        average_chart_over_time_data,
        workload_chart_data,
    )

    return_structure = {}

    if (
        plot_study_mean_dlp
        or plot_study_mean_ctdi
        or plot_study_freq
        or plot_study_num_events
        or plot_study_mean_dlp_over_time
        or plot_study_per_day_and_hour
        or plot_request_mean_dlp
        or plot_request_freq
        or plot_request_num_events
    ):
        prefetch_list = [
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name"
        ]
        if plot_study_mean_ctdi:
            prefetch_list.append(
                "ctradiationdose__ctirradiationeventdata__mean_ctdivol"
            )

        if (
            "acquisition_protocol" in f.form.data
            and f.form.data["acquisition_protocol"]
        ) or (
            "ct_acquisition_type" in f.form.data and f.form.data["ct_acquisition_type"]
        ):
            # The user has filtered on acquisition_protocol, so need to use the slow method of querying the database
            # to avoid studies being duplicated when there is more than one of a particular acquisition type in a
            # study.
            try:
                exp_include = f.qs.values_list("study_instance_uid")
                study_and_request_events = (
                    GeneralStudyModuleAttr.objects.exclude(total_dlp__isnull=True)
                    .filter(study_instance_uid__in=exp_include)
                    .values(*prefetch_list)
                )
            except KeyError:
                study_and_request_events = f.qs.values(*prefetch_list)
        else:
            # The user hasn't filtered on acquisition, so we can use the faster database querying.
            study_and_request_events = f.qs.values(*prefetch_list)

    if plot_acquisition_mean_dlp or plot_acquisition_freq or plot_acquisition_mean_ctdi:
        prefetch_list = [
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
            "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
        ]
        if plot_acquisition_mean_dlp:
            prefetch_list.append("ctradiationdose__ctirradiationeventdata__dlp")
        if plot_acquisition_mean_ctdi:
            prefetch_list.append(
                "ctradiationdose__ctirradiationeventdata__mean_ctdivol"
            )

        if (
            plot_histograms
            and "ct_acquisition_type" in f.form.data
            and f.form.data["ct_acquisition_type"]
        ):
            # The user has filtered on acquisition_protocol, so need to use the slow method of querying the database
            # to avoid studies being duplicated when there is more than one of a particular acquisition type in a
            # study.
            try:
                exp_include = f.qs.values_list("study_instance_uid")
                acquisition_events = (
                    GeneralStudyModuleAttr.objects.exclude(total_dlp__isnull=True)
                    .filter(
                        study_instance_uid__in=exp_include,
                        ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning__iexact=f.form.data[
                            "ct_acquisition_type"
                        ],
                    )
                    .values(*prefetch_list)
                )
            except KeyError:
                acquisition_events = f.qs.values(*prefetch_list)
        else:
            acquisition_events = f.qs.values(*prefetch_list)

    #######################################################################
    # Prepare acquisition-level Pandas DataFrame to use for charts
    if "acquisition_events" in locals():

        name_fields = []
        if plot_acquisition_mean_dlp or plot_acquisition_mean_ctdi or plot_acquisition_freq:
            name_fields.append("ctradiationdose__ctirradiationeventdata__acquisition_protocol")

        value_fields = []
        if plot_acquisition_mean_dlp:
            value_fields.append("ctradiationdose__ctirradiationeventdata__dlp")
        if plot_acquisition_mean_ctdi:
            value_fields.append("ctradiationdose__ctirradiationeventdata__mean_ctdivol")

        date_fields = []

        system_field = None
        if plot_series_per_systems:
            system_field = "generalequipmentmoduleattr__unique_equipment_name_id__display_name"

        df = create_dataframe(
            acquisition_events,
            data_point_name_fields=name_fields,
            data_point_value_fields=value_fields,
            data_point_date_fields=date_fields,
            system_name_field=system_field,
            data_point_name_lowercase=plot_case_insensitive_categories
        )
        #######################################################################

        #######################################################################
        # Create the required acquisition-level charts
        if plot_acquisition_mean_dlp:
            if plot_average_choice in ["mean", "both"]:
                return_structure["acquisitionMeanDLPData"] = plotly_barchart(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Acquisition protocol"
                )

            if plot_average_choice in ["median", "both"]:
                return_structure["acquisitionBoxplotDLPData"] = plotly_boxplot(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Acquisition protocol"
                )

            if plot_histograms:
                return_structure["acquisitionHistDLPData"] = plotly_histogram(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Acquisition protocol",
                    n_bins=plot_histogram_bins
                )

        if plot_acquisition_mean_ctdi:
            if plot_average_choice in ["mean", "both"]:
                return_structure["acquisitionMeanCTDIData"] = plotly_barchart(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI (mGy.cm)",
                    name_axis_title="Acquisition protocol"
                )

            if plot_average_choice in ["median", "both"]:
                return_structure["acquisitionBoxplotCTDIData"] = plotly_boxplot(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI (mGy.cm)",
                    name_axis_title="Acquisition protocol"
                )

            if plot_histograms:
                return_structure["acquisitionHistCTDIData"] = plotly_histogram(
                    df,
                    "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI (mGy.cm)",
                    name_axis_title="Acquisition protocol",
                    n_bins=plot_histogram_bins
                )

        if plot_acquisition_freq:
            return_structure["acquisitionFreqData"] = plotly_stacked_histogram(
                df,
                "ctradiationdose__ctirradiationeventdata__acquisition_protocol",
                name_axis_title="Acquisition protocol"
            )
    #######################################################################
    # Prepare study- and request-level Pandas DataFrame to use for charts
    if "study_and_request_events" in locals():

        name_fields = []
        if plot_study_mean_dlp or plot_study_freq or plot_study_mean_dlp_over_time or plot_study_per_day_and_hour or plot_study_num_events or plot_study_mean_ctdi:
            name_fields.append("study_description")
        if plot_request_mean_dlp or plot_request_freq or plot_request_num_events:
            name_fields.append("requested_procedure_code_meaning")

        value_fields = []
        if plot_study_mean_dlp or plot_study_mean_dlp_over_time or plot_request_mean_dlp:
            value_fields.append("total_dlp")
        if plot_study_mean_ctdi:
            value_fields.append("ctradiationdose__ctirradiationeventdata__mean_ctdivol")
        if plot_study_num_events or plot_request_num_events:
            value_fields.append("number_of_events")

        date_fields = []
        if plot_study_mean_dlp_over_time or plot_study_per_day_and_hour:
            date_fields.append("study_date")

        system_field = None
        if plot_series_per_systems:
            system_field = "generalequipmentmoduleattr__unique_equipment_name_id__display_name"

        df = create_dataframe(
            study_and_request_events,
            data_point_name_fields=name_fields,
            data_point_value_fields=value_fields,
            data_point_date_fields=date_fields,
            system_name_field=system_field,
            data_point_name_lowercase=plot_case_insensitive_categories
        )
        #######################################################################

        #######################################################################
        # Create the required study- and request-level charts
        if plot_study_mean_dlp:
            if plot_average_choice in ["mean", "both"]:
                return_structure["studyMeanDLPData"] = plotly_barchart(
                    df,
                    "study_description",
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Study description"
                )

            if plot_average_choice in ["median", "both"]:
                return_structure["studyBoxplotDLPData"] = plotly_boxplot(
                    df,
                    "study_description",
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Study description"
                )

            if plot_histograms:
                return_structure["studyHistDLPData"] = plotly_histogram(
                    df,
                    "study_description",
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Study description",
                    n_bins=plot_histogram_bins
                )

        if plot_study_mean_ctdi:
            if plot_average_choice in ["mean", "both"]:
                return_structure["studyMeanCTDIData"] = plotly_barchart(
                    df,
                    "study_description",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI (mGy)",
                    name_axis_title="Study description"
                )

            if plot_average_choice in ["median", "both"]:
                return_structure["studyBoxplotCTDIData"] = plotly_boxplot(
                    df,
                    "study_description",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI (mGy)",
                    name_axis_title="Study description"
                )

            if plot_histograms:
                return_structure["studyHistCTDIData"] = plotly_histogram(
                    df,
                    "study_description",
                    "ctradiationdose__ctirradiationeventdata__mean_ctdivol",
                    value_axis_title="CTDI (mGy)",
                    name_axis_title="Study description",
                    n_bins=plot_histogram_bins
                )

        if plot_study_num_events:
            if plot_average_choice in ["mean", "both"]:
                return_structure["studyMeanNumEventsData"] = plotly_barchart(
                    df,
                    "study_description",
                    "number_of_events",
                    value_axis_title="Events",
                    name_axis_title="Study description"
                )

            if plot_average_choice in ["median", "both"]:
                return_structure["studyBoxplotNumEventsData"] = plotly_boxplot(
                    df,
                    "study_description",
                    "number_of_events",
                    value_axis_title="Events",
                    name_axis_title="Study description"
                )

            if plot_histograms:
                return_structure["studyHistNumEventsData"] = plotly_histogram(
                    df,
                    "study_description",
                    "number_of_events",
                    value_axis_title="Events",
                    name_axis_title="Study description",
                    n_bins=plot_histogram_bins
                )

        if plot_request_mean_dlp:
            if plot_average_choice in ["mean", "both"]:
                return_structure["requestMeanData"] = plotly_barchart(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Requested procedure"
                )

            if plot_average_choice in ["median", "both"]:
                return_structure["requestBoxplotData"] = plotly_boxplot(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Requested procedure"
                )

            if plot_histograms:
                return_structure["requestHistData"] = plotly_histogram(
                    df,
                    "requested_procedure_code_meaning",
                    "total_dlp",
                    value_axis_title="DLP (mGy.cm)",
                    name_axis_title="Requested procedure",
                    n_bins=plot_histogram_bins
                )

        if plot_request_num_events:
            if plot_average_choice in ["mean", "both"]:
                return_structure["requestMeanNumEventsData"] = plotly_barchart(
                    df,
                    "requested_procedure_code_meaning",
                    "number_of_events",
                    value_axis_title="Events",
                    name_axis_title="Requested procedure"
                )

            if plot_average_choice in ["median", "both"]:
                return_structure["requestBoxplotNumEventsData"] = plotly_boxplot(
                    df,
                    "requested_procedure_code_meaning",
                    "number_of_events",
                    value_axis_title="Events",
                    name_axis_title="Requested procedure"
                )

            if plot_histograms:
                return_structure["requestHistNumEventsData"] = plotly_histogram(
                    df,
                    "requested_procedure_code_meaning",
                    "number_of_events",
                    value_axis_title="Events",
                    name_axis_title="Requested procedure",
                    n_bins=plot_histogram_bins
                )

        if plot_study_freq:
            return_structure["studyFreqData"] = plotly_stacked_histogram(
                df,
                "study_description",
                name_axis_title="Study description"
            )

        if plot_request_freq:
            return_structure["requestFreqData"] = plotly_stacked_histogram(
                df,
                "requested_procedure_code_meaning",
                name_axis_title="Requested procedure"
            )

        if plot_study_mean_dlp_over_time:
            df_time_series = create_dataframe_time_series(
                df,
                "study_description",
                "total_dlp",
                df_date_col="study_date",
                time_period=plot_study_mean_dlp_over_time_period,
                average=plot_average_choice
            )

            if plot_average_choice in ["mean", "both"]:
                return_structure["studyMeanDLPoverTime"] = plotly_timeseries_linechart(
                    df_time_series,
                    "study_description",
                    "meantotal_dlp",
                    "study_date",
                    value_axis_title="Mean DLP (mGy.cm)",
                    name_axis_title="Study date",
                    legend_title="Study description"
                )

            if plot_average_choice in ["median", "both"]:
                return_structure["studyMedianDLPoverTime"] = plotly_timeseries_linechart(
                    df_time_series,
                    "study_description",
                    "mediantotal_dlp",
                    "study_date",
                    value_axis_title="Median DLP (mGy.cm)",
                    name_axis_title="Study date",
                    legend_title="Study description"
                )

        if plot_study_per_day_and_hour:
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
                value_axis_title="Frequency"
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
            user_profile.plotMGkVpvsThickness = chart_options_form.cleaned_data[
                "plotMGkVpvsThickness"
            ]
            user_profile.plotMGmAsvsThickness = chart_options_form.cleaned_data[
                "plotMGmAsvsThickness"
            ]
            user_profile.plotSeriesPerSystem = chart_options_form.cleaned_data[
                "plotSeriesPerSystem"
            ]
            # Uncomment the following line when there's at least one bar chart for mammo
            # user_profile.plotHistograms = chart_options_form.cleaned_data['plotHistograms']
            user_profile.save()

        else:
            form_data = {
                "plotCharts": user_profile.plotCharts,
                "plotMGStudyPerDayAndHour": user_profile.plotMGStudyPerDayAndHour,
                "plotMGAGDvsThickness": user_profile.plotMGAGDvsThickness,
                "plotMGkVpvsThickness": user_profile.plotMGkVpvsThickness,
                "plotMGmAsvsThickness": user_profile.plotMGmAsvsThickness,
                "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
            }
            # Uncomment the following line when there's at least one bar chart for mammo
            #'plotHistograms': user_profile.plotHistograms}
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

    return render(request, "remapp/mgfiltered.html", return_structure,)


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

    if (
        user_profile.median_available
        and "postgresql" in settings.DATABASES["default"]["ENGINE"]
    ):
        median_available = True
    elif "postgresql" in settings.DATABASES["default"]["ENGINE"]:
        user_profile.median_available = True
        user_profile.save()
        median_available = True
    else:
        user_profile.median_available = False
        user_profile.save()
        median_available = False

    if settings.DEBUG:
        from datetime import datetime

        start_time = datetime.now()

    return_structure = mg_plot_calculations(
        f,
        median_available,
        user_profile.plotAverageChoice,
        user_profile.plotSeriesPerSystem,
        user_profile.plotHistogramBins,
        user_profile.plotMGStudyPerDayAndHour,
        user_profile.plotMGAGDvsThickness,
        user_profile.plotMGkVpvsThickness,
        user_profile.plotMGmAsvsThickness,
    )

    if settings.DEBUG:
        logger.debug(f"Elapsed time is {datetime.now() - start_time}")

    return JsonResponse(return_structure, safe=False)


def mg_plot_calculations(
    f,
    median_available,
    plot_average_choice,
    plot_series_per_systems,
    plot_histogram_bins,
    plot_study_per_day_and_hour,
    plot_agd_vs_thickness,
    plot_kvp_vs_thickness,
    plot_mas_vs_thickness,
):
    """Calculations for mammography charts
    """
    from .interface.chart_functions import workload_chart_data, scatter_plot_data

    return_structure = {}

    if plot_study_per_day_and_hour:
        # No acquisition-level filters, so can use f.qs for all charts at the moment.
        # exp_include = f.qs.values_list('study_instance_uid')
        # study_events = GeneralStudyModuleAttr.objects.filter(study_instance_uid__in=exp_include)
        study_events = f.qs

        result = workload_chart_data(study_events)
        return_structure["studiesPerHourInWeekdays"] = result["workload"]

    if plot_agd_vs_thickness:
        result = scatter_plot_data(
            f.qs,
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__average_glandular_dose",
            1,
            plot_series_per_systems,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
        )
        return_structure["AGDvsThickness"] = result["scatterData"]
        return_structure["maxThicknessAndAGD"] = result["maxXandY"]
        return_structure["AGDvsThicknessSystems"] = result["system_list"]

    if plot_kvp_vs_thickness:
        result = scatter_plot_data(
            f.qs,
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__kvp__kvp",
            1,
            plot_series_per_systems,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
        )
        return_structure["kVpvsThickness"] = result["scatterData"]
        return_structure["maxThicknessAndkVp"] = result["maxXandY"]
        return_structure["kVpvsThicknessSystems"] = result["system_list"]

    if plot_mas_vs_thickness:
        result = scatter_plot_data(
            f.qs,
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraymechanicaldata__compression_thickness",
            "projectionxrayradiationdose__irradeventxraydata__irradeventxraysourcedata__exposure__exposure",
            0.001,
            plot_series_per_systems,
            "generalequipmentmoduleattr__unique_equipment_name_id__display_name",
        )
        return_structure["mAsvsThickness"] = result["scatterData"]
        return_structure["maxThicknessAndmAs"] = result["maxXandY"]
        return_structure["mAsvsThicknessSystems"] = result["system_list"]

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
    from datetime import datetime
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
        home_config = {
            "display_workload_stats": display_workload_stats,
            "day_delta_a": day_delta_a,
            "day_delta_b": day_delta_b,
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
