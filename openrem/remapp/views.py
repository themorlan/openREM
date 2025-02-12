# pylint: disable=too-many-lines
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

import os
import gzip
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import pickle  # nosec
from collections import OrderedDict

from django.db.models import (
    Sum,
    Q,
    Max,
    Count,
    Subquery,
    OuterRef,
)
from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import Group
from django.core.exceptions import ObjectDoesNotExist
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.template.defaultfilters import register
from django.urls import reverse_lazy
from django.utils.translation import gettext as _
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import numpy as np

from .forms import itemsPerPageForm
from .interface.mod_filters import (
    RFSummaryListFilter,
    RFFilterPlusStdNames,
    RFFilterPlusPid,
    RFFilterPlusPidPlusStdNames,
    dx_acq_filter,
    ct_acq_filter,
    MGSummaryListFilter,
    MGFilterPlusPid,
    MGFilterPlusStdNames,
    MGFilterPlusPidPlusStdNames,
    nm_filter,
)
from .tools.make_skin_map import (
    make_skin_map,
    skin_dose_maps_enabled_for_xray_system,
)
from .tools.background import run_in_background_with_limits
from .tools.check_standard_name_status import are_standard_names_enabled
from .views_charts_ct import (
    generate_required_ct_charts_list,
    ct_chart_form_processing,
)
from .views_charts_dx import (
    generate_required_dx_charts_list,
    dx_chart_form_processing,
)
from .views_charts_mg import (
    generate_required_mg_charts_list,
    mg_chart_form_processing,
)
from .views_charts_rf import (
    generate_required_rf_charts_list,
    rf_chart_form_processing,
)
from .views_charts_nm import nm_chart_form_processing, generate_required_nm_charts_list
from .models import (
    GeneralStudyModuleAttr,
    create_user_profile,
    HighDoseMetricAlertSettings,
    SkinDoseMapCalcSettings,
    PatientIDSettings,
    DicomDeleteSettings,
    AdminTaskQuestions,
    HomePageAdminSettings,
    UpgradeStatus,
    StandardNameSettings,
    BackgroundTask,
    CTIrradiationEventData,
)
from .version import __version__, __docs_version__, __skin_map_version__

os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"


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
    """Log users out and re-direct them to the main page."""
    logout(request)
    return HttpResponseRedirect(reverse_lazy("home"))


def update_items_per_page_form(request, user_profile):
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
    return items_per_page_form


def get_or_create_user(request):
    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile
    return user_profile


def create_admin_info(request):
    admin = {
        "openremversion": __version__,
        "docsversion": __docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True
    return admin


def create_paginated_study_list(request, f, user_profile):
    paginator = Paginator(f.qs, user_profile.itemsPerPage)
    page = request.GET.get("page")
    try:
        study_list = paginator.page(page)
    except PageNotAnInteger:
        study_list = paginator.page(1)
    except EmptyPage:
        study_list = paginator.page(paginator.num_pages)
    return study_list


def generate_return_structure(request, f):
    user_profile = get_or_create_user(request)
    items_per_page_form = update_items_per_page_form(request, user_profile)
    admin = create_admin_info(request)
    
    # Berechne max_ctdi für alle Studien in der QuerySet, aber nur für CT-Studien
    if f.qs and hasattr(f.qs.first(), 'ctradiationdose_set'):  # Prüfe ob QuerySet nicht leer ist
        for study in f.qs:
            try:
                max_ctdi = study.ctradiationdose_set.get().ctirradiationeventdata_set.filter(
                    acquisition_protocol__in=['Stationary Acquisition', 'Spiral Acquisition']
                ).aggregate(
                    max_ctdivol=Max('mean_ctdivol')
                )['max_ctdivol']
                study.max_ctdi = max_ctdi
            except (ObjectDoesNotExist, AttributeError):
                study.max_ctdi = None
    
    study_list = create_paginated_study_list(request, f, user_profile)
    enable_standard_names = are_standard_names_enabled()
    return_structure = {
        "filter": f,
        "study_list": study_list,
        "admin": admin,
        "itemsPerPageForm": items_per_page_form,
        "showStandardNames": enable_standard_names,
    }
    return user_profile, return_structure


@login_required
def dx_summary_list_filter(request):
    """Obtain data for radiographic summary view."""
    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = dx_acq_filter(request.GET, pid=pid)

    user_profile, return_structure = generate_return_structure(request, f)
    chart_options_form = dx_chart_form_processing(request, user_profile)
    return_structure["chartOptionsForm"] = chart_options_form

    if user_profile.plotCharts:
        return_structure["required_charts"] = generate_required_dx_charts_list(
            user_profile
        )

    return render(request, "remapp/dxfiltered.html", return_structure)


@login_required
def dx_detail_view(request, pk=None):
    """Detail view for a DX study."""

    try:
        study = GeneralStudyModuleAttr.objects.get(pk=pk)
    except:
        messages.error(request, "That study was not found")
        return redirect(reverse_lazy("dx_summary_list_filter"))

    admin = create_admin_info(request)
    enable_standard_names = are_standard_names_enabled()

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
            "showStandardNames": enable_standard_names,
        },
    )


@login_required
def rf_summary_list_filter(request):
    """Obtain data for radiographic summary view."""

    enable_standard_names = are_standard_names_enabled()
    queryset = (
        GeneralStudyModuleAttr.objects.filter(modality_type="RF")
        .order_by("-study_date", "-study_time")
        .distinct()
    )

    if request.user.groups.filter(name="pidgroup"):
        if enable_standard_names:
            f = RFFilterPlusPidPlusStdNames(
                request.GET,
                queryset=queryset,
            )
        else:
            f = RFFilterPlusPid(
                request.GET,
                queryset=queryset,
            )
    else:
        if enable_standard_names:
            f = RFFilterPlusStdNames(
                request.GET,
                queryset=queryset,
            )
        else:
            f = RFSummaryListFilter(
                request.GET,
                queryset=queryset,
            )

    user_profile, return_structure = generate_return_structure(request, f)
    chart_options_form = rf_chart_form_processing(request, user_profile)

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

    return_structure["chartOptionsForm"] = chart_options_form
    return_structure["alertLevels"] = alert_levels

    if user_profile.plotCharts:
        return_structure["required_charts"] = generate_required_rf_charts_list(
            user_profile
        )

    return render(request, "remapp/rffiltered.html", return_structure)


@login_required
def rf_detail_view(request, pk=None):
    """Detail view for an RF study."""

    enable_standard_names = are_standard_names_enabled()

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
    accumxraydose_set_all_planes = (
        projection_xray_dose_set.accumxraydose_set.select_related(
            "acquisition_plane"
        ).all()
    )
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
        try:
            stu_time_totals[0] = stu_time_totals[0] + accum_dose_ds.total_fluoro_time
        except TypeError:
            pass
        try:
            stu_time_totals[1] = (
                stu_time_totals[1] + accum_dose_ds.total_acquisition_time
            )
        except TypeError:
            pass
        accum_integrated = (
            accum_dose_ds.accumulated_xray_dose.accumintegratedprojradiogdose_set.get()
        )
        try:
            total_dap = total_dap + accum_integrated.dose_area_product_total
        except TypeError:
            pass
        try:
            total_dose = total_dose + accum_dose_ds.dose_rp_total
        except TypeError:
            pass

    # get info for different Acquisition Types
    stu_inc_totals = (  # pylint: disable=line-too-long
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
        stu_time_totals.append(  # pylint: disable=line-too-long
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
                modality_type="RF",
                patientmoduleattr__patient_id=patient_id,
                study_date__range=[oldest_date, study_date],
            )
        else:
            included_studies = None
    else:
        included_studies = None

    admin = create_admin_info(request)
    admin["enable_skin_dose_maps"] = SkinDoseMapCalcSettings.objects.values_list(
        "enable_skin_dose_maps", flat=True
    )[0]

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
            "showStandardNames": enable_standard_names,
        },
    )


@login_required
def rf_detail_view_skin_map(request, pk=None):
    """View to calculate a skin dose map."""
    try:
        GeneralStudyModuleAttr.objects.get(pk=pk)
    except ObjectDoesNotExist:
        messages.error(request, "That study was not found")
        return redirect(reverse_lazy("rf_summary_list_filter"))

    return_structure = {}

    # Get the study
    study = GeneralStudyModuleAttr.objects.get(pk=pk)

    # Find the latest task corresponding to this study. This is done in two
    # stages because using .latest("task_type") on an empty queryset throws
    # a DoesNotExist error
    matching_latest_task = None
    matching_tasks = BackgroundTask.objects.filter(
        task_type="make_skin_map",
        info__contains=pk,
    )
    if matching_tasks:
        matching_latest_task = matching_tasks.latest("task_type")

    latest_task_failed = False
    if matching_latest_task:
        if (
            matching_latest_task.error is not None
        ):  # Avoid TypeError in next line if error is None
            if (
                matching_latest_task.completed_successfully is False
                and "failed" in matching_latest_task.error
            ):
                latest_task_failed = True

    # Check if skin dose maps are enabled for the x-ray system used for the study
    skin_maps_enabled = skin_dose_maps_enabled_for_xray_system(study)
    if skin_maps_enabled is False:
        # Some code to return something that says they are disabled for this system
        return_structure["disabled_skin_maps"] = True

    elif (
        latest_task_failed
        and "force_recalculation" not in request.resolver_match.url_name
    ):
        if (
            matching_latest_task.completed_successfully is False
            and "failed" in matching_latest_task.error
        ):
            return_structure["skin_map_calculation_failed"] = True

            # Find out if there is a task running to re-calculate the skin dose map for this study
            matching_ongoing_task = BackgroundTask.objects.filter(
                task_type="make_skin_map", info__contains=pk, complete=False
            )

            # Set the skin_map_progress and in_progress entries if there is a match to a running task.
            if matching_ongoing_task.count() != 0:
                return_structure["skin_map_progress"] = (
                    matching_ongoing_task.values_list("info", flat=True)[0].split(
                        "irradiation ", 1
                    )[1]

            return_structure["in_progress"] = True

    else:
        # Check to see if there is already a skin map pickle with the same study ID.
        try:
            study_date = GeneralStudyModuleAttr.objects.get(pk=pk).study_date
            if study_date:
                skin_map_path = os.path.join(
                    settings.MEDIA_ROOT,
                    "skin_maps",
                    "{0:0>4}".format(study_date.year),
                    "{0:0>2}".format(study_date.month),
                    "{0:0>2}".format(study_date.day),
                    "skin_map_" + str(pk) + ".p",
                )
            else:
                skin_map_path = os.path.join(
                    settings.MEDIA_ROOT, "skin_maps", "skin_map_" + str(pk) + ".p"
                )
        except:
            skin_map_path = os.path.join(
                settings.MEDIA_ROOT, "skin_maps", "skin_map_" + str(pk) + ".p"
            )

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

        return_structure["in_progress"] = False

        if os.path.exists(skin_map_path):
            with gzip.open(skin_map_path, "rb") as f:
                existing_skin_map_data = pickle.load(f)  # nosec
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
                        return_structure["in_progress"] = False
                        loaded_existing_data = True
            except KeyError:
                pass

        if not loaded_existing_data:
            # Check to see if there is already a background task running to calculate a skin dose map for this study.
            # Need to have a Django query that matches the following SQL:
            #   SELECT info
            #   FROM remapp_backgroundtask
            #   WHERE task_type='make_skin_map' [just make_skin_map tasks]
            #     AND info LIKE '%1880069%'     [info contains the pk of the study, 1880069 in this case]
            #     AND complete=FALSE;           [it is not yet complete]
            #
            # If there are no rows returned from the above query then there is not already a job being run to calculate
            # the skin dose map for this study, and we can go ahead and create a background task to run it.

            # The following line is equivalent to the above SQL:
            matching_ongoing_task = BackgroundTask.objects.filter(
                task_type="make_skin_map", info__contains=pk, complete=False
            )

            # Only run make_skin_map if matching_ongoing_task is empty.
            if matching_ongoing_task.count() == 0:
                run_in_background_with_limits(
                    make_skin_map,
                    "make_skin_map",
                    0,
                    {"make_skin_map": 1},
                    pk,
                )
            else:
                return_structure["skin_map_progress"] = (
                    matching_ongoing_task.values_list("info", flat=True)[0].split(
                        "irradiation ", 1
                    )[1]

            return_structure["in_progress"] = True

    return_structure["primary_key"] = pk
    return JsonResponse(return_structure, safe=False)


@login_required
def nm_summary_list_filter(request):
    """Obtain data for NM summary view."""
    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = nm_filter(request.GET, pid=pid)

    user_profile, return_structure = generate_return_structure(request, f)
    chart_options_form = nm_chart_form_processing(request, user_profile)
    return_structure["chartOptionsForm"] = chart_options_form

    if user_profile.plotCharts:
        return_structure["required_charts"] = generate_required_nm_charts_list(
            user_profile
        )

    return render(request, "remapp/nmfiltered.html", return_structure)


@login_required
def nm_detail_view(request, pk=None):
    """Detail view for a NM study."""
    try:
        study = GeneralStudyModuleAttr.objects.get(pk=pk)
    except ObjectDoesNotExist:
        messages.error(request, "That study was not found")
        return redirect(reverse_lazy("nm_summary_list_filter"))

    associated_ct = GeneralStudyModuleAttr.objects.filter(
        Q(study_instance_uid=study.study_instance_uid) & Q(modality_type="CT")
    ).first()

    admin = create_admin_info(request)
    enable_standard_names = are_standard_names_enabled()

    return render(
        request,
        "remapp/nmdetail.html",
        {
            "generalstudymoduleattr": study,
            "admin": admin,
            "associated_ct": associated_ct,
            "showStandardNames": enable_standard_names,
        },
    )


@login_required
def ct_summary_list_filter(request):
    """Obtain data for CT summary view."""
    pid = bool(request.user.groups.filter(name="pidgroup"))
    f = ct_acq_filter(request.GET, pid=pid)

    user_profile, return_structure = generate_return_structure(request, f)
    chart_options_form = ct_chart_form_processing(request, user_profile)
    return_structure["chartOptionsForm"] = chart_options_form

    if user_profile.plotCharts:
        return_structure["required_charts"] = generate_required_ct_charts_list(
            user_profile
        )

    return render(request, "remapp/ctfiltered.html", return_structure)


@login_required
def ct_detail_view(request, pk=None):
    """Detail view for a CT study."""

    enable_standard_names = are_standard_names_enabled()

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

    associated_nm = GeneralStudyModuleAttr.objects.filter(
        Q(study_instance_uid=study.study_instance_uid) & Q(modality_type="NM")
    ).first()

    admin = create_admin_info(request)

    return render(
        request,
        "remapp/ctdetail.html",
        {
            "generalstudymoduleattr": study,
            "admin": admin,
            "events_all": events_all,
            "associated_nm": associated_nm,
            "showStandardNames": enable_standard_names,
        },
    )


@login_required
def mg_summary_list_filter(request):
    """Mammography data for summary view."""

    enable_standard_names = are_standard_names_enabled()
    filter_data = request.GET.copy()
    if "page" in filter_data:
        del filter_data["page"]

    queryset = (
        GeneralStudyModuleAttr.objects.filter(modality_type="MG")
        .order_by("-study_date", "-study_time")
        .distinct()
    )

    if request.user.groups.filter(name="pidgroup"):
        if enable_standard_names:
            f = MGFilterPlusPidPlusStdNames(
                filter_data,
                queryset=queryset,
            )
        else:
            f = MGFilterPlusPid(
                filter_data,
                queryset=queryset,
            )
    else:
        if enable_standard_names:
            f = MGFilterPlusStdNames(
                filter_data,
                queryset=queryset,
            )
        else:
            f = MGSummaryListFilter(
                filter_data,
                queryset=queryset,
            )

    user_profile, return_structure = generate_return_structure(request, f)
    chart_options_form = mg_chart_form_processing(request, user_profile)
    return_structure["chartOptionsForm"] = chart_options_form

    if user_profile.plotCharts:
        return_structure["required_charts"] = generate_required_mg_charts_list(
            user_profile
        )

    return render(request, "remapp/mgfiltered.html", return_structure)


@login_required
def mg_detail_view(request, pk=None):
    """Detail view for a CT study."""

    enable_standard_names = are_standard_names_enabled()

    try:
        study = GeneralStudyModuleAttr.objects.get(pk=pk)
    except:
        messages.error(request, "That study was not found")
        return redirect(reverse_lazy("mg_summary_list_filter"))

    admin = create_admin_info(request)

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
            "showStandardNames": enable_standard_names,
        },
    )


def openrem_home(request):
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
    study_counts = allstudies.aggregate(
        all_count=Count("pk"),
        ct_count=Count("pk", filter=Q(modality_type="CT")),
        rf_count=Count("pk", filter=Q(modality_type="RF")),
        mg_count=Count("pk", filter=Q(modality_type="MG")),
        nm_count=Count("pk", filter=Q(modality_type="NM")),
        dx_count=Count("pk", filter=Q(modality_type__in=["DX", "CR", "PX"])),
    )

    modalities = OrderedDict()
    if study_counts["ct_count"]:
        modalities["CT"] = {"name": _("CT"), "count": study_counts["ct_count"]}
    if study_counts["rf_count"]:
        modalities["RF"] = {"name": _("Fluoroscopy"), "count": study_counts["rf_count"]}
    if study_counts["mg_count"]:
        modalities["MG"] = {"name": _("Mammography"), "count": study_counts["mg_count"]}
    if study_counts["dx_count"]:
        modalities["DX"] = {"name": _("Radiography"), "count": study_counts["dx_count"]}
    if study_counts["nm_count"]:
        modalities["NM"] = {
            "name": _("Nuclear Medicine"),
            "count": study_counts["nm_count"],
        }

    homedata = {"total": study_counts["all_count"]}

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

    admin = dict(openremversion=__version__, docsversion=__docs_version__)

    for group in request.user.groups.all():
        admin[group.name] = True

    admin_questions = {}
    admin_questions_true = False
    if request.user.groups.filter(name="admingroup"):
        not_patient_indicator_question = (
            AdminTaskQuestions.get_solo().ask_revert_to_074_question
        )
        admin_questions["not_patient_indicator_question"] = (
            not_patient_indicator_question
        )
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
    # send_rf_high_dose_alert_email(881397)
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
    # send_alert_emails = HighDoseMetricAlertSettings.objects.values_list(
    #     'send_high_dose_metric_alert_emails', flat=True
    # )[0]
    # if send_alert_emails:
    #     recipients = User.objects.filter(
    #         highdosemetricalertrecipients__receive_high_dose_metric_alerts=True
    #     ).values_list('email', flat=True)
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
def update_latest_studies(request):
    """
    AJAX function to calculate the latest studies for each display name for a particular modality.

    :param request: Request object
    :return: HTML table of modalities
    """
    if request.is_ajax():
        data = request.POST
        modality = data.get("modality")
        if modality == "DX":
            studies = GeneralStudyModuleAttr.objects.filter(
                Q(modality_type__in=["DX", "CR", "PX"])
            ).all()
        else:
            studies = GeneralStudyModuleAttr.objects.filter(
                modality_type=modality
            ).all()

        today = datetime.now()
        study_data = (
            studies.values(
                "generalequipmentmoduleattr__unique_equipment_name__display_name"
            )
            .annotate(
                num_studies=Count("pk"),
                latest_entry_date_time=Max("test_date_time"),
                # timedelta=ExpressionWrapper(today - F("latest_entry_date_time"), output_field=DurationField()),
            )
            .order_by("-latest_entry_date_time")
        )

        display_workload_stats = HomePageAdminSettings.objects.values_list(
            "enable_workload_stats", flat=True
        )[0]
        if request.user.is_authenticated:
            day_delta_a = request.user.userprofile.summaryWorkloadDaysA
            day_delta_b = request.user.userprofile.summaryWorkloadDaysB
        else:
            day_delta_a = 7
            day_delta_b = 28

        date_a = today.date() - timedelta(days=day_delta_a)
        date_b = today.date() - timedelta(days=day_delta_b)
        if display_workload_stats:
            study_data = study_data.annotate(
                studies_since_delta_a=Count("pk", filter=Q(test_date_time__gte=date_a)),
                studies_since_delta_b=Count("pk", filter=Q(test_date_time__gte=date_b)),
            ).order_by("-latest_entry_date_time")

        admin = {}
        for group in request.user.groups.all():
            admin[group.name] = True

        template = "remapp/home-list-modalities.html"

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
                "study_data": study_data,
                "modality": modality.lower(),
                "home_config": home_config,
                "admin": admin,
            },
        )


def ct_acq_filter(request_get, pid=False):
    """Filter for CT acquisition data according to query."""
    queryset = GeneralStudyModuleAttr.objects.filter(
        modality_type="CT"
    ).order_by("-study_date", "-study_time").annotate(
        max_ctdi=Subquery(
            CTIrradiationEventData.objects.filter(
                ct_radiation_dose__ct_study=OuterRef('pk'),
                acquisition_protocol__in=['Stationary Acquisition', 'Spiral Acquisition']
            ).values('ct_radiation_dose__ct_study').annotate(
                max_ctdivol=Max('mean_ctdivol')
            ).values('max_ctdivol')[:1]
        )
    )
    
    if pid:
        return CTFilterPlusPid(request_get, queryset=queryset)
    return CTSummaryListFilter(request_get, queryset=queryset)
