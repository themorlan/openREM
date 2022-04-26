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
..  module:: views_admin.
    :synopsis: Module to render appropriate content according to request.

..  moduleauthor:: Ed McDonagh

"""
from __future__ import absolute_import

import os
import json
import logging
from datetime import datetime, timedelta
import requests
from builtins import map  # pylint: disable=redefined-builtin

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User  # pylint: disable=all
from django.core.exceptions import ObjectDoesNotExist
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q, Sum
from django.db.utils import OperationalError as AvoidDataMigrationErrorSQLite
from django.db.utils import ProgrammingError as AvoidDataMigrationErrorPostgres
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.template.loader import render_to_string
from django.urls import reverse_lazy
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_exempt
from django.views.generic.edit import CreateView, UpdateView, DeleteView

from .extractors.extract_common import populate_rf_delta_weeks_summary
from .forms import (
    CTChartOptionsDisplayForm,
    DXChartOptionsDisplayForm,
    DicomDeleteSettingsForm,
    GeneralChartOptionsDisplayForm,
    HomepageOptionsForm,
    MGChartOptionsDisplayForm,
    MergeOnDeviceObserverUIDForm,
    NotPatientIDForm,
    NotPatientNameForm,
    RFChartOptionsDisplayForm,
    RFHighDoseFluoroAlertsForm,
    UpdateDisplayNamesForm,
)
from .models import (
    AccumIntegratedProjRadiogDose,
    AdminTaskQuestions,
    BackgroundTask,
    DicomDeleteSettings,
    DicomQuery,
    Exports,
    GeneralStudyModuleAttr,
    HighDoseMetricAlertRecipients,
    HighDoseMetricAlertSettings,
    HomePageAdminSettings,
    MergeOnDeviceObserverUIDSettings,
    NotPatientIndicatorsID,
    NotPatientIndicatorsName,
    PKsForSummedRFDoseStudiesInDeltaWeeks,
    PatientIDSettings,
    SizeUpload,
    SummaryFields,
    UniqueEquipmentNames,
    UpgradeStatus,
    create_user_profile,
    CommonVariables,
)
from .tools.get_values import get_keys_by_value
from .tools.hash_id import hash_id
from .tools.populate_summary import (
    populate_summary_ct,
    populate_summary_mg,
    populate_summary_dx,
    populate_summary_rf,
)
from remapp.tools.background import run_in_background
from .tools.send_high_dose_alert_emails import send_rf_high_dose_alert_email
from remapp.tools.background import terminate_background
from .version import __version__, __docs_version__


os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"

logger = logging.getLogger(__name__)


@login_required
def study_delete(request, pk, template_name="remapp/study_confirm_delete.html"):
    study = get_object_or_404(GeneralStudyModuleAttr, pk=pk)

    if request.method == "POST":
        if request.user.groups.filter(name="admingroup"):
            study.delete()
            messages.success(request, "Study deleted")
        else:
            messages.error(
                request, "Only members of the admingroup are allowed to delete studies"
            )
        return redirect(request.POST["return_url"])

    if request.user.groups.filter(name="admingroup"):
        return render(
            request,
            template_name,
            {"exam": study, "return_url": request.META["HTTP_REFERER"]},
        )

    if "HTTP_REFERER" in list(request.META.keys()):
        return redirect(request.META["HTTP_REFERER"])
    else:
        return redirect(reverse_lazy("home"))


def charts_toggle(request):
    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        if request.user.is_authenticated:
            # Create a default userprofile for the user if one doesn't exist
            create_user_profile(
                sender=request.user, instance=request.user, created=True
            )
            user_profile = request.user.userprofile

    # Toggle chart plotting
    user_profile.plotCharts = not user_profile.plotCharts
    user_profile.save()
    if request.user.get_full_name():
        name = request.user.get_full_name()
    else:
        name = request.user.get_username()
    if user_profile.plotCharts:
        messages.success(
            request, "Chart plotting has been turned on for {0}".format(name)
        )
        # Redirect to the calling page
        return redirect(request.META["HTTP_REFERER"])
    else:
        messages.warning(
            request, "Chart plotting has been turned off for {0}".format(name)
        )
        # Redirect to the calling page, removing '&plotCharts=on' from the url
        return redirect((request.META["HTTP_REFERER"]).replace("&plotCharts=on", ""))


@login_required
def display_names_view(request):
    try:
        match_on_device_observer_uid = (
            MergeOnDeviceObserverUIDSettings.objects.values_list(
                "match_on_device_observer_uid", flat=True
            )[0]
        )
    except IndexError:
        match_on_device_observer_uid = False
        m = MergeOnDeviceObserverUIDSettings(match_on_device_observer_uid=False)
        m.save()

    if request.method == "POST":
        merge_options_form = MergeOnDeviceObserverUIDForm(request.POST)
        if merge_options_form.is_valid():
            if (
                merge_options_form.cleaned_data["match_on_device_observer_uid"]
                != match_on_device_observer_uid
            ):
                merge_options_settings = MergeOnDeviceObserverUIDSettings.objects.all()[
                    0
                ]
                merge_options_settings.match_on_device_observer_uid = (
                    merge_options_form.cleaned_data["match_on_device_observer_uid"]
                )
                merge_options_settings.save()
                if merge_options_form.cleaned_data["match_on_device_observer_uid"]:
                    messages.info(
                        request,
                        "Display name and Modality type are set automatically based on "
                        "Device Observer UID",
                    )
                else:
                    messages.info(
                        request,
                        "Display name and Modality type are NOT set automatically",
                    )
        return HttpResponseRedirect(reverse_lazy("display_names_view"))

    f = UniqueEquipmentNames.objects.order_by("display_name")

    # if user_defined_modality is filled, we should use this value, otherwise the value of modality type in the
    # general_study module. So we look if the concatenation of the user_defined_modality (empty if not used) and
    # modality_type starts with a specific modality type
    ct_names = f.filter(
        generalequipmentmoduleattr__general_study_module_attributes__modality_type="CT"
    ).distinct()
    mg_names = f.filter(
        generalequipmentmoduleattr__general_study_module_attributes__modality_type="MG"
    ).distinct()
    dx_names = f.filter(
        Q(user_defined_modality="DX")
        | Q(user_defined_modality="dual")
        | (
            Q(user_defined_modality__isnull=True)
            & (
                Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="DX"
                )
                | Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="CR"
                )
                | Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="PX"
                )
            )
        )
    ).distinct()
    rf_names = f.filter(
        Q(user_defined_modality="RF")
        | Q(user_defined_modality="dual")
        | (
            Q(user_defined_modality__isnull=True)
            & Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="RF"
            )
        )
    ).distinct()
    ot_names = f.filter(
        ~Q(user_defined_modality__isnull=True)
        | (
            ~Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="RF"
            )
            & ~Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="MG"
            )
            & ~Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="CT"
            )
            & ~Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="DX"
            )
            & ~Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="CR"
            )
            & ~Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="PX"
            )
        )
    ).distinct()

    admin = {
        "openremversion": __version__,
        "docsversion": __docs_version__,
    }

    merge_options_form = MergeOnDeviceObserverUIDForm(
        {"match_on_device_observer_uid": match_on_device_observer_uid}
    )

    for group in request.user.groups.all():
        admin[group.name] = True

    return_structure = {
        "name_list": f,
        "admin": admin,
        "MergeOptionsForm": merge_options_form,
        "ct_names": ct_names,
        "mg_names": mg_names,
        "dx_names": dx_names,
        "rf_names": rf_names,
        "ot_names": ot_names,
        "modalities": ["CT", "RF", "MG", "DX", "OT"],
    }

    return render(request, "remapp/displaynameview.html", return_structure)


def display_name_gen_hash(eq):
    eq.manufacturer_hash = hash_id(eq.manufacturer)
    eq.institution_name_hash = hash_id(eq.institution_name)
    eq.station_name_hash = hash_id(eq.station_name)
    eq.institutional_department_name_hash = hash_id(eq.institutional_department_name)
    eq.manufacturer_model_name_hash = hash_id(eq.manufacturer_model_name)
    eq.device_serial_number_hash = hash_id(eq.device_serial_number)
    eq.software_versions_hash = hash_id(eq.software_versions)
    eq.gantry_id_hash = hash_id(eq.gantry_id)
    eq.hash_generated = True
    eq.save()


@login_required
def display_name_update(request):
    if request.method == "POST":
        error_message = ""
        new_display_name = request.POST.get("new_display_name")
        new_user_defined_modality = request.POST.get("new_user_defined_modality")
        for pk in request.POST.get("pks").split(","):
            display_name_data = UniqueEquipmentNames.objects.get(pk=int(pk))
            if not display_name_data.hash_generated:
                display_name_gen_hash(display_name_data)
            if new_display_name:
                display_name_data.display_name = new_display_name
            if new_user_defined_modality and (
                not display_name_data.user_defined_modality == new_user_defined_modality
            ):
                # See if change is valid otherwise return validation error
                # Assuming modality is always the same, so we take the first
                try:
                    modality = GeneralStudyModuleAttr.objects.filter(
                        generalequipmentmoduleattr__unique_equipment_name__pk=pk
                    )[0].modality_type
                except:
                    modality = ""
                if modality in {"DX", "CR", "RF", "dual", "OT"}:
                    display_name_data.user_defined_modality = new_user_defined_modality
                    # We can't reimport as new modality type, instead we just change the modality type value
                    if new_user_defined_modality == "dual":
                        status_message = reset_dual(pk=pk)
                        messages.info(request, status_message)
                        display_name_data.save()
                        continue
                    GeneralStudyModuleAttr.objects.filter(
                        generalequipmentmoduleattr__unique_equipment_name__pk=pk
                    ).update(modality_type=new_user_defined_modality)
                elif not modality:
                    error_message = (
                        error_message + "Can't determine modality type for"
                        " " + display_name_data.display_name + ", "
                        "user defined modality type not set.\n"
                    )
                else:
                    error_message = (
                        error_message + "Modality type change is not allowed for"
                        " "
                        + display_name_data.display_name
                        + ", modality "
                        + modality
                        + ". Only changing from DX "
                        "to RF and vice versa is allowed.\n"
                    )
            display_name_data.save()

        if error_message:
            messages.error(request, error_message)
        return HttpResponseRedirect(reverse_lazy("display_names_view"))

    else:
        if request.GET.__len__() == 0:
            return HttpResponseRedirect(reverse_lazy("display_names_view"))

        max_pk = (
            UniqueEquipmentNames.objects.all().order_by("-pk").values_list("pk")[0][0]
        )
        for current_pk in request.GET:
            if int(current_pk) > max_pk:
                return HttpResponseRedirect(reverse_lazy("display_names_view"))

        f = UniqueEquipmentNames.objects.filter(
            pk__in=list(map(int, list(request.GET.values())))
        )

        form = UpdateDisplayNamesForm(
            initial={
                "display_names": [
                    x.encode("utf-8") for x in f.values_list("display_name", flat=True)
                ]
            },
            auto_id=False,
        )

        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }

        for group in request.user.groups.all():
            admin[group.name] = True

        return_structure = {"name_list": f, "admin": admin, "form": form}

    return render(request, "remapp/displaynameupdate.html", return_structure)


def display_name_populate(request):
    """AJAX view to populate the modality tables for the display names view

    :param request: Request object containing modality
    :return: HTML table
    """
    if request.is_ajax():
        data = request.POST
        modality = data.get("modality")
        f = UniqueEquipmentNames.objects.order_by("display_name")
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in request.user.groups.all():
            admin[group.name] = True
        if modality in ["MG", "CT"]:
            name_set = f.filter(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type=modality
            ).distinct()
            dual = False
        elif modality == "DX":
            name_set = f.filter(
                Q(user_defined_modality="DX")
                | Q(user_defined_modality="dual")
                | (
                    Q(user_defined_modality__isnull=True)
                    & (
                        Q(
                            generalequipmentmoduleattr__general_study_module_attributes__modality_type="DX"
                        )
                        | Q(
                            generalequipmentmoduleattr__general_study_module_attributes__modality_type="CR"
                        )
                        | Q(
                            generalequipmentmoduleattr__general_study_module_attributes__modality_type="PX"
                        )
                    )
                )
            ).distinct()
            dual = True
        elif modality == "RF":
            name_set = f.filter(
                Q(user_defined_modality="RF")
                | Q(user_defined_modality="dual")
                | (
                    Q(user_defined_modality__isnull=True)
                    & Q(
                        generalequipmentmoduleattr__general_study_module_attributes__modality_type="RF"
                    )
                )
            ).distinct()
            dual = True
        elif modality == "OT":
            name_set = f.filter(  # ~Q(user_defined_modality__isnull=True) | (
                ~Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="RF"
                )
                & ~Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="MG"
                )
                & ~Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="CT"
                )
                & ~Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="DX"
                )
                & ~Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="CR"
                )
                & ~Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="PX"
                )
            ).distinct()
            dual = False
        else:
            name_set = None
            dual = False
        template = "remapp/displayname-modality.html"
        return render(
            request,
            template,
            {"name_set": name_set, "admin": admin, "modality": modality, "dual": dual},
        )


def display_name_modality_filter(equip_name_pk=None, modality=None):
    """Function to filter the studies to a particular unique_name entry and particular modality.

    :param equip_name_pk: Primary key of entry in unique names table
    :param modality: Modality to filter on
    :return: Reduced queryset of studies, plus count of pre-modality filtered studies for modality OT
    """
    if not equip_name_pk:
        logger.error(
            "Display name modality filter function called without a primary key ID for the unique names table"
        )
        return
    if not modality or modality not in ["CT", "RF", "MG", "DX", "OT"]:
        logger.error(
            "Display name modality filter function called without an appropriate modality specified"
        )
        return

    studies_all = GeneralStudyModuleAttr.objects.filter(
        generalequipmentmoduleattr__unique_equipment_name__pk=equip_name_pk
    )
    count_all = studies_all.count()
    if modality in ["CT", "MG", "RF"]:
        studies = studies_all.filter(modality_type__exact=modality)
    elif modality == "DX":
        studies = studies_all.filter(
            Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type__exact="DX"
            )
            | Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type__exact="CR"
            )
            | Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type__exact="PX"
            )
        )
    else:  # modality == 'OT'
        studies = (
            studies_all.exclude(modality_type__exact="CT")
            .exclude(modality_type__exact="MG")
            .exclude(modality_type__exact="DX")
            .exclude(modality_type__exact="CR")
            .exclude(modality_type__exact="PX")
            .exclude(modality_type__exact="RF")
        )
    return studies, count_all


def display_name_last_date_and_count(request):
    """AJAX view to return the most recent study date associated with an entry in the equipment database along with
    the number of studies

    :param request: Request object containing modality and equipment table ID
    :return: HTML table data element
    """

    if request.is_ajax():
        data = request.POST
        modality = data.get("modality")
        equip_name_pk = data.get("equip_name_pk")
        latest = None
        studies, count_all = display_name_modality_filter(
            equip_name_pk=equip_name_pk, modality=modality
        )
        count = studies.count()
        if count:
            latest = studies.latest("study_date").study_date
        template_latest = "remapp/displayname-last-date.html"
        template_count = "remapp/displayname-count.html"
        count_html = render_to_string(
            template_count, {"count": count, "count_all": count_all}, request=request
        )
        latest_html = render_to_string(
            template_latest, {"latest": latest}, request=request
        )
        return_html = {"count_html": count_html, "latest_html": latest_html}
        html_dict = json.dumps(return_html)
        return HttpResponse(html_dict, content_type="application/json")


@login_required
def review_summary_list(request, equip_name_pk=None, modality=None, delete_equip=None):
    """View to list partial and broken studies

    :param request:
    :param equip_name_pk: UniqueEquipmentNames primary key
    :param modality: modality to filter by
    :return:
    """
    if not equip_name_pk:
        logger.error("Attempt to load review_summary_list without equip_name_pk")
        messages.error(
            request,
            "Partial and broken imports can only be reviewed with the correct "
            "link from the display name page",
        )
        return HttpResponseRedirect(reverse_lazy("display_names_view"))

    if not request.user.groups.filter(name="admingroup"):
        messages.error(
            request,
            "You are not in the administrator group - please contact your administrator",
        )
        return redirect(reverse_lazy("display_names_view"))

    if request.method == "GET":
        equipment = UniqueEquipmentNames.objects.get(pk=equip_name_pk)
        studies_list, count_all = display_name_modality_filter(
            equip_name_pk=equip_name_pk, modality=modality
        )
        paginator = Paginator(studies_list, 25)
        page = request.GET.get("page")
        try:
            studies = paginator.page(page)
        except PageNotAnInteger:
            studies = paginator.page(1)
        except EmptyPage:
            studies = paginator.page(paginator.num_pages)

        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }

        for group in request.user.groups.all():
            admin[group.name] = True

        template = "remapp/review_summary_list.html"
        return render(
            request,
            template,
            {
                "modality": modality,
                "equipment": equipment,
                "equip_name_pk": equip_name_pk,
                "studies": studies,
                "studies_count": studies_list.count(),
                "count_all": count_all,
                "admin": admin,
            },
        )

    if (
        request.method == "POST"
        and request.user.groups.filter(name="admingroup")
        and equip_name_pk
        and modality
    ):
        delete_equip = bool(request.POST["delete_equip"] == "True")
        if not delete_equip:
            studies, count_all = display_name_modality_filter(
                equip_name_pk=equip_name_pk, modality=modality
            )
            studies.delete()
            messages.info(request, "Studies deleted")
            return redirect(
                reverse_lazy(
                    "review_summary_list",
                    kwargs={"equip_name_pk": equip_name_pk, "modality": modality},
                )
            )
        else:
            studies, count_all = display_name_modality_filter(
                equip_name_pk=equip_name_pk, modality=modality
            )
            if count_all > studies.count():
                messages.warning(
                    request,
                    "Can't delete table entry - non-{0} studies are associated with it".format(
                        modality
                    ),
                )
                logger.warning(
                    "Can't delete table entry - non-{0} studies are associated with it".format(
                        modality
                    )
                )
                return redirect(
                    reverse_lazy(
                        "review_summary_list",
                        kwargs={"equip_name_pk": equip_name_pk, "modality": modality},
                    )
                )
            else:
                studies.delete()
                UniqueEquipmentNames.objects.get(pk=equip_name_pk).delete()
                messages.info(request, "Studies and equipment name table entry deleted")
                return redirect(reverse_lazy("display_names_view"))
    else:
        messages.error(request, "Incorrect attempt to delete studies.")
        return redirect(
            reverse_lazy(
                "review_summary_list",
                kwargs={"equip_name_pk": equip_name_pk, "modality": modality},
            )
        )


@login_required
def review_studies_delete(request):
    """AJAX function to replace Delete button with delete form for associated studies

    :param request:
    :return:
    """
    if request.is_ajax() and request.user.groups.filter(name="admingroup"):
        data = request.POST
        template = "remapp/review_studies_delete_button.html"
        return render(
            request,
            template,
            {
                "delete_equip": False,
                "modality": data["modality"],
                "equip_name_pk": data["equip_name_pk"],
            },
        )


@login_required
def review_studies_equip_delete(request):
    """AJAX function to replace Delete button with delete form for equipment table entry and studies

    :param request:
    :return:
    """
    if request.is_ajax() and request.user.groups.filter(name="admingroup"):
        data = request.POST
        template = "remapp/review_studies_delete_button.html"
        return render(
            request,
            template,
            {
                "delete_equip": True,
                "modality": data["modality"],
                "equip_name_pk": data["equip_name_pk"],
            },
        )


@login_required
def review_failed_studies_delete(request):
    """AJAX function to replace Delete button with delete form for studies without ubique_equipment_name table

    :param request:
    :return:
    """
    if request.is_ajax() and request.user.groups.filter(name="admingroup"):
        data = request.POST
        template = "remapp/review_studies_delete_button.html"
        return render(
            request,
            template,
            {
                "delete_equip": False,
                "modality": data["modality"],
                "equip_name_pk": "n/a",
            },
        )


def reset_dual(pk=None):
    """function to set modality to DX or RF depending on presence of fluoro information.

    :param pk: Unique equipment names table prmary key
    :return: status message
    """

    if not pk:
        logger.error("Reset dual called with no primary key")
        return

    studies = GeneralStudyModuleAttr.objects.filter(
        generalequipmentmoduleattr__unique_equipment_name__pk=pk
    )
    not_dx_rf_cr = (
        studies.exclude(modality_type__exact="DX")
        .exclude(modality_type__exact="RF")
        .exclude(modality_type__exact="CR")
        .exclude(modality_type__exact="PX")
    )
    message_start = (
        "Reprocessing dual for {0}. Number of studies is {1}, of which {2} are "
        "DX, {3} are CR, {4} are PX, {5} are RF and {6} are something else before processing,".format(  # pylint: disable=line-too-long
            studies[0]
            .generalequipmentmoduleattr_set.get()
            .unique_equipment_name.display_name,
            studies.count(),
            studies.filter(modality_type__exact="DX").count(),
            studies.filter(modality_type__exact="CR").count(),
            studies.filter(modality_type__exact="PX").count(),
            studies.filter(modality_type__exact="RF").count(),
            not_dx_rf_cr.count(),
        )
    )

    logger.debug(message_start)

    for study in studies:
        try:
            projection_xray_dose = study.projectionxrayradiationdose_set.get()
            if projection_xray_dose.acquisition_device_type_cid:
                device_type = (
                    projection_xray_dose.acquisition_device_type_cid.code_meaning
                )
                if "Fluoroscopy-Guided" in device_type:
                    study.modality_type = "RF"
                    study.save()
                    continue
                elif any(x in device_type for x in ["Integrated", "Cassette-based"]):
                    study.modality_type = "DX"
                    study.save()
                    continue
            try:
                accum_xray_dose = projection_xray_dose.accumxraydose_set.order_by("pk")[
                    0
                ]  # consider just first plane
                try:
                    accum_fluoro_proj = accum_xray_dose.accumprojxraydose_set.get()
                    if (
                        accum_fluoro_proj.fluoro_dose_area_product_total
                        or accum_fluoro_proj.total_fluoro_time
                    ):
                        study.modality_type = "RF"
                        study.save()
                        continue
                    else:
                        study.modality_type = "DX"
                        study.save()
                        continue
                except ObjectDoesNotExist:
                    try:
                        if accum_xray_dose.accumintegratedprojradiogdose_set.get():
                            study.modality_type = "DX"
                            study.save()
                            continue
                    except ObjectDoesNotExist:
                        study.modality_type = "OT"
                        study.save()
                        logger.debug(
                            "Unable to reprocess study - no device type or accumulated data to go on. "
                            "Modality set to OT."
                        )
                study.modality_type = "OT"
                study.save()
                logger.debug(
                    "Unable to reprocess study - no device type or accumulated data to go on. Modality set to OT."
                )
            except ObjectDoesNotExist:
                study.modality_type = "OT"
                study.save()
                logger.debug(
                    "Unable to reprocess study - no device type or accumulated data to go on. Modality set to OT."
                )
        except ObjectDoesNotExist:
            study.modality_type = "OT"
            study.save()
            logger.debug(
                "Unable to reprocess study - no device type or accumulated data to go on. Modality set to OT."
            )

    not_dx_rf_cr = (
        studies.exclude(modality_type__exact="DX")
        .exclude(modality_type__exact="RF")
        .exclude(modality_type__exact="CR")
        .exclude(modality_type__exact="PX")
    )
    message_finish = "and after processing  {0} are DX, {1} are CR, {2} are PX, {3} are RF and {4} are something else".format(
        studies.filter(modality_type="DX").count(),
        studies.filter(modality_type="CR").count(),
        studies.filter(modality_type="PX").count(),
        studies.filter(modality_type="RF").count(),
        not_dx_rf_cr.count(),
    )
    logger.debug(message_finish)
    return " ".join([message_start, message_finish])


@login_required
def reprocess_dual(request, pk=None):
    """View to reprocess the studies from a modality that produces planar radiography and fluoroscopy to recategorise
    them to DX or RF.

    :param request: Request object
    :return: Redirect back to display names view
    """

    if not request.user.groups.filter(name="admingroup"):
        messages.error(
            request,
            "You are not in the administrator group - please contact your administrator",
        )
        return redirect(reverse_lazy("display_names_view"))

    if request.method == "GET" and pk:
        status_message = reset_dual(pk=pk)
        messages.info(request, status_message)

    return HttpResponseRedirect(reverse_lazy("display_names_view"))


def _get_review_study_data(study):
    """Get study data common to normal review and failed study review

    :param study: GeneralStudyModuleAttr object
    :return: Dict of study data
    """
    study_data = {
        "study_date": study.study_date,
        "study_time": study.study_time,
        "accession_number": study.accession_number,
        "study_description": study.study_description,
    }
    try:
        patient = study.patientmoduleattr_set.get()
        study_data["patientmoduleattr"] = "Yes"
        if patient.not_patient_indicator:
            study_data["patientmoduleattr"] += "<br>?not patient"
    except ObjectDoesNotExist:
        study_data["patientmoduleattr"] = "Missing"
    try:
        patientstudymoduleattr = study.patientstudymoduleattr_set.get()
        age = patientstudymoduleattr.patient_age_decimal
        if age:
            study_data["patientstudymoduleattr"] = "Yes. Age {0:.1f}".format(
                patientstudymoduleattr.patient_age_decimal
            )
        else:
            study_data["patientstudymoduleattr"] = "Yes."
    except ObjectDoesNotExist:
        study_data["patientstudymoduleattr"] = "Missing"
    try:
        ctradiationdose = study.ctradiationdose_set.get()
        study_data["ctradiationdose"] = "Yes"
        try:
            ctaccumulateddosedata = ctradiationdose.ctaccumulateddosedata_set.get()
            num_events = ctaccumulateddosedata.total_number_of_irradiation_events
            study_data["ctaccumulateddosedata"] = "Yes, {0} events".format(num_events)
        except ObjectDoesNotExist:
            study_data["ctaccumulateddosedata"] = ""
        try:
            ctirradiationeventdata_set = (
                ctradiationdose.ctirradiationeventdata_set.order_by("pk")
            )

            study_data["cteventdata"] = "{0} events.<br>".format(
                ctirradiationeventdata_set.count()
            )
            for index, event in enumerate(ctirradiationeventdata_set):
                if event.acquisition_protocol:
                    protocol = event.acquisition_protocol
                else:
                    protocol = ""
                if event.dlp:
                    study_data[
                        "cteventdata"
                    ] += "e{0}: {1} {2:.2f}&nbsp;mGycm<br>".format(
                        index, protocol, event.dlp
                    )
                else:
                    study_data["cteventdata"] += "e{0}: {1}<br>".format(index, protocol)
        except ObjectDoesNotExist:
            study_data["cteventdata"] = ""
    except ObjectDoesNotExist:
        study_data["ctradiationdose"] = ""
        study_data["ctaccumulateddosedata"] = ""
        study_data["cteventdata"] = ""
    try:
        projectionxraydata = study.projectionxrayradiationdose_set.get()
        study_data["projectionxraydata"] = "Yes"
        try:
            accumxraydose_set = projectionxraydata.accumxraydose_set.order_by("pk")
            accumxraydose_set_count = accumxraydose_set.count()
            if accumxraydose_set_count == 1:
                study_data["accumxraydose"] = "Yes"
            elif accumxraydose_set_count:
                study_data["accumxraydose"] = "{0} present".format(
                    accumxraydose_set_count
                )
            else:
                study_data["accumxraydose"] = ""
            try:
                accumfluoroproj = {}
                study_data["accumfluoroproj"] = ""
                for index, accumxraydose in enumerate(accumxraydose_set):
                    accumfluoroproj[index] = accumxraydose.accumprojxraydose_set.get()
                    study_data["accumfluoroproj"] += "P{0} ".format(index + 1)
                    if accumfluoroproj[index].fluoro_dose_area_product_total:
                        study_data["accumfluoroproj"] += (
                            "Total fluoro DA: {0:.2f}&nbsp;cGy.cm<sup>2</sup>"
                            "; ".format(accumfluoroproj[index].fluoro_gym2_to_cgycm2())
                        )
                    if accumfluoroproj[index].acquisition_dose_area_product_total:
                        study_data[
                            "accumfluoroproj"
                        ] += "Acq: {0:.2f}&nbsp;cGy.cm<sup>2</sup>. ".format(
                            accumfluoroproj[index].acq_gym2_to_cgycm2()
                        )
            except ObjectDoesNotExist:
                study_data["accumfluoroproj"] = ""
            try:
                accummammo_set = accumxraydose_set[
                    0
                ].accummammographyxraydose_set.order_by("pk")
                if accummammo_set.count() == 0:
                    study_data["accummammo"] = ""
                else:
                    study_data["accummammo"] = ""
                    for accummammo in accummammo_set:
                        study_data["accummammo"] += "{0}: {1:.3f}&nbsp;mGy".format(
                            accummammo.laterality,
                            accummammo.accumulated_average_glandular_dose,
                        )
            except ObjectDoesNotExist:
                study_data["accummammo"] = ""
            try:
                accumcassproj = {}
                study_data["accumcassproj"] = ""
                for index, accumxraydose in enumerate(accumxraydose_set):
                    accumcassproj[
                        index
                    ] = accumxraydose.accumcassettebsdprojradiogdose_set.get()
                    study_data["accumcassproj"] += "Number of frames {0}".format(
                        accumcassproj[index].total_number_of_radiographic_frames
                    )
            except ObjectDoesNotExist:
                study_data["accumcassproj"] = ""
            try:
                accumproj = {}
                study_data["accumproj"] = ""
                for index, accumxraydose in enumerate(accumxraydose_set):
                    accumproj[
                        index
                    ] = accumxraydose.accumintegratedprojradiogdose_set.get()
                    study_data[
                        "accumproj"
                    ] += "DAP total {0:.2f}&nbsp;cGy.cm<sup>2</sup> ".format(
                        accumproj[index].convert_gym2_to_cgycm2()
                    )
            except ObjectDoesNotExist:
                study_data["accumproj"] = ""
        except ObjectDoesNotExist:
            study_data["accumxraydose"] = ""
            study_data["accumfluoroproj"] = ""
            study_data["accummammo"] = ""
            study_data["accumcassproj"] = ""
            study_data["accumproj"] = ""
        try:
            study_data["eventdetector"] = ""
            study_data["eventsource"] = ""
            study_data["eventmech"] = ""
            irradevent_set = projectionxraydata.irradeventxraydata_set.order_by("pk")
            irradevent_set_count = irradevent_set.count()
            if irradevent_set_count == 1:
                study_data["irradevent"] = "{0} event. ".format(irradevent_set_count)
            else:
                study_data["irradevent"] = "{0} events. <br>".format(
                    irradevent_set_count
                )
            for index, irradevent in enumerate(irradevent_set):
                if index == 4:
                    study_data["irradevent"] += "...etc"
                    study_data["eventdetector"] += "...etc"
                    study_data["eventsource"] += "...etc"
                    study_data["eventmech"] += "...etc"
                    break
                if irradevent.dose_area_product:
                    study_data[
                        "irradevent"
                    ] += "e{0}: {1} {2:.2f}&nbsp;cGy.cm<sup>2</sup> <br>".format(
                        index + 1,
                        irradevent.acquisition_protocol,
                        irradevent.convert_gym2_to_cgycm2(),
                    )
                elif irradevent.entrance_exposure_at_rp:
                    study_data["irradevent"] += "RP dose {0}: {1:.2f} mGy  <br>".format(
                        index + 1, irradevent.entrance_exposure_at_rp
                    )
                try:
                    eventdetector = irradevent.irradeventxraydetectordata_set.get()
                    if eventdetector.exposure_index:
                        study_data[
                            "eventdetector"
                        ] += "e{0}: EI&nbsp;{1:.1f},<br>".format(
                            index + 1, eventdetector.exposure_index
                        )
                    else:
                        study_data["eventdetector"] += "e{0} present,<br>".format(
                            index + 1
                        )
                except ObjectDoesNotExist:
                    study_data["eventdetector"] += ""
                try:
                    eventsource = irradevent.irradeventxraysourcedata_set.get()
                    if eventsource.dose_rp:
                        study_data[
                            "eventsource"
                        ] += "e{0} RP Dose {1:.3f}&nbsp;mGy,<br>".format(
                            index + 1, eventsource.convert_gy_to_mgy()
                        )
                    elif eventsource.average_glandular_dose:
                        study_data[
                            "eventsource"
                        ] += "e{0} AGD {1:.2f}&nbsp;mGy,<br>".format(
                            index + 1, eventsource.average_glandular_dose
                        )
                    else:
                        study_data["eventsource"] += "e{0} present,<br>".format(
                            index + 1
                        )
                except ObjectDoesNotExist:
                    study_data["eventsource"] += ""
                try:
                    eventmech = irradevent.irradeventxraymechanicaldata_set.get()
                    if eventmech.positioner_primary_angle:
                        study_data["eventmech"] += "e{0} {1:.1f}&deg;<br>".format(
                            index + 1, eventmech.positioner_primary_angle
                        )
                    else:
                        study_data["eventmech"] += "e{0} present,<br>".format(index + 1)
                except ObjectDoesNotExist:
                    study_data["eventmech"] = ""
        except ObjectDoesNotExist:
            study_data["irradevent"] = ""
    except ObjectDoesNotExist:
        study_data["projectionxraydata"] = ""
        study_data["accumxraydose"] = ""
        study_data["accumfluoroproj"] = ""
        study_data["accummammo"] = ""
        study_data["accumcassproj"] = ""
        study_data["accumproj"] = ""
        study_data["irradevent"] = ""
        study_data["eventdetector"] = ""
        study_data["eventdetector"] = ""
        study_data["eventsource"] = ""
        study_data["eventmech"] = ""
        study_data["eventmech"] = ""
    return study_data


def review_study_details(request):
    """AJAX function to populate row in table with details of study for review

    :param request: Request object containing study pk
    :return: HTML row data
    """

    if request.is_ajax():
        data = request.POST
        study_pk = data.get("study_pk")
        study = GeneralStudyModuleAttr.objects.get(pk__exact=study_pk)
        study_data = _get_review_study_data(study)
        template = "remapp/review_study.html"
        return render(request, template, study_data)


def review_failed_study_details(request):
    """AJAX function to populate row in table with details of study for review

    :param request: Request object containing study pk
    :return: HTML row data
    """

    if request.is_ajax():
        data = request.POST
        study_pk = data.get("study_pk")
        study = GeneralStudyModuleAttr.objects.get(pk__exact=study_pk)
        study_data = _get_review_study_data(study)

        try:
            equipment = study.generalequipmentmoduleattr_set.get()
            study_data["station_name"] = equipment.station_name
            study_data["manufacturer"] = equipment.manufacturer
            study_data["manufacturer_model_name"] = equipment.manufacturer_model_name
            study_data["institution_name"] = equipment.institution_name
            study_data[
                "institution_department_name"
            ] = equipment.institutional_department_name
            study_data["device_serial_number"] = equipment.device_serial_number
            study_data["equipmentattr"] = True
        except ObjectDoesNotExist:
            study_data["equipmentattr"] = False
            study_data["station_name"] = ""
            study_data["manufacturer"] = ""
            study_data["manufacturer_model_name"] = ""
            study_data["institution_name"] = ""
            study_data["institution_department_name"] = ""
            study_data["device_serial_number"] = ""

        template = "remapp/review_failed_study.html"
        return render(request, template, study_data)


def _get_broken_studies(modality=None):
    """Filter studies with no unique_equipment_name table entry
    :param modality: modality to filter by
    :return: Query filter of studies
    """
    if modality == "DX":
        all_mod = GeneralStudyModuleAttr.objects.filter(
            Q(modality_type__exact="DX")
            | Q(modality_type__exact="CR")
            | Q(modality_type__exact="PX")
        )
    else:
        all_mod = GeneralStudyModuleAttr.objects.filter(modality_type__exact=modality)

    return all_mod.filter(
        generalequipmentmoduleattr__unique_equipment_name__display_name__isnull=True
    )


def failed_list_populate(request):
    """View for failed import section of display name view

    :return: render request with modality specific numbers of studies
    """

    if request.is_ajax():
        failed = {}
        for modality in ["CT", "RF", "MG", "DX"]:
            failed[modality] = _get_broken_studies(modality).count()
        template = "remapp/failed_summary_list.html"
        return render(request, template, {"failed": failed})


@login_required
def review_failed_imports(request, modality=None):
    """View to list 'failed import' studies

    :param request:
    :param modality: modality to filter by
    :return:
    """
    if not modality in ["CT", "RF", "MG", "DX"]:
        logger.error("Attempt to load review_failed_imports without suitable modality")
        messages.error(
            request,
            "Failed study imports can only be reviewed with the correct "
            "link from the display name page",
        )
        return HttpResponseRedirect(reverse_lazy("display_names_view"))

    if not request.user.groups.filter(name="admingroup"):
        messages.error(
            request,
            "You are not in the administrator group - please contact your administrator",
        )
        return redirect(reverse_lazy("display_names_view"))

    if request.method == "GET":
        broken_studies = _get_broken_studies(modality)

        paginator = Paginator(broken_studies, 25)
        page = request.GET.get("page")
        try:
            studies = paginator.page(page)
        except PageNotAnInteger:
            studies = paginator.page(1)
        except EmptyPage:
            studies = paginator.page(paginator.num_pages)

        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }

        for group in request.user.groups.all():
            admin[group.name] = True

        template = "remapp/review_failed_imports.html"
        return render(
            request,
            template,
            {
                "modality": modality,
                "studies": studies,
                "studies_count": broken_studies.count(),
                "admin": admin,
            },
        )

    if (
        request.method == "POST"
        and request.user.groups.filter(name="admingroup")
        and modality
    ):
        broken_studies = _get_broken_studies(modality)
        broken_studies.delete()
        messages.info(request, "Studies deleted")
        return redirect(
            reverse_lazy("review_failed_imports", kwargs={"modality": modality})
        )
    else:
        messages.error(request, "Incorrect attempt to delete studies.")
        return redirect(
            reverse_lazy("review_failed_imports", kwargs={"modality": modality})
        )


@login_required
def chart_options_view(request):
    if request.method == "POST":
        general_form = GeneralChartOptionsDisplayForm(request.POST)
        ct_form = CTChartOptionsDisplayForm(request.POST)
        dx_form = DXChartOptionsDisplayForm(request.POST)
        rf_form = RFChartOptionsDisplayForm(request.POST)
        mg_form = MGChartOptionsDisplayForm(request.POST)
        if (
            general_form.is_valid()
            and ct_form.is_valid()
            and dx_form.is_valid()
            and rf_form.is_valid()
            and mg_form.is_valid()
        ):
            try:
                # See if the user has plot settings in userprofile
                user_profile = request.user.userprofile
            except ObjectDoesNotExist:
                # Create a default userprofile for the user if one doesn't exist
                create_user_profile(
                    sender=request.user, instance=request.user, created=True
                )
                user_profile = request.user.userprofile

            user_profile.plotThemeChoice = general_form.cleaned_data["plotThemeChoice"]
            user_profile.plotColourMapChoice = general_form.cleaned_data[
                "plotColourMapChoice"
            ]
            user_profile.plotFacetColWrapVal = general_form.cleaned_data[
                "plotFacetColWrapVal"
            ]
            user_profile.plotHistogramBins = general_form.cleaned_data[
                "plotHistogramBins"
            ]
            user_profile.plotHistogramGlobalBins = general_form.cleaned_data[
                "plotHistogramGlobalBins"
            ]
            user_profile.plotCaseInsensitiveCategories = general_form.cleaned_data[
                "plotCaseInsensitiveCategories"
            ]
            user_profile.plotRemoveCategoryWhitespacePadding = (
                general_form.cleaned_data["plotRemoveCategoryWhitespacePadding"]
            )

            user_profile.plotLabelCharWrap = general_form.cleaned_data[
                "plotLabelCharWrap"
            ]

            set_common_chart_options(general_form, user_profile)

            set_average_chart_options(general_form, user_profile)

            set_ct_chart_options(ct_form, user_profile)

            set_dx_chart_options(dx_form, user_profile)

            set_rf_chart_options(rf_form, user_profile)

            set_mg_chart_options(mg_form, user_profile)

            user_profile.save()

        messages.success(request, "Chart options have been updated")

    admin = {
        "openremversion": __version__,
        "docsversion": __docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    try:
        # See if the user has plot settings in userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile

    average_choices = required_average_choices(user_profile)

    general_form_data = {
        "plotCharts": user_profile.plotCharts,
        "plotAverageChoice": average_choices,
        "plotInitialSortingDirection": user_profile.plotInitialSortingDirection,
        "plotSeriesPerSystem": user_profile.plotSeriesPerSystem,
        "plotHistogramBins": user_profile.plotHistogramBins,
        "plotHistograms": user_profile.plotHistograms,
        "plotHistogramGlobalBins": user_profile.plotHistogramGlobalBins,
        "plotCaseInsensitiveCategories": user_profile.plotCaseInsensitiveCategories,
        "plotRemoveCategoryWhitespacePadding": user_profile.plotRemoveCategoryWhitespacePadding,
        "plotLabelCharWrap": user_profile.plotLabelCharWrap,
        "plotThemeChoice": user_profile.plotThemeChoice,
        "plotColourMapChoice": user_profile.plotColourMapChoice,
        "plotFacetColWrapVal": user_profile.plotFacetColWrapVal,
    }

    ct_acquisition_types = required_ct_acquisition_types(user_profile)

    ct_form_data = initialise_ct_form_data(ct_acquisition_types, user_profile)

    dx_form_data = initialise_dx_form_data(user_profile)

    rf_form_data = initialise_rf_form_data(user_profile)

    mg_form_data = initialise_mg_form_data(user_profile)

    general_chart_options_form = GeneralChartOptionsDisplayForm(general_form_data)
    ct_chart_options_form = CTChartOptionsDisplayForm(ct_form_data)
    dx_chart_options_form = DXChartOptionsDisplayForm(dx_form_data)
    rf_chart_options_form = RFChartOptionsDisplayForm(rf_form_data)
    mg_chart_options_form = MGChartOptionsDisplayForm(mg_form_data)

    return_structure = {
        "admin": admin,
        "GeneralChartOptionsForm": general_chart_options_form,
        "CTChartOptionsForm": ct_chart_options_form,
        "DXChartOptionsForm": dx_chart_options_form,
        "RFChartOptionsForm": rf_chart_options_form,
        "MGChartOptionsForm": mg_chart_options_form,
    }

    return render(request, "remapp/displaychartoptions.html", return_structure)


def set_common_chart_options(general_form, user_profile):
    user_profile.plotCharts = general_form.cleaned_data["plotCharts"]
    user_profile.plotGroupingChoice = general_form.cleaned_data["plotGrouping"]
    user_profile.plotSeriesPerSystem = general_form.cleaned_data["plotSeriesPerSystem"]
    user_profile.plotHistograms = general_form.cleaned_data["plotHistograms"]
    user_profile.plotInitialSortingDirection = general_form.cleaned_data[
        "plotInitialSortingDirection"
    ]


def set_rf_chart_options(rf_form, user_profile):
    user_profile.plotRFStudyPerDayAndHour = rf_form.cleaned_data[
        "plotRFStudyPerDayAndHour"
    ]
    user_profile.plotRFStudyFreq = rf_form.cleaned_data["plotRFStudyFreq"]
    user_profile.plotRFStudyDAP = rf_form.cleaned_data["plotRFStudyDAP"]
    user_profile.plotRFStudyDAPOverTime = rf_form.cleaned_data["plotRFStudyDAPOverTime"]
    user_profile.plotRFRequestFreq = rf_form.cleaned_data["plotRFRequestFreq"]
    user_profile.plotRFRequestDAP = rf_form.cleaned_data["plotRFRequestDAP"]
    user_profile.plotRFRequestDAPOverTime = rf_form.cleaned_data[
        "plotRFRequestDAPOverTime"
    ]
    user_profile.plotRFOverTimePeriod = rf_form.cleaned_data["plotRFOverTimePeriod"]
    user_profile.plotRFSplitByPhysician = rf_form.cleaned_data["plotRFSplitByPhysician"]
    user_profile.plotRFInitialSortingChoice = rf_form.cleaned_data[
        "plotRFInitialSortingChoice"
    ]


def initialise_rf_form_data(user_profile):
    rf_form_data = {
        "plotRFStudyPerDayAndHour": user_profile.plotRFStudyPerDayAndHour,
        "plotRFStudyFreq": user_profile.plotRFStudyFreq,
        "plotRFStudyDAP": user_profile.plotRFStudyDAP,
        "plotRFStudyDAPOverTime": user_profile.plotRFStudyDAPOverTime,
        "plotRFRequestFreq": user_profile.plotRFRequestFreq,
        "plotRFRequestDAP": user_profile.plotRFRequestDAP,
        "plotRFRequestDAPOverTime": user_profile.plotRFRequestDAPOverTime,
        "plotRFOverTimePeriod": user_profile.plotRFOverTimePeriod,
        "plotRFSplitByPhysician": user_profile.plotRFSplitByPhysician,
        "plotRFInitialSortingChoice": user_profile.plotRFInitialSortingChoice,
    }
    return rf_form_data


def set_mg_chart_options(mg_form, user_profile):
    user_profile.plotMGacquisitionFreq = mg_form.cleaned_data["plotMGacquisitionFreq"]
    user_profile.plotMGaverageAGD = mg_form.cleaned_data["plotMGaverageAGD"]
    user_profile.plotMGaverageAGDvsThickness = mg_form.cleaned_data[
        "plotMGaverageAGDvsThickness"
    ]
    user_profile.plotMGAcquisitionAGDOverTime = mg_form.cleaned_data[
        "plotMGAcquisitionAGDOverTime"
    ]
    user_profile.plotMGAGDvsThickness = mg_form.cleaned_data["plotMGAGDvsThickness"]
    user_profile.plotMGkVpvsThickness = mg_form.cleaned_data["plotMGkVpvsThickness"]
    user_profile.plotMGmAsvsThickness = mg_form.cleaned_data["plotMGmAsvsThickness"]
    user_profile.plotMGStudyPerDayAndHour = mg_form.cleaned_data[
        "plotMGStudyPerDayAndHour"
    ]
    user_profile.plotMGOverTimePeriod = mg_form.cleaned_data["plotMGOverTimePeriod"]
    user_profile.plotMGInitialSortingChoice = mg_form.cleaned_data[
        "plotMGInitialSortingChoice"
    ]


def initialise_mg_form_data(user_profile):
    mg_form_data = {
        "plotMGacquisitionFreq": user_profile.plotMGacquisitionFreq,
        "plotMGaverageAGD": user_profile.plotMGaverageAGD,
        "plotMGaverageAGDvsThickness": user_profile.plotMGaverageAGDvsThickness,
        "plotMGAcquisitionAGDOverTime": user_profile.plotMGAcquisitionAGDOverTime,
        "plotMGAGDvsThickness": user_profile.plotMGAGDvsThickness,
        "plotMGkVpvsThickness": user_profile.plotMGkVpvsThickness,
        "plotMGmAsvsThickness": user_profile.plotMGmAsvsThickness,
        "plotMGStudyPerDayAndHour": user_profile.plotMGStudyPerDayAndHour,
        "plotMGOverTimePeriod": user_profile.plotMGOverTimePeriod,
        "plotMGInitialSortingChoice": user_profile.plotMGInitialSortingChoice,
    }
    return mg_form_data


def set_dx_chart_options(dx_form, user_profile):
    user_profile.plotDXAcquisitionMeanDAP = dx_form.cleaned_data[
        "plotDXAcquisitionMeanDAP"
    ]
    user_profile.plotDXAcquisitionFreq = dx_form.cleaned_data["plotDXAcquisitionFreq"]
    user_profile.plotDXStudyMeanDAP = dx_form.cleaned_data["plotDXStudyMeanDAP"]
    user_profile.plotDXStudyFreq = dx_form.cleaned_data["plotDXStudyFreq"]
    user_profile.plotDXRequestMeanDAP = dx_form.cleaned_data["plotDXRequestMeanDAP"]
    user_profile.plotDXRequestFreq = dx_form.cleaned_data["plotDXRequestFreq"]
    user_profile.plotDXAcquisitionMeankVp = dx_form.cleaned_data[
        "plotDXAcquisitionMeankVp"
    ]
    user_profile.plotDXAcquisitionMeanmAs = dx_form.cleaned_data[
        "plotDXAcquisitionMeanmAs"
    ]
    user_profile.plotDXStudyPerDayAndHour = dx_form.cleaned_data[
        "plotDXStudyPerDayAndHour"
    ]
    user_profile.plotDXAcquisitionMeankVpOverTime = dx_form.cleaned_data[
        "plotDXAcquisitionMeankVpOverTime"
    ]
    user_profile.plotDXAcquisitionMeanmAsOverTime = dx_form.cleaned_data[
        "plotDXAcquisitionMeanmAsOverTime"
    ]
    user_profile.plotDXAcquisitionMeanDAPOverTime = dx_form.cleaned_data[
        "plotDXAcquisitionMeanDAPOverTime"
    ]
    user_profile.plotDXAcquisitionMeanDAPOverTimePeriod = dx_form.cleaned_data[
        "plotDXAcquisitionMeanDAPOverTimePeriod"
    ]
    user_profile.plotDXAcquisitionDAPvsMass = dx_form.cleaned_data[
        "plotDXAcquisitionDAPvsMass"
    ]
    user_profile.plotDXStudyDAPvsMass = dx_form.cleaned_data["plotDXStudyDAPvsMass"]
    user_profile.plotDXRequestDAPvsMass = dx_form.cleaned_data["plotDXRequestDAPvsMass"]
    user_profile.plotDXInitialSortingChoice = dx_form.cleaned_data[
        "plotDXInitialSortingChoice"
    ]


def initialise_dx_form_data(user_profile):
    dx_form_data = {
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
        "plotDXInitialSortingChoice": user_profile.plotDXInitialSortingChoice,
    }
    return dx_form_data


def set_average_chart_options(general_form, user_profile):
    if CommonVariables.MEAN in general_form.cleaned_data["plotAverageChoice"]:
        user_profile.plotMean = True
    else:
        user_profile.plotMean = False
    if CommonVariables.MEDIAN in general_form.cleaned_data["plotAverageChoice"]:
        user_profile.plotMedian = True
    else:
        user_profile.plotMedian = False
    if CommonVariables.BOXPLOT in general_form.cleaned_data["plotAverageChoice"]:
        user_profile.plotBoxplots = True
    else:
        user_profile.plotBoxplots = False


def set_ct_chart_options(ct_form, user_profile):
    user_profile.plotCTAcquisitionMeanDLP = ct_form.cleaned_data[
        "plotCTAcquisitionMeanDLP"
    ]
    user_profile.plotCTAcquisitionMeanCTDI = ct_form.cleaned_data[
        "plotCTAcquisitionMeanCTDI"
    ]
    user_profile.plotCTAcquisitionFreq = ct_form.cleaned_data["plotCTAcquisitionFreq"]
    user_profile.plotCTAcquisitionCTDIvsMass = ct_form.cleaned_data[
        "plotCTAcquisitionCTDIvsMass"
    ]
    user_profile.plotCTAcquisitionDLPvsMass = ct_form.cleaned_data[
        "plotCTAcquisitionDLPvsMass"
    ]
    user_profile.plotCTAcquisitionCTDIOverTime = ct_form.cleaned_data[
        "plotCTAcquisitionCTDIOverTime"
    ]
    user_profile.plotCTAcquisitionDLPOverTime = ct_form.cleaned_data[
        "plotCTAcquisitionDLPOverTime"
    ]
    if (
        CommonVariables.CT_SEQUENCED_ACQUISITION_TYPE
        in ct_form.cleaned_data["plotCTAcquisitionTypes"]
    ):
        user_profile.plotCTSequencedAcquisition = True
    else:
        user_profile.plotCTSequencedAcquisition = False
    if (
        CommonVariables.CT_SPIRAL_ACQUISITION_TYPE
        in ct_form.cleaned_data["plotCTAcquisitionTypes"]
    ):
        user_profile.plotCTSpiralAcquisition = True
    else:
        user_profile.plotCTSpiralAcquisition = False
    if (
        CommonVariables.CT_CONSTANT_ANGLE_ACQUISITION_TYPE
        in ct_form.cleaned_data["plotCTAcquisitionTypes"]
    ):
        user_profile.plotCTConstantAngleAcquisition = True
    else:
        user_profile.plotCTConstantAngleAcquisition = False
    if (
        CommonVariables.CT_STATIONARY_ACQUISITION_TYPE
        in ct_form.cleaned_data["plotCTAcquisitionTypes"]
    ):
        user_profile.plotCTStationaryAcquisition = True
    else:
        user_profile.plotCTStationaryAcquisition = False
    if (
        CommonVariables.CT_FREE_ACQUISITION_TYPE
        in ct_form.cleaned_data["plotCTAcquisitionTypes"]
    ):
        user_profile.plotCTFreeAcquisition = True
    else:
        user_profile.plotCTFreeAcquisition = False
    if (
        CommonVariables.CT_CONE_BEAM_ACQUISITION
        in ct_form.cleaned_data["plotCTAcquisitionTypes"]
    ):
        user_profile.plotCTConeBeamAcquisition = True
    else:
        user_profile.plotCTConeBeamAcquisition = False
    user_profile.plotCTStudyMeanDLP = ct_form.cleaned_data["plotCTStudyMeanDLP"]
    user_profile.plotCTStudyMeanCTDI = ct_form.cleaned_data["plotCTStudyMeanCTDI"]
    user_profile.plotCTStudyFreq = ct_form.cleaned_data["plotCTStudyFreq"]
    user_profile.plotCTStudyNumEvents = ct_form.cleaned_data["plotCTStudyNumEvents"]
    user_profile.plotCTStudyPerDayAndHour = ct_form.cleaned_data[
        "plotCTStudyPerDayAndHour"
    ]
    user_profile.plotCTStudyMeanDLPOverTime = ct_form.cleaned_data[
        "plotCTStudyMeanDLPOverTime"
    ]
    user_profile.plotCTRequestMeanDLP = ct_form.cleaned_data["plotCTRequestMeanDLP"]
    user_profile.plotCTRequestFreq = ct_form.cleaned_data["plotCTRequestFreq"]
    user_profile.plotCTRequestNumEvents = ct_form.cleaned_data["plotCTRequestNumEvents"]
    user_profile.plotCTRequestDLPOverTime = ct_form.cleaned_data[
        "plotCTRequestDLPOverTime"
    ]
    user_profile.plotCTOverTimePeriod = ct_form.cleaned_data["plotCTOverTimePeriod"]
    user_profile.plotCTInitialSortingChoice = ct_form.cleaned_data[
        "plotCTInitialSortingChoice"
    ]


def initialise_ct_form_data(ct_acquisition_types, user_profile):
    ct_form_data = {
        "plotCTAcquisitionMeanDLP": user_profile.plotCTAcquisitionMeanDLP,
        "plotCTAcquisitionMeanCTDI": user_profile.plotCTAcquisitionMeanCTDI,
        "plotCTAcquisitionFreq": user_profile.plotCTAcquisitionFreq,
        "plotCTAcquisitionCTDIvsMass": user_profile.plotCTAcquisitionCTDIvsMass,
        "plotCTAcquisitionDLPvsMass": user_profile.plotCTAcquisitionDLPvsMass,
        "plotCTAcquisitionCTDIOverTime": user_profile.plotCTAcquisitionCTDIOverTime,
        "plotCTAcquisitionDLPOverTime": user_profile.plotCTAcquisitionDLPOverTime,
        "plotCTAcquisitionTypes": ct_acquisition_types,
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
        "plotCTInitialSortingChoice": user_profile.plotCTInitialSortingChoice,
    }
    return ct_form_data


def required_ct_acquisition_types(user_profile):
    ct_acquisition_types = []
    if user_profile.plotCTSequencedAcquisition:
        ct_acquisition_types.append(CommonVariables.CT_SEQUENCED_ACQUISITION_TYPE)
    if user_profile.plotCTSpiralAcquisition:
        ct_acquisition_types.append(CommonVariables.CT_SPIRAL_ACQUISITION_TYPE)
    if user_profile.plotCTConstantAngleAcquisition:
        ct_acquisition_types.append(CommonVariables.CT_CONSTANT_ANGLE_ACQUISITION_TYPE)
    if user_profile.plotCTStationaryAcquisition:
        ct_acquisition_types.append(CommonVariables.CT_STATIONARY_ACQUISITION_TYPE)
    if user_profile.plotCTFreeAcquisition:
        ct_acquisition_types.append(CommonVariables.CT_FREE_ACQUISITION_TYPE)
    if user_profile.plotCTConeBeamAcquisition:
        ct_acquisition_types.append(CommonVariables.CT_CONE_BEAM_ACQUISITION)

    return ct_acquisition_types


def required_average_choices(user_profile):
    average_choices = []
    if user_profile.plotMean:
        average_choices.append(CommonVariables.MEAN)
    if user_profile.plotMedian:
        average_choices.append(CommonVariables.MEDIAN)
    if user_profile.plotBoxplots:
        average_choices.append(CommonVariables.BOXPLOT)
    return average_choices


@login_required
def homepage_options_view(request):
    """View to enable user to see and update home page options

    :param request: request object
    :return: dictionary of home page settings, html template location and request object
    """
    try:
        HomePageAdminSettings.objects.get()
    except ObjectDoesNotExist:
        HomePageAdminSettings.objects.create()

    display_workload_stats = HomePageAdminSettings.objects.values_list(
        "enable_workload_stats", flat=True
    )[0]
    if not display_workload_stats:
        if not request.user.groups.filter(name="admingroup"):
            messages.info(
                request,
                mark_safe(
                    "The display of homepage workload stats is disabled; only a member of the admin group can change this setting"  # pylint: disable=line-too-long
                ),
            )  # nosec

    if request.method == "POST":
        homepage_options_form = HomepageOptionsForm(request.POST)
        if homepage_options_form.is_valid():
            try:
                # See if the user has a userprofile
                user_profile = request.user.userprofile
            except ObjectDoesNotExist:
                # Create a default userprofile for the user if one doesn't exist
                create_user_profile(
                    sender=request.user, instance=request.user, created=True
                )
                user_profile = request.user.userprofile

            user_profile.summaryWorkloadDaysA = homepage_options_form.cleaned_data[
                "dayDeltaA"
            ]
            user_profile.summaryWorkloadDaysB = homepage_options_form.cleaned_data[
                "dayDeltaB"
            ]

            user_profile.save()

            if request.user.groups.filter(name="admingroup"):
                if (
                    homepage_options_form.cleaned_data["enable_workload_stats"]
                    != display_workload_stats
                ):
                    homepage_admin_settings = HomePageAdminSettings.objects.all()[0]
                    homepage_admin_settings.enable_workload_stats = (
                        homepage_options_form.cleaned_data["enable_workload_stats"]
                    )
                    homepage_admin_settings.save()
                    if homepage_options_form.cleaned_data["enable_workload_stats"]:
                        messages.info(request, "Display of workload stats enabled")
                    else:
                        messages.info(request, "Display of workload stats disabled")

        messages.success(request, "Home page options have been updated")
        return HttpResponseRedirect(reverse_lazy("homepage_options_view"))

    admin = {
        "openremversion": __version__,
        "docsversion": __docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    try:
        # See if the user has a userprofile
        user_profile = request.user.userprofile
    except ObjectDoesNotExist:
        # Create a default userprofile for the user if one doesn't exist
        create_user_profile(sender=request.user, instance=request.user, created=True)
        user_profile = request.user.userprofile

    homepage_form_data = {
        "dayDeltaA": user_profile.summaryWorkloadDaysA,
        "dayDeltaB": user_profile.summaryWorkloadDaysB,
        "enable_workload_stats": display_workload_stats,
    }

    homepage_options_form = HomepageOptionsForm(homepage_form_data)

    home_config = {"display_workload_stats": display_workload_stats}

    return_structure = {
        "admin": admin,
        "HomepageOptionsForm": homepage_options_form,
        "home_config": home_config,
    }

    return render(request, "remapp/displayhomepageoptions.html", return_structure)


@login_required
def not_patient_indicators(request):
    """Displays current not-patient indicators"""
    not_patient_ids = NotPatientIndicatorsID.objects.all()
    not_patient_names = NotPatientIndicatorsName.objects.all()

    admin = {
        "openremversion": __version__,
        "docsversion": __docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    # Render list page with the documents and the form
    return render(
        request,
        "remapp/notpatient.html",
        {"ids": not_patient_ids, "names": not_patient_names, "admin": admin},
    )


@login_required
def not_patient_indicators_as_074(request):
    """Add patterns to no-patient indicators to replicate 0.7.4 behaviour"""
    if request.user.groups.filter(name="admingroup"):
        not_patient_ids = NotPatientIndicatorsID.objects.all()
        not_patient_names = NotPatientIndicatorsName.objects.all()

        id_indicators = ["*phy*", "*test*", "*qa*"]
        name_indicators = ["*phys*", "*test*", "*qa*"]

        for id_indicator in id_indicators:
            if not not_patient_ids.filter(not_patient_id__iexact=id_indicator):
                NotPatientIndicatorsID(not_patient_id=id_indicator).save()

        for name_indicator in name_indicators:
            if not not_patient_names.filter(not_patient_name__iexact=name_indicator):
                NotPatientIndicatorsName(not_patient_name=name_indicator).save()

        messages.success(request, "0.7.4 style not-patient indicators restored")
        return redirect(reverse_lazy("not_patient_indicators"))

    else:
        messages.error(
            request,
            "Only members of the admingroup are allowed to modify not-patient indicators",
        )
    return redirect(reverse_lazy("not_patient_indicators"))


@login_required
def admin_questions_hide_not_patient(request):
    """Hides the not-patient revert to 0.7.4 question"""
    if request.user.groups.filter(name="admingroup"):
        admin_question = AdminTaskQuestions.objects.all()[0]
        admin_question.ask_revert_to_074_question = False
        admin_question.save()
        messages.success(
            request, "Identifying not-patient exposure question won't be shown again"
        )
        return redirect(reverse_lazy("home"))
    else:
        messages.error(
            request, "Only members of the admingroup are allowed config this question"
        )
    return redirect(reverse_lazy("not_patient_indicators"))


def _create_admin_dict(request):
    """Function to factor out creating admin dict with admin true/false

    :return: dict containing version numbers and admin group membership
    """
    admin = {
        "openremversion": __version__,
        "docsversion": __docs_version__,
    }
    for group in request.user.groups.all():
        admin[group.name] = True
    return admin


@login_required
def display_tasks(request):
    """View to show Celery tasks. Content generated using AJAX"""

    admin = _create_admin_dict(request)

    template = "remapp/task_admin.html"
    return render(request, template, {"admin": admin})


def tasks(request, stage=None):
    """AJAX function to get current task details"""
    if request.is_ajax() and request.user.groups.filter(name="admingroup"):
        active_tasks = []
        recent_tasks = []
        older_tasks = []
        tasks = BackgroundTask.objects.all()
        datetime_now = datetime.now()

        for task in tasks:
            recent_time_delta = timedelta(hours=6)
            if not task.complete:
                active_tasks.append(task)
            elif datetime_now - recent_time_delta < task.started_at:
                recent_tasks.append(task)
            else:
                older_tasks.append(task)

        if "active" in stage:
            tinfo = {"tasks": active_tasks, "type": "active"}
        elif "recent" in stage:
            tinfo = {"tasks": recent_tasks, "type": "recent"}
        elif "older" in stage:
            tinfo = {"tasks": older_tasks, "type": "older"}

        return render(request, "remapp/tasks.html", tinfo)


def task_abort(request, task_id=None):
    """Function to abort one of the tasks"""
    if task_id and request.user.groups.filter(name="admingroup"):
        task = get_object_or_404(BackgroundTask, uuid=task_id)

        try:
            if task.task_type == "query" or task.task_type == "move":
                abort_logger = logging.getLogger("remapp.netdicom.qrscu")
                abort_logger.info(
                    "Query or move task {0} terminated from the Tasks interface".format(
                        task_id
                    )
                )
            else:
                abort_logger = logging.getLogger("remapp")
                abort_logger.info(
                    "Task {0} of type {1} terminated from the Tasks interface".format(
                        task_id, task.task_type
                    )
                )
        except ObjectDoesNotExist:
            pass
        terminate_background(task)
        messages.success(
            request,
            "Task {0} terminated".format(task_id),
        )

    return redirect(reverse_lazy("task_admin"))


class PatientIDSettingsUpdate(UpdateView):  # pylint: disable=unused-variable
    """UpdateView to update the patient ID settings"""

    model = PatientIDSettings
    fields = [
        "name_stored",
        "name_hashed",
        "id_stored",
        "id_hashed",
        "accession_hashed",
        "dob_stored",
    ]

    def get_context_data(self, **context):
        context = super(PatientIDSettingsUpdate, self).get_context_data(**context)
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context


class DicomDeleteSettingsUpdate(UpdateView):  # pylint: disable=unused-variable
    """UpdateView tp update the settings relating to deleting DICOM after import"""

    model = DicomDeleteSettings
    form_class = DicomDeleteSettingsForm

    def get_context_data(self, **context):
        context = super(DicomDeleteSettingsUpdate, self).get_context_data(**context)
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context


class RFHighDoseAlertSettings(UpdateView):  # pylint: disable=unused-variable
    """UpdateView for configuring the fluoroscopy high dose alert settings"""

    try:
        HighDoseMetricAlertSettings.get_solo()  # will create item if it doesn't exist
    except (AvoidDataMigrationErrorPostgres, AvoidDataMigrationErrorSQLite):
        pass

    model = HighDoseMetricAlertSettings
    form_class = RFHighDoseFluoroAlertsForm

    def get_context_data(self, **context):
        context = super(RFHighDoseAlertSettings, self).get_context_data(**context)
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context

    def form_valid(self, form):
        if form.has_changed():
            if "show_accum_dose_over_delta_weeks" in form.changed_data:
                msg = (
                    "Display of summed total DAP and total dose at RP on summary page "
                )
                if form.cleaned_data["show_accum_dose_over_delta_weeks"]:
                    msg += "enabled"
                else:
                    msg += " disabled"
                messages.info(self.request, msg)
            if "calc_accum_dose_over_delta_weeks_on_import" in form.changed_data:
                msg = "Calculation of summed total DAP and total dose at RP for incoming studies "
                if form.cleaned_data["calc_accum_dose_over_delta_weeks_on_import"]:
                    msg += "enabled"
                else:
                    msg += " disabled"
                messages.info(self.request, msg)
            if "send_high_dose_metric_alert_emails" in form.changed_data:
                msg = "E-mail notification of high doses "
                if form.cleaned_data["send_high_dose_metric_alert_emails"]:
                    msg += "enabled"
                else:
                    msg += " disabled"
                messages.info(self.request, msg)
            if "alert_total_dap_rf" in form.changed_data:
                messages.info(
                    self.request,
                    "Total DAP alert level has been changed to {0}".format(
                        form.cleaned_data["alert_total_dap_rf"]
                    ),
                )
            if "alert_total_rp_dose_rf" in form.changed_data:
                messages.info(
                    self.request,
                    "Total dose at reference point alert level has been changed to {0}".format(
                        form.cleaned_data["alert_total_rp_dose_rf"]
                    ),
                )
            if "accum_dose_delta_weeks" in form.changed_data:
                messages.warning(
                    self.request,
                    'The time period used to sum total DAP and total dose at RP has changed. The summed data must be recalculated: click on the "Recalculate all summed data" button below. The recalculation can take several minutes',  # pylint: disable=line-too-long
                )
            return super(RFHighDoseAlertSettings, self).form_valid(form)
        else:
            messages.info(self.request, "No changes made")
        return super(RFHighDoseAlertSettings, self).form_valid(form)


@login_required
@csrf_exempt
def rf_alert_notifications_view(request):
    """View for display and modification of fluoroscopy high dose alert recipients"""
    if request.method == "POST" and request.user.groups.filter(name="admingroup"):
        # Check to see if we need to send a test message
        if "Send test" in list(request.POST.values()):
            recipient = get_keys_by_value(request.POST, "Send test")[0]
            email_response = send_rf_high_dose_alert_email(
                study_pk=None, test_message=True, test_user=recipient
            )
            if email_response is None:
                messages.success(request, "Test e-mail sent to {0}".format(recipient))
            else:
                messages.error(
                    request, "Test e-mail failed: {0}".format(email_response)
                )

        all_users = User.objects.all()
        for user in all_users:
            if str(user.pk) in list(request.POST.values()):
                if not hasattr(user, "highdosemetricalertrecipients"):
                    new_objects = HighDoseMetricAlertRecipients.objects.create(
                        user=user
                    )
                    new_objects.save()
                user.highdosemetricalertrecipients.receive_high_dose_metric_alerts = (
                    True
                )
            else:
                if not hasattr(user, "highdosemetricalertrecipients"):
                    new_objects = HighDoseMetricAlertRecipients.objects.create(
                        user=user
                    )
                    new_objects.save()
                user.highdosemetricalertrecipients.receive_high_dose_metric_alerts = (
                    False
                )
            user.save()

    f = User.objects.order_by("username")

    admin = {
        "openremversion": __version__,
        "docsversion": __docs_version__,
    }

    for group in request.user.groups.all():
        admin[group.name] = True

    return_structure = {"user_list": f, "admin": admin}

    return render(request, "remapp/rfalertnotificationsview.html", return_structure)


@login_required
def rf_recalculate_accum_doses(request):  # pylint: disable=unused-variable
    """View to recalculate the summed total DAP and total dose at RP for all RF studies"""
    if not request.user.groups.filter(name="admingroup"):
        # Send the user to the home page
        return HttpResponseRedirect(reverse_lazy("home"))
    else:
        # Empty the PKsForSummedRFDoseStudiesInDeltaWeeks table
        PKsForSummedRFDoseStudiesInDeltaWeeks.objects.all().delete()

        # In the AccumIntegratedProjRadiogDose table delete all dose_area_product_total_over_delta_weeks
        # and dose_rp_total_over_delta_weeks entries
        AccumIntegratedProjRadiogDose.objects.all().update(
            dose_area_product_total_over_delta_weeks=None,
            dose_rp_total_over_delta_weeks=None,
        )

        # For each RF study recalculate dose_area_product_total_over_delta_weeks and dose_rp_total_over_delta_weeks
        try:
            HighDoseMetricAlertSettings.objects.get()
        except ObjectDoesNotExist:
            HighDoseMetricAlertSettings.objects.create()
        week_delta = HighDoseMetricAlertSettings.objects.values_list(
            "accum_dose_delta_weeks", flat=True
        )[0]

        all_rf_studies = GeneralStudyModuleAttr.objects.filter(
            modality_type__exact="RF"
        ).all()

        for study in all_rf_studies:
            try:
                study.patientmoduleattr_set.get()
                patient_id = study.patientmoduleattr_set.values_list(
                    "patient_id", flat=True
                )[0]
            except ObjectDoesNotExist:
                patient_id = None

            if patient_id:
                study_date = study.study_date
                oldest_date = study_date - timedelta(weeks=week_delta)

                # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # The try and except parts of this code are here because some of the studies in my database didn't have
                # the expected data in the related fields - not sure why. Perhaps an issue with the extractor routine?
                try:
                    study.projectionxrayradiationdose_set.get().accumxraydose_set.all()
                except ObjectDoesNotExist:
                    study.projectionxrayradiationdose_set.get().accumxraydose_set.create()

                for (
                    accumxraydose
                ) in (
                    study.projectionxrayradiationdose_set.get().accumxraydose_set.all()
                ):
                    try:
                        accumxraydose.accumintegratedprojradiogdose_set.get()
                    except:
                        accumxraydose.accumintegratedprojradiogdose_set.create()

                for (
                    accumxraydose
                ) in (
                    study.projectionxrayradiationdose_set.get().accumxraydose_set.all()
                ):
                    accum_int_proj_pk = (
                        accumxraydose.accumintegratedprojradiogdose_set.get().pk
                    )

                    accum_int_proj_to_update = (
                        AccumIntegratedProjRadiogDose.objects.get(pk=accum_int_proj_pk)
                    )

                    included_studies = all_rf_studies.filter(
                        patientmoduleattr__patient_id__exact=patient_id,
                        study_date__range=[oldest_date, study_date],
                    )

                    bulk_entries = []
                    for pk in included_studies.values_list("pk", flat=True):
                        if not PKsForSummedRFDoseStudiesInDeltaWeeks.objects.filter(
                            general_study_module_attributes_id__exact=study.pk
                        ).filter(study_pk_in_delta_weeks__exact=pk):
                            new_entry = PKsForSummedRFDoseStudiesInDeltaWeeks()
                            new_entry.general_study_module_attributes_id = study.pk
                            new_entry.study_pk_in_delta_weeks = pk
                            bulk_entries.append(new_entry)

                    if len(bulk_entries):
                        PKsForSummedRFDoseStudiesInDeltaWeeks.objects.bulk_create(
                            bulk_entries
                        )

                    accum_totals = included_studies.aggregate(
                        Sum(
                            "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_area_product_total"
                        ),
                        Sum(
                            "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_rp_total"
                        ),
                    )
                    accum_int_proj_to_update.dose_area_product_total_over_delta_weeks = accum_totals[
                        "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_area_product_total__sum"
                    ]
                    accum_int_proj_to_update.dose_rp_total_over_delta_weeks = accum_totals[
                        "projectionxrayradiationdose__accumxraydose__accumintegratedprojradiogdose__dose_rp_total__sum"
                    ]
                    accum_int_proj_to_update.save()
                populate_rf_delta_weeks_summary(study)

        HighDoseMetricAlertSettings.objects.all().update(
            changed_accum_dose_delta_weeks=False
        )

        messages.success(
            request,
            "All summed total DAP and total dose at RP doses have been re-calculated",
        )

        django_messages = []
        for message in messages.get_messages(request):
            django_messages.append(
                {
                    "level": message.level_tag,
                    "message": message.message,
                    "extra_tags": message.tags,
                }
            )

        return_structure = {"success": True, "messages": django_messages}

        return JsonResponse(return_structure, safe=False)


class NotPatientNameCreate(CreateView):  # pylint: disable=unused-variable
    """CreateView for configuration of indicators a study might not be a patient study"""

    model = NotPatientIndicatorsName
    form_class = NotPatientNameForm

    def get_context_data(self, **context):
        context = super(NotPatientNameCreate, self).get_context_data(**context)
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context


class NotPatientNameUpdate(UpdateView):  # pylint: disable=unused-variable
    """UpdateView to update choices regarding not-patient indicators"""

    model = NotPatientIndicatorsName
    form_class = NotPatientNameForm

    def get_context_data(self, **context):
        context = super(NotPatientNameUpdate, self).get_context_data(**context)
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context


class NotPatientNameDelete(DeleteView):  # pylint: disable=unused-variable
    """DeleteView for the not-patient name indicator table"""

    model = NotPatientIndicatorsName
    success_url = reverse_lazy("not_patient_indicators")

    def get_context_data(self, **context):
        context[self.context_object_name] = self.object
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context


class NotPatientIDCreate(CreateView):  # pylint: disable=unused-variable
    """CreateView for not-patient ID indicators"""

    model = NotPatientIndicatorsID
    form_class = NotPatientIDForm

    def get_context_data(self, **context):
        context = super(NotPatientIDCreate, self).get_context_data(**context)
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context


class NotPatientIDUpdate(UpdateView):  # pylint: disable=unused-variable
    """UpdateView for non-patient ID indicators"""

    model = NotPatientIndicatorsID
    form_class = NotPatientIDForm

    def get_context_data(self, **context):
        context = super(NotPatientIDUpdate, self).get_context_data(**context)
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context


class NotPatientIDDelete(DeleteView):  # pylint: disable=unused-variable
    """DeleteView for non-patient ID indicators"""

    model = NotPatientIndicatorsID
    success_url = reverse_lazy("not_patient_indicators")

    def get_context_data(self, **context):
        context[self.context_object_name] = self.object
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context


def populate_summary(request):
    """Populate the summary fields in GeneralStudyModuleAttr table for existing studies

    :param request:
    :return:
    """
    if request.user.groups.filter(name="admingroup"):
        try:
            task_ct = SummaryFields.objects.get(modality_type__exact="CT")
        except ObjectDoesNotExist:
            task_ct = SummaryFields.objects.create(modality_type="CT")
        if not task_ct.complete:
            run_in_background(
                populate_summary_ct,
                "populate_summary_ct",
            )
        try:
            task_mg = SummaryFields.objects.get(modality_type__exact="MG")
        except ObjectDoesNotExist:
            task_mg = SummaryFields.objects.create(modality_type="MG")
        if not task_mg.complete:
            run_in_background(
                populate_summary_mg,
                "populate_summary_mg",
            )
        try:
            task_dx = SummaryFields.objects.get(modality_type__exact="DX")
        except ObjectDoesNotExist:
            task_dx = SummaryFields.objects.create(modality_type="DX")
        if not task_dx.complete:
            run_in_background(
                populate_summary_dx,
                "populate_summary_dx",
            )
        try:
            task_rf = SummaryFields.objects.get(modality_type__exact="RF")
        except ObjectDoesNotExist:
            task_rf = SummaryFields.objects.create(modality_type="RF")
        if not task_rf.complete:
            run_in_background(
                populate_summary_rf,
                "populate_summary_rf",
            )

        return redirect(reverse_lazy("home"))


def populate_summary_progress(request):
    """AJAX function to get populate summary fields progress"""
    if request.is_ajax():
        if request.user.groups.filter(name="admingroup"):
            try:
                ct_status = SummaryFields.objects.get(modality_type__exact="CT")
                rf_status = SummaryFields.objects.get(modality_type__exact="RF")
                mg_status = SummaryFields.objects.get(modality_type__exact="MG")
                dx_status = SummaryFields.objects.get(modality_type__exact="DX")
            except ObjectDoesNotExist:
                return render(
                    request,
                    "remapp/populate_summary_progress_error.html",
                    {"not_admin": False},
                )

            if (
                ct_status.complete
                and rf_status.complete
                and mg_status.complete
                and dx_status.complete
            ):
                upgrade_status = UpgradeStatus.get_solo()
                upgrade_status.from_0_9_1_summary_fields = True
                upgrade_status.save()
                return HttpResponse("")
            try:
                ct = GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT")
                if ct.filter(number_of_const_angle__isnull=True).count() > 0:
                    ct_complete = ct.filter(number_of_const_angle__isnull=False).count()
                    ct_total = ct.count()
                    ct_pc = 100 * (float(ct_complete) / ct_total)
                else:
                    ct_status.complete = True
                    ct_status.save()
                    ct_complete = None
                    ct_total = None
                    ct_pc = 0
            except ObjectDoesNotExist:
                ct_complete = None
                ct_total = None
                ct_pc = 0
            try:
                rf = GeneralStudyModuleAttr.objects.filter(modality_type__exact="RF")
                if rf.filter(number_of_events_a__isnull=True).count() > 0:
                    rf_complete = rf.filter(number_of_events_a__isnull=False).count()
                    rf_total = rf.count()
                    rf_pc = 100 * (float(rf_complete) / rf_total)
                else:
                    rf_status.complete = True
                    rf_status.save()
                    rf_complete = None
                    rf_total = None
                    rf_pc = 0
            except ObjectDoesNotExist:
                rf_complete = None
                rf_total = None
                rf_pc = 0
            try:
                mg = GeneralStudyModuleAttr.objects.filter(modality_type__exact="MG")
                if (
                    mg.filter(total_agd_right__isnull=True)
                    .filter(total_agd_left__isnull=True)
                    .filter(total_agd_both__isnull=True)
                    .count()
                    > 0
                ):
                    mg_complete = mg.filter(
                        Q(total_agd_right__isnull=False)
                        | Q(total_agd_left__isnull=False)
                        | Q(total_agd_both__isnull=False)
                    ).count()
                    mg_total = mg.count()
                    mg_pc = 100 * (float(mg_complete) / mg_total)
                else:
                    mg_status.complete = True
                    mg_status.save()
                    mg_complete = None
                    mg_total = None
                    mg_pc = 0
            except ObjectDoesNotExist:
                mg_complete = None
                mg_total = None
                mg_pc = 0
            try:
                dx = GeneralStudyModuleAttr.objects.filter(
                    Q(modality_type__exact="DX")
                    | Q(modality_type__exact="CR")
                    | Q(modality_type__exact="PX")
                )
                if dx.filter(number_of_events_a__isnull=True).count() > 0:
                    dx_complete = dx.filter(number_of_events_a__isnull=False).count()
                    dx_total = dx.count()
                    dx_pc = 100 * (float(dx_complete) / dx_total)
                else:
                    dx_status.complete = True
                    dx_status.save()
                    dx_complete = None
                    dx_total = None
                    dx_pc = 0
            except ObjectDoesNotExist:
                dx_complete = None
                dx_total = None
                dx_pc = 0
            try:
                dx_pc = 100 * (float(dx_status.current_study) / dx_status.total_studies)
            except ObjectDoesNotExist:
                dx_status = None

            return render(
                request,
                "remapp/populate_summary_progress.html",
                {
                    "ct_complete": ct_complete,
                    "ct_total": ct_total,
                    "ct_pc": ct_pc,
                    "ct_status": ct_status,
                    "rf_complete": rf_complete,
                    "rf_total": rf_total,
                    "rf_pc": rf_pc,
                    "rf_status": rf_status,
                    "mg_complete": mg_complete,
                    "mg_total": mg_total,
                    "mg_pc": mg_pc,
                    "mg_status": mg_status,
                    "dx_complete": dx_complete,
                    "dx_total": dx_total,
                    "dx_pc": dx_pc,
                    "dx_status": dx_status,
                },
            )
        else:
            return render(
                request,
                "remapp/populate_summary_progress_error.html",
                {"not_admin": True},
            )
