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

"""
..  module:: views_openskin.
    :synopsis: Module to render views relating to openSkin.

..  moduleauthor:: Ed McDonagh

"""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned
from django.db.models import Q
from django.db.utils import OperationalError as AvoidDataMigrationErrorSQLite
from django.db.utils import ProgrammingError as AvoidDataMigrationErrorPostgres
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView, UpdateView, DeleteView

from .forms import (
    SkinDoseMapCalcSettingsForm,
    SkinSafeListForm,
)
from .models import (
    SkinDoseMapCalcSettings,
    UniqueEquipmentNames,
    OpenSkinSafeList,
)
from .version import __version__, __docs_version__


def check_skin_safe_model(skin_safe_models):
    try:
        safe_list_model_pk = skin_safe_models.get(software_version="").pk
        model_enabled = True
    except ObjectDoesNotExist:
        model_enabled = False
        safe_list_model_pk = None
    except MultipleObjectsReturned:
        model_enabled = True
        safe_list_model_pk = skin_safe_models.filter(software_version="").order_by("pk").first().pk
    return safe_list_model_pk, model_enabled


@login_required
def display_name_skin_enabled(request):
    """AJAX view to return whether an entry in the equipment database is enabled for skin dose map calculations

    :param request: Request object containing modality and equipment table ID
    :return: HTML table data element
    """
    if request.is_ajax() and request.method == "POST":
        data = request.POST
        equip_name_pk = data.get("equip_name_pk")

        model_only = False
        version_only = False
        model_and_version = False
        safe_list_pk = None
        equipment = UniqueEquipmentNames.objects.get(pk=int(equip_name_pk))
        skin_safe_models = OpenSkinSafeList.objects.filter(
            manufacturer=equipment.manufacturer,
            manufacturer_model_name=equipment.manufacturer_model_name,
        )
        if skin_safe_models:
            try:
                skin_safe_version = skin_safe_models.get(software_version=equipment.software_versions)
                safe_list_pk = skin_safe_version.pk
                all_model_safe_list_pk = check_skin_safe_model(skin_safe_models)
                if all_model_safe_list_pk[0]:
                    model_and_version = True
                else:
                    version_only = True
            except ObjectDoesNotExist:
                safe_list_pk, model_only = check_skin_safe_model(skin_safe_models)
            except MultipleObjectsReturned:
                safe_list_pk = skin_safe_models.filter(
                    software_version=equipment.software_versions).order_by("pk").first().pk
                all_model_safe_list_pk = check_skin_safe_model(skin_safe_models)
                if all_model_safe_list_pk[0]:
                    model_and_version = True
                else:
                    version_only = True

        template = "remapp/displayname-skinmap.html"

        context = {
            "safe_list_pk": safe_list_pk,
            "equip_name_pk": equip_name_pk,
            "model_only": model_only,
            "version_only": version_only,
            "model_and_version": model_and_version,
        }
        return render(
            request,
            template,
            context,
        )


class SkinDoseMapCalcSettingsUpdate(UpdateView):  # pylint: disable=unused-variable
    """UpdateView for configuring the skin dose map calculation choices"""

    try:
        SkinDoseMapCalcSettings.get_solo()  # will create item if it doesn't exist
    except (AvoidDataMigrationErrorPostgres, AvoidDataMigrationErrorSQLite):
        pass

    model = SkinDoseMapCalcSettings
    form_class = SkinDoseMapCalcSettingsForm

    def get_context_data(self, **context):
        context = super(SkinDoseMapCalcSettingsUpdate, self).get_context_data(**context)
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
            messages.success(self.request, "Skin dose map settings have been updated")
        else:
            messages.info(self.request, "No changes made")
        return super(SkinDoseMapCalcSettingsUpdate, self).form_valid(form)


class SkinSafeListCreate(CreateView):
    model = OpenSkinSafeList
    form_class = SkinSafeListForm
    template_name_suffix = '_add'

    def get_context_data(self, **context):
        context = super(SkinSafeListCreate, self).get_context_data(**context)
        equipment = None
        if self.kwargs["equip_name_pk"]:
            equipment = UniqueEquipmentNames.objects.get(pk=int(self.kwargs["equip_name_pk"]))
            context["form"].initial["manufacturer"] = equipment.manufacturer
            context["form"].initial["manufacturer_model_name"] = equipment.manufacturer_model_name
            context["form"].initial["software_version"] = equipment.software_versions
        context["equipment"] = equipment

        rf_names = UniqueEquipmentNames.objects.order_by("display_name").filter(
            Q(user_defined_modality="RF")
            | Q(user_defined_modality="dual")
            | (
                    Q(user_defined_modality__isnull=True)
                    & Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="RF"
            )
            )
        ).distinct()
        manufacturer_model = rf_names.filter(
            manufacturer__exact=equipment.manufacturer
        ).filter(
            manufacturer_model_name__exact=equipment.manufacturer_model_name
        )
        manufacturer_model_version = manufacturer_model.filter(
            software_versions__exact=equipment.software_versions
        )
        context["manufacturer_model"] = manufacturer_model
        context["manufacturer_model_version"] = manufacturer_model_version
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context

    def form_valid(self, form):
        if self.request.POST.get("model"):
            form.instance.software_version = ""
        return super().form_valid(form)


class SkinSafeListUpdate(UpdateView):
    model = OpenSkinSafeList
    form_class = SkinSafeListForm

    def get_context_data(self, **context):
        context = super(SkinSafeListUpdate, self).get_context_data(**context)
        equipment = None
        if self.kwargs["equip_name_pk"]:
            equipment = UniqueEquipmentNames.objects.get(pk=int(self.kwargs["equip_name_pk"]))
            if not context["form"].initial['software_version']:
                context["form"].initial['software_version'] = equipment.software_versions
            else:
                context["form"].initial['software_version'] = None
        context["equipment"] = equipment

        rf_names = UniqueEquipmentNames.objects.order_by("display_name").filter(
            Q(user_defined_modality="RF")
            | Q(user_defined_modality="dual")
            | (
                    Q(user_defined_modality__isnull=True)
                    & Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="RF"
            )
            )
        ).distinct()
        manufacturer_model = rf_names.filter(
            manufacturer__exact=self.object.manufacturer
        ).filter(
            manufacturer_model_name__exact=self.object.manufacturer_model_name
        )
        manufacturer_model_version = manufacturer_model.filter(
            software_versions__exact=equipment.software_versions
        )
        model_exists = False
        if self.object.software_version:
            model_exists = bool(OpenSkinSafeList.objects.filter(
                manufacturer=self.object.manufacturer).filter(
                manufacturer_model_name=self.object.manufacturer_model_name).filter(
                software_version=None))
        context["manufacturer_model"] = manufacturer_model
        context["manufacturer_model_version"] = manufacturer_model_version
        context["model_exists"] = model_exists
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context


class SkinSafeListDelete(DeleteView):  # pylint: disable=unused-variable

    model = OpenSkinSafeList
    success_url = reverse_lazy("display_names_view")

    def get_context_data(self, **context):
        context[self.context_object_name] = self.object

        model_and_version = False
        rf_names = UniqueEquipmentNames.objects.order_by("display_name").filter(
            Q(user_defined_modality="RF")
            | Q(user_defined_modality="dual")
            | (
                    Q(user_defined_modality__isnull=True)
                    & Q(
                generalequipmentmoduleattr__general_study_module_attributes__modality_type="RF"
            )
            )
        ).distinct().filter(
            manufacturer__exact=self.object.manufacturer
        ).filter(
            manufacturer_model_name__exact=self.object.manufacturer_model_name
        )
        if self.object.software_version:
            rf_names = rf_names.filter(software_versions__exact=self.object.software_version)
            skin_safe_models = OpenSkinSafeList.objects.filter(
                manufacturer=self.object.manufacturer,
                manufacturer_model_name=self.object.manufacturer_model_name,
            )
            all_model_safe_list_pk = check_skin_safe_model(skin_safe_models)
            if all_model_safe_list_pk[0]:
                model_and_version = True
        context["equipment"] = rf_names
        context["model_and_version"] = model_and_version
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context
