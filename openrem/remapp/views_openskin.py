# Copyright 2012-2021 The Royal Marsden NHS Foundation Trust. See LICENSE file for details.

"""openSkin related views."""

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned
from django.db.models import Q
from django.db.utils import OperationalError as AvoidDataMigrationErrorSQLite
from django.db.utils import ProgrammingError as AvoidDataMigrationErrorPostgres
from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.utils.translation import gettext as _
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
    """Check if device matches on manufacturer and model without version restriction.

    openSkin safe list `OpenSkinSafeList` is checked against manufacturer and model. This function is then used to check
    if there are any entries on the list where `software_version` is blank.

    Parameters
    ----------
    skin_safe_models : OpenSkinSafeList queryset
        Queryset of safe list entries matching manufacturer and model

    Returns
    -------
    safe_list_model_pk: int or None
        Primary key of entry if match found, ``None`` otherwise
    model_enabled: bool
        ``True`` if match found with blank `software_version`, otherwise ``False``

    """
    try:
        safe_list_model_pk = skin_safe_models.get(software_version="").pk
        model_enabled = True
    except ObjectDoesNotExist:
        model_enabled = False
        safe_list_model_pk = None
    except MultipleObjectsReturned:
        model_enabled = True
        safe_list_model_pk = (
            skin_safe_models.filter(software_version="").order_by("pk").first().pk
        )
    return safe_list_model_pk, model_enabled


def get_matching_equipment_names(manufacturer, model_name):
    """Get queryset of unique equipment names that match the manufacturer and model name being reviewed.

    Filters the `UniqueEquipmentNames` table for fluoroscopy entries (or dual fluoro + radiography) that match the
    manufacturer and model that has been selected.

    Parameters
    ----------
    manufacturer : str
        Name of manufacturer from `UniqueEquipmentNames` table
    model_name : str
        Model name from `UniqueEquipmentNames` table

    Returns
    -------
    UniqueEquipmentNames queryset
        Queryset filtered for fluoro systems matching the manufacturer and model name

    """
    rf_names = (
        UniqueEquipmentNames.objects.order_by("display_name")
        .filter(
            Q(user_defined_modality="RF")
            | Q(user_defined_modality="dual")
            | (
                Q(user_defined_modality__isnull=True)
                & Q(
                    generalequipmentmoduleattr__general_study_module_attributes__modality_type="RF"
                )
            )
        )
        .distinct()
        .filter(manufacturer__exact=manufacturer)
        .filter(manufacturer_model_name__exact=model_name)
    )
    return rf_names


@login_required
def display_name_skin_enabled(request):
    """AJAX view to display if skin map calculations are enabled and links to change the configuration."""
    template = "remapp/displayname-skinmap.html"
    if request.is_ajax() and request.method == "POST":
        data = request.POST
        equip_name_pk = data.get("equip_name_pk")

        try:
            SkinDoseMapCalcSettings.get_solo()  # will create item if it doesn't exist
        except (AvoidDataMigrationErrorPostgres, AvoidDataMigrationErrorSQLite):
            pass
        allow_safelist_modify = SkinDoseMapCalcSettings.get_solo().allow_safelist_modify
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
                skin_safe_version = skin_safe_models.get(
                    software_version=equipment.software_versions
                )
                safe_list_pk = skin_safe_version.pk
                all_model_safe_list_pk = check_skin_safe_model(skin_safe_models)
                if all_model_safe_list_pk[0]:
                    model_and_version = True
                else:
                    version_only = True
            except ObjectDoesNotExist:
                safe_list_pk, model_only = check_skin_safe_model(skin_safe_models)
            except MultipleObjectsReturned:
                safe_list_pk = (
                    skin_safe_models.filter(
                        software_version=equipment.software_versions
                    )
                    .order_by("pk")
                    .first()
                    .pk
                )
                all_model_safe_list_pk = check_skin_safe_model(skin_safe_models)
                if all_model_safe_list_pk[0]:
                    model_and_version = True
                else:
                    version_only = True

        context = {
            "allow_safelist_modify": allow_safelist_modify,
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
    else:
        return render(request, template, {"illegal": True})


class SkinDoseMapCalcSettingsUpdate(UpdateView):  # pylint: disable=unused-variable

    """Update skin dose map calculation settings."""

    try:
        SkinDoseMapCalcSettings.get_solo()  # will create item if it doesn't exist
    except (AvoidDataMigrationErrorPostgres, AvoidDataMigrationErrorSQLite):
        pass

    model = SkinDoseMapCalcSettings
    form_class = SkinDoseMapCalcSettingsForm

    def get_context_data(self, **context):
        context = super().get_context_data(**context)
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
            messages.success(
                self.request, _("Skin dose map settings have been updated")
            )
        else:
            messages.info(self.request, _("No changes made"))
        return super().form_valid(form)


class SkinSafeListCreate(CreateView):

    """Enable skin map calculations by adding model, or model and software version to `OpenSkinSafeList`."""

    model = OpenSkinSafeList
    form_class = SkinSafeListForm
    template_name_suffix = "_add"

    def get_context_data(self, **context):
        context = super().get_context_data(**context)
        equipment = None
        if self.kwargs["equip_name_pk"]:
            equipment = UniqueEquipmentNames.objects.get(
                pk=int(self.kwargs["equip_name_pk"])
            )
            context["form"].initial["manufacturer"] = equipment.manufacturer
            context["form"].initial[
                "manufacturer_model_name"
            ] = equipment.manufacturer_model_name
            context["form"].initial["software_version"] = equipment.software_versions
        context["equipment"] = equipment

        manufacturer_model = get_matching_equipment_names(
            equipment.manufacturer, equipment.manufacturer_model_name
        )
        manufacturer_model_version = manufacturer_model.filter(
            software_versions__exact=equipment.software_versions
        )
        context["manufacturer_model"] = manufacturer_model
        context["manufacturer_model_version"] = manufacturer_model_version
        context[
            "allow_safelist_modify"
        ] = SkinDoseMapCalcSettings.get_solo().allow_safelist_modify
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context

    def form_valid(self, form):
        allow_safelist_modify = SkinDoseMapCalcSettings.get_solo().allow_safelist_modify
        if not allow_safelist_modify:
            messages.error(
                self.request, _("Skin dose map set to not allow safelist modification")
            )
            return redirect(reverse_lazy("display_names_view"))
        if self.request.user.groups.filter(name="admingroup"):
            if self.request.POST.get("model"):
                form.instance.software_version = ""
            return super().form_valid(form)
        else:
            messages.error(
                self.request,
                _("Only members of the admin group can change the openSkin safe list"),
            )
            return redirect(reverse_lazy("display_names_view"))


class SkinSafeListUpdate(UpdateView):

    """Add or remove the software version restriction."""

    model = OpenSkinSafeList
    form_class = SkinSafeListForm

    def get_context_data(self, **context):
        context = super().get_context_data(**context)
        equipment = None
        if self.kwargs["equip_name_pk"]:
            equipment = UniqueEquipmentNames.objects.get(
                pk=int(self.kwargs["equip_name_pk"])
            )
            if not context["form"].initial["software_version"]:
                context["form"].initial[
                    "software_version"
                ] = equipment.software_versions
            else:
                context["form"].initial["software_version"] = None
        context["equipment"] = equipment

        manufacturer_model = get_matching_equipment_names(
            manufacturer=self.object.manufacturer,
            model_name=self.object.manufacturer_model_name,
        )
        manufacturer_model_version = manufacturer_model.filter(
            software_versions__exact=equipment.software_versions
        )
        model_exists = False
        if self.object.software_version:
            model_exists = bool(
                OpenSkinSafeList.objects.filter(manufacturer=self.object.manufacturer)
                .filter(manufacturer_model_name=self.object.manufacturer_model_name)
                .filter(software_version=None)
            )
        context["manufacturer_model"] = manufacturer_model
        context["manufacturer_model_version"] = manufacturer_model_version
        context["model_exists"] = model_exists
        context[
            "allow_safelist_modify"
        ] = SkinDoseMapCalcSettings.get_solo().allow_safelist_modify
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context

    def form_valid(self, form):
        allow_safelist_modify = SkinDoseMapCalcSettings.get_solo().allow_safelist_modify
        if not allow_safelist_modify:
            messages.error(
                self.request, _("Skin dose map set to not allow safelist modification")
            )
            return redirect(reverse_lazy("display_names_view"))
        if self.request.user.groups.filter(name="admingroup"):
            return super().form_valid(form)
        else:
            messages.error(
                self.request,
                _("Only members of the admin group can change the openSkin safe list"),
            )
            return redirect(reverse_lazy("display_names_view"))


class SkinSafeListDelete(DeleteView):  # pylint: disable=unused-variable

    """Disable skin map calculations for particular model or model and software version."""

    model = OpenSkinSafeList
    success_url = reverse_lazy("display_names_view")

    def get_context_data(self, **context):
        context[self.context_object_name] = self.object

        model_and_version = False
        rf_names = get_matching_equipment_names(
            manufacturer=self.object.manufacturer,
            model_name=self.object.manufacturer_model_name,
        )
        if self.object.software_version:
            rf_names = rf_names.filter(
                software_versions__exact=self.object.software_version
            )
            skin_safe_models = OpenSkinSafeList.objects.filter(
                manufacturer=self.object.manufacturer,
                manufacturer_model_name=self.object.manufacturer_model_name,
            )
            all_model_safe_list_pk = check_skin_safe_model(skin_safe_models)
            if all_model_safe_list_pk[0]:
                model_and_version = True
        context["equipment"] = rf_names
        context["model_and_version"] = model_and_version
        context[
            "allow_safelist_modify"
        ] = SkinDoseMapCalcSettings.get_solo().allow_safelist_modify
        admin = {
            "openremversion": __version__,
            "docsversion": __docs_version__,
        }
        for group in self.request.user.groups.all():
            admin[group.name] = True
        context["admin"] = admin
        return context

    def delete(self, request, *args, **kwargs):
        allow_safelist_modify = SkinDoseMapCalcSettings.get_solo().allow_safelist_modify
        if not allow_safelist_modify:
            messages.error(
                self.request, _("Skin dose map set to not allow safelist modification")
            )
            return redirect(reverse_lazy("display_names_view"))
        if self.request.user.groups.filter(name="admingroup"):
            self.object = self.get_object()
            self.object.delete()
            return redirect(reverse_lazy("display_names_view"))
        else:
            messages.error(
                self.request,
                _("Only members of the admin group can change the openSkin safe list"),
            )
            return redirect(reverse_lazy("display_names_view"))
