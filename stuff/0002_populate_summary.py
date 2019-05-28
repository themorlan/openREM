# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models
from django.db.models import ObjectDoesNotExist


def populate_ct_summary_fields(apps, schema_editor):
    GeneralStudyModuleAttr = apps.get_model("remapp", "GeneralStudyModuleAttr")

    for study in GeneralStudyModuleAttr.objects.filter(modality_type__exact="CT"):
        try:
            study.number_of_events = study.ctradiationdose_set.get().ctirradiationeventdata_set.count()
            study.save()
        except ObjectDoesNotExist:
            pass
        study.number_of_axial = 0
        study.number_of_spiral = 0
        study.number_of_stationary = 0
        study.number_of_const_angle = 0
        try:
            events = study.ctradiationdose_set.get().ctirradiationeventdata_set.order_by('pk')
            study.number_of_axial += events.filter(ct_acquisition_type__code_value__exact='113804').count()
            study.number_of_spiral += events.filter(ct_acquisition_type__code_value__exact='116152004').count()
            study.number_of_spiral += events.filter(ct_acquisition_type__code_value__exact='P5-08001').count()
            study.number_of_spiral += events.filter(ct_acquisition_type__code_value__exact='C0860888').count()
            study.number_of_stationary += events.filter(ct_acquisition_type__code_value__exact='113806').count()
            study.number_of_const_angle += events.filter(ct_acquisition_type__code_value__exact='113805').count()
        except ObjectDoesNotExist:
            pass
        study.save()
        try:
            study.total_dlp = study.ctradiationdose_set.get().ctaccumulateddosedata_set.get(
                ).ct_dose_length_product_total
            study.save()
        except ObjectDoesNotExist:
            pass


class Migration(migrations.Migration):

    dependencies = [
        ('remapp', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(populate_ct_summary_fields),
    ]
