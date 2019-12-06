# This Python file uses the following encoding: utf-8
# test_rf_detail.py

import os
from decimal import Decimal
from django.contrib.auth.models import User, Group
from django.urls import reverse
from django.test import RequestFactory, TestCase, override_settings
from remapp.extractors import rdsr
from remapp.models import PatientIDSettings, GeneralStudyModuleAttr, HighDoseMetricAlertSettings
from remapp.interface.mod_filters import RFSummaryListFilter


@override_settings(LANGUAGE_CODE='en-us')
class SummaryTotalDoses(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.user = User.objects.create_user(
            username='temporary', email='temporary@…', password='temporary')
        vg = Group(name="admingroup")
        vg.save()
        vg.user_set.add(self.user)
        vg.save()

        pid = PatientIDSettings.objects.create()
        pid.save()

        rf_siemens_zee_20160512 = os.path.join("test_files", "RF-RDSR-Siemens-Zee.dcm")
        rf_philips_allura = os.path.join("test_files", "RF-RDSR-Philips_Allura.dcm")

        root_tests = os.path.dirname(os.path.abspath(__file__))
        rdsr.rdsr(os.path.join(root_tests, rf_siemens_zee_20160512))
        rdsr.rdsr(os.path.join(root_tests, rf_philips_allura))

    def test_summary_total_dose_table(self):
        """Test the summary total dose table
        """
        self.client.login(username='temporary', password='temporary')

        # Obtain the private keys of the two studies.
        # The Siemens test study date is 2016-05-12
        # The Philips test study date is 2016-03-15
        filter_set = ''
        f = RFSummaryListFilter(filter_set, queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact='RF').order_by().distinct())
        pk_20160512 = f.qs.filter(study_date='2016-05-12').values_list('pk', flat=True)[0]
        pk_20160315 = f.qs.filter(study_date='2016-03-15').values_list('pk', flat=True)[0]

        response = self.client.get(reverse('rf_detail_view', kwargs={'pk':pk_20160512}), follow=True)

        summary_table_text = [[u'Fluoroscopy',
                               Decimal('16.000000000000'),
                               Decimal('0.002520000000'),
                               Decimal('28.00')],
                              [u'Acquisition', Decimal('0E-12'), Decimal('0E-12'), Decimal('0E-8')],
                              [u'Total',
                               Decimal('16.000000000000'),
                               Decimal('0.002520000000'),
                               Decimal('28.00000000')]]

        self.assertEqual(response.context['study_totals'], summary_table_text)

        summary_philips = [[u'Fluoroscopy',
                            Decimal('10.558274000000'),
                            Decimal('0.000293081169'),
                            Decimal('13.00')],
                           [u'Acquisition',
                            Decimal('143.010366000000'),
                            Decimal('0.003978199182'),
                            Decimal('14.75000000')],
                           [u'- Stationary Acquisition',
                            Decimal('143.0104000000'),
                            Decimal('0.003978199182'),
                            Decimal('14.75000000')],
                           [u'Total',
                            Decimal('153.568640000000'),
                            Decimal('0.004271280351'),
                            Decimal('27.75000000')]]

        response_philips = self.client.get(reverse('rf_detail_view', kwargs={'pk': pk_20160315}), follow=True)
        self.assertEqual(response_philips.context['study_totals'], summary_philips)