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
..  module:: urls.
    :synopsis: Module to match URLs and pass over to views or export modules.

..  moduleauthor:: Ed McDonagh

"""

from django.conf.urls import url
from django.contrib.auth import views as auth_views

from . import views
from .exports import exportviews
from .netdicom import dicomviews

urlpatterns = [
    url(r'^$', views.openrem_home, name='home'),
    url(r'^hometotals/$', views.update_modality_totals, name='update_modality_totals'),
    url(r'^homestudies/$', views.update_latest_studies, name='update_latest_studies'),
    url(r'^homeworkload/$', views.update_study_workload, name='update_study_workload'),

    url(r'^rf/$', views.rf_summary_list_filter, name='rf_summary_list_filter'),
    url(r'^rf/chart/$', views.rf_summary_chart_data, name='rf_summary_chart_data'),
    url(r'^rf/(?P<pk>\d+)/$', views.rf_detail_view, name='rf_detail_view'),
    url(r'^rf/(?P<pk>\d+)/skin_map/$', views.rf_detail_view_skin_map, name='rf_detail_view_skin_map'),

    url(r'^ct/$', views.ct_summary_list_filter, name='ct_summary_list_filter'),
    url(r'^ct/chart/$', views.ct_summary_chart_data, name='ct_summary_chart_data'),
    url(r'^ct/(?P<pk>\d+)/$', views.ct_detail_view, name='ct_detail_view'),

    url(r'^dx/$', views.dx_summary_list_filter, name='dx_summary_list_filter'),
    url(r'^dx/chart/$', views.dx_summary_chart_data, name='dx_summary_chart_data'),
    url(r'^dx/(?P<pk>\d+)/$', views.dx_detail_view, name='dx_detail_view'),

    url(r'^mg/$', views.mg_summary_list_filter, name='mg_summary_list_filter'),
    url(r'^mg/chart/$', views.mg_summary_chart_data, name='mg_summary_chart_data'),
    url(r'^mg/(?P<pk>\d+)/$', views.mg_detail_view, name='mg_detail_view'),

    url(r'^viewdisplaynames/$', views.display_names_view, name='display_names_view'),

    url(r'^delete/(?P<pk>\d+)$', views.study_delete, name='study_delete'),
    url(r'^ptsize/sizeupload$', views.size_upload, name='size_upload'),
    url(r'^ptsize/sizeprocess/(?P<pk>\d+)/$', views.size_process, name='size_process'),
    url(r'^ptsize/sizeimports', views.size_imports, name='size_imports'),
    url(r'^ptsize/sizedelete', views.size_delete, name='size_delete'),
    url(r'^ptsize/sizeimport/abort/(?P<pk>\d+)$', views.size_abort, name='size_abort'),
    url(r'^ptsize/sizelogs/(?P<task_id>[a-f0-9-]{36})$', views.size_download, name='size_download'),
    url(r'^updatedisplaynames/$', views.display_name_update, name='display_name_update'),
    url(r'^populatedisplaynames$', views.display_name_populate, name='display_name_populate'),
    url(r'^populatefailedimportlist', views.failed_list_populate, name='failed_list_populate'),
    url(r'^misc/reprocessdual/(?P<pk>\d+)/$', views.reprocess_dual, name='reprocess_dual'),
    url(r'^review/(?P<equip_name_pk>\d+)/(?P<modality>\w+)/$', views.review_summary_list,
        name='review_summary_list'),
    url(r'^review/study$', views.review_study_details, name='review_study_details'),
    url(r'^review/studiesdelete$', views.review_studies_delete, name='review_studies_delete'),
    url(r'^review/equipmentlastdateandcount$', views.display_name_last_date_and_count,
        name='display_name_last_date_and_count'),
    url(r'^review/studiesequipdelete$', views.review_studies_equip_delete, name='review_studies_equip_delete'),
    url(r'^review/failed/(?P<modality>\w+)/$', views.review_failed_imports, name='review_failed_imports'),
    url(r'^review/failed/study$', views.review_failed_study_details, name='review_failed_study_details'),
    url(r'^review/studiesdeletefailed$', views.review_failed_studies_delete, name='review_failed_studies_delete'),
    url(r'^chartoptions/$', views.chart_options_view, name='chart_options_view'),
    url(r'^homepageoptions/$', views.homepage_options_view, name='homepage_options_view'),
    url(r'^settings/patientidsettings/(?P<pk>\d+)/$', views.PatientIDSettingsUpdate.as_view(),
        name='patient_id_settings_update'),
    url(r'^settings/dicomdelsettings/(?P<pk>\d+)/$', views.DicomDeleteSettingsUpdate.as_view(),
        name='dicom_delete_settings_update'),
    url(r'^settings/skindosemapsettings/(?P<pk>\d+)/$', views.SkinDoseMapCalcSettingsUpdate.as_view(),
        name='skin_dose_map_settings_update'),
    url(r'^settings/notpatientindicators/$', views.not_patient_indicators, name='not_patient_indicators'),
    url(r'^settings/notpatientindicators/restore074/$', views.not_patient_indicators_as_074,
        name='not_patient_indicators_as_074'),
    url(r'^settings/notpatientindicators/names/add/$', views.NotPatientNameCreate.as_view(),
        name='notpatientname_add'),
    url(r'^settings/notpatientindicators/names/(?P<pk>\d+)/$', views.NotPatientNameUpdate.as_view(),
        name='notpatientname_update'),
    url(r'^settings/notpatientindicators/names/(?P<pk>\d+)/delete/$', views.NotPatientNameDelete.as_view(),
        name='notpatientname_delete'),
    url(r'^settings/notpatientindicators/id/add/$', views.NotPatientIDCreate.as_view(),
        name='notpatienid_add'),
    url(r'^settings/notpatientindicators/id/(?P<pk>\d+)/$', views.NotPatientIDUpdate.as_view(),
        name='notpatientid_update'),
    url(r'^settings/notpatientindicators/id/(?P<pk>\d+)/delete/$', views.NotPatientIDDelete.as_view(),
        name='notpatientid_delete'),
    url(r'^settings/adminquestions/hide_not_patient/$', views.admin_questions_hide_not_patient,
        name='admin_questions_hide_not_patient'),
    url(r'^settings/rfalertsettings/(?P<pk>\d+)/$', views.RFHighDoseAlertSettings.as_view(),
        name='rf_alert_settings_update'),
    url(r'^settings/rfalertnotifications/$', views.rf_alert_notifications_view,
        name='rf_alert_notifications_view'),
    url(r'^settings/rfrecalculateaccumdoses/', views.rf_recalculate_accum_doses,
        name='rf_recalculate_accum_doses'),
    # url(r'^password/$', views.change_password, name='change_password'),
    url(r'^change_password/$', auth_views.password_change,
        {'template_name': 'registration/changepassword.html'}, name='password_change'),
    url(r'^change_password/done/$', auth_views.password_change_done,
        {'template_name': 'registration/changepassworddone.html'}, name='password_change_done'),
    url(r'^tasks/rabbitmq/purge_queue/(?P<queue>[0-9a-zA-Z.@-]+)$', views.rabbitmq_purge,
        name='rabbitmq_purge'),
    url(r'^tasks/celery/$', views.celery_admin, name='celery_admin'),
    url(r'^tasks/celery/tasks/(?P<stage>\w+)$', views.celery_tasks, name='celery_tasks'),
    url(r'^tasks/celery/abort_task/(?P<task_id>[0-9a-zA-Z.@-]+)/(?P<type>\w+)$', views.celery_abort,
        name='celery_abort'),
    url(r'^tasks/celery/service_status/$', views.task_service_status, name='task_service_status'),
    url(r'^migrate/populate_summary/$', views.populate_summary, name='populate_summary'),
    url(r'^migrate/populate_summary_progress/$', views.populate_summary_progress,
        name='populate_summary_progress'),
    url(r'^charts_off/$', views.charts_off, name='charts_off')
]

urlpatterns += [
    url(r'^export/$', exportviews.export, name='export'),
    url(r'^exportctcsv1/(?P<name>\w+)/(?P<pat_id>\w+)/$', exportviews.ctcsv1, name='ctcsv1'),
    url(r'^exportctxlsx1/(?P<name>\w+)/(?P<pat_id>\w+)/$', exportviews.ctxlsx1, name='ctxlsx1'),
    url(r'^exportctphe2019/$', exportviews.ct_xlsx_phe2019, name='ct_xlsx_phe2019'),
    url(r'^exportdxcsv1/(?P<name>\w+)/(?P<pat_id>\w+)/$', exportviews.dxcsv1, name='dxcsv1'),
    url(r'^exportdxxlsx1/(?P<name>\w+)/(?P<pat_id>\w+)/$', exportviews.dxxlsx1, name='dxxlsx1'),
    url(r'^exportdxphe2019/(?P<export_type>\w+)/$', exportviews.dx_xlsx_phe2019, name='dx_xlsx_phe2019'),
    url(r'^exportflcsv1/(?P<name>\w+)/(?P<pat_id>\w+)/$', exportviews.flcsv1, name='flcsv1'),
    url(r'^exportrfxlsx1/(?P<name>\w+)/(?P<pat_id>\w+)/$', exportviews.rfxlsx1, name='rfxlsx1'),
    url(r'^exportrfopenskin/(?P<pk>\d+)$', exportviews.rfopenskin, name='rfopenskin'),
    url(r'^exportrfphe2019/$', exportviews.rf_xlsx_phe2019, name='rf_xlsx_phe2019'),
    url(r'^exportmgcsv1/(?P<name>\w+)/(?P<pat_id>\w+)/$', exportviews.mgcsv1, name='mgcsv1'),
    url(r'^exportmgxlsx1/(?P<name>\w+)/(?P<pat_id>\w+)/$', exportviews.mgxlsx1, name='mgxlsx1'),
    url(r'^exportmgnhsbsp/$', exportviews.mgnhsbsp, name='mgnhsbsp'),
    url(r'^download/(?P<task_id>[a-f0-9-]{36})$', exportviews.download, name='download'),
    url(r'^deletefile/$', exportviews.deletefile, name='deletefile'),
    url(r'^export/abort/(?P<pk>\d+)$', exportviews.export_abort, name='export_abort'),
    url(r'^export/updateactive$', exportviews.update_active, name='update_active'),
    url(r'^export/updateerror$', exportviews.update_error, name='update_error'),
    url(r'^export/updatecomplete$', exportviews.update_complete, name='update_complete'),
]

urlpatterns += [
    url(r'^dicom/summary', views.dicom_summary, name='dicom_summary'),
    url(r'^dicom/store/add/$', views.DicomStoreCreate.as_view(), name='dicomstore_add'),
    url(r'^dicom/store/(?P<pk>\d+)/$', views.DicomStoreUpdate.as_view(), name='dicomstore_update'),
    url(r'^dicom/store/(?P<pk>\d+)/delete/$', views.DicomStoreDelete.as_view(), name='dicomstore_delete'),
    url(r'^dicom/store/(?P<pk>\d+)/start/$', dicomviews.run_store, name='run_store'),
    url(r'^dicom/store/(?P<pk>\d+)/stop/$', dicomviews.stop_store, name='stop_store'),
    url(r'^dicom/store/statusupdate', dicomviews.status_update_store, name='status_update_store'),
    url(r'^dicom/qr/add/$', views.DicomQRCreate.as_view(), name='dicomqr_add'),
    url(r'^dicom/qr/(?P<pk>\d+)/$', views.DicomQRUpdate.as_view(), name='dicomqr_update'),
    url(r'^dicom/qr/(?P<pk>\d+)/delete/$', views.DicomQRDelete.as_view(), name='dicomqr_delete'),
    url(r'^dicom/queryupdate$', dicomviews.q_update, name='query_update'),
    url(r'^dicom/queryprocess$', dicomviews.q_process, name='q_proces'),
    url(r'^dicom/queryremote$', dicomviews.dicom_qr_page, name='dicom_qr_page'),
    url(r'^dicom/queryretrieve$', dicomviews.r_start, name='start_retrieve'),
    url(r'^dicom/moveupdate$', dicomviews.r_update, name='move_update'),
]
