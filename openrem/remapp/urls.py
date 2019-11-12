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

from django.contrib.auth import views as auth_views
from django.urls import include, path

from . import views
from .exports import exportviews
from .netdicom import dicomviews

main_patterns = [
    path('rf/<int:pk>/', views.rf_detail_view, name='rf_detail_view'),
    path('', views.openrem_home, name='home'),
    path('hometotals/', views.update_modality_totals, name='update_modality_totals'),
    path('homestudies/', views.update_latest_studies, name='update_latest_studies'),
    path('homeworkload/', views.update_study_workload, name='update_study_workload'),
    path('rf/', views.rf_summary_list_filter, name='rf_summary_list_filter'),
    path('rf/chart/', views.rf_summary_chart_data, name='rf_summary_chart_data'),
    path('rf/<int:pk>/skin_map/', views.rf_detail_view_skin_map, name='rf_detail_view_skin_map'),
    path('ct/', views.ct_summary_list_filter, name='ct_summary_list_filter'),
    path('ct/chart/', views.ct_summary_chart_data, name='ct_summary_chart_data'),
    path('ct/<int:pk>/', views.ct_detail_view, name='ct_detail_view'),
    path('dx/', views.dx_summary_list_filter, name='dx_summary_list_filter'),
    path('dx/chart/', views.dx_summary_chart_data, name='dx_summary_chart_data'),
    path('dx/<int:pk>/', views.dx_detail_view, name='dx_detail_view'),
    path('mg/', views.mg_summary_list_filter, name='mg_summary_list_filter'),
    path('mg/chart/', views.mg_summary_chart_data, name='mg_summary_chart_data'),
    path('mg/<int:pk>/', views.mg_detail_view, name='mg_detail_view'),
    path('viewdisplaynames/', views.display_names_view, name='display_names_view'),
    path('delete/<int:pk>', views.study_delete, name='study_delete'),
    path('updatedisplaynames/', views.display_name_update, name='display_name_update'),
    path('populatedisplaynames', views.display_name_populate, name='display_name_populate'),
    path('populatefailedimportlist', views.failed_list_populate, name='failed_list_populate'),
    path('misc/reprocessdual/<int:pk>/', views.reprocess_dual, name='reprocess_dual'),
    path('change_password/', auth_views.PasswordChangeView, {'template_name': 'registration/changepassword.html'}, name='password_change'),
    path('change_password/done/', auth_views.PasswordChangeDoneView, {'template_name': 'registration/changepassworddone.html'}, name='password_change_done'),
    path('migrate/populate_summary/', views.populate_summary, name='populate_summary'),
    path('migrate/populate_summary_progress/', views.populate_summary_progress, name='populate_summary_progress'),
]


tasks_patterns = [
    path('rabbitmq/purge_queue/<str:queue>/', views.rabbitmq_purge, name='rabbitmq_purge'),
    path('celery/', views.celery_admin, name='celery_admin'),
    path('celery/tasks/<str:stage>/', views.celery_tasks, name='celery_tasks'),
    path('celery/abort_task/<uuid:task_id>/<str:type>/', views.celery_abort, name='celery_abort'),
    path('celery/service_status/', views.task_service_status, name='task_service_status'),
]


review_patterns = [
    path('<int:equip_name_pk>/<str:modality>/', views.review_summary_list, name='review_summary_list'),
    path('study', views.review_study_details, name='review_study_details'),
    path('studiesdelete', views.review_studies_delete, name='review_studies_delete'),
    path('equipmentlastdateandcount', views.display_name_last_date_and_count, name='display_name_last_date_and_count'),
    path('studiesequipdelete', views.review_studies_equip_delete, name='review_studies_equip_delete'),
    path('failed/<str:modality>/', views.review_failed_imports, name='review_failed_imports'),
    path('failed/study', views.review_failed_study_details, name='review_failed_study_details'),
    path('studiesdeletefailed', views.review_failed_studies_delete, name='review_failed_studies_delete'),
]


patient_size_patterns = [
    path('sizeupload/', views.size_upload, name='size_upload'),
    path('sizeprocess/<int:pk>/', views.size_process, name='size_process'),
    path('sizeimports/', views.size_imports, name='size_imports'),
    path('sizedelete/', views.size_delete, name='size_delete'),
    path('sizeimport/abort/<int:pk>/', views.size_abort, name='size_abort'),
    path('sizelogs/<uuid:task_id>/', views.size_download, name='size_download'),
]


settings_not_patient_indicators_patterns = [
    path('', views.not_patient_indicators, name='not_patient_indicators'),
    path('restore074/', views.not_patient_indicators_as_074, name='not_patient_indicators_as_074'),
    path('names/add/', views.NotPatientNameCreate.as_view(), name='notpatientname_add'),
    path('names/<int:pk>/', views.NotPatientNameUpdate.as_view(), name='notpatientname_update'),
    path('names/<int:pk>/delete/', views.NotPatientNameDelete.as_view(), name='notpatientname_delete'),
    path('id/add/', views.NotPatientIDCreate.as_view(), name='notpatienid_add'),
    path('id/<int:pk>/', views.NotPatientIDUpdate.as_view(), name='notpatientid_update'),
    path('id/<int:pk>/delete/', views.NotPatientIDDelete.as_view(), name='notpatientid_delete'),
]

settings_patterns = [
    path('charts_off/', views.charts_off, name='charts_off'),
    path('chartoptions/', views.chart_options_view, name='chart_options_view'),
    path('homepageoptions/', views.homepage_options_view, name='homepage_options_view'),
    path('patientidsettings/<int:pk>/', views.PatientIDSettingsUpdate.as_view(), name='patient_id_settings_update'),
    path('dicomdelsettings/<int:pk>/', views.DicomDeleteSettingsUpdate.as_view(), name='dicom_delete_settings_update'),
    path('skindosemapsettings/<int:pk>/', views.SkinDoseMapCalcSettingsUpdate.as_view(), name='skin_dose_map_settings_update'),
    path('adminquestions/hide_not_patient/', views.admin_questions_hide_not_patient, name='admin_questions_hide_not_patient'),
    path('rfalertsettings/<int:pk>/', views.RFHighDoseAlertSettings.as_view(), name='rf_alert_settings_update'),
    path('rfalertnotifications/', views.rf_alert_notifications_view, name='rf_alert_notifications_view'),
    path('rfrecalculateaccumdoses/', views.rf_recalculate_accum_doses, name='rf_recalculate_accum_doses'),
    path('notpatientindicators/', include(settings_not_patient_indicators_patterns)),
]


export_patterns = [
    path('', exportviews.export, name='export'),
    path('ctcsv1/<int:name>/<int:pat_id>/', exportviews.ctcsv1, name='ctcsv1'),
    path('ctxlsx1/<int:name>/<int:pat_id>/', exportviews.ctxlsx1, name='ctxlsx1'),
    path('ctphe2019/', exportviews.ct_xlsx_phe2019, name='ct_xlsx_phe2019'),
    path('dxcsv1/<int:name>/<int:pat_id>/', exportviews.dxcsv1, name='dxcsv1'),
    path('dxxlsx1/<int:name>/<int:pat_id>/', exportviews.dxxlsx1, name='dxxlsx1'),
    path('dxphe2019/<str:export_type>/', exportviews.dx_xlsx_phe2019, name='dx_xlsx_phe2019'),
    path('flcsv1/<int:name>/<int:pat_id>/', exportviews.flcsv1, name='flcsv1'),
    path('rfxlsx1/<int:name>/<int:pat_id>/', exportviews.rfxlsx1, name='rfxlsx1'),
    path('rfopenskin/<int:pk>/', exportviews.rfopenskin, name='rfopenskin'),
    path('rfphe2019/', exportviews.rf_xlsx_phe2019, name='rf_xlsx_phe2019'),
    path('mgcsv1/<int:name>/<int:pat_id>/', exportviews.mgcsv1, name='mgcsv1'),
    path('mgxlsx1/<int:name>/<int:pat_id>/', exportviews.mgxlsx1, name='mgxlsx1'),
    path('mgnhsbsp/', exportviews.mgnhsbsp, name='mgnhsbsp'),
    path('download/<uuid:task_id>/', exportviews.download, name='download'),
    path('deletefile/', exportviews.deletefile, name='deletefile'),
    path('abort/<int:pk>/', exportviews.export_abort, name='export_abort'),
    path('updateactive/', exportviews.update_active, name='update_active'),
    path('updateerror/', exportviews.update_error, name='update_error'),
    path('updatecomplete/', exportviews.update_complete, name='update_complete'),
]

dicom_patterns = [
    path('summary', views.dicom_summary, name='dicom_summary'),
    path('store/add/', views.DicomStoreCreate.as_view(), name='dicomstore_add'),
    path('store/<int:pk>/', views.DicomStoreUpdate.as_view(), name='dicomstore_update'),
    path('store/<int:pk>/delete/', views.DicomStoreDelete.as_view(), name='dicomstore_delete'),
    path('store/<int:pk>/start/', dicomviews.run_store, name='run_store'),
    path('store/<int:pk>/stop/', dicomviews.stop_store, name='stop_store'),
    path('store/statusupdate', dicomviews.status_update_store, name='status_update_store'),
    path('qr/add/', views.DicomQRCreate.as_view(), name='dicomqr_add'),
    path('qr/<int:pk>/', views.DicomQRUpdate.as_view(), name='dicomqr_update'),
    path('qr/<int:pk>/delete/', views.DicomQRDelete.as_view(), name='dicomqr_delete'),
    path('queryupdate', dicomviews.q_update, name='query_update'),
    path('queryprocess', dicomviews.q_process, name='q_proces'),
    path('queryremote', dicomviews.dicom_qr_page, name='dicom_qr_page'),
    path('queryretrieve', dicomviews.r_start, name='start_retrieve'),
    path('moveupdate', dicomviews.r_update, name='move_update'),
]

urlpatterns = [
    path('', include(main_patterns)),
    path('export/', include(export_patterns)),
    path('dicom/', include(dicom_patterns)),
    path('settings/', include(settings_patterns)),
    path('ptsize/', include(patient_size_patterns)),
    path('review/', include(review_patterns)),
    path('tasks/', include(tasks_patterns)),
]
