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
from django.conf.urls import url
from django import views as django_views

from . import (
    views,
    views_admin,
    views_charts_ct,
    views_charts_nm,
    views_charts_dx,
    views_charts_mg,
    views_charts_rf,
    views_openskin,
)
from .exports import exportviews
from .netdicom import dicomviews
from .extractors import import_views


main_patterns = [
    path(
        "rf/<int:pk>/force_recalculation/", views.rf_detail_view, name="rf_detail_view"
    ),
    path("rf/<int:pk>/", views.rf_detail_view, name="rf_detail_view"),
    path("", views.openrem_home, name="home"),
    path("homestudies/", views.update_latest_studies, name="update_latest_studies"),
    path("rf/", views.rf_summary_list_filter, name="rf_summary_list_filter"),
    path(
        "rf/chart/", views_charts_rf.rf_summary_chart_data, name="rf_summary_chart_data"
    ),
    path(
        "rf/<int:pk>/skin_map/",
        views.rf_detail_view_skin_map,
        name="rf_detail_view_skin_map",
    ),
    path(
        "rf/<int:pk>/force_recalculation/skin_map/",
        views.rf_detail_view_skin_map,
        name="rf_detail_view_skin_map_force_recalculation",
    ),
    path("nm/", views.nm_summary_list_filter, name="nm_summary_list_filter"),
    path(
        "nm/chart/", views_charts_nm.nm_summary_chart_data, name="nm_summary_chart_data"
    ),
    path("nm/<int:pk>/", views.nm_detail_view, name="nm_detail_view"),
    path("ct/", views.ct_summary_list_filter, name="ct_summary_list_filter"),
    path(
        "ct/chart/", views_charts_ct.ct_summary_chart_data, name="ct_summary_chart_data"
    ),
    path("ct/<int:pk>/", views.ct_detail_view, name="ct_detail_view"),
    path("dx/", views.dx_summary_list_filter, name="dx_summary_list_filter"),
    path(
        "dx/chart/", views_charts_dx.dx_summary_chart_data, name="dx_summary_chart_data"
    ),
    path("dx/<int:pk>/", views.dx_detail_view, name="dx_detail_view"),
    path("mg/", views.mg_summary_list_filter, name="mg_summary_list_filter"),
    path(
        "mg/chart/", views_charts_mg.mg_summary_chart_data, name="mg_summary_chart_data"
    ),
    path("mg/<int:pk>/", views.mg_detail_view, name="mg_detail_view"),
    path(
        "viewdisplaynames/", views_admin.display_names_view, name="display_names_view"
    ),
    path("delete/<int:pk>", views_admin.study_delete, name="study_delete"),
    path(
        "updatedisplaynames/",
        views_admin.display_name_update,
        name="display_name_update",
    ),
    path(
        "populatedisplaynames",
        views_admin.display_name_populate,
        name="display_name_populate",
    ),
    path(
        "populatefailedimportlist",
        views_admin.failed_list_populate,
        name="failed_list_populate",
    ),
    path(
        "misc/reprocessdual/<int:pk>/",
        views_admin.reprocess_dual,
        name="reprocess_dual",
    ),
    path(
        "change_password/",
        auth_views.PasswordChangeView.as_view(
            template_name="registration/changepassword.html"
        ),
        name="password_change",
    ),
    path(
        "change_password/done/",
        auth_views.PasswordChangeDoneView.as_view(
            template_name="registration/changepassworddone.html"
        ),
        name="password_change_done",
    ),
    path(
        "migrate/populate_summary/",
        views_admin.populate_summary,
        name="populate_summary",
    ),
    path(
        "migrate/populate_summary_progress/",
        views_admin.populate_summary_progress,
        name="populate_summary_progress",
    ),
]


tasks_patterns = [
    path("tasks/task_admin/", views_admin.display_tasks, name="task_admin"),
    path("tasks/get_tasks/<str:stage>/", views_admin.tasks, name="get_tasks"),
    path(
        "tasks/abort_task/<uuid:task_id>/",
        views_admin.task_abort,
        name="abort_task",
    ),
    path(
        "tasks/remove_task/<uuid:task_id>/", views_admin.task_remove, name="remove_task"
    ),
]


review_patterns = [
    path(
        "<int:equip_name_pk>/<str:modality>/",
        views_admin.review_summary_list,
        name="review_summary_list",
    ),
    path("study", views_admin.review_study_details, name="review_study_details"),
    path(
        "studiesdelete", views_admin.review_studies_delete, name="review_studies_delete"
    ),
    path(
        "equipmentlastdateandcount",
        views_admin.display_name_last_date_and_count,
        name="display_name_last_date_and_count",
    ),
    path(
        "studiesequipdelete",
        views_admin.review_studies_equip_delete,
        name="review_studies_equip_delete",
    ),
    path(
        "failed/<str:modality>/",
        views_admin.review_failed_imports,
        name="review_failed_imports",
    ),
    path(
        "failed/study",
        views_admin.review_failed_study_details,
        name="review_failed_study_details",
    ),
    path(
        "studiesdeletefailed",
        views_admin.review_failed_studies_delete,
        name="review_failed_studies_delete",
    ),
    path(
        "equipmentskinmapenabled/",
        views_openskin.display_name_skin_enabled,
        name="display_name_skin_enabled",
    ),
    path(
        "openskinsafelist/add/<int:equip_name_pk>/",
        views_openskin.SkinSafeListCreate.as_view(),
        name="skinsafelist_add",
    ),
    path(
        "openskinsafelist/<int:pk>/<int:equip_name_pk>/",
        views_openskin.SkinSafeListUpdate.as_view(),
        name="skinsafelist_update",
    ),
    path(
        "openskinsafelist/<int:pk>/delete/",
        views_openskin.SkinSafeListDelete.as_view(),
        name="skinsafelist_delete",
    ),
]


patient_size_patterns = [
    path("sizeupload/", import_views.size_upload, name="size_upload"),
    path("sizeprocess/<int:pk>/", import_views.size_process, name="size_process"),
    path("sizeimports/", import_views.size_imports, name="size_imports"),
    path("sizedelete/", import_views.size_delete, name="size_delete"),
    path("sizeimport/abort/<int:pk>/", import_views.size_abort, name="size_abort"),
    path("sizelogs/<uuid:task_id>/", import_views.size_download, name="size_download"),
]


settings_not_patient_indicators_patterns = [
    path("", views_admin.not_patient_indicators, name="not_patient_indicators"),
    path(
        "restore074/",
        views_admin.not_patient_indicators_as_074,
        name="not_patient_indicators_as_074",
    ),
    path(
        "names/add/",
        views_admin.NotPatientNameCreate.as_view(),
        name="notpatientname_add",
    ),
    path(
        "names/<int:pk>/",
        views_admin.NotPatientNameUpdate.as_view(),
        name="notpatientname_update",
    ),
    path(
        "names/<int:pk>/delete/",
        views_admin.NotPatientNameDelete.as_view(),
        name="notpatientname_delete",
    ),
    path("id/add/", views_admin.NotPatientIDCreate.as_view(), name="notpatienid_add"),
    path(
        "id/<int:pk>/",
        views_admin.NotPatientIDUpdate.as_view(),
        name="notpatientid_update",
    ),
    path(
        "id/<int:pk>/delete/",
        views_admin.NotPatientIDDelete.as_view(),
        name="notpatientid_delete",
    ),
]

settings_patterns = [
    path("charts_toggle/", views_admin.charts_toggle, name="charts_toggle"),
    path("chartoptions/", views_admin.chart_options_view, name="chart_options_view"),
    path(
        "homepageoptions/",
        views_admin.homepage_options_view,
        name="homepage_options_view",
    ),
    path(
        "patientidsettings/<int:pk>/",
        views_admin.PatientIDSettingsUpdate.as_view(),
        name="patient_id_settings_update",
    ),
    path(
        "dicomdelsettings/<int:pk>/",
        views_admin.DicomDeleteSettingsUpdate.as_view(),
        name="dicom_delete_settings_update",
    ),
    path(
        "skindosemapsettings/<int:pk>/",
        views_openskin.SkinDoseMapCalcSettingsUpdate.as_view(),
        name="skin_dose_map_settings_update",
    ),
    path(
        "adminquestions/hide_not_patient/",
        views_admin.admin_questions_hide_not_patient,
        name="admin_questions_hide_not_patient",
    ),
    path(
        "rfalertsettings/<int:pk>/",
        views_admin.RFHighDoseAlertSettings.as_view(),
        name="rf_alert_settings_update",
    ),
    path(
        "rfalertnotifications/",
        views_admin.rf_alert_notifications_view,
        name="rf_alert_notifications_view",
    ),
    path(
        "rfrecalculateaccumdoses/",
        views_admin.rf_recalculate_accum_doses,
        name="rf_recalculate_accum_doses",
    ),
    path("notpatientindicators/", include(settings_not_patient_indicators_patterns)),
    path(
        "background_task_settings/<int:pk>/",
        views_admin.BackgroundTaskMaximumRowsUpdate.as_view(),
        name="background_task_settings",
    ),
]


export_patterns = [
    path("", exportviews.export, name="export"),
    path("ctcsv1/<int:name>/<int:pat_id>/", exportviews.ctcsv1, name="ctcsv1"),
    path("ctxlsx1/<int:name>/<int:pat_id>/", exportviews.ctxlsx1, name="ctxlsx1"),
    path("ctphe2019/", exportviews.ct_xlsx_phe2019, name="ct_xlsx_phe2019"),
    path("nmcsv1/<int:name>/<int:pat_id>/", exportviews.nmcsv1, name="nmcsv1"),
    path("nmxlsx1/<int:name>/<int:pat_id>/", exportviews.nmxlsx1, name="nmxlsx1"),
    path("dxcsv1/<int:name>/<int:pat_id>/", exportviews.dxcsv1, name="dxcsv1"),
    path("dxxlsx1/<int:name>/<int:pat_id>/", exportviews.dxxlsx1, name="dxxlsx1"),
    path(
        "dxphe2019/<str:export_type>/",
        exportviews.dx_xlsx_phe2019,
        name="dx_xlsx_phe2019",
    ),
    path("flcsv1/<int:name>/<int:pat_id>/", exportviews.flcsv1, name="flcsv1"),
    path("rfxlsx1/<int:name>/<int:pat_id>/", exportviews.rfxlsx1, name="rfxlsx1"),
    path("rfopenskin/<int:pk>/", exportviews.rfopenskin, name="rfopenskin"),
    path("rfphe2019/", exportviews.rf_xlsx_phe2019, name="rf_xlsx_phe2019"),
    path("mgcsv1/<int:name>/<int:pat_id>/", exportviews.mgcsv1, name="mgcsv1"),
    path("mgxlsx1/<int:name>/<int:pat_id>/", exportviews.mgxlsx1, name="mgxlsx1"),
    path("mgnhsbsp/", exportviews.mgnhsbsp, name="mgnhsbsp"),
    path("download/<uuid:task_id>/", exportviews.download, name="download"),
    path("deletefile/", exportviews.deletefile, name="deletefile"),
    path("abort/<str:pk>/", exportviews.export_abort, name="export_abort"),
    path("remove/<uuid:task_id>/", exportviews.export_remove, name="export_remove"),
    path("updatequeue/", exportviews.update_queue, name="update_queue"),
    path("updateactive/", exportviews.update_active, name="update_active"),
    path("updateerror/", exportviews.update_error, name="update_error"),
    path("updatecomplete/", exportviews.update_complete, name="update_complete"),
]

dicom_patterns = [
    path("summary", dicomviews.dicom_summary, name="dicom_summary"),
    path("store/add/", dicomviews.DicomStoreCreate.as_view(), name="dicomstore_add"),
    path(
        "store/<int:pk>/",
        dicomviews.DicomStoreUpdate.as_view(),
        name="dicomstore_update",
    ),
    path(
        "store/<int:pk>/delete/",
        dicomviews.DicomStoreDelete.as_view(),
        name="dicomstore_delete",
    ),
    path(
        "store/statusupdate", dicomviews.status_update_store, name="status_update_store"
    ),
    path("qr/add/", dicomviews.DicomQRCreate.as_view(), name="dicomqr_add"),
    path("qr/<int:pk>/", dicomviews.DicomQRUpdate.as_view(), name="dicomqr_update"),
    path(
        "qr/<int:pk>/delete/", dicomviews.DicomQRDelete.as_view(), name="dicomqr_delete"
    ),
    path("queryupdate", dicomviews.q_update, name="query_update"),
    path("queryprocess", dicomviews.q_process, name="q_process"),
    path("queryremote", dicomviews.dicom_qr_page, name="dicom_qr_page"),
    path("queryretrieve", dicomviews.r_start, name="start_retrieve"),
    path("moveupdate", dicomviews.r_update, name="move_update"),
    path("qrnodestatus", dicomviews.get_qr_status, name="get_qr_status"),
    path("storenodestatus", dicomviews.get_store_status, name="get_store_status"),
    path("querysummary", dicomviews.get_query_summary, name="get_query_summary"),
    path(
        "querydetails/<int:pk>", dicomviews.get_query_details, name="get_query_details"
    ),
    path("queryseries/<int:pk>", dicomviews.get_query_series, name="get_query_series"),
    path("queryimages/<int:pk>", dicomviews.get_query_images, name="get_query_images"),
    path("deletequeries/", dicomviews.delete_queries, name="delete_queries"),
]

import_patterns = [  # pylint: disable=invalid-name
    path("from_docker/", import_views.import_from_docker, name="import_docker")
]

standard_name_patterns = [  # pylint: disable=invalid-name
    path("add_name_ct/", views_admin.StandardNameAddCT.as_view(), name="add_name_ct"),
    path("add_name_dx/", views_admin.StandardNameAddDX.as_view(), name="add_name_dx"),
    path("add_name_rf/", views_admin.StandardNameAddRF.as_view(), name="add_name_rf"),
    path("add_name_mg/", views_admin.StandardNameAddMG.as_view(), name="add_name_mg"),
    path(
        "update_all_std_names/<str:modality>/",
        views_admin.standard_name_update_all,
        name="update_all_std_names",
    ),
    path(
        "update_all_std_names_form/<str:modality>/",
        views_admin.standard_name_update_all_form,
        name="update_all_std_names_form",
    ),
    path(
        "update_name/<int:std_name_pk>/<str:modality>/",
        views_admin.standard_name_update,
        name="update_name",
    ),
    path(
        "update_name_ct/<int:pk>/",
        views_admin.StandardNameUpdateCT.as_view(),
        name="update_name_ct",
    ),
    path(
        "update_name_dx/<int:pk>/",
        views_admin.StandardNameUpdateDX.as_view(),
        name="update_name_dx",
    ),
    path(
        "update_name_rf/<int:pk>/",
        views_admin.StandardNameUpdateRF.as_view(),
        name="update_name_rf",
    ),
    path(
        "update_name_mg/<int:pk>/",
        views_admin.StandardNameUpdateMG.as_view(),
        name="update_name_mg",
    ),
    path(
        "view_standard_names/",
        views_admin.standard_names_view,
        name="standard_names_view",
    ),
    path(
        "populate_standard_names",
        views_admin.standard_names_populate,
        name="standard_names_populate",
    ),
    path(
        "standard_name/<int:pk>/delete/",
        views_admin.StandardNameDelete.as_view(),
        name="standard_name_delete",
    ),
    path(
        "standard_name_settings/<int:pk>/",
        views_admin.StandardNameSettingsUpdate.as_view(),
        name="standard_name_settings",
    ),
]

urlpatterns = [
    path("", include(main_patterns)),
    path("export/", include(export_patterns)),
    path("dicom/", include(dicom_patterns)),
    path("settings/", include(settings_patterns)),
    path("ptsize/", include(patient_size_patterns)),
    path("review/", include(review_patterns)),
    path("tasks/", include(tasks_patterns)),
    path("import/", include(import_patterns)),
    path("names/", include(standard_name_patterns)),
    path('save-ct-comment/', views.save_ct_comment, name='save_ct_comment'),
    url(r"^jsi18n/$", django_views.i18n.JavaScriptCatalog.as_view(), name="jsi18n"),
]
