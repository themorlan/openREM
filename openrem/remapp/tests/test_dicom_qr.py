# This Python file uses the following encoding: utf-8
# dicom_qr.py

import collections
import os
import uuid

from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from django.test import TestCase
from mock import patch, MagicMock
from testfixtures import LogCapture

from ..extractors import rdsr
from ..models import (
    DicomQuery,
    DicomQRRspStudy,
    DicomQRRspSeries,
    DicomQRRspImage,
    DicomRemoteQR,
    DicomStoreSCP,
    GeneralStudyModuleAttr,
    PatientIDSettings,
)
from ..netdicom import qrscu


def _fake_check_sr_type_in_study_with_rdsr(
    ae, remote, assoc, study, query, get_empty_sr
):
    return "RDSR"


fake_responses = [
    [["MG", "SR"], ["MG"], ["OT", "MG"], ["PR", "MG"]],
    [["CT"], ["OT", "CT", "SR"], ["SR", "CT"]],
]


def _fake_two_modalities(ae, remote, assoc, d, query, study_query_id, *args, **kwargs):
    """
    Mock routine that returns a set of four MG studies the first time it is called, and a set of three CT studies the
    second time  it is called.

    Used by test_modality_matching

    :param my_ae:       Not used in mock
    :param remote_ae:   Not used in mock
    :param d:           Not used in mock
    :param query:       Database foreign key to create DicomQRRspStudy objects
    :param study_query_id:    Query ID to tie DicomQRRspStudy from this query together
    :param args:        Not used in mock
    :param kwargs:      Not used in mock
    :return:            Seven MG and CT DicomQRRspStudy objects in the database
    """
    mods = fake_responses.pop()
    for mod_list in mods:
        rsp = DicomQRRspStudy.objects.create(dicom_query=query)
        rsp.query_id = study_query_id
        rsp.set_modalities_in_study(mod_list)
        rsp.save()


def _fake_all_modalities(ae, remote, assoc, d, query, study_query_id, *args, **kwargs):
    """
    Mock routine to return a modality response that includes a study with a 'modalities in study' that does not have
    the requested modality in.

    Used by test_non_modality_matching

    :param my_ae:       Not used in mock
    :param remote_ae:   Not used in mock
    :param d:           Not used in mock
    :param query:       Database foreign key to create DicomQRRspStudy objects
    :param study_query_id:    Query ID to tie DicomQRRspStudy from this query together
    :param args:        Not used in mock
    :param kwargs:      Not used in mock
    :return:            Two DicomQRRspStudy objects in the database
    """
    mods = [["MG", "SR"], ["US", "SR"]]
    for mod_list in mods:
        rsp = DicomQRRspStudy.objects.create(dicom_query=query)
        rsp.query_id = study_query_id
        rsp.set_modalities_in_study(mod_list)
        rsp.save()


class StudyQueryLogic(TestCase):
    def setUp(self):
        # Remote find/move node details
        qr_scp = DicomRemoteQR.objects.create()
        qr_scp.hostname = "qrserver"
        qr_scp.port = 104
        qr_scp.aetitle = "qrserver"
        qr_scp.callingaet = "openrem"
        qr_scp.save()
        # Local store node details
        store_scp = DicomStoreSCP.objects.create()
        store_scp.aetitle = "openremstore"
        store_scp.port = 104
        store_scp.save()
        # Query db object
        query_id = uuid.uuid4()
        query = DicomQuery.objects.create()
        query.query_id = query_id
        query.complete = False
        query.store_scp_fk = store_scp
        query.qr_scp_fk = qr_scp
        query.save()

    @patch("remapp.netdicom.qrscu._query_study", side_effect=_fake_all_modalities)
    def test_non_modality_matching(self, study_query_mock):
        """
        Tests the study level query for each modality. Fake responses include a study with just US in, indicating the
        study filter doesn't work and there is no point querying for any further modalities as we'll already have the
        responses.
        :param study_query_mock: Mocked study level response routine
        :return: Nothing
        """
        from ..netdicom.qrscu import _query_for_each_modality

        all_mods = {
            "CT": {"inc": True, "mods": ["CT"]},
            "MG": {"inc": True, "mods": ["MG"]},
            "FL": {"inc": False, "mods": ["RF", "XA"]},
            "DX": {"inc": False, "mods": ["DX", "CR"]},
            "NM": {"inc": False, "mods": ["NM", "PT"]},
        }
        query = DicomQuery.objects.get()

        d = Dataset()
        assoc = None
        ae = None
        remote = None
        modalities_returned, modality_matching = _query_for_each_modality(
            all_mods, query, d, assoc, ae, remote
        )

        self.assertEqual(DicomQRRspStudy.objects.filter(deleted_flag=False).count(), 2)
        self.assertEqual(study_query_mock.call_count, 1)
        self.assertEqual(modality_matching, False)
        self.assertEqual(modalities_returned, True)

    @patch("remapp.netdicom.qrscu._query_study", side_effect=_fake_two_modalities)
    def test_modality_matching(self, study_query_mock):
        """
        Tests the study level query for each modality. Fake responses only include appropriate modalities, so
        _query_for_each_modality should return modality_matching as True
        :param study_query_mock: Mocked study level response routine
        :return:  Nothing
        """
        from ..netdicom.qrscu import _query_for_each_modality

        all_mods = collections.OrderedDict()
        all_mods["CT"] = {"inc": True, "mods": ["CT"]}
        all_mods["MG"] = {"inc": True, "mods": ["MG"]}
        all_mods["FL"] = {"inc": False, "mods": ["RF", "XA"]}
        all_mods["DX"] = {"inc": False, "mods": ["DX", "CR"]}
        all_mods["NM"] = {"inc": False, "mods": ["NM", "PT"]}

        query = DicomQuery.objects.get()
        qr_scp = DicomRemoteQR.objects.get()

        d = Dataset()
        assoc = None
        ae = None
        remote = None
        modalities_returned, modality_matching = _query_for_each_modality(
            all_mods, query, d, assoc, ae, remote
        )

        self.assertEqual(DicomQRRspStudy.objects.filter(deleted_flag=False).count(), 7)
        self.assertEqual(study_query_mock.call_count, 2)
        self.assertEqual(modality_matching, True)


class QRPhilipsCT(TestCase):
    def setUp(self):
        """"""

        query = DicomQuery.objects.create()
        query.query_id = uuid.uuid4()
        query.save()

        rst1 = DicomQRRspStudy.objects.create(dicom_query=query)
        rst1.query_id = query.query_id
        rst1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1.study_description = "test response 1"
        rst1.station_name = ""
        rst1.save()

        rst1s1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rst1)
        rst1s1.query_id = query.query_id
        rst1s1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1s1.modality = "CT"
        rst1s1.series_number = 1
        rst1s1.series_description = "scan projection radiograph"
        rst1s1.number_of_series_related_instances = 1
        rst1s1.save()

        rst1s2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rst1)
        rst1s2.query_id = query.query_id
        rst1s2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1s2.modality = "CT"
        rst1s2.series_number = 3
        rst1s2.series_description = "thorax and abdomen"
        rst1s2.number_of_series_related_instances = 300
        rst1s2.save()

        rst1s3 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rst1)
        rst1s3.query_id = query.query_id
        rst1s3.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1s3.modality = "SC"
        rst1s3.series_number = 2394
        rst1s3.series_description = "dose info"
        rst1s3.number_of_series_related_instances = 1
        rst1s3.save()

        rst1_series_rsp = rst1.dicomqrrspseries_set.filter(deleted_flag=False).all()
        rst1.set_modalities_in_study(
            list(
                set(
                    val
                    for dic in rst1_series_rsp.values("modality")
                    for val in list(dic.values())
                )
            )
        )
        rst1.save()

    def test_response_sorting_ct_philips_with_desc(self):
        """
        Study response contains a Philips style 'dose info' series, with study descriptions available, and no structured
        report series. Expect a single series to be left after pruning.
        """
        all_mods = {
            "CT": {"inc": True, "mods": ["CT"]},
            "MG": {"inc": False, "mods": ["MG"]},
            "FL": {"inc": False, "mods": ["RF", "XA"]},
            "DX": {"inc": False, "mods": ["DX", "CR"]},
            "NM": {"inc": False, "mods": ["NM", "PT"]},
        }
        filters = {
            "stationname_inc": None,
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
        }

        query = DicomQuery.objects.get()
        rst1 = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()[0]

        self.assertEqual(
            rst1.dicomqrrspseries_set.filter(deleted_flag=False).count(), 3
        )

        assoc = None
        ae = None
        remote = None
        qrscu._prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )

        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).count(), 1
        )
        self.assertEqual(
            rst1.dicomqrrspseries_set.filter(deleted_flag=False).count(), 1
        )
        self.assertEqual(
            rst1.dicomqrrspseries_set.filter(deleted_flag=False)
            .all()[0]
            .series_description,
            "dose info",
        )

    def test_response_sorting_ct_philips_no_desc(self):
        """
        Study response contains a Philips style 'dose info' series, but without study descriptions available, and no
        structured report series. Expect two series to be left after pruning, with the main series removed.
        """
        all_mods = {
            "CT": {"inc": True, "mods": ["CT"]},
            "MG": {"inc": False, "mods": ["MG"]},
            "FL": {"inc": False, "mods": ["RF", "XA"]},
            "DX": {"inc": False, "mods": ["DX", "CR"]},
            "NM": {"inc": False, "mods": ["NM", "PT"]},
        }
        filters = {
            "stationname_inc": None,
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
        }

        query = DicomQuery.objects.get()
        rst1 = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()[0]

        rst1_series_rsp = rst1.dicomqrrspseries_set.filter(deleted_flag=False).all()
        rst1s1 = rst1_series_rsp[0]
        rst1s2 = rst1_series_rsp[1]
        rst1s3 = rst1_series_rsp[2]
        rst1s1.series_description = None
        rst1s2.series_description = None
        rst1s3.series_description = None
        rst1s1.save()
        rst1s2.save()
        rst1s3.save()

        # Before pruning, three series
        self.assertEqual(
            rst1.dicomqrrspseries_set.filter(deleted_flag=False).count(), 3
        )

        assoc = None
        ae = None
        remote = None
        qrscu._prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )

        # After pruning, two series
        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).all().count(), 1
        )
        self.assertEqual(
            rst1.dicomqrrspseries_set.filter(deleted_flag=False).all().count(), 2
        )

    def test_response_sorting_ct_philips_with_desc_no_dose_info(self):
        """
        Study response doesn't contain a Philips style 'dose info' series or an SR series, and study descriptions
        are returned. Expect no series to be left after pruning, and the study response record deleted.
        """
        all_mods = {
            "CT": {"inc": True, "mods": ["CT"]},
            "MG": {"inc": False, "mods": ["MG"]},
            "FL": {"inc": False, "mods": ["RF", "XA"]},
            "DX": {"inc": False, "mods": ["DX", "CR"]},
            "NM": {"inc": False, "mods": ["NM", "PT"]},
        }
        filters = {
            "stationname_inc": None,
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
        }

        query = DicomQuery.objects.get()
        rst1 = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()[0]
        rst1_series_rsp = rst1.dicomqrrspseries_set.filter(deleted_flag=False).order_by(
            "id"
        )
        rst1s3 = rst1_series_rsp[2]

        # Remove the third series with the 'dose info' description
        rst1s3.delete()

        # Before the pruning, two series
        self.assertEqual(
            rst1.dicomqrrspseries_set.filter(deleted_flag=False).all().count(), 2
        )

        assoc = None
        ae = None
        remote = None
        qrscu._prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )

        # After pruning, there should be no studies left
        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).all().count(), 0
        )

    @patch(
        "remapp.netdicom.qrscu._check_sr_type_in_study",
        _fake_check_sr_type_in_study_with_rdsr,
    )
    def test_response_pruning_ct_philips_with_desc_and_sr(self):
        """
        Study response contains a Philips style 'dose info' series, with study descriptions available, and a structured
        report series. Expect a single SR series to be left after pruning.
        """
        all_mods = {
            "CT": {"inc": True, "mods": ["CT"]},
            "MG": {"inc": False, "mods": ["MG"]},
            "FL": {"inc": False, "mods": ["RF", "XA"]},
            "DX": {"inc": False, "mods": ["DX", "CR"]},
            "NM": {"inc": False, "mods": ["NM", "PT"]},
        }
        filters = {
            "stationname_inc": None,
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
        }

        query = DicomQuery.objects.get()
        rst1 = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()[0]

        # Add in a fourth series with modality SR
        rst1s4 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rst1)
        rst1s4.query_id = query.query_id
        rst1s4.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1s4.modality = "SR"
        rst1s4.series_number = 999
        rst1s4.series_description = "radiation dose report"
        rst1s4.number_of_series_related_instances = 1
        rst1s4.save()

        # Re-generate the modality list
        rst1_series_rsp = rst1.dicomqrrspseries_set.filter(deleted_flag=False).all()
        rst1.set_modalities_in_study(
            list(
                set(
                    val
                    for dic in rst1_series_rsp.values("modality")
                    for val in list(dic.values())
                )
            )
        )
        rst1.save()

        # Now starting with four series
        self.assertEqual(
            rst1.dicomqrrspseries_set.filter(deleted_flag=False).all().count(), 4
        )

        assoc = None
        ae = None
        remote = None
        qrscu._prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )

        # Should now have one SR series left, identified by the series description for the purposes of this test
        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).all().count(), 1
        )
        self.assertEqual(
            rst1.dicomqrrspseries_set.filter(deleted_flag=False).all().count(), 1
        )
        self.assertEqual(
            rst1.dicomqrrspseries_set.filter(deleted_flag=False)
            .all()[0]
            .series_description,
            "radiation dose report",
        )

    def test_modalities_in_study_generation(self):
        """
        Testing that ModalitiesInStudy is generated if not returned by remote C-Find SCP
        """
        from collections import Counter
        from ..netdicom.qrscu import _generate_modalities_in_study

        query = DicomQuery.objects.get()
        rst1 = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()[0]

        # Add in a fourth series with modality SR
        rst1s4 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rst1)
        rst1s4.query_id = query.query_id
        rst1s4.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1s4.modality = "SR"
        rst1s4.series_number = 999
        rst1s4.series_description = "radiation dose report"
        rst1s4.number_of_series_related_instances = 1
        rst1s4.save()

        # Delete the modalities in study data
        rst1.set_modalities_in_study(None)
        rst1.save()

        _generate_modalities_in_study(rst1, query.query_id)

        # reload study, else _generate_modalities_in_study appears to work without save. See #627
        rst2 = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()[0]

        # Modalities in study should now be available again
        self.assertEqual(
            Counter(rst2.get_modalities_in_study()), Counter(["CT", "SC", "SR"])
        )


class ResponseFiltering(TestCase):
    """
    Test case for the study or series level filtering for desired or otherwise station names, study descriptions etc
    Function tested is qrscu._filter
    """

    def setUp(self):
        """"""

        query = DicomQuery.objects.create()
        query.query_id = uuid.uuid4()
        query.save()

        rst1 = DicomQRRspStudy.objects.create(dicom_query=query)
        rst1.query_id = query.query_id
        rst1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1.study_description = "Imported  CT studies"
        rst1.station_name = "badstation"
        rst1.save()

        rst1s1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rst1)
        rst1s1.query_id = query.query_id
        rst1s1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1s1.modality = "CT"
        rst1s1.series_number = 1
        rst1s1.series_description = "scan projection radiograph"
        rst1s1.number_of_series_related_instances = 1
        rst1s1.save()

        rst1s2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rst1)
        rst1s2.query_id = query.query_id
        rst1s2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1s2.modality = "CT"
        rst1s2.series_number = 3
        rst1s2.series_description = "thorax and abdomen"
        rst1s2.number_of_series_related_instances = 300
        rst1s2.save()

        rst1s3 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rst1)
        rst1s3.query_id = query.query_id
        rst1s3.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1s3.modality = "SC"
        rst1s3.series_number = 2394
        rst1s3.series_description = "dose info"
        rst1s3.number_of_series_related_instances = 1
        rst1s3.save()

        rst1_series_rsp = rst1.dicomqrrspseries_set.filter(deleted_flag=False).all()
        rst1.set_modalities_in_study(
            list(
                set(
                    val
                    for dic in rst1_series_rsp.values("modality")
                    for val in list(dic.values())
                )
            )
        )
        rst1.save()

        rst2 = DicomQRRspStudy.objects.create(dicom_query=query)
        rst2.query_id = query.query_id
        rst2.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst2.study_description = "Test Response 2"
        rst2.station_name = "goodstation"
        rst2.save()

        rst3 = DicomQRRspStudy.objects.create(dicom_query=query)
        rst3.query_id = query.query_id
        rst3.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst3.study_description = "test response 3"
        rst3.station_name = "goodstation2"
        rst3.save()

        rst4 = DicomQRRspStudy.objects.create(dicom_query=query)
        rst4.query_id = query.query_id
        rst4.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst4.save()

        rst5 = DicomQRRspStudy.objects.create(dicom_query=query)
        rst5.query_id = query.query_id
        rst5.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst5.study_description = ""
        rst5.station_name = None
        rst5.save()

    def test_filter_include_station_name(self):
        """
        Testing _filter with include station name of 'goodstation'. Expect four responses goodstation, goodstation2,
        and two studies with the station name not returned or None
        :return: None
        """
        from ..netdicom.qrscu import _filter

        query = DicomQuery.objects.get()
        _filter(query, "study", "station_name", ["goodstation"], "include")

        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).all().count(), 4
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        for study in studies:
            if study.station_name is not None:
                self.assertTrue("goodstation" in study.station_name)

    def test_filter_exclude_station_name(self):
        """
        Testing _filter with exclude station name of 'badstation'. Expect four responses goodstation, goodstation2
        and two studies with the station name not returned or None
        :return: None
        """
        from ..netdicom.qrscu import _filter

        query = DicomQuery.objects.get()
        _filter(query, "study", "station_name", ["badstation"], "exclude")

        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).all().count(), 4
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        for study in studies:
            if study.station_name is not None and study.station_name != "":
                self.assertFalse("badstation" in study.station_name)

    def test_filter_exclude_study_description(self):
        """
        Testing _filter with exclude two study descriptions. Expect three responses - goodstation and the two studies
        with no study description or empty study description
        :return: None
        """
        from ..netdicom.qrscu import _filter

        query = DicomQuery.objects.get()
        _filter(
            query,
            "study",
            "study_description",
            ["import", "test response 3"],
            "exclude",
        )

        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).all().count(), 3
        )
        study = query.dicomqrrspstudy_set.filter(deleted_flag=False).order_by("pk")
        self.assertTrue(study[0].station_name == "goodstation")

    def test_filter_include_study_description(self):
        """
        Testing _filter with include study description 'test'. Expect four responses of goodstation, goodstation2,
        and the two studies with no study description or empty study description
        :return: None
        """
        from ..netdicom.qrscu import _filter

        query = DicomQuery.objects.get()
        _filter(
            query,
            "study",
            "study_description",
            [
                "test",
            ],
            "include",
        )

        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).all().count(), 4
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        for study in studies:
            if study.station_name is not None and study.station_name != "":
                self.assertTrue("goodstation" in study.station_name)


def _fake_image_query(ae, remote, assoc, sr, query, initial_image_only=False):
    return


class PruneSeriesResponses(TestCase):
    """
    Test case for filtering series responses depending on availability and type of SR series, including using -emptysr
    flag
    """

    def setUp(self):
        """"""

        self.all_mods = {
            "CT": {"inc": True, "mods": ["CT"]},
            "MG": {"inc": True, "mods": ["MG"]},
            "FL": {"inc": True, "mods": ["RF", "XA"]},
            "DX": {"inc": True, "mods": ["DX", "CR"]},
            "NM": {"inc": True, "mods": ["NM", "PT"]},
        }
        self.filters = {
            "stationname_inc": None,
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
        }

    def test_prune_ser_resp_mg_no_sr(self):
        """
        Test _prune_series_responses with mammo exam with no SR.
        :return: No change to response
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.create()
        query.query_id = "MammoNoSR"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1.study_description = "MG study no SR"
        st1.set_modalities_in_study(["MG"])
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se1.modality = "MG"
        st1_se1.series_number = 1
        st1_se1.number_of_series_related_instances = 1
        st1_se1.save()

        query = DicomQuery.objects.get(query_id__exact="MammoNoSR")
        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_mg_with_sr(self):
        """
        Test _prune_series_responses with mammo exam with two SRs, one RDSR and one Basic SR.
        :return: MG series and basic SR series should be deleted.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.create()
        query.query_id = "MammoWithSR"
        query.save()

        st2 = DicomQRRspStudy.objects.create(dicom_query=query)
        st2.query_id = query.query_id
        st2.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2.study_description = "MG study with SR"
        st2.set_modalities_in_study(["MG", "SR"])
        st2.save()

        st2_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st2)
        st2_se1.query_id = query.query_id
        st2_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2_se1.modality = "MG"
        st2_se1.series_number = 1
        st2_se1.number_of_series_related_instances = 1
        st2_se1.save()

        st2_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st2)
        st2_se2.query_id = query.query_id
        st2_se2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2_se2.modality = "SR"
        st2_se2.series_number = 2
        st2_se2.number_of_series_related_instances = 1
        st2_se2.save()

        st2_se2_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st2_se2)
        st2_se2_im1.query_id = query.query_id
        st2_se2_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2_se2_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.67"
        st2_se2_im1.save()

        st2_se3 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st2)
        st2_se3.query_id = query.query_id
        st2_se3.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2_se3.modality = "SR"
        st2_se3.series_number = 3
        st2_se3.number_of_series_related_instances = 1
        st2_se3.save()

        st2_se3_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st2_se3)
        st2_se3_im1.query_id = query.query_id
        st2_se3_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2_se3_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"
        st2_se3_im1.save()

        query = DicomQuery.objects.get(query_id__exact="MammoWithSR")
        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        sr_instance = series[0].dicomqrrspimage_set.filter(deleted_flag=False).get()
        self.assertEqual(sr_instance.sop_class_uid, "1.2.840.10008.5.1.4.1.1.88.67")

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_cr_no_rdsr(self):
        """
        Test _prune_series_responses with CR exam with no RDSR but with Basic SR.
        :return: Basic SR deleted, study.modality set to "DX"
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.create()
        query.query_id = "CRNoRDSR"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1.study_description = "CR study no SR"
        st1.set_modalities_in_study(["CR", "SR"])
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se1.modality = "CR"
        st1_se1.series_number = 1
        st1_se1.number_of_series_related_instances = 1
        st1_se1.save()

        st1_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se2.query_id = query.query_id
        st1_se2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2.modality = "SR"
        st1_se2.series_number = 2
        st1_se2.number_of_series_related_instances = 1
        st1_se2.save()

        st1_se2_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se2)
        st1_se2_im1.query_id = query.query_id
        st1_se2_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"
        st1_se2_im1.save()

        query = DicomQuery.objects.get(query_id__exact="CRNoRDSR")
        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        self.assertEqual(series[0].modality, "CR")
        self.assertEqual(studies[0].modality, "DX")

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_dx_with_sr(self):
        """
        Test _prune_series_responses with DX exam with three SRs, one RDSR, one ESR and one Basic SR.
        :return: DX series, ESR and basic SR series should be deleted.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.create()
        query.query_id = "DXWithSR"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1.study_description = "DX study with RDSR"
        st1.set_modalities_in_study(["DX", "SR"])
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se1.modality = "DX"
        st1_se1.series_number = 1
        st1_se1.number_of_series_related_instances = 1
        st1_se1.save()

        st1_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se2.query_id = query.query_id
        st1_se2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2.modality = "SR"
        st1_se2.series_number = 2
        st1_se2.number_of_series_related_instances = 1
        st1_se2.save()

        st1_se2_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se2)
        st1_se2_im1.query_id = query.query_id
        st1_se2_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.67"
        st1_se2_im1.save()

        st1_se3 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se3.query_id = query.query_id
        st1_se3.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se3.modality = "SR"
        st1_se3.series_number = 3
        st1_se3.number_of_series_related_instances = 1
        st1_se3.save()

        st1_se3_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se3)
        st1_se3_im1.query_id = query.query_id
        st1_se3_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se3_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"
        st1_se3_im1.save()

        st1_se4 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se4.query_id = query.query_id
        st1_se4.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se4.modality = "SR"
        st1_se4.series_number = 4
        st1_se4.number_of_series_related_instances = 1
        st1_se4.save()

        st1_se4_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se4)
        st1_se4_im1.query_id = query.query_id
        st1_se4_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se4_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.22"
        st1_se4_im1.save()

        query = DicomQuery.objects.get(query_id__exact="DXWithSR")
        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        sr_instance = series[0].dicomqrrspimage_set.filter(deleted_flag=False).get()
        self.assertEqual(sr_instance.sop_class_uid, "1.2.840.10008.5.1.4.1.1.88.67")

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_rf_no_sr(self):
        """
        Test _prune_series_responses with fluoro exam with no ESR or RDSR.
        :return: Whole study response deleted
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.create()
        query.query_id = "RFNoSR"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1.study_description = "RF study no SR"
        st1.set_modalities_in_study(["RF", "SR"])
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se1.modality = "RF"
        st1_se1.series_number = 1
        st1_se1.number_of_series_related_instances = 1
        st1_se1.save()

        st1_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se2.query_id = query.query_id
        st1_se2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2.modality = "SR"
        st1_se2.series_number = 2
        st1_se2.number_of_series_related_instances = 1
        st1_se2.save()

        st1_se2_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se2)
        st1_se2_im1.query_id = query.query_id
        st1_se2_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"
        st1_se2_im1.save()

        query = DicomQuery.objects.get(query_id__exact="RFNoSR")
        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 0)

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_xa_with_esr(self):
        """
        Test _prune_series_responses with XA exam with an ESR, and one Basic SR.
        :return: XA series and basic SR series should be deleted.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.create()
        query.query_id = "XAWithESRBSR"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1.study_description = "XA study with ESR and Basic SR"
        st1.set_modalities_in_study(["XA", "SR"])
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se1.modality = "XA"
        st1_se1.series_number = 1
        st1_se1.number_of_series_related_instances = 1
        st1_se1.save()

        st1_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se2.query_id = query.query_id
        st1_se2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2.modality = "SR"
        st1_se2.series_number = 2
        st1_se2.number_of_series_related_instances = 1
        st1_se2.save()

        st1_se2_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se2)
        st1_se2_im1.query_id = query.query_id
        st1_se2_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.22"
        st1_se2_im1.save()

        st1_se3 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se3.query_id = query.query_id
        st1_se3.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se3.modality = "SR"
        st1_se3.series_number = 3
        st1_se3.number_of_series_related_instances = 1
        st1_se3.save()

        st1_se3_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se3)
        st1_se3_im1.query_id = query.query_id
        st1_se3_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se3_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"
        st1_se3_im1.save()

        query = DicomQuery.objects.get(query_id__exact="XAWithESRBSR")
        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        sr_instance = series[0].dicomqrrspimage_set.filter(deleted_flag=False).get()
        self.assertEqual(sr_instance.sop_class_uid, "1.2.840.10008.5.1.4.1.1.88.22")

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_empty_sr_no_flag(self):
        """
        Test _prune_series_responses with mammo exam with one SR series & no -emptysr flag, simulating a PACS that
        doesn't return any image level responses.
        :return: image series should remain, SR series should be deleted.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.create()
        query.query_id = "MammoWithSR"
        query.save()

        st2 = DicomQRRspStudy.objects.create(dicom_query=query)
        st2.query_id = query.query_id
        st2.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2.study_description = "MG study with SR"
        st2.set_modalities_in_study(["MG", "SR"])
        st2.save()

        st2_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st2)
        st2_se1.query_id = query.query_id
        st2_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2_se1.modality = "MG"
        st2_se1.series_number = 1
        st2_se1.number_of_series_related_instances = 1
        st2_se1.save()

        st2_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st2)
        st2_se2.query_id = query.query_id
        st2_se2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2_se2.modality = "SR"
        st2_se2.series_number = 2
        st2_se2.number_of_series_related_instances = 1
        st2_se2.save()

        query = DicomQuery.objects.get(query_id__exact="MammoWithSR")
        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        self.assertEqual(series[0].modality, "MG")

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_empty_sr_with_flag(self):
        """
        Test _prune_series_responses with mammo exam with one SR series & the -emptysr flag, simulating a PACS that
        doesn't return any image level responses.
        :return: SR series should remain, image series should be deleted.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.create()
        query.query_id = "MammoWithSR"
        query.save()

        st2 = DicomQRRspStudy.objects.create(dicom_query=query)
        st2.query_id = query.query_id
        st2.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2.study_description = "MG study with SR"
        st2.set_modalities_in_study(["MG", "SR"])
        st2.save()

        st2_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st2)
        st2_se1.query_id = query.query_id
        st2_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2_se1.modality = "MG"
        st2_se1.series_number = 1
        st2_se1.number_of_series_related_instances = 1
        st2_se1.save()

        st2_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st2)
        st2_se2.query_id = query.query_id
        st2_se2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st2_se2.modality = "SR"
        st2_se2.series_number = 2
        st2_se2.number_of_series_related_instances = 1
        st2_se2.save()

        query = DicomQuery.objects.get(query_id__exact="MammoWithSR")
        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=True,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        self.assertEqual(series[0].modality, "SR")


class PETCTStudyDuplication(TestCase):
    """
    Tests the _duplicate_ct_pet_studies function which duplicates studies containing
    both CT and PET/NM modalities
    """

    def setUp(self):
        self.all_mods = {
            "CT": {"inc": True, "mods": ["CT"]},
            "MG": {"inc": True, "mods": ["MG"]},
            "FL": {"inc": True, "mods": ["RF", "XA"]},
            "DX": {"inc": True, "mods": ["DX", "CR"]},
            "NM": {"inc": True, "mods": ["NM", "PT"]},
        }

    def test_prune_ser_resp_nm_pet_image(self):
        """
        Test _prune_series_responses with NM exam containing PET and CT Study with RDSR
        :return: No change to response
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.create()
        query.query_id = "NMPETCT"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1.study_description = "NM Study with PET Image and CT RDSR"
        st1.set_modalities_in_study(["PT", "SR", "CT"])
        st1.save()

        se_rdsr = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        se_rdsr.query_id = query.query_id
        se_rdsr.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        se_rdsr.modality = "SR"
        se_rdsr.series_number = 1
        se_rdsr.number_of_series_related_instances = 1
        se_rdsr.save()

        st1_se2_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=se_rdsr)
        st1_se2_im1.query_id = query.query_id
        st1_se2_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.67"
        st1_se2_im1.save()

        se_pt = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        se_pt.query_id = query.query_id
        se_pt.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        se_pt.modality = "PT"
        se_pt.series_number = 1
        se_pt.number_of_series_related_instances = 1
        se_pt.save()

        from ..netdicom.qrscu import _duplicate_ct_pet_studies

        _duplicate_ct_pet_studies(query, self.all_mods)
        studies = (
            query.dicomqrrspstudy_set.filter(deleted_flag=False).order_by("pk").all()
        )
        self.assertEqual(studies.count(), 2)
        self.assertNotIn("NM", studies[0].get_modalities_in_study())
        self.assertNotIn("CT", studies[1].get_modalities_in_study())


class PruneSeriesResponseNM(TestCase):
    """
    Tests _prune_series_responses for nuclear medicine
    """

    def setUp(self):
        self.all_mods = {
            "CT": {"inc": True, "mods": ["CT"]},
            "MG": {"inc": True, "mods": ["MG"]},
            "FL": {"inc": True, "mods": ["RF", "XA"]},
            "DX": {"inc": True, "mods": ["DX", "CR"]},
            "NM": {"inc": True, "mods": ["NM", "PT"]},
        }
        self.filters = {
            "stationname_inc": None,
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
        }

        query = DicomQuery.objects.create()
        query.query_id = "NMSR"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1.study_description = "NM Study with RRDSR"
        st1.save()

        se_sr = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        se_sr.query_id = query.query_id
        se_sr.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        se_sr.modality = "SR"
        se_sr.series_number = 1
        se_sr.number_of_series_related_instances = 1
        se_sr.save()

        se_sr_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=se_sr)
        se_sr_im1.query_id = query.query_id
        se_sr_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        se_sr_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.68"
        se_sr_im1.save()

        se_pt = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        se_pt.query_id = query.query_id
        se_pt.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        se_pt.modality = "PT"
        se_pt.series_number = 1
        se_pt.number_of_series_related_instances = 1
        se_pt.save()

        se_pt_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=se_pt)
        se_pt_im1.query_id = query.query_id
        se_pt_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        se_pt_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.128"
        se_pt_im1.save()

        se_nm = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        se_nm.query_id = query.query_id
        se_nm.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        se_nm.modality = "NM"
        se_nm.series_number = 1
        se_nm.number_of_series_related_instances = 1
        se_nm.save()

        se_nm_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=se_nm)
        se_nm_im1.query_id = query.query_id
        se_nm_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        se_nm_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.20"
        se_nm_im1.save()

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_nm_rrdsr(self):
        """
        Test _prune_series_responses with NM exam containing rrdsr
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.get()
        s = DicomQRRspStudy.objects.get()
        s.set_modalities_in_study(["PT", "SR", "NM"])
        s.save()

        _prune_series_responses(
            None,
            None,
            None,
            query,
            self.all_mods,
            self.filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 2)
        for serie in series.all():
            self.assertIn(serie.modality, ["SR", "PT"])

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_pt_image(self):
        """
        Test _prune_series_responses with NM exam containing PET and NM images
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.get()
        DicomQRRspSeries.objects.filter(modality__exact="SR").delete()
        s = DicomQRRspStudy.objects.get()
        s.set_modalities_in_study(["PT", "NM"])
        s.save()

        _prune_series_responses(
            None,
            None,
            None,
            query,
            self.all_mods,
            self.filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        for serie in series.all():
            self.assertEqual(serie.modality, "PT")

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_nm_image(self):
        """
        Test _prune_series_responses with NM exam containing NM image
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.get()
        DicomQRRspSeries.objects.exclude(modality__exact="NM").delete()
        s = DicomQRRspStudy.objects.get()
        s.set_modalities_in_study(["NM"])
        s.save()

        _prune_series_responses(
            None,
            None,
            None,
            query,
            self.all_mods,
            self.filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        self.assertEqual(series.get().modality, "NM")


class PruneSeriesResponsesCT(TestCase):
    """
    Test case for filtering series responses of CT studies, depending on availability and type of SR series,
    including using -emptysr flag
    """

    def setUp(self):
        """"""

        self.all_mods = {
            "CT": {"inc": True, "mods": ["CT"]},
            "MG": {"inc": True, "mods": ["MG"]},
            "FL": {"inc": True, "mods": ["RF", "XA"]},
            "DX": {"inc": True, "mods": ["DX", "CR"]},
            "NM": {"inc": True, "mods": ["NM", "PT"]},
        }
        self.filters = {
            "stationname_inc": None,
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
        }

        query = DicomQuery.objects.create()
        query.query_id = "CT"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1.study_description = "CT study"
        st1.set_modalities_in_study(["CT", "SR"])
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se1.modality = "CT"
        st1_se1.series_number = 1
        st1_se1.number_of_series_related_instances = 15
        st1_se1.series_description = "TAP"
        st1_se1.save()

        st1_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se2.query_id = query.query_id
        st1_se2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2.modality = "SR"
        st1_se2.series_number = 2
        st1_se2.number_of_series_related_instances = 1
        st1_se2.save()

        st1_se2_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se2)
        st1_se2_im1.query_id = query.query_id
        st1_se2_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se2_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.22"
        st1_se2_im1.save()

        st1_se3 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se3.query_id = query.query_id
        st1_se3.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se3.modality = "SR"
        st1_se3.series_number = 3
        st1_se3.number_of_series_related_instances = 1
        st1_se3.save()

        st1_se3_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se3)
        st1_se3_im1.query_id = query.query_id
        st1_se3_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se3_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.11"
        st1_se3_im1.save()

        st1_se4 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se4.query_id = query.query_id
        st1_se4.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se4.modality = "CT"
        st1_se4.series_number = 4
        st1_se4.number_of_series_related_instances = 1
        st1_se4.series_description = "Dose Info"
        st1_se4.save()

        st1_se5 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se5.query_id = query.query_id
        st1_se5.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se5.modality = "SR"
        st1_se5.series_number = 5
        st1_se5.number_of_series_related_instances = 1
        st1_se5.save()

        st1_se5_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se5)
        st1_se5_im1.query_id = query.query_id
        st1_se5_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        st1_se5_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.67"
        st1_se5_im1.save()

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_ct_with_rdsr(self):
        """
        Test _prune_series_responses with CT exam with a RDSR, ESR, Basic SR, Dose info and an axial series.
        :return: RDSR series.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.get(query_id__exact="CT")
        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        sr_instance = series[0].dicomqrrspimage_set.filter(deleted_flag=False).get()
        self.assertEqual(sr_instance.sop_class_uid, "1.2.840.10008.5.1.4.1.1.88.67")

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_ct_with_esr(self):
        """
        Test _prune_series_responses with CT exam with a ESR, Basic SR, Dose info and an axial series.
        :return: ESR series.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.get(query_id__exact="CT")

        study = query.dicomqrrspstudy_set.filter(deleted_flag=False).get()
        rdsr_series = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=5
        )
        rdsr_series.delete()

        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        sr_instance = series[0].dicomqrrspimage_set.filter(deleted_flag=False).get()
        self.assertEqual(sr_instance.sop_class_uid, "1.2.840.10008.5.1.4.1.1.88.22")

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_ct_with_dose_info(self):
        """
        Test _prune_series_responses with CT exam with a Basic SR, Dose info and an axial series.
        :return: Dose info series.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.get(query_id__exact="CT")

        study = query.dicomqrrspstudy_set.filter(deleted_flag=False).get()
        rdsr_series = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=5
        )
        rdsr_series.delete()
        esr_series = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=2
        )
        esr_series.delete()

        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        self.assertEqual(series[0].series_number, 4)

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_ct_with_dose_info_no_desc(self):
        """
        Test _prune_series_responses with CT exam with a Basic SR, Dose info and an axial series, but no series desc.
        :return: Dose info series.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.get(query_id__exact="CT")

        study = query.dicomqrrspstudy_set.filter(deleted_flag=False).get()
        rdsr_series = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=5
        )
        rdsr_series.delete()
        esr_series = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=2
        )
        esr_series.delete()
        dose_info_series = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=4
        )
        dose_info_series[0].series_description = ""

        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        self.assertEqual(series[0].series_number, 4)

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_ct_empty_sr_no_flag(self):
        """
        Test _prune_series_responses with CT exam with one SR series & no -emptysr flag, simulating a PACS that
        doesn't return any image level responses.
        :return: Dose info series.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.get(query_id__exact="CT")

        study = query.dicomqrrspstudy_set.filter(deleted_flag=False).get()

        esr_series = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=2
        )
        esr_series[0].dicomqrrspimage_set.filter(deleted_flag=False).get().delete()
        study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=3
        ).all().delete()
        study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=5
        ).all().delete()

        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=False,
        )

        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        self.assertEqual(series[0].series_number, 4)

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_prune_ser_resp_ct_empty_sr_with_flag(self):
        """
        Test _prune_series_responses with CT exam with one SR series with -emptysr flag, simulating a PACS that
        doesn't return any image level responses.
        :return: SR series.
        """
        from ..netdicom.qrscu import _prune_series_responses

        query = DicomQuery.objects.get(query_id__exact="CT")

        study = query.dicomqrrspstudy_set.filter(deleted_flag=False).get()

        esr_series = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=2
        )
        esr_series[0].dicomqrrspimage_set.filter(deleted_flag=False).get().delete()
        study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=3
        ).all().delete()
        study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
            series_number__exact=5
        ).all().delete()

        all_mods = self.all_mods
        filters = self.filters
        assoc = None
        ae = None
        remote = None
        _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images=False,
            get_empty_sr=True,
        )

        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        self.assertEqual(studies.count(), 1)
        series = studies[0].dicomqrrspseries_set.filter(deleted_flag=False).all()
        self.assertEqual(series.count(), 1)
        self.assertEqual(series[0].series_number, 2)


def _fake_qrscu(
    qr_scp_pk=None,
    store_scp_pk=None,
    implicit=False,
    explicit=False,
    move=False,
    query_id=None,
    date_from=None,
    date_until=None,
    modalities=None,
    inc_sr=False,
    remove_duplicates=True,
    filters=None,
):
    """
    Check that the parsing has worked
    """
    pass


def _fake_echo_success(scp_pk=None, store_scp=False, qr_scp=False):
    """
    Fake success return for echoscu
    :param scp_pk:
    :param store_scp:
    :param qr_scp:
    :return: str "Success"
    """
    return "Success"


class QRSCUScriptArgParsing(TestCase):
    """
    Test the args passed on the command line are parsed properly
    """

    @patch("remapp.netdicom.tools.echoscu", _fake_echo_success)
    def test_ct_mg(self):
        """
        Test the arg parser with modalities CT and MG
        :return:
        """

        from ..netdicom.qrscu import _create_parser, _process_args

        parser = _create_parser()
        parsed_args = _process_args(parser.parse_args(["1", "2", "-ct", "-mg"]), parser)

        self.assertEqual(parsed_args["qr_id"], 1)
        self.assertEqual(parsed_args["store_id"], 2)
        self.assertEqual(parsed_args["modalities"].sort(), ["MG", "CT"].sort())
        filters = {
            "study_desc_exc": None,
            "stationname_exc": None,
            "study_desc_inc": None,
            "stationname_inc": None,
            "stationname_study": False,
        }
        self.assertEqual(parsed_args["filters"], filters)

    @patch("remapp.netdicom.tools.echoscu", _fake_echo_success)
    def test_ct_std_exc(self):
        """
        Test the arg parser with modalities CT and MG
        :return:
        """

        from ..netdicom.qrscu import _create_parser, _process_args

        parser = _create_parser()
        parsed_args = _process_args(
            parser.parse_args(["1", "2", "-ct", "-e Thorax, Neck "]), parser
        )

        self.assertEqual(parsed_args["qr_id"], 1)
        self.assertEqual(parsed_args["store_id"], 2)
        self.assertEqual(parsed_args["modalities"].sort(), ["MG", "CT"].sort())
        filters = {
            "study_desc_exc": ["thorax", "neck"],
            "study_desc_inc": None,
            "stationname_exc": None,
            "stationname_inc": None,
            "stationname_study": False,
        }
        self.assertEqual(parsed_args["filters"], filters)

    @patch("remapp.netdicom.tools.echoscu", _fake_echo_success)
    def test_ct_std_exc_stn_inc(self):
        """
        Test the arg parser with modalities CT and MG
        :return:
        """

        from ..netdicom.qrscu import _create_parser, _process_args

        parser = _create_parser()
        parsed_args = _process_args(
            parser.parse_args(
                ["1", "2", "-ct", "--desc_exclude", "Thorax, Neck ", "-sni", "MyStn"]
            ),
            parser,
        )

        self.assertEqual(parsed_args["qr_id"], 1)
        self.assertEqual(parsed_args["store_id"], 2)
        self.assertEqual(parsed_args["modalities"].sort(), ["MG", "CT"].sort())
        filters = {
            "study_desc_exc": ["thorax", "neck"],
            "study_desc_inc": None,
            "stationname_exc": None,
            "stationname_inc": ["mystn"],
            "stationname_study": False,
        }
        self.assertEqual(parsed_args["filters"], filters)


class RemoveDuplicates(TestCase):
    """
    Test the routine to remove any responses that correspond to information already in the database
    """

    def test_rdsr_new(self):
        """Inital test that _remove_duplicates doesn't remove new RDSR"""

        from ..netdicom.qrscu import _remove_duplicates

        PatientIDSettings.objects.create()

        # Nothing imported into the database

        query = DicomQuery.objects.create()
        query.query_id = "CT"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.3.0"
        )
        st1.study_description = "CT study"
        st1.set_modalities_in_study(["CT", "SR"])
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.12.0"
        )
        st1_se1.modality = "SR"
        st1_se1.series_number = 502
        st1_se1.number_of_series_related_instances = 1
        st1_se1.save()

        st1_se1_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se1)
        st1_se1_im1.query_id = query.query_id
        st1_se1_im1.sop_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.11.0"
        )
        st1_se1_im1.save()

        study_responses_pre = DicomQRRspStudy.objects.all()
        self.assertEqual(study_responses_pre.count(), 1)
        self.assertEqual(
            study_responses_pre[0]
            .dicomqrrspseries_set.filter(deleted_flag=False)
            .count(),
            1,
        )

        study_rsp = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        assoc = None
        ae = None
        remote = None
        _remove_duplicates(ae, remote, query, study_rsp, assoc)

        study_responses_post = DicomQRRspStudy.objects.all()
        self.assertEqual(study_responses_post.count(), 1)
        self.assertEqual(
            study_responses_post[0]
            .dicomqrrspseries_set.filter(deleted_flag=False)
            .count(),
            1,
        )

    def test_rdsr_same(self):
        """Now testing _remove_duplicates will remove an identical RDSR, but retain a new one."""

        from ..netdicom.qrscu import _remove_duplicates

        PatientIDSettings.objects.create()

        dicom_file_1 = "test_files/CT-RDSR-Siemens-Multi-1.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path_1 = os.path.join(root_tests, dicom_file_1)
        rdsr.rdsr(dicom_path_1)

        query = DicomQuery.objects.create()
        query.query_id = "CT"
        query.save()

        # Same RDSR - expect study response to be deleted (post count = 0)
        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.3.0"
        )
        st1.study_description = "CT study"
        st1.set_modalities_in_study(["CT", "SR"])
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.12.0"
        )
        st1_se1.modality = "SR"
        st1_se1.series_number = 501
        st1_se1.number_of_series_related_instances = 1
        st1_se1.save()

        st1_se1_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se1)
        st1_se1_im1.query_id = query.query_id
        st1_se1_im1.sop_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.11.0"
        )
        st1_se1_im1.save()

        st1_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se2.query_id = query.query_id
        st1_se2.series_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.7.0"
        )
        st1_se2.modality = "SR"
        st1_se2.series_number = 501
        st1_se2.number_of_series_related_instances = 1
        st1_se2.save()

        st1_se2_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se2)
        st1_se2_im1.query_id = query.query_id
        st1_se2_im1.sop_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.6.0"
        )
        st1_se2_im1.save()

        study_responses_pre = DicomQRRspStudy.objects.all()
        self.assertEqual(study_responses_pre.count(), 1)
        self.assertEqual(
            study_responses_pre[0]
            .dicomqrrspseries_set.filter(deleted_flag=False)
            .count(),
            2,
        )

        study_rsp = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        assoc = None
        ae = None
        remote = None
        _remove_duplicates(ae, remote, query, study_rsp, assoc)

        study_responses_post = DicomQRRspStudy.objects.all()
        self.assertEqual(study_responses_post.count(), 1)
        self.assertEqual(
            study_responses_post[0]
            .dicomqrrspseries_set.filter(deleted_flag=False)
            .count(),
            1,
        )
        self.assertEqual(
            study_responses_post[0]
            .dicomqrrspseries_set.filter(deleted_flag=False)
            .all()[0]
            .series_instance_uid,
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.7.0",
        )

    def test_rdsr_no_objectuids(self):
        """
        Test importing RDSR where
        * same study, ObjectUIDsProcessed not populated
        :return:
        """

        from ..netdicom.qrscu import _remove_duplicates

        PatientIDSettings.objects.create()

        dicom_file_1 = "test_files/CT-RDSR-Siemens-Multi-2.dcm"
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dicom_path_1 = os.path.join(root_tests, dicom_file_1)
        rdsr.rdsr(dicom_path_1)
        imported_study = GeneralStudyModuleAttr.objects.order_by("pk")[0]
        imported_study.objectuidsprocessed_set.all().delete()
        imported_study.save()

        query = DicomQuery.objects.create()
        query.query_id = "CT"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.3.0"
        )
        st1.study_description = "CT study"
        st1.set_modalities_in_study(["CT", "SR"])
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.7.0"
        )
        st1_se1.modality = "SR"
        st1_se1.series_number = 501
        st1_se1.number_of_series_related_instances = 1
        st1_se1.save()

        st1_se1_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se1)
        st1_se1_im1.query_id = query.query_id
        st1_se1_im1.sop_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.792239193.1702185591.1516915727449.6.0"
        )
        st1_se1_im1.save()

        study_responses_pre = DicomQRRspStudy.objects.all()
        self.assertEqual(study_responses_pre.count(), 1)
        self.assertEqual(
            study_responses_pre[0]
            .dicomqrrspseries_set.filter(deleted_flag=False)
            .count(),
            1,
        )

        study_rsp = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        assoc = None
        ae = None
        remote = None
        _remove_duplicates(ae, remote, query, study_rsp, assoc)

        study_responses_post = DicomQRRspStudy.objects.all()
        self.assertEqual(study_responses_post.count(), 1)
        self.assertEqual(
            study_responses_post[0]
            .dicomqrrspseries_set.filter(deleted_flag=False)
            .count(),
            1,
        )

    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_dx(self):
        """
        Test remove duplicates with DX images
        :return:
        """

        from ..extractors import dx
        from ..netdicom.qrscu import _remove_duplicates

        PatientIDSettings.objects.create()

        dx_ge_xr220_1 = os.path.join("test_files", "DX-Im-GE_XR220-1.dcm")
        root_tests = os.path.dirname(os.path.abspath(__file__))
        dx.dx(os.path.join(root_tests, dx_ge_xr220_1))

        query = DicomQuery.objects.create()
        query.query_id = "DX"
        query.save()

        st1 = DicomQRRspStudy.objects.create(dicom_query=query)
        st1.query_id = query.query_id
        st1.study_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.2282339064.1266597797.1479751121656.24.0"
        )
        st1.study_description = "DX study"
        st1.set_modalities_in_study(
            [
                "DX",
            ]
        )
        st1.save()

        st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=st1)
        st1_se1.query_id = query.query_id
        st1_se1.series_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.2282339064.1266597797.1479751121656.25.0"
        )
        st1_se1.modality = "DX"
        st1_se1.series_number = 1
        st1_se1.number_of_series_related_instances = 1
        st1_se1.save()

        # Image responses won't be there yet, but image level query is faked
        st1_se5_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se1)
        st1_se5_im1.query_id = query.query_id
        st1_se5_im1.sop_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.2282339064.1266597797.1479751121656.20.0"
        )
        st1_se5_im1.save()

        st1_se5_im2 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se1)
        st1_se5_im2.query_id = query.query_id
        st1_se5_im2.sop_instance_uid = (
            "1.3.6.1.4.1.5962.99.1.2282339064.1266597797.1479751121656.26.0"
        )
        st1_se5_im2.save()

        study_responses_pre = DicomQRRspStudy.objects.all()
        self.assertEqual(study_responses_pre.count(), 1)
        self.assertEqual(
            study_responses_pre[0]
            .dicomqrrspseries_set.filter(deleted_flag=False)
            .count(),
            1,
        )
        self.assertEqual(
            study_responses_pre[0]
            .dicomqrrspseries_set.get()
            .dicomqrrspimage_set.filter(deleted_flag=False)
            .count(),
            2,
        )

        study_rsp = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        assoc = None
        ae = None
        remote = None
        _remove_duplicates(ae, remote, query, study_rsp, assoc)

        # One image response should have been deleted, one remain
        study_responses_post = DicomQRRspStudy.objects.all()
        self.assertEqual(study_responses_post.count(), 1)
        self.assertEqual(
            study_responses_post[0]
            .dicomqrrspseries_set.filter(deleted_flag=False)
            .count(),
            1,
        )
        self.assertEqual(
            study_responses_pre[0]
            .dicomqrrspseries_set.get()
            .dicomqrrspimage_set.filter(deleted_flag=False)
            .count(),
            1,
        )
        remaining_image_rsp = (
            study_responses_pre[0]
            .dicomqrrspseries_set.get()
            .dicomqrrspimage_set.filter(deleted_flag=False)
            .get()
        )
        self.assertEqual(
            remaining_image_rsp.sop_instance_uid,
            "1.3.6.1.4.1.5962.99.1.2282339064.1266597797.1479751121656.26.0",
        )


class InvalidMove(TestCase):
    """Small test class to check passing an invalid query ID to movescu fails gracefully"""

    def test_invalid_query_id(self):
        """Pass invalid query_id to movescu, expect log update and return False/0"""
        from ..netdicom.qrscu import movescu

        PatientIDSettings.objects.create()

        with LogCapture("remapp.netdicom.qrscu") as log:
            movestatus = movescu("not_a_query_ID")
            self.assertEqual(movestatus, False)

            log.check_present(
                (
                    "remapp.netdicom.qrscu",
                    "WARNING",
                    "Move called with invalid query_id not_a_query_ID. Move abandoned.",
                )
            )


class DuplicatesInStudyResponse(TestCase):
    """Test function that removes duplicates within the response at study level"""

    def test_remove_duplicates_in_study_response(self):
        """Test to ensure duplicates are removed from the response query set"""
        from ..netdicom.qrscu import _remove_duplicates_in_study_response

        query = DicomQuery.objects.create()
        query.query_id = uuid.uuid4()
        query.save()

        rst1 = DicomQRRspStudy.objects.create(dicom_query=query)
        rst1.query_id = query.query_id
        rst1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst1.save()

        rst2 = DicomQRRspStudy.objects.create(dicom_query=query)
        rst2.query_id = query.query_id
        rst2.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
        rst2.save()

        rst3 = DicomQRRspStudy.objects.create(dicom_query=query)
        rst3.query_id = query.query_id
        rst3.study_instance_uid = rst1.study_instance_uid
        rst3.save()

        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).count(), 3
        )

        current_count = _remove_duplicates_in_study_response(query, 3)
        self.assertEqual(current_count, 2)
        self.assertEqual(
            query.dicomqrrspstudy_set.filter(deleted_flag=False).count(), 2
        )


def _fake_query_each_mod(all_mods, query, d, assoc, ae, remote):
    rsp1 = DicomQRRspStudy.objects.create(dicom_query=query)
    rsp1.query_id = uuid.uuid4()
    rsp1.study_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
    rsp1.station_name = "MIXEDCTNM"
    rsp1.set_modalities_in_study(["PT", "CT", "SR"])
    rsp1.modality = None
    rsp1.save()
    return True, True


def _fake_associate(*args, **kwargs):
    associate = MagicMock()
    associate.is_established = True
    return associate


def _fake_query_series(ae, remote, assoc, d2, rsp, query):
    st1_se1 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rsp)
    st1_se1.query_id = query.query_id
    st1_se1.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
    st1_se1.modality = "CT"
    st1_se1.series_number = 1
    st1_se1.number_of_series_related_instances = 100
    st1_se1.station_name = "MEDPC"
    st1_se1.save()

    st1_se2 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rsp)
    st1_se2.query_id = query.query_id
    st1_se2.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
    st1_se2.modality = "PET"
    st1_se2.series_number = 2
    st1_se2.number_of_series_related_instances = 100
    st1_se2.station_name = "SYMBIAT16"
    st1_se2.save()

    st1_se3 = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=rsp)
    st1_se3.query_id = query.query_id
    st1_se3.series_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
    st1_se3.modality = "SR"
    st1_se3.series_number = 3
    st1_se3.number_of_series_related_instances = 1
    st1_se3.station_name = "MEDPC"
    st1_se3.save()

    st1_se3_im1 = DicomQRRspImage.objects.create(dicom_qr_rsp_series=st1_se3)
    st1_se3_im1.query_id = query.query_id
    st1_se3_im1.sop_instance_uid = generate_uid(prefix="1.3.6.1.4.1.45593.999.")
    st1_se3_im1.sop_class_uid = "1.2.840.10008.5.1.4.1.1.88.67"
    st1_se3_im1.save()


class DifferentStationNamesAtStudySeriesLevel(TestCase):
    """Test fix for issue #772 with differing station names at study and series level"""

    def setUp(self):
        # Remote find/move node details
        qr_scp = DicomRemoteQR.objects.create()
        qr_scp.hostname = "localhost"
        qr_scp.port = 11112
        qr_scp.aetitle = "teststnnameqrsvr"
        qr_scp.callingaet = "openrem"
        qr_scp.save()
        # Local store node details
        store_scp = DicomStoreSCP.objects.create()
        store_scp.aetitle = "openremstore"
        store_scp.port = 104
        store_scp.save()

    @patch("remapp.netdicom.qrscu._query_for_each_modality", _fake_query_each_mod)
    @patch("pynetdicom.ae.ApplicationEntity.associate", _fake_associate)
    @patch("remapp.netdicom.qrscu._query_series", _fake_query_series)
    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_mixed_ct_nm_no_filter(self):
        """Test study level MIXEDCTNM, series level CT MEDPC and series level PET SYMBIAT16"""
        from ..netdicom.qrscu import qrscu

        qr_scp = DicomRemoteQR.objects.last()
        store_scp = DicomStoreSCP.objects.last()

        qrscu(qr_scp.pk, store_scp.pk, query_id="no_filter", modalities="CT")

        query = DicomQuery.objects.filter(query_id__exact="no_filter").last()
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()

        self.assertEqual(studies.count(), 1)
        self.assertEqual(
            studies[0].dicomqrrspseries_set.filter(deleted_flag=False).count(), 1
        )

        qr_scp.delete()
        store_scp.delete()

    @patch("remapp.netdicom.qrscu._query_for_each_modality", _fake_query_each_mod)
    @patch("pynetdicom.ae.ApplicationEntity.associate", _fake_associate)
    @patch("remapp.netdicom.qrscu._query_series", _fake_query_series)
    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_mixed_ct_nm_series_filter_medpc(self):
        """Test study level MIXEDCTNM, series level CT MEDPC and series level PET SYMBIAT16"""
        from ..netdicom.qrscu import qrscu

        qr_scp = DicomRemoteQR.objects.last()
        store_scp = DicomStoreSCP.objects.last()

        filters = {
            "stationname_inc": [
                "medpc",
            ],
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
            "stationname_study": None,
        }

        qrscu(
            qr_scp.pk,
            store_scp.pk,
            query_id="ser_medpc",
            modalities="CT",
            filters=filters,
        )

        query = DicomQuery.objects.filter(query_id__exact="ser_medpc").last()
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()

        self.assertEqual(1, studies.count())
        self.assertEqual(
            1, studies[0].dicomqrrspseries_set.filter(deleted_flag=False).count()
        )

        qr_scp.delete()
        store_scp.delete()

    @patch("remapp.netdicom.qrscu._query_for_each_modality", _fake_query_each_mod)
    @patch("pynetdicom.ae.ApplicationEntity.associate", _fake_associate)
    @patch("remapp.netdicom.qrscu._query_series", _fake_query_series)
    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_mixed_ct_nm_series_filter_mixedctnm(self):
        """Test study level MIXEDCTNM, series level CT MEDPC and series level PET SYMBIAT16"""
        from ..netdicom.qrscu import qrscu

        qr_scp = DicomRemoteQR.objects.last()
        store_scp = DicomStoreSCP.objects.last()

        filters = {
            "stationname_inc": [
                "mixedctnm",
            ],
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
            "stationname_study": None,
        }

        qrscu(
            qr_scp.pk,
            store_scp.pk,
            query_id="ser_mixed",
            modalities="CT",
            filters=filters,
        )

        query = DicomQuery.objects.filter(query_id__exact="ser_mixed").last()
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()

        self.assertEqual(studies.count(), 0)

        qr_scp.delete()
        store_scp.delete()

    @patch("remapp.netdicom.qrscu._query_for_each_modality", _fake_query_each_mod)
    @patch("pynetdicom.ae.ApplicationEntity.associate", _fake_associate)
    @patch("remapp.netdicom.qrscu._query_series", _fake_query_series)
    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_mixed_ct_nm_study_filter_mixedctnm(self):
        """Test study level MIXEDCTNM, series level CT MEDPC and series level PET SYMBIAT16"""
        from ..netdicom.qrscu import qrscu

        qr_scp = DicomRemoteQR.objects.last()
        store_scp = DicomStoreSCP.objects.last()

        filters = {
            "stationname_inc": [
                "mixedctnm",
            ],
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
            "stationname_study": True,
        }

        qrscu(
            qr_scp.pk,
            store_scp.pk,
            query_id="stdy_mix",
            modalities="CT",
            filters=filters,
        )

        query = DicomQuery.objects.filter(query_id__exact="stdy_mix").last()
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()

        self.assertEqual(studies.count(), 1)
        self.assertEqual(
            studies[0].dicomqrrspseries_set.filter(deleted_flag=False).count(), 1
        )

        qr_scp.delete()
        store_scp.delete()

    @patch("remapp.netdicom.qrscu._query_for_each_modality", _fake_query_each_mod)
    @patch("pynetdicom.ae.ApplicationEntity.associate", _fake_associate)
    @patch("remapp.netdicom.qrscu._query_series", _fake_query_series)
    @patch("remapp.netdicom.qrscu._query_images", _fake_image_query)
    def test_mixed_ct_nm_study_filter_medpc(self):
        """Test study level MIXEDCTNM, series level CT MEDPC and series level PET SYMBIAT16"""
        from ..netdicom.qrscu import qrscu

        qr_scp = DicomRemoteQR.objects.last()
        store_scp = DicomStoreSCP.objects.last()

        filters = {
            "stationname_inc": [
                "medpc",
            ],
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
            "stationname_study": True,
        }

        qrscu(
            qr_scp.pk,
            store_scp.pk,
            query_id="stdy_medpc",
            modalities="CT",
            filters=filters,
        )

        query = DicomQuery.objects.filter(query_id__exact="stdy_medpc").last()
        studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()

        self.assertEqual(studies.count(), 0)

        qr_scp.delete()
        store_scp.delete()
