#!/usr/bin/python

"""
Query/Retrieve SCU AE

Specialised QR routine to get just the objects that might be useful for dose related metrics from a remote PACS or
modality

The qrscu routine does basically act using c-find queries to the remote PACS/DICOM node. It progressively asks
for information about studies, then series, then images storing the (high level) info acquired about them
in the database as a DicomQuery object.
On each level data that we are not interested in for some reason (studies that are duplicates; series that are
excluded by our filters; and so on) are removed from the DicomQuery object.
The movescu routing in the end initiates a c-move of all the objects that were not deleted from the query, such that
those objects are sent to the local DICOM node (e.g. Orthanc). The local dicom node has to be configured such that it
uses the openrem extractor scripts on receive of DICOM files.
"""

import argparse
import collections
from datetime import datetime
import logging
import os
import sys
import uuid
from copy import deepcopy


import django
from django.core.exceptions import ObjectDoesNotExist
from django.db.models import Q
from django.utils.translation import gettext as _
from pydicom.dataset import Dataset
from pynetdicom import AE, _config  # , debug_logger
from pynetdicom.sop_class import (
    StudyRootQueryRetrieveInformationModelFind,
    StudyRootQueryRetrieveInformationModelMove,
)
from pynetdicom.status import QR_FIND_SERVICE_CLASS_STATUS, QR_MOVE_SERVICE_CLASS_STATUS

from ..templatetags.remappduration import naturalduration
from ..tools.dcmdatetime import get_time, make_dcm_date_range, make_dcm_time_range
from ..tools.get_values import get_value_kw

logger = logging.getLogger(
    "remapp.netdicom.qrscu"
)  # Explicitly named so that it is still handled when using __main__
# setup django/OpenREM
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from remapp.models import (  # pylint: disable=wrong-import-order, wrong-import-position
    DicomQuery,
    DicomQRRspImage,
    DicomQRRspSeries,
    DicomQRRspStudy,
    DicomRemoteQR,
    DicomStoreSCP,
    GeneralStudyModuleAttr,
)
from openrem.remapp.tools.background import (
    get_current_task,
    get_or_generate_task_uuid,
    record_task_error_exit,
    run_in_background,
    wait_task,
)

_config.LOG_RESPONSE_IDENTIFIERS = False
_config.LOG_HANDLER_LEVEL = "none"
_config.LOG_REQUEST_IDENTIFIERS = False


def _generate_modalities_in_study(study_rsp, query_id):
    """Generates modalities in study from series level Modality information

    :param study_rsp: study level C-Find response object in database
    :return: response updated with ModalitiesInStudy
    """
    try:
        query_id_8 = query_id.hex[:8]
    except AttributeError:
        query_id_8 = query_id[:8]
    logger.debug(
        f"{query_id_8} modalities_returned = False, so building from series info"
    )
    series_rsp = study_rsp.dicomqrrspseries_set.all()
    study_rsp.set_modalities_in_study(
        list(
            set(
                val
                for dic in series_rsp.values("modality")
                for val in list(dic.values())
            )
        )
    )
    study_rsp.save()


def _make_query_deleted_reasons_consistent(query):
    """
    When the parent of an object is marked as deleted all it's children
    still exist and are potentially not marked as deleted. This routine marks
    those children as deleted.
    """

    DicomQRRspSeries.objects.filter(
        dicom_qr_rsp_study__deleted_flag=True,
        deleted_flag=False,
        dicom_qr_rsp_study__dicom_query__pk=query.pk,
    ).update(
        deleted_flag=True, deleted_reason="Ignored, since parent study was ignored"
    )
    DicomQRRspImage.objects.filter(
        dicom_qr_rsp_series__deleted_flag=True,
        deleted_flag=False,
        dicom_qr_rsp_series__dicom_qr_rsp_study__dicom_query__pk=query.pk,
    ).update(
        deleted_flag=True, deleted_reason="Ignored, since parent series was ignored"
    )


def _remove_image_sop_uids(
    series_rsp, query_id_8, study_number, study_instance_uid, existing_sop_instance_uids
):
    for image_rsp in series_rsp.dicomqrrspimage_set.filter(deleted_flag=False).all():
        logger.debug(
            f"{query_id_8} Study {study_number} {study_instance_uid} Checking for"
            f"SOPInstanceUID {image_rsp.sop_instance_uid}"
        )
        if image_rsp.sop_instance_uid in existing_sop_instance_uids:
            logger.debug(
                f"{query_id_8} Study {study_number} {study_instance_uid} Found "
                f"SOPInstanceUID processed before, won't ask for this one"
            )
            image_rsp.deleted_flag = True
            image_rsp.deleted_reason = (
                "SOP instance of this object is present in database"
            )
            image_rsp.save()
            series_rsp.image_level_move = (
                True  # If we have deleted images we need to set this flag
            )
            series_rsp.save()
    if not series_rsp.dicomqrrspimage_set.filter(deleted_flag=False):
        series_rsp.deleted_flag = True
        series_rsp.deleted_reason = "All files of this series ignored"
        series_rsp.save()


def _query_id_8(query):
    try:
        query_id_8 = query.query_id.hex[:8]
    except AttributeError:
        query_id_8 = query.query_id[:8]
    return query_id_8


def _remove_duplicates(ae, remote, query, study_rsp, assoc):
    """
    Checks for objects in C-Find response already being in the OpenREM database to remove them from the C-Move request
    :param query: Query object in database
    :param study_rsp: study level C-Find response object in database
    :param assoc: current DICOM Query object
    :return: Study, series and image level responses deleted if not useful
    """

    query_id_8 = _query_id_8(query)
    logger.debug(
        f"{query_id_8} About to remove any studies we already have in the database"
    )
    query.stage = _(
        "Checking to see if any response studies are already in the OpenREM database"
    )
    try:
        query.save()
    except Exception as e:
        logger.error(
            f"{query_id_8} query.save in remove duplicates didn't work because of {e}"
        )
    logger.debug(
        f"{query_id_8} Checking to see if any of the {study_rsp.count()} studies are already in the "
        f"OpenREM database"
    )
    for study_number, study in enumerate(study_rsp):
        existing_studies = GeneralStudyModuleAttr.objects.filter(
            study_instance_uid=study.study_instance_uid
        )
        if existing_studies.exists():
            logger.debug(
                f"{query_id_8} Study {study_number} {study.study_instance_uid} exists in database already"
            )
            for existing_study in existing_studies:
                existing_sop_instance_uids = set()
                for previous_object in existing_study.objectuidsprocessed_set.all():
                    existing_sop_instance_uids.add(previous_object.sop_instance_uid)
                logger.debug(
                    f"{query_id_8} Study {study_number} {study.study_instance_uid} has previously processed "
                    f"the following SOPInstanceUIDs: {existing_sop_instance_uids}"
                )
                for series_rsp in study.dicomqrrspseries_set.filter(
                    deleted_flag=False
                ).all():
                    if series_rsp.modality == "SR":
                        _remove_image_sop_uids(
                            series_rsp,
                            query_id_8,
                            study_number,
                            study.study_instance_uid,
                            existing_sop_instance_uids,
                        )
                    elif series_rsp.modality in ["MG", "DX", "CR", "PX", "PT", "NM"]:
                        logger.debug(
                            f"{query_id_8} Study {study_number} {study.study_instance_uid} about to query at "
                            f"image level to get SOPInstanceUID"
                        )
                        _query_images(
                            ae, remote, assoc, series_rsp, query, True
                        )  # Only check first image of series
                        _remove_image_sop_uids(
                            series_rsp,
                            query_id_8,
                            study_number,
                            study.study_instance_uid,
                            existing_sop_instance_uids,
                        )
                    else:
                        series_rsp.deleted_flag = True
                        series_rsp.deleted_reason = (
                            "Does not have modality SR, MG, DX, CR or PX"
                        )
                        series_rsp.save()
        if not study.dicomqrrspseries_set.filter(deleted_flag=False):
            study.deleted_flag = True
            study.deleted_reason = "Study does not have any series left"
            study.save()

    logger.info(
        f"{query_id_8} After removing studies we already have in the db, "
        f"{query.dicomqrrspstudy_set.filter(deleted_flag=False).count()} studies are left"
    )


def _filter(query, level, filter_name, filter_list, filter_type):
    """
    Reduces Study or Series level UIDs that will have a Move command sent for by filtering against one of three
    variables that can be 'include only' or 'exclude'
    :param query: Query object in database
    :param level: 'series' or 'study'
    :param filter_name: 'station_name', 'sop_classes_in_study', or 'study_description'
    :param filter_list: list of lower case search words or phrases, or SOP classes
    :param filter_type: 'exclude', 'include'
    :return: None
    """
    query_id_8 = _query_id_8(query)
    if filter_type == "exclude":
        filtertype = True
    elif filter_type == "include":
        filtertype = False
    else:
        logger.error(f"{query_id_8} _filter called without filter_type. Cannot filter!")
        return

    study_rsp = query.dicomqrrspstudy_set.all()
    query.stage = _(
        "Filter at {level} level on {filter_name} that {filter_type} {filter_list}".format(
            level=level,
            filter_name=filter_name,
            filter_type=filter_type,
            filter_list=filter_list,
        )
    )
    logger.debug(
        f"{query_id_8} Filter at {level} level on {filter_name} that {filter_type} {filter_list}"
    )
    for study in study_rsp:
        if level == "study":
            if (
                getattr(study, filter_name) is not None
                and getattr(study, filter_name) != ""
                and (
                    any(
                        term in getattr(study, filter_name).lower()
                        for term in filter_list
                    )
                    is filtertype
                )
            ):
                study.deleted_flag = True
                study.deleted_reason = (
                    f"Filter {filter_name} {filter_type} active and matched here"
                )
                study.save()
        elif level == "series":
            series = study.dicomqrrspseries_set.all()
            for s in series:
                if (
                    getattr(s, filter_name) is not None
                    and getattr(s, filter_name) != ""
                    and (
                        any(
                            term in getattr(s, filter_name).lower()
                            for term in filter_list
                        )
                        is filtertype
                    )
                ):
                    s.deleted_flag = True
                    s.deleted_reason = (
                        f"Filter {filter_name} {filter_type} active and matched here"
                    )
                    s.save()
            nr_series_remaining = study.dicomqrrspseries_set.all().count()
            if nr_series_remaining == 0:
                study.deleted_flag = True
                study.deleted_reason = (
                    "All Series of this studies where ignored due to some active filter"
                )
                study.save()
    logger.info(
        f"{query_id_8} Now have {query.dicomqrrspstudy_set.filter(deleted_flag=False).count()} studies"
    )


def _prune_series_responses(
    ae, remote, assoc, query, all_mods, filters, get_toshiba_images, get_empty_sr
):
    """
    For each study level response, remove any series that we know can't be used.
    :param query: Current DicomQuery object
    :param all_mods: Ordered dict of dicts detailing modalities we are interested in
    :param filters: Include and exclude lists for StationName (StudyDescription not considered at series level)
    :param get_toshiba_images: Bool, whether to try to get Toshiba dose summary images
    :param get_empty_sr: Bool, whether to get SR series that return nothing at image level query
    :return Series level response database rows are deleted if not useful
    """
    query.stage = _(
        "Getting series and image level information and deleting series we can't use"
    )
    query.save()
    query_id = query.query_id
    query_id_8 = _query_id_8(query)
    logger.debug(
        f"{query_id_8} Getting series and image level information and deleting series we can't use"
    )

    deleted_studies = {"RF": 0, "CT": 0, "SR": 0}
    kept_ct = {"SR": 0, "philips": 0, "toshiba": 0, "maybe_philips": 0}
    deleted_studies_filters = {"stationname_inc": 0, "stationname_exc": 0}

    if filters["stationname_inc"] and not filters["stationname_study"]:
        before_count = query.dicomqrrspstudy_set.filter(deleted_flag=False).count()
        _filter(
            query,
            level="series",
            filter_name="station_name",
            filter_list=filters["stationname_inc"],
            filter_type="include",
        )
        after_count = query.dicomqrrspstudy_set.filter(deleted_flag=False).count()
        if after_count < before_count:
            deleted_studies_filters["stationname_inc"] = before_count - after_count
            logger.debug(
                f"{query_id_8} stationname_inc removed {deleted_studies_filters['stationname_inc']} studies"
            )

    if filters["stationname_exc"] and not filters["stationname_study"]:
        before_count = query.dicomqrrspstudy_set.filter(deleted_flag=False).count()
        _filter(
            query,
            level="series",
            filter_name="station_name",
            filter_list=filters["stationname_exc"],
            filter_type="exclude",
        )
        after_count = query.dicomqrrspstudy_set.filter(deleted_flag=False).count()
        if after_count < before_count:
            deleted_studies_filters["stationname_exc"] = before_count - after_count
            logger.debug(
                f"{query_id_8} stationname_exc removed {deleted_studies_filters['stationname_exc']} studies"
            )

    study_rsp = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()

    for study in study_rsp:
        logger.debug(
            f"{query_id_8} Modalities in this study are: {study.get_modalities_in_study()}"
        )
        if all_mods["MG"]["inc"] and "MG" in study.get_modalities_in_study():
            # If _check_sr_type_in_study returns an RDSR, all other SR series will have been deleted and then all images
            # are deleted. If _check_sr_type_in_study returns an ESR or no_dose_report, everything else is kept.
            study.modality = "MG"
            study.save()

            if "SR" in study.get_modalities_in_study():
                if _check_sr_type_in_study(
                    ae, remote, assoc, study, query, get_empty_sr
                ) in [
                    "RDSR",
                    "null_response",
                ]:
                    logger.debug(
                        f"{query_id_8} RDSR in MG study, keep SR, delete all other series"
                    )
                    series = study.dicomqrrspseries_set.filter(deleted_flag=False).all()
                    series.exclude(modality__exact="SR").update(
                        deleted_flag=True,
                        deleted_reason="RDSR present, all other series ignored",
                    )
                else:
                    logger.debug(
                        f"{query_id_8} no RDSR in MG study, deleting other SR series"
                    )
                    series = study.dicomqrrspseries_set.filter(deleted_flag=False).all()
                    series.filter(modality__exact="SR").update(
                        deleted_flag=True,
                        deleted_reason="No RDSR, ignored all SR series",
                    )
            # ToDo: see if there is a mechanism to remove duplicate 'for processing' 'for presentation' images.

        elif all_mods["DX"]["inc"] and any(
            mod in study.get_modalities_in_study() for mod in ("CR", "DX", "PX")
        ):
            # If _check_sr_type_in_study returns an RDSR, all other SR series will have been deleted and then all images
            # are deleted. If _check_sr_type_in_study returns an ESR or no_dose_report, everything else is kept.
            study.modality = "DX"
            study.save()

            if "SR" in study.get_modalities_in_study():
                if _check_sr_type_in_study(
                    ae, remote, assoc, study, query, get_empty_sr
                ) in [
                    "RDSR",
                    "null_response",
                ]:
                    logger.debug(
                        f"{query_id_8} RDSR in DX study, keep SR, delete all other series"
                    )
                    series = study.dicomqrrspseries_set.filter(deleted_flag=False).all()
                    series.exclude(modality__exact="SR").update(
                        deleted_flag=True,
                        deleted_reason="RDSR present, all non SR series ignored",
                    )
                else:
                    logger.debug(
                        f"{query_id_8} no RDSR in DX study, deleting other SR series"
                    )
                    series = study.dicomqrrspseries_set.filter(deleted_flag=False).all()
                    series.filter(modality__exact="SR").update(
                        deleted_flag=True,
                        deleted_reason="No RDSR, ignored all SR series",
                    )

        elif all_mods["FL"]["inc"] and any(
            mod in study.get_modalities_in_study() for mod in ("XA", "RF")
        ):
            # Only RDSR or some ESR useful here
            study.modality = "FL"
            study.save()
            sr_type = _check_sr_type_in_study(
                ae, remote, assoc, study, query, get_empty_sr
            )
            if sr_type == "no_dose_report":
                logger.debug(
                    f"{query_id_8} No usable SR in RF study. Deleting from query."
                )
                study.deleted_flag = True
                study.deleted_reason = "No usable SR in study"
                study.save()
                deleted_studies["RF"] += 1
            else:
                logger.debug(
                    f"{query_id_8} {sr_type} in RF study, keep SR, delete all other series"
                )
                series = study.dicomqrrspseries_set.filter(deleted_flag=False).all()
                series.exclude(modality__exact="SR").update(
                    deleted_flag=True,
                    deleted_reason="RDSR present, all not SR series ignored",
                )

        elif all_mods["CT"]["inc"] and "CT" in study.get_modalities_in_study():
            # If _check_sr_type_in_study returns RDSR, all other SR series responses will have been deleted and then all
            # other series responses will be deleted too.
            # If _check_sr_type_in_study returns ESR, all other SR series responses will have been deleted and then all
            # other series responses will be deleted too.
            # Otherwise, we pass the study response to _get_philips_dose_images to see if there is a Philips dose info
            # series and optionally get samples from each series for the Toshiba RDSR creation routine.
            study.modality = "CT"
            study.save()
            series = study.dicomqrrspseries_set.filter(deleted_flag=False).all()
            sr_type = None
            if "SR" in study.get_modalities_in_study():
                sr_type = _check_sr_type_in_study(
                    ae, remote, assoc, study, query, get_empty_sr
                )
            if sr_type in ("RDSR", "ESR", "null_response"):
                logger.debug(
                    f"{query_id_8} {sr_type} in CT study, keep SR, delete all other series"
                )
                series.exclude(modality__exact="SR").update(
                    deleted_flag=True,
                    deleted_reason="RDSR present, all not SR series ignored",
                )
                kept_ct["SR"] += 1
            else:
                logger.debug(
                    f"{query_id_8} No usable SR in CT study, checking for Philips dose images"
                )
                philips_desc, philips_found = _get_philips_dose_images(
                    series, get_toshiba_images, query_id
                )
                if philips_desc and philips_found:
                    kept_ct["philips"] += 1
                elif not philips_found and get_toshiba_images:
                    logger.debug(
                        f"{query_id_8} No usable CT SR, no Philips dose image found, preparing study for "
                        f"Toshiba option"
                    )
                    _get_toshiba_dose_images(ae, remote, series, assoc, query)
                    kept_ct["toshiba"] += 1
                elif not philips_desc and philips_found:
                    logger.debug(
                        f"{query_id_8} No usable CT SR, series descriptions, retaining small series in "
                        f"case useful."
                    )
                    kept_ct["maybe_philips"] += 1
                else:
                    logger.debug(
                        f"{query_id_8} No usable CT information available, deleting study from query"
                    )
                    study.deleted_flag = True
                    study.deleted_reason = "Found no usuable information in this study"
                    study.save()
                    deleted_studies["CT"] += 1

        elif all_mods["NM"]["inc"] and (
            any(
                mod in study.get_modalities_in_study() for mod in all_mods["NM"]["mods"]
            )
        ):
            study.modality = "NM"
            study.save()
            # SOP ids: RRDSR, PET Image, NM Image. We will try to
            # get all of those (In case of NM and PT only the first
            # object of the series). NM will not be taken unless
            # nothing else present
            nm_img_sop_ids = [
                "1.2.840.10008.5.1.4.1.1.88.68",
                "1.2.840.10008.5.1.4.1.1.128",
                "1.2.840.10008.5.1.4.1.1.20",
            ]
            possible_modalities = {
                nm_img_sop_ids[0]: ["SR"],
                nm_img_sop_ids[1]: ["PT"],
                nm_img_sop_ids[2]: ["NM"],
            }

            loaded_sop_classes = set()
            for sop_class in nm_img_sop_ids:
                logger.debug(f"{query_id_8} Now checking for {sop_class} available")
                for load_mod in possible_modalities[sop_class]:
                    loaded_sop_classes = loaded_sop_classes.union(
                        _get_series_sop_class(
                            ae, remote, assoc, study, query, get_empty_sr, load_mod
                        )
                    )
                loaded_sop_classes.intersection_update(nm_img_sop_ids)
                # Don't download NM images when anything else is available
                if sop_class == nm_img_sop_ids[1] and len(loaded_sop_classes) > 0:
                    break

            series = study.dicomqrrspseries_set.filter(deleted_flag=False).all()
            series.exclude(sop_class_in_series__in=nm_img_sop_ids).update(
                deleted_flag=True,
                deleted_reason="Excluding all series without SOP class we can use",
            )
            keep = series.filter(sop_class_in_series=nm_img_sop_ids[2]).first()
            if keep is not None:
                series.filter(
                    Q(sop_class_in_series=nm_img_sop_ids[2]) & ~Q(pk=keep.pk)
                ).update(
                    deleted_flag=True,
                    deleted_reason="All NM images of a series contain the same NM data. Only first downloaded.",
                )  # Only take the first NM imgage

            series = series.filter(deleted_flag=False)

            if series.count() == 0:
                logger.debug(
                    f"{query_id_8} No usable NM information available, deleting study from query"
                )
                continue
            logger.debug(f"{query_id_8} Found {loaded_sop_classes}. Keeping them all.")
            for serie in series:
                if serie.sop_class_in_series in nm_img_sop_ids[1:2]:  # PET or NM Images
                    serie.image_level_move = (
                        True  # We set this so only the first img of the series is moved
                    )
                    serie.save()

        elif all_mods["SR"]["inc"]:
            sr_type = _check_sr_type_in_study(
                ae, remote, assoc, study, query, get_empty_sr
            )
            if sr_type == "RDSR":
                logger.debug(
                    f"{query_id_8} SR only query, found RDSR, deleted other SRs"
                )
            elif sr_type == "ESR":
                logger.debug(
                    f"{query_id_8} SR only query, found ESR, deleted other SRs"
                )
            elif sr_type == "null_response":
                logger.debug(f"{query_id_8} SR type unknown, -emptysr=True")
            elif sr_type == "no_dose_report":
                logger.debug(
                    f"{query_id_8} No RDSR or ESR found. Study will be deleted."
                )
                study.deleted_flag = True
                study.deleted_reason = "No RDSR or ESR found."
                study.save()
                deleted_studies["SR"] += 1

        if (
            study.id is not None
            and study.dicomqrrspseries_set.filter(deleted_flag=False).all().count() == 0
        ):
            logger.debug(
                f"{query_id_8} Deleting empty study with suid {study.study_instance_uid}"
            )
            study.deleted_flag = True
            study.deleted_reason = "There are no series left"
            study.save()

    return deleted_studies, deleted_studies_filters, kept_ct


def _get_philips_dose_images(series, get_toshiba_images, query_id):
    """
    Remove series that are not likely to be Philips Dose Info series
    :param series: database set
    :param get_toshiba_images: Bool, whether to try to get Toshiba dose summary images
    :param query_id: UID for this query
    :return: Bool, Bool representing if series_descriptions are available and if a Philips image was or might be found
    """
    try:
        query_id_8 = query_id.hex[:8]
    except AttributeError:
        query_id_8 = query_id[:8]

    series_descriptions = set(
        val for dic in series.values("series_description") for val in list(dic.values())
    )
    logger.debug(
        f"{query_id_8} Get Philips:  series_descriptions are {series_descriptions}"
    )
    if series_descriptions != {None}:
        if series.filter(series_description__iexact="dose info"):
            logger.debug(
                f"{query_id_8} Get Philips: found likely Philips dose image, no SR, delete all other series"
            )
            series.exclude(series_description__iexact="dose info").update(
                deleted_flag=True,
                deleted_reason="Found what is probably Philips dose image, no SR, all other series ignored",
            )
            return True, True
        else:
            logger.debug(f"{query_id_8} Get Philips: not matched Philips dose image")
            return True, False
    elif get_toshiba_images:
        return False, False
    else:
        logger.debug(
            f"{query_id_8} Get Philips: no series descriptions, keeping only series with < 6 images in "
            f"case we might get a Philips dose info image"
        )
        series.filter(number_of_series_related_instances__gt=5).update(
            deleted_flag=True,
            deleted_reason="No series descriptions when checking for philips dose image."
            "Keeping only series with < 6 images.",
        )
        return False, True


def _get_toshiba_dose_images(ae, remote, study_series, assoc, query):
    """
    Get images for Toshiba studies with no RDSR
    :param study_series: database set
    :return: None. Non-useful entries will be removed from database
    """

    query_id_8 = _query_id_8(query)

    for index, series in enumerate(study_series):
        _query_images(
            ae, remote, assoc, series, query, initial_image_only=True, msg_id=index + 1
        )
        images = series.dicomqrrspimage_set.all()
        if images.count() == 0:
            logger.debug(
                f"{query_id_8} Toshiba option: No images in series, deleting series."
            )
            series.deleted_flag = True
            series.deleted_reason = "Searching for Toshiba: No images in series found."
            series.save()
        else:
            if images[0].sop_class_uid != "1.2.840.10008.5.1.4.1.1.7":
                logger.debug(
                    f"{query_id_8} Toshiba option: Image series, SOPClassUID {images[0].sop_class_uid}, "
                    f"delete all but first image."
                )
                images.exclude(
                    sop_instance_uid__exact=images[0].sop_instance_uid
                ).update(
                    deleted_flag=True,
                    deleted_reason="Toshiba option selected. Ignoring all but the first image.",
                )
                logger.debug(
                    f"{query_id_8} Toshiba option: Deleted other images, "
                    f"now {images.filter(deleted_flag=False).count()} "
                    f"remaining (should be 1)"
                )
                series.image_level_move = True
                series.save()
            else:
                logger.debug(
                    f"{query_id_8} Toshiba option: Secondary capture series, "
                    f"keep the {images.filter(deleted_flag=False).count()} "
                    f"images in this series."
                )


def _prune_study_responses(query, filters):

    query_id_8 = _query_id_8(query)

    deleted_studies_filters = {
        "study_desc_inc": 0,
        "study_desc_exc": 0,
        "stationname_inc": 0,
        "stationname_exc": 0,
    }
    for (
        apply_current_filter,
        current_filter,
        current_filter_name,
        current_filter_type,
        short_name,
    ) in [
        (
            filters["study_desc_inc"],
            filters["study_desc_inc"],
            "study_description",
            "include",
            "study_desc_inc",
        ),
        (
            filters["study_desc_exc"],
            filters["study_desc_exc"],
            "study_description",
            "exclude",
            "study_desc_exc",
        ),
        (
            filters["stationname_inc"] and filters["stationname_study"],
            filters["stationname_inc"],
            "station_name",
            "include",
            "stationname_inc",
        ),
        (
            filters["stationname_exc"] and filters["stationname_study"],
            filters["stationname_exc"],
            "station_name",
            "exclude",
            "stationname_exc",
        ),
    ]:
        if apply_current_filter:
            before_count = query.dicomqrrspstudy_set.filter(deleted_flag=False).count()
            logger.debug(
                f"{query_id_8} About to filter on {current_filter_name} with {current_filter_type}: {current_filter}, "
                f"currently have {before_count} studies."
            )
            _filter(
                query,
                level="study",
                filter_name=current_filter_name,
                filter_list=current_filter,
                filter_type=current_filter_type,
            )
            after_count = query.dicomqrrspstudy_set.filter(deleted_flag=False).count()
            if after_count < before_count:
                deleted_studies_filters[short_name] = before_count - after_count
                logger.debug(
                    f"{query_id_8} study_desc_inc removed {deleted_studies_filters[short_name]} studies"
                )

    return deleted_studies_filters


def _get_series_sop_class(ae, remote, assoc, study, query, get_empty_sr, modality="SR"):
    """
    Checks for all SR (or other modality) Series what their SOP-Class is and stores it to their DB entry.

    :param assoc: Current DICOM query object
    :param study: study level C-Find response object in database
    :param get_empty_sr: Whether to get SR series that return empty at image level query
    :param modality: Basically a filter, the method will only check the SOP classes of series with this modality
    :return: set of SOP classes found for SR/All series
    """
    query_id_8 = _query_id_8(query)
    series_selected = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
        modality__exact=modality
    )
    logger.debug(
        f"{query_id_8} Check {modality} type: Number of series with {modality} {series_selected.count()}"
    )

    if modality == "SR":
        initial_image_only = False
    else:
        initial_image_only = True
    sop_classes = set()
    for sr in series_selected:
        _query_images(
            ae, remote, assoc, sr, query, initial_image_only=initial_image_only
        )
        images = sr.dicomqrrspimage_set.all()
        if images.count() == 0:
            if get_empty_sr and modality == "SR":
                logger.debug(
                    f"{query_id_8} Check SR type: studyuid: {study.study_instance_uid} "
                    f"seriesuid: {sr.series_instance_uid}. Image level response returned null, "
                    f"-emptysr=True so assuming SR is RDSR"
                )
                sop_classes.add("null_response")
            logger.warning(
                f"{query_id_8} Check SR type: Oops, series {sr.series_number} of study instance "
                f"UID {study.study_instance_uid} returned null at image level query. Try '-emptysr' option?"
            )
            continue
        for image in images:
            sop_classes.add(image.sop_class_uid)
        sr.sop_class_in_series = images[0].sop_class_uid
        sr.save()
        logger.debug(
            f"{query_id_8} Check {modality} type: studyuid: {study.study_instance_uid}   "
            f"seriesuid: {sr.series_instance_uid}  sop_classes: {sop_classes}"
        )
    return sop_classes


def _remove_sr_objects(series_sr, target_sop_class, query_id_8):
    if target_sop_class == "1.2.840.10008.5.1.4.1.1.88.67":
        debug_string = f"{query_id_8} Check SR type: Have RDSR, deleting non-RDSR SR"
        deleted_reason = "RDSR present, ignoring all non-RDSR SR"
    else:
        debug_string = (
            f"{query_id_8} Check SR type: Have ESR, deleting non-RDSR, non-ESR SR"
        )
        deleted_reason = "ESR present, no RDSR found, all other SR series ignored"

    for series_rsp in series_sr:
        # create Set of SOP classes for all objects in series
        sop_classes_in_series = {
            image.sop_class_uid for image in series_rsp.dicomqrrspimage_set.all()
        }
        if target_sop_class not in sop_classes_in_series:
            logger.debug(debug_string)
            series_rsp.deleted_flag = True
            series_rsp.deleted_reason = deleted_reason
            series_rsp.save()
        else:
            if len(sop_classes_in_series) > 1:
                series_rsp.image_level_move = True
                series_rsp.save()
                other_sop_class_objects = series_rsp.dicomqrrspimage_set.exclude(
                    sop_class_uid__exact=target_sop_class
                )
                for image_rsp in other_sop_class_objects:
                    image_rsp.deleted_flag = True
                    image_rsp.deleted_reason = deleted_reason
                    image_rsp.save()


# returns SR-type: RDSR or ESR; otherwise returns 'no_dose_report'
def _check_sr_type_in_study(ae, remote, assoc, study, query, get_empty_sr):
    """Checks at an image level whether SR in study is RDSR, ESR, or something else (Radiologist's report for example)

    * If RDSR is found, all non-RDSR SR series responses are deleted
    * Otherwise, if an ESR is found, all non-ESR series responses are deleted
    * Otherwise, all SR series responses are retained *(change in 0.9.0)*
    * If get_empty_sr=True, no RDSR or ESR objects are found and series image count is 0, SR series is kept

    The function returns one of 'RDSR', 'ESR', 'no_dose_report', 'null_response'.

    :param assoc: Current DICOM query object
    :param study: study level C-Find response object in database
    :param get_empty_sr: Whether to get SR series that return empty at image level query
    :return: string indicating SR type remaining in study
    """
    query_id_8 = _query_id_8(query)
    sop_classes = _get_series_sop_class(ae, remote, assoc, study, query, get_empty_sr)
    series_sr = study.dicomqrrspseries_set.filter(deleted_flag=False).filter(
        modality__exact="SR"
    )

    logger.debug(f"{query_id_8} Check SR type: sop_classes: {sop_classes}")

    if "1.2.840.10008.5.1.4.1.1.88.67" in sop_classes:
        _remove_sr_objects(series_sr, "1.2.840.10008.5.1.4.1.1.88.67", query_id_8)
        return "RDSR"
    elif "1.2.840.10008.5.1.4.1.1.88.22" in sop_classes:
        _remove_sr_objects(series_sr, "1.2.840.10008.5.1.4.1.1.88.22", query_id_8)
        return "ESR"
    elif "null_response" in sop_classes:
        logger.debug(
            f"{query_id_8} Check SR type: Image level response was null and -emptysr=True, "
            f"assuming SR series are RDSR."
        )
        return "null_response"
    else:
        logger.debug(
            f"{query_id_8} Check SR type: {series_sr.filter(deleted_flag=False).count()} "
            f"non-RDSR, non-ESR SR series remain"
        )
        return "no_dose_report"


def _get_responses(ae, remote, assoc, query, query_details):

    if assoc.is_established:
        responses = assoc.send_c_find(
            query_details["dataset"], StudyRootQueryRetrieveInformationModelFind
        )
        return assoc, responses
    elif assoc.is_aborted:
        logger.warning(
            f"{query_details['query_id_8']}/{query_details['query_level_id_8']} Association "
            f"aborted during query, trying again"
        )
    else:
        logger.error(
            f"{query_details['query_id_8']}/{query_details['query_level_id_8']} Association "
            f"not established, and not aborted!"
        )
    assoc = ae.associate(remote["host"], remote["port"], ae_title=remote["aet"])
    if assoc.is_established:
        responses = assoc.send_c_find(
            query_details["dataset"], StudyRootQueryRetrieveInformationModelFind
        )
        return assoc, responses
    elif assoc.is_aborted:
        msg = _("Association aborted twice in succession. Aborting query")
    elif assoc.is_rejected:
        msg = _("Association rejected after being aborted. Aborting query")
    else:
        msg = _("Association failed. Aborting query.")
    logger.warning(
        f"{query_details['query_id_8']}/{query_details['query_level_id_8']} {msg}"
    )
    query.stage = msg
    query.failed = True
    query.save()
    record_task_error_exit(msg)
    sys.exit()


def _failure_statuses(query, status, query_id_8, level_query_id):
    try:
        result_type = QR_FIND_SERVICE_CLASS_STATUS[status][0]
        result_status = QR_FIND_SERVICE_CLASS_STATUS[status][1]
    except KeyError:
        result_type = "Unknown"
        result_status = "Unknown status"
    logger.error(
        f"{query_id_8}/{level_query_id} Result: {result_type} (0x{status:04x}) - {result_status} "
    )
    query.errors = _(
        f"{result_type} (0x{status:04x}) - {result_status}. See logs for details."
    )
    query.save()


def _query_images(
    ae,
    remote,
    assoc,
    seriesrsp,
    query,
    initial_image_only=False,
    msg_id=None,
    instance_number=None,
):

    query_id_8 = _query_id_8(query)

    logger.debug(f"Query_id {query_id_8}: In _query_images")

    d3 = Dataset()
    d3.QueryRetrieveLevel = "IMAGE"
    d3.SeriesInstanceUID = seriesrsp.series_instance_uid
    d3.StudyInstanceUID = seriesrsp.dicom_qr_rsp_study.study_instance_uid
    d3.SOPInstanceUID = ""
    d3.SOPClassUID = ""
    d3.InstanceNumber = ""
    d3.SpecificCharacterSet = ""

    if initial_image_only:
        d3.InstanceNumber = "1"
    if instance_number is not None:
        d3.InstanceNumber = str(instance_number)
    if not msg_id:
        msg_id = 1

    logger.debug(
        f"{query_id_8} query is {d3}, initial_image_only is {initial_image_only}, msg_id is {msg_id}"
    )

    image_query_id = uuid.uuid4()

    # responses = assoc.send_c_find(d3, StudyRootQueryRetrieveInformationModelFind)
    assoc, responses = _get_responses(
        ae,
        remote,
        assoc,
        query,
        query_details={
            "dataset": d3,
            "query_id_8": query_id_8,
            "query_level_id_8": image_query_id.hex[:8],
        },
    )

    im_rsp_no = 0

    for (status, identifier) in responses:
        if status:
            if status.Status == 0x0000:
                logger.debug(
                    f"{query_id_8}/{image_query_id.hex[:8]} Image level matching is complete for this study"
                )
                query.stage = _(
                    "Image level matching for this study is complete (there many be more)"
                )
                query.save()

                # Some older systems start their instance numbers with 0. If == 1 does not deliver, retry with 0
                if initial_image_only and instance_number is None:
                    image_count = seriesrsp.dicomqrrspimage_set.count()
                    if image_count == 0:
                        _query_images(
                            ae, remote, assoc, seriesrsp, query, True, msg_id, 0
                        )

                return
            if status.Status in (0xFF00, 0xFF01):
                logger.debug(
                    f"{query_id_8}/{image_query_id.hex[:8]} Image level matches are continuing "
                    f"(0x{status.Status:04x})"
                )
                query.stage = _("Image level matches are continuing.")
                query.save()
                im_rsp_no += 1
                logger.debug(
                    f"{query_id_8}/{image_query_id.hex[:8]} Image Response {im_rsp_no}: {identifier}"
                )
                imagesrsp = DicomQRRspImage.objects.create(
                    dicom_qr_rsp_series=seriesrsp
                )
                imagesrsp.query_id = image_query_id
                # Mandatory tags
                imagesrsp.sop_instance_uid = identifier.SOPInstanceUID
                try:
                    imagesrsp.sop_class_uid = identifier.SOPClassUID
                except AttributeError:
                    logger.debug(
                        f"{query_id_8}/{image_query_id.hex[:8]} StudyInstUID {d3.StudyInstanceUID} Image "
                        f"Response {im_rsp_no}: no SOPClassUID. If CT, might need to use Toshiba Advanced option"
                        f" (additional config required)"
                    )
                    imagesrsp.sop_class_uid = ""
                try:
                    imagesrsp.instance_number = int(identifier.InstanceNumber)
                except (ValueError, TypeError, AttributeError):
                    logger.warning(
                        f"{query_id_8}/{image_query_id.hex[:8]} Image Response {im_rsp_no}: illegal response, "
                        f"no InstanceNumber"
                    )
                    imagesrsp.instance_number = None  # integer so can't be ''
                imagesrsp.save()
            else:
                _failure_statuses(
                    query, status.Status, query_id_8, image_query_id.hex[:8]
                )
        else:
            logger.info(
                f"{query_id_8}/{image_query_id.hex[:8]} Connection timed out, was aborted or received invalid"
                f" response"
            )
            query.stage = _(
                "Connection timed out, was aborted or received invalid response"
            )
            query.save()


def _query_series(ae, remote, assoc, d2, studyrsp, query):
    """Query for series level data for each study

    :param query:
    :param assoc: DICOM association with C-FIND SCP
    :param d2: DICOM dataset containing StudyInstanceUID to be used for series level query
    :param studyrsp: database entry for the study
    :return: None
    """

    d2.QueryRetrieveLevel = "SERIES"
    d2.SeriesDescription = ""
    d2.SeriesNumber = ""
    d2.SeriesInstanceUID = ""
    d2.Modality = ""
    d2.NumberOfSeriesRelatedInstances = ""
    d2.StationName = ""
    d2.SpecificCharacterSet = ""
    d2.SeriesTime = ""

    query_id_8 = _query_id_8(query)

    logger.debug(f"{query_id_8} In _query_series")
    logger.debug(f"{query_id_8} series query is {d2}")

    series_query_id = uuid.uuid4()

    # responses = assoc.send_c_find(d2, StudyRootQueryRetrieveInformationModelFind)
    assoc, responses = _get_responses(
        ae,
        remote,
        assoc,
        query,
        query_details={
            "dataset": d2,
            "query_id_8": query_id_8,
            "query_level_id_8": series_query_id.hex[:8],
        },
    )

    se_rsp_no = 0

    for (status, identifier) in responses:
        if status:
            if status.Status == 0x0000:
                logger.debug(
                    f"{query_id_8}/{series_query_id.hex[:8]} Series level matching is complete for this study"
                )
                query.stage = _(
                    "Series level matching for this study is complete (there may be more)"
                )
                query.save()
                return
            if status.Status in (0xFF00, 0xFF01):
                logger.debug(
                    f"{query_id_8}/{series_query_id.hex[:8]} Series level matches are"
                    f" continuing (0x{status.Status:04x})"
                )
                query.stage = _("Series level matches are continuing.")
                query.save()
                se_rsp_no += 1
                seriesrsp = DicomQRRspSeries.objects.create(dicom_qr_rsp_study=studyrsp)
                seriesrsp.query_id = series_query_id
                # Mandatory tags
                seriesrsp.series_instance_uid = identifier.SeriesInstanceUID
                try:
                    seriesrsp.modality = identifier.Modality
                except AttributeError:
                    seriesrsp.modality = "OT"  # not sure why a series is returned without, assume we don't want it.
                    logger.warning(
                        f"{query_id_8}/{series_query_id.hex[:8]} Illegal response with no modality at"
                        f" series level"
                    )
                try:
                    seriesrsp.series_number = int(identifier.SeriesNumber)
                except (ValueError, TypeError, AttributeError):
                    seriesrsp.series_number = None
                    # integer so can't be '' (ValueError). If missing will be (AttributeError).
                    # None would be ok, but now gives (TypeError)
                # Optional useful tags
                seriesrsp.series_description = get_value_kw(
                    "SeriesDescription", identifier
                )
                if seriesrsp.series_description:
                    seriesrsp.series_description = (
                        "".join(seriesrsp.series_description).strip().lower()
                    )
                seriesrsp.number_of_series_related_instances = get_value_kw(
                    "NumberOfSeriesRelatedInstances", identifier
                )
                if not seriesrsp.number_of_series_related_instances:
                    seriesrsp.number_of_series_related_instances = (
                        None  # integer so can't be ''
                    )
                seriesrsp.station_name = get_value_kw("StationName", identifier)
                seriesrsp.series_time = get_time("SeriesTime", identifier)
                logger.debug(
                    f"{query_id_8}/{series_query_id.hex[:8]} Series Response {se_rsp_no}: "
                    f"Modality {seriesrsp.modality}, StationName {seriesrsp.station_name}, "
                    f"StudyUID {d2.StudyInstanceUID}, Series No. {seriesrsp.series_number}, "
                    f"Series description {seriesrsp.series_description}"
                )
                seriesrsp.save()
            else:
                _failure_statuses(
                    query, status.Status, query_id_8, series_query_id.hex[:8]
                )
        else:
            logger.info(
                f"{query_id_8}/{series_query_id.hex[:8]} Connection timed out, was aborted or received "
                f"invalid response"
            )
            query.stage = _(
                "Connection timed out, was aborted or received invalid response"
            )
            query.save()


def _query_study(ae, remote, assoc, d, query, study_query_id):

    d.QueryRetrieveLevel = "STUDY"
    d.PatientName = ""
    d.PatientID = ""
    d.AccessionNumber = ""
    d.StudyDescription = ""
    d.StudyID = ""
    d.StudyInstanceUID = ""
    d.NumberOfStudyRelatedSeries = ""
    d.StationName = ""
    d.SpecificCharacterSet = ""

    query_id_8 = _query_id_8(query)

    logger.debug(
        f"{query_id_8}/{study_query_id.hex[:8]} Study level association requested"
    )
    logger.debug(f"{query_id_8}/{study_query_id.hex[:8]} Study level query is {d}")
    assoc, responses = _get_responses(
        ae,
        remote,
        assoc,
        query,
        query_details={
            "dataset": d,
            "query_id_8": query_id_8,
            "query_level_id_8": study_query_id.hex[:8],
        },
    )
    # responses = assoc.send_c_find(d, StudyRootQueryRetrieveInformationModelFind)

    logger.debug(
        f"{query_id_8}/{study_query_id.hex[:8]} _query_study done with status {responses}"
    )

    rspno = 0

    for (status, identifier) in responses:
        if status:
            if status.Status == 0x0000:
                query.stage = _(
                    "Study level matching for {modalities} is complete".format(
                        modalities=d.ModalitiesInStudy
                    )
                )
                query.save()
                logger.info(f"{query_id_8}/{study_query_id.hex[:8]} {query.stage}")
                return
            if status.Status in (0xFF00, 0xFF01):
                logger.debug(
                    f"{query_id_8}/{study_query_id.hex[:8]} Matches are continuing (0x{status.Status:04x})"
                )
                query.stage = _(
                    "Matches are continuing for {modalities} studies".format(
                        modalities=d.ModalitiesInStudy
                    )
                )
                query.save()
                # Next line commented to avoid patient information being logged
                # logger.debug(identifier)
                rspno += 1
                rsp = DicomQRRspStudy.objects.create(dicom_query=query)
                rsp.query_id = study_query_id
                # Unique key
                rsp.study_instance_uid = identifier.StudyInstanceUID
                # Required keys - none of interest
                logger.debug(
                    f"{query_id_8}/{study_query_id.hex[:8]} Response {rspno}, StudyUID: {rsp.study_instance_uid}"
                )

                # Optional and special keys
                rsp.study_description = get_value_kw("StudyDescription", identifier)
                rsp.station_name = get_value_kw("StationName", identifier)
                logger.debug(
                    f"{query_id_8}/{study_query_id.hex[:8]} Study Description: {rsp.study_description}; "
                    f"Station Name: {rsp.station_name}"
                )

                # Populate modalities_in_study, stored as JSON
                try:
                    if isinstance(
                        identifier.ModalitiesInStudy, str
                    ):  # if single modality, then type = string ('XA')
                        rsp.set_modalities_in_study(
                            identifier.ModalitiesInStudy.split(",")
                        )
                    else:  # if multiple modalities, type = MultiValue (['XA', 'RF'])
                        rsp.set_modalities_in_study(identifier.ModalitiesInStudy)
                    logger.debug(
                        f"{query_id_8}/{study_query_id.hex[:8]} "
                        f"ModalitiesInStudy: {rsp.get_modalities_in_study()}"
                    )
                except AttributeError:
                    rsp.set_modalities_in_study([""])
                    logger.debug(
                        f"{query_id_8}/{study_query_id.hex[:8]} ModalitiesInStudy was not in response"
                    )

                rsp.modality = None  # Used later
                rsp.save()
            else:
                _failure_statuses(
                    query, status.Status, query_id_8, study_query_id.hex[:8]
                )
        else:
            if assoc.is_aborted():
                status_msg = "Connection was aborted - check remote server logs."
            else:
                status_msg = "Connection timed out or received an invalid response. Check remote server logs"
            logger.error(f"{query_id_8}/{study_query_id.hex[:8]} {status_msg} ")
            query.stage = _(
                "Connection timed out, was aborted or received invalid response"
            )
            query.save()


def _query_for_each_modality(all_mods, query, d, assoc, ae, remote):
    """
    Uses _query_study for each modality we've asked for, and populates study level response data in the database
    :param all_mods: dict of dicts indicating which modalities to request
    :param query: DicomQuery object
    :param d: Dataset object containing StudyDate
    :param assoc: Established association with remote host
    :return: modalities_returned = whether ModalitiesInStudy is returned populated; modality_matching = whether
             responses have been filtered based on requested modality
    """

    # Assume that ModalitiesInStudy is a Matching Key Attribute
    # If not, 1 query is sufficient to retrieve all relevant studies
    modality_matching = True
    modalities_returned = False
    query_id_8 = _query_id_8(query)

    # query for all requested studies
    # if ModalitiesInStudy is not supported by the PACS set modality_matching to False and stop querying further
    for selection, details in list(all_mods.items()):
        if details["inc"]:
            for mod in details["mods"]:
                if modality_matching:
                    query.stage = _(
                        "Currently querying for {modality} studies…".format(
                            modality=mod
                        )
                    )
                    query.save()
                    logger.debug(f"{query_id_8} Currently querying for {mod} studies…")
                    d.ModalitiesInStudy = mod
                    if query.qr_scp_fk.use_modality_tag:
                        logger.debug(
                            f"{query_id_8} Using modality tag in study level query."
                        )
                        d.Modality = ""
                    study_query_id = uuid.uuid4()
                    _query_study(ae, remote, assoc, d, query, study_query_id)
                    study_rsp = query.dicomqrrspstudy_set.filter(
                        query_id__exact=study_query_id
                    )
                    logger.debug(
                        f"{query_id_8}/{study_query_id.hex[:8]} Queried for {mod}, now have "
                        f"{study_rsp.count()} study level responses"
                    )
                    for (
                        rsp
                    ) in (
                        study_rsp
                    ):  # First check if modalities in study has been populated
                        if (
                            rsp.get_modalities_in_study()
                            and rsp.get_modalities_in_study()[0] != ""
                        ):
                            modalities_returned = True
                            # Then check for inappropriate responses
                            if mod not in rsp.get_modalities_in_study():
                                modality_matching = False
                                logger.debug(
                                    f"{query_id_8}/{study_query_id.hex[:8]} Remote node returns but doesn't"
                                    f" match against ModalitiesInStudy"
                                )
                                break  # This indicates that there was no modality match, so we have everything already
                        else:  # modalities in study is empty
                            modalities_returned = False
                            if query.qr_scp_fk.use_modality_tag:
                                logger.debug(
                                    f"{query_id_8}/{study_query_id.hex[:8]} Remote node doesn't support"
                                    f" ModalitiesInStudy, but is configured to 'use_modality_tags' so assume it has"
                                    f" filtered and query again with next modality"
                                )
                                break
                            # ModalitiesInStudy not supported, therefore assume not matched on key
                            modality_matching = False
                            logger.debug(
                                f"{query_id_8}/{study_query_id.hex[:8]} Remote node doesn't support"
                                f" ModalitiesInStudy, assume we have everything"
                            )
                            break
    logger.debug(
        f"{query_id_8} modalities_returned: {modalities_returned}; "
        f"modality_matching: {modality_matching}"
    )
    return modalities_returned, modality_matching


def _remove_duplicates_in_study_response(query, initial_count):
    """Remove duplicates in study response on basis of StudyInstanceUID

    :param query: DICOM query response database object
    :return: response count after removing duplicates
    """

    query_id = query.query_id
    try:
        query_id_8 = query_id.hex[:8]
    except AttributeError:
        query_id_8 = query_id[:8]

    logger.debug(
        f"{query_id_8} {initial_count} study responses returned, removing duplicates."
    )
    try:
        for study_rsp in (
            query.dicomqrrspstudy_set.order_by("study_instance_uid")
            .values_list("study_instance_uid", flat=True)
            .distinct()
        ):
            query.dicomqrrspstudy_set.filter(
                pk__in=query.dicomqrrspstudy_set.filter(
                    study_instance_uid=study_rsp
                ).values_list("id", flat=True)[1:]
            ).update(
                deleted_flag=True, deleted_reason="Somehow this study was sent twice"
            )
        query.save()
        current_count = query.dicomqrrspstudy_set.filter(deleted_flag=False).count()
        logger.info(
            f"{query_id_8} Removed {initial_count - current_count} duplicates from response, {current_count} remain."
        )
        return current_count
    except NotImplementedError:
        logger.info(
            f"{query_id_8} Unable to remove duplicates - works with PostgreSQL only"
        )
        return initial_count


def _duplicate_ct_pet_studies(query, all_mods):
    """
    CT and Radipharmaceutical Studies (i.e PET/CT) often occur in the same study.
    Even though it's the same study, it has completly different data and is stored as 2
    studies with same id in openREM. We start this already here by duplicating studies which
    contain both modalities.
    """
    study_rsp = query.dicomqrrspstudy_set.all()
    for study in study_rsp:
        has_ct = all_mods["CT"]["inc"] and "CT" in study.get_modalities_in_study()
        has_nm = all_mods["NM"]["inc"] and (
            any(
                mod in study.get_modalities_in_study() for mod in all_mods["NM"]["mods"]
            )
        )
        if has_ct and has_nm:
            if "SR" in study.get_modalities_in_study():
                base = ["SR"]
            else:
                base = []
            study.set_modalities_in_study(base + ["CT"])
            study.save()
            study_nm = deepcopy(study)
            study_nm.pk = None
            study_nm.set_modalities_in_study(base + ["NM"])
            study_nm.save()
            for series in study.dicomqrrspseries_set.all():
                series_nm = deepcopy(series)
                series_nm.dicom_qr_rsp_study = study_nm
                series_nm.pk = None
                series_nm.save()


def qrscu(
    qr_scp_pk=None,
    store_scp_pk=None,
    implicit=False,
    explicit=False,
    move=False,
    query_id=None,
    date_from=None,
    date_until=None,
    single_date=False,
    time_from=None,
    time_until=None,
    modalities=None,
    inc_sr=False,
    remove_duplicates=True,
    filters=None,
    get_toshiba_images=False,
    get_empty_sr=False,
):
    """Query retrieve service class user function

    Queries a pre-configured remote query retrieve service class provider for dose metric related objects,
    making use of the filter parameters provided. Can automatically trigger a c-move (retrieve) operation.

    Args:
      qr_scp_pk(int, optional): Database ID/pk of the remote QR SCP (Default value = None)
      store_scp_pk(int, optional): Database ID/pk of the local store SCP (Default value = None)
      implicit(bool, optional): Prefer implicit transfer syntax (preference possibly not implemented) (Default value = False)
      explicit(bool, optional): Prefer explicit transfer syntax (preference possibly not implemented) (Default value = False)
      move(bool, optional): Automatically trigger move request when query is complete (Default value = False)
      query_id(str, optional): UID of query if generated by web interface (Default value = None)
      date_from(str, optional): Date to search from, format yyyy-mm-dd (Default value = None)
      date_until(str, optional): Date to search until, format yyyy-mm-dd (Default value = None)
      single_date(bool, optional): search only on date_from, allows time_from/time_until (Default value = False)
      time_from(str, optional): Time of day to search from, format hhmm 24 hour clock, single date only (Default value = None)
      time_until(str, optional): Time of day to search until, format hhmm 24 hour clock, single date only (Default value = None)
      modalities(list, optional): Modalities to search for, options are CT, MG, DX and FL (Default value = None)
      inc_sr(bool, optional): Only include studies that only have structured reports in (unknown modality) (Default value = False)
      remove_duplicates(bool, optional): If True, studies that already exist in the database are removed from the query results (Default value = True)
      filters(dictionary list, optional): lowercase include and exclude lists for StationName and StudyDescription (Default value = None)
      get_toshiba_images(bool, optional): Whether to try to get Toshiba dose summary images
      get_empty_sr(bool, optional): Whether to get SR series that return nothing at image level

    Returns:
      : Series Instance UIDs are stored as rows in the database to be used by a move request. Move request is
      optionally triggered automatically.

    """

    debug_timer = datetime.now()
    if not query_id:
        query_id = get_or_generate_task_uuid()
    try:
        query_id_8 = query_id.hex[:8]
    except AttributeError:
        query_id_8 = query_id[:8]

    if filters is None:
        filters = {
            "stationname_inc": None,
            "stationname_exc": None,
            "study_desc_inc": None,
            "study_desc_exc": None,
            "stationname_study": None,
        }

    logger.debug(f"Query_id is {query_id_8}")
    logger.debug(
        "{12} qrscu args passed: qr_scp_pk={0}, store_scp_pk={1}, implicit={2}, explicit={3}, move={4}, "
        "queryID={5}, date_from={6}, date_until={7}, modalities={8}, inc_sr={9}, remove_duplicates={10}, "
        "filters={11}".format(
            qr_scp_pk,
            store_scp_pk,
            implicit,
            explicit,
            move,
            query_id,
            date_from,
            date_until,
            modalities,
            inc_sr,
            remove_duplicates,
            filters,
            query_id,
        )
    )

    task = get_current_task()
    if task is None:
        logger.debug("qrscu is running in synchronous mode (no task id)")
    else:
        logger.debug(f"task id is {task.uuid}")

    # Currently, if called from qrscu_script modalities will either be a list of modalities or it will be "SR".
    # Web interface hasn't changed, so will be a list of modalities and or the inc_sr flag
    # Need to normalise one way or the other.
    logger.debug(f"{query_id_8} Checking for modality selection and sr_only clash")
    if modalities is None and inc_sr is False:
        logger.error(
            f"{query_id_8} Query retrieve routine called with no modalities selected"
        )
        return
    elif modalities is not None and inc_sr is True:
        logger.error(
            f"{query_id_8} Query retrieve routine should be called with a modality selection _or_ SR only query,"
            f" not both. Modalities is {modalities}, inc_sr is {inc_sr}"
        )
        return
    elif modalities is None and inc_sr is True:
        modalities = ["SR"]

    qr_scp = DicomRemoteQR.objects.get(pk=qr_scp_pk)
    remote = {}
    if qr_scp.hostname:
        remote["host"] = qr_scp.hostname
    else:
        remote["host"] = qr_scp.ip
    remote["port"] = qr_scp.port
    remote["aet"] = qr_scp.aetitle
    our_aet = qr_scp.callingaet
    if not our_aet:
        our_aet = "OPENREMDEFAULT"

    ae = AE()
    ae.add_requested_context(StudyRootQueryRetrieveInformationModelFind)
    ae.ae_title = our_aet
    ae.dimse_timeout = 300

    logger.debug(f"{query_id_8} Remote AE is {remote['aet']}")

    query = DicomQuery.objects.create()
    query.started_at = datetime.now()
    query.query_id = query_id
    query.complete = False
    query.store_scp_fk = DicomStoreSCP.objects.get(pk=store_scp_pk)
    query.qr_scp_fk = qr_scp
    query.move_completed_sub_ops = 0
    query.move_warning_sub_ops = 0
    query.move_failed_sub_ops = 0
    query.query_task = task
    study_date = str(
        make_dcm_date_range(date1=date_from, date2=date_until, single_date=single_date)
        or ""
    )
    if len(study_date) == 8:
        study_time = str(make_dcm_time_range(time1=time_from, time2=time_until) or "")
    else:
        study_time = ""
    if study_time:
        study_date_time = f"{study_date} {study_time}"
    else:
        study_date_time = study_date
    if inc_sr:
        modality_text = "SR 'studies' only"
    else:
        modality_text = f"Modalities = {modalities}"
    active_filters = {k: v for k, v in filters.items() if v is not None}
    query_summary_1 = (
        f"QR SCP PK = {qr_scp_pk} ({qr_scp.name}). "
        f"Store SCP PK = {store_scp_pk} ({query.store_scp_fk.name})."
    )
    query_summary_2 = f"{study_date_time}. {modality_text}. Filters = {active_filters}."
    query_summary_3 = (
        f"Advanced options: Remove duplicates = {remove_duplicates}, "
        f"Get Toshiba images = {get_toshiba_images}, Get 'empty' SR series = {get_empty_sr}"
    )
    query.query_summary = (
        f"{query_summary_1} <br> {query_summary_2} <br> {query_summary_3}"
    )
    query.save()

    assoc = ae.associate(remote["host"], remote["port"], ae_title=remote["aet"])

    if assoc.is_established:

        logger.info(
            f"{query_id_8} "
            f"DICOM FindSCU: {query_summary_1} \n    {query_summary_2} \n    {query_summary_3}"
        )
        d = Dataset()
        d.StudyDate = study_date
        d.StudyTime = study_time

        all_mods = collections.OrderedDict()
        all_mods["CT"] = {"inc": False, "mods": ["CT"]}
        all_mods["MG"] = {"inc": False, "mods": ["MG"]}
        all_mods["FL"] = {"inc": False, "mods": ["RF", "XA"]}
        all_mods["DX"] = {"inc": False, "mods": ["DX", "CR", "PX"]}
        all_mods["NM"] = {"inc": False, "mods": ["NM", "PT"]}
        all_mods["SR"] = {"inc": False, "mods": ["SR"]}

        # Reasoning regarding PET-CT: Some PACS allocate study modality PT, some CT, some depending on order received.
        # If ModalitiesInStudy is used for matching on C-Find, the CT from PET-CT will be picked up.
        # If not, then the PET-CT will be returned with everything else, and the CT will show up in the series level
        # query. Therefore, there is no need to search for PT at any stage.
        for m in all_mods:
            if m in modalities:
                all_mods[m]["inc"] = True

        # query for all requested studies
        modalities_returned, modality_matching = _query_for_each_modality(
            all_mods,
            query,
            d,
            assoc,
            ae,
            remote,
        )

        study_count = query.dicomqrrspstudy_set.count()
        removed_study = study_count - _remove_duplicates_in_study_response(
            query, study_count
        )

        _duplicate_ct_pet_studies(query, all_mods)
        study_rsp = query.dicomqrrspstudy_set.all()
        study_numbers = {"initial": query.dicomqrrspstudy_set.count()}
        study_numbers["current"] = study_numbers["initial"] - removed_study

        # Performing some cleanup if modality_matching=True (prevents having to retrieve unnecessary series)
        # We are assuming that if remote matches on modality it will populate ModalitiesInStudy and conversely
        # if remote doesn't match on modality it won't return a populated ModalitiesInStudy.
        study_rsp = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        if modalities_returned and inc_sr:
            logger.debug(
                f"{query_id_8} Modalities_returned is true and we only want studies with only SR in;"
                " removing everything else."
            )
            for study in study_rsp:
                mods = study.get_modalities_in_study()
                if mods != ["SR"]:
                    study.deleted_flag = True
                    study.deleted_reason = (
                        f"SR only checked, but this study contains {mods}"
                    )
                    study.save()
            study_rsp = study_rsp.filter(deleted_flag=False)
            study_numbers["current"] = study_rsp.count()
            study_numbers["sr_only_removed"] = (
                study_numbers["initial"] - study_numbers["current"]
            )
            logger.info(
                f"{query_id_8} Finished removing studies that have anything other than SR in, "
                f"{study_numbers['sr_only_removed']} removed, {study_numbers['current']} remain"
            )

        filter_logs = []
        if filters["study_desc_inc"]:
            filter_logs += [
                "study description includes {0}, ".format(
                    ", ".join(filters["study_desc_inc"])
                )
            ]
        if filters["study_desc_exc"]:
            filter_logs += [
                "study description excludes {0}, ".format(
                    ", ".join(filters["study_desc_exc"])
                )
            ]
        if filters["stationname_inc"]:
            filter_logs += [
                "station name includes {0}, ".format(
                    ", ".join(filters["stationname_inc"])
                )
            ]
        if filters["stationname_exc"]:
            filter_logs += [
                "station name excludes {0}, ".format(
                    ", ".join(filters["stationname_exc"])
                )
            ]

        if filter_logs:
            query.stage = _("Pruning study responses based on inc/exc options")
            query.save()
            logger.debug(
                f"{query_id_8} Pruning study responses based on inc/exc options: {''.join(filter_logs)}"
            )
            before_study_prune = study_numbers["current"]
            deleted_studies_filters = _prune_study_responses(query, filters)
            study_rsp = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
            study_numbers["current"] = study_rsp.count()
            study_numbers["inc_exc_removed"] = (
                before_study_prune - study_numbers["current"]
            )
            logger.info(
                f"{query_id_8} Pruning studies based on inc/exc has removed "
                f"{study_numbers['inc_exc_removed']} studies, {study_numbers['current']} studies remain."
            )
        else:
            deleted_studies_filters = {
                "study_desc_inc": 0,
                "study_desc_exc": 0,
                "stationname_inc": 0,
                "stationname_exc": 0,
            }
            logger.debug(f"{query_id_8} No inc/exc options selected")

        query.stage = _("Querying at series level to get more details about studies")
        query.save()
        logger.debug(
            f"{query_id_8} Querying at series level to get more details about studies"
        )
        for rsp in study_rsp:
            # Series level query
            d2 = Dataset()
            d2.StudyInstanceUID = rsp.study_instance_uid
            _query_series(ae, remote, assoc, d2, rsp, query)
            if not modalities_returned:
                _generate_modalities_in_study(rsp, query_id)
        logger.debug(f"{query_id_8} Series level query complete.")

        if (not modality_matching) or qr_scp.use_modality_tag:
            before_not_modality_matching = study_numbers["current"]
            mods_in_study_set = set(
                val
                for dic in study_rsp.values("modalities_in_study")
                for val in list(dic.values())
            )
            logger.debug(
                f"{query_id_8} mods in study are: {study_rsp.values('modalities_in_study')}"
            )
            query.stage = _("Deleting studies we didn't ask for")
            query.save()
            logger.debug(f"{query_id_8} Deleting studies we didn't ask for")
            logger.debug(f"{query_id_8} mods_in_study_set is {mods_in_study_set}")
            for mod_set in mods_in_study_set:
                logger.debug(f"{query_id_8} mod_set is {mod_set}")
                delete = True
                for mod_choice, details in list(all_mods.items()):
                    if details["inc"]:
                        logger.debug(
                            f"{query_id_8} mod_choice {mod_choice}, details {details}"
                        )
                        for mod in details["mods"]:
                            logger.debug(
                                f"{query_id_8} mod is {mod}, mod_set is {mod_set}"
                            )
                            if mod in mod_set:
                                delete = False
                                continue
                            if inc_sr and mod_set == ["SR"]:
                                delete = False
                if delete:
                    study_rsp.filter(modalities_in_study__exact=mod_set).update(
                        deleted_flag=True,
                        deleted_reason=f"The study only contained modalities we do not care about ({mod_set})",
                    )
            study_rsp = study_rsp.filter(deleted_flag=False).all()
            study_numbers["current"] = study_rsp.count()
            study_numbers["wrong_modality_removed"] = (
                before_not_modality_matching - study_numbers["current"]
            )
            logger.info(
                f"{query_id_8} Removing studies of modalities not asked for removed "
                f"{study_numbers['wrong_modality_removed']} studies, "
                f"{study_numbers['current']} studies remain"
            )

        query.stage = _("Pruning series responses")
        query.save()
        logger.debug(f"{query_id_8} Pruning series responses")
        before_series_pruning = study_numbers["current"]
        (
            deleted_studies,
            deleted_studies_filters_series,
            kept_ct,
        ) = _prune_series_responses(
            ae,
            remote,
            assoc,
            query,
            all_mods,
            filters,
            get_toshiba_images,
            get_empty_sr,
        )
        deleted_studies_filters["stationname_inc"] += deleted_studies_filters_series[
            "stationname_inc"
        ]
        deleted_studies_filters["stationname_exc"] += deleted_studies_filters_series[
            "stationname_exc"
        ]

        series_pruning_log = ""
        if all_mods["FL"]["inc"]:
            series_pruning_log += _(
                "{num_del_studies} RF studies were deleted from query due to no suitable RDSR being found. ".format(
                    num_del_studies=deleted_studies["RF"]
                )
            )
        if all_mods["CT"]["inc"]:
            series_pruning_log += _(
                "{ct_del_studies} CT studies were deleted from query due to no suitable images or reports being "
                "found. Of the remaining CT studies, {kept_ct_sr} have RDSR or ESR, {kept_ct_philips} have Philips "
                "dose images, {kept_ct_toshiba} have been prepared for the Toshiba import option and "
                "{kept_ct_maybe_philips} have been prepared as possibly containing Philips dose images. ".format(
                    ct_del_studies=deleted_studies["CT"],
                    kept_ct_sr=kept_ct["SR"],
                    kept_ct_philips=kept_ct["philips"],
                    kept_ct_toshiba=kept_ct["toshiba"],
                    kept_ct_maybe_philips=kept_ct["maybe_philips"],
                )
            )
        if all_mods["SR"]["inc"]:
            series_pruning_log += _(
                "{del_sr_studies} SR studies were deleted from query due to no suitable SR"
                " being found. ".format(del_sr_studies=deleted_studies["SR"])
            )
        logger.debug(f"{query_id_8} {series_pruning_log}")

        study_rsp = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
        study_numbers["current"] = study_rsp.count()
        study_numbers["series_pruning_removed"] = (
            before_series_pruning - study_numbers["current"]
        )
        logger.info(
            f"{query_id_8} Pruning series responses removed {study_numbers['series_pruning_removed']} "
            f"studies, leaving {study_numbers['current']} studies"
        )

        if remove_duplicates:
            study_rsp = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
            before_remove_duplicates = study_rsp.count()
            query.stage = _(
                "Removing any responses that match data we already have in the database"
            )
            logger.debug(
                f"{query_id_8} Removing any responses that match data we already have in the database"
            )
            query.save()
            _remove_duplicates(ae, remote, query, study_rsp, assoc)
            study_numbers["current"] = query.dicomqrrspstudy_set.filter(
                deleted_flag=False
            ).count()
            study_numbers["duplicates_removed"] = (
                before_remove_duplicates - study_numbers["current"]
            )
            logger.info(
                f"{query_id_8} Removing duplicates of previous objects removed "
                f"{study_numbers['duplicates_removed']}, "
                f"leaving {study_numbers['current']}"
            )

        # done
        assoc.release()
        _make_query_deleted_reasons_consistent(query)
        query.complete = True

        time_took = (datetime.now() - debug_timer).total_seconds()
        study_numbers["current"] = query.dicomqrrspstudy_set.filter(
            deleted_flag=False
        ).count()
        query.stage = _(
            "Query complete. Query took {time} and we are left with {studies_left} studies to move.<br>"
            "Of the original {studies_initial} study responses, ".format(
                time=naturalduration(time_took),
                studies_left=study_numbers["current"],
                studies_initial=study_numbers["initial"],
            )
        )
        query.stage += series_pruning_log
        filter_pruning_logs = ""
        filter_level = "series"
        if filters["stationname_study"]:
            filter_level = "study"
        if filters["study_desc_inc"]:
            filter_pruning_logs += _(
                "only studies with description that include '{text}' removed {num} studies, ".format(
                    text=", ".join(filters["study_desc_inc"]),
                    num=deleted_studies_filters["study_desc_inc"],
                )
            )
        if filters["study_desc_exc"]:
            filter_pruning_logs += _(
                "studies with description that do not include '{text}' removed {num} studies, ".format(
                    text=", ".join(filters["study_desc_exc"]),
                    num=deleted_studies_filters["study_desc_exc"],
                )
            )
        if filters["stationname_inc"]:
            filter_pruning_logs += _(
                "only studies with station names that include '{text}' at {filter_level} level"
                " removed {num} studies, ".format(
                    filter_level=filter_level,
                    text=", ".join(filters["stationname_inc"]),
                    num=deleted_studies_filters["stationname_inc"],
                )
            )
        if filters["stationname_exc"]:
            filter_pruning_logs += _(
                "studies with station names that do not include '{text}' at {filter_level} level"
                " removed {num} studies, ".format(
                    filter_level=filter_level,
                    text=", ".join(filters["stationname_exc"]),
                    num=deleted_studies_filters["stationname_exc"],
                )
            )
        if filter_pruning_logs:
            query.stage += _(
                "<br>Filtering for {pruning_logs}.".format(
                    pruning_logs=filter_pruning_logs[:-2]
                )
            )
        if remove_duplicates:
            query.stage += _(
                "<br>Removing duplicates of previous objects removed {duplicates_removed} studies.".format(
                    duplicates_removed=study_numbers["duplicates_removed"]
                )
            )
        if query.errors:
            query.stage += _(
                "<br>The following errors were received: {errors}".format(
                    errors=query.errors
                )
            )
        query.save()
        stage_text = query.stage.replace("<br>", "\n -- ")
        logger.info(f"{query_id_8} {stage_text}")

        logger.debug(
            f"{query_id_8} Query complete. Move is {move}. Query took {time_took}"
        )

        if move:
            movescu(str(query.query_id))

    else:
        if assoc.is_rejected:
            msg = "{0}: {1}".format(
                assoc.acceptor.primitive.result_str, assoc.acceptor.primitive.reason_str
            )
            logger.warning(
                f"{query_id_8} Association rejected from {remote['host']} {remote['port']}"
                f" {remote['aet']}. {msg}"
            )
        elif assoc.is_aborted:
            msg = _("Association aborted or never connected")
            logger.warning(
                f"{query_id_8} {msg} to {remote['host']} {remote['port']} {remote['aet']}"
            )
        else:
            msg = _("Association Failed")
            logger.warning(
                f"{query_id_8} {msg} with {remote['host']} {remote['port']} {remote['aet']}"
            )
        query.message = msg
        query.stage = msg
        query.failed = True
        query.save()


def _move_req(my_ae, assoc, d, study_no, series_no, query):

    responses = assoc.send_c_move(
        d, my_ae.ae_title, StudyRootQueryRetrieveInformationModelMove
    )

    completed_sub_ops = 0
    failed_sub_ops = 0
    warning_sub_ops = 0
    for (status, identifier) in responses:
        if status:
            # LO: Changed from status.NumberOfCompletedSubOperations to getattr
            #     status is actually the move-response and may not contain these items (e.g. in case of error)
            completed_sub_ops = getattr(status, "NumberOfCompletedSuboperations", None)
            failed_sub_ops = getattr(status, "NumberOfFailedSuboperations", None)
            warning_sub_ops = getattr(status, "NumberOfWarningSuboperations", None)
            status_msg = "Status undefined."
            # If the status is 'Pending' then the identifier is the C-MOVE response
            if status.Status in (0xFF00, 0xFF01):
                # print(identifier)
                status_msg = "Match returned, further matches are continuing."
            elif status.Status == 0x0000:
                status_msg = "All matches returned."
            else:
                if status.Status == 0xFE00:
                    status.msg = "Failure 0xFE00 - Sub-operations terminated due to Cancel indication."
                else:
                    try:
                        status_type = QR_MOVE_SERVICE_CLASS_STATUS[status.Status][0]
                        status_message = QR_MOVE_SERVICE_CLASS_STATUS[status.Status][1]
                    except KeyError:
                        status_type = "Unknown"
                        status_message = "Unknown status"
                    status_msg = (
                        f"{status_type} (0x{status.Status:04x}) - {status_message}."
                    )
                msg = (
                    f"Move of study {study_no}, series {series_no}: {status_msg} "
                    f"Sub-ops completed: {completed_sub_ops}, failed: {failed_sub_ops}, "
                    f"warning: {warning_sub_ops}."
                )
                logger.error(msg)
                query.move_summary = msg
                query.save()
                return False
            msg = (
                f"Move of study {study_no}, series {series_no}: {status_msg} "
                f"Sub-ops completed: {completed_sub_ops}, failed: {failed_sub_ops}, "
                f"warning: {warning_sub_ops}."
            )
            query.move_summary = msg
            query.save()
        else:
            if assoc.acse.is_aborted():
                status_msg = "Connection was aborted - check remote server logs."
            else:
                status_msg = "Connection timed out or received an invalid response. Check remote server logs"
            msg = (
                f"Move of study {study_no}, series {series_no}: {status_msg} "
                f"Cumulative sub-ops completed: {query.move_completed_sub_ops}, "
                f"failed: {query.move_failed_sub_ops}, warning: {query.move_warning_sub_ops}."
            )
            logger.error(msg)
            query.move_summary = msg
            query.save()
            return False
    query.move_completed_sub_ops += completed_sub_ops
    query.move_failed_sub_ops += failed_sub_ops
    query.move_warning_sub_ops += warning_sub_ops
    query.save()
    return True


def _remove_duplicate_images(series):
    """
    Removes duplicate images from a series. Duplication can happen if
    for some reason the images for the same series are queried
    multiple times.
    """
    # distinct with specific fields only works on postgres. Therefore not used here.
    seen = set()
    for img in series.dicomqrrspimage_set.all():
        if img.sop_instance_uid in seen:
            img.deleted_flag = True
            img.deleted_reason = (
                "It seems like this image was found twice or somewhere duplicated"
            )
            img.save()
        seen.add(img.sop_instance_uid)


def _move_if_established(ae, assoc, d, study_no, series_no, query, remote):
    if assoc.is_established:
        move = _move_req(ae, assoc, d, study_no, series_no, query)
        if move:
            return True, None
    elif assoc.is_aborted:
        logger.warning(
            f"Query_id {query.query_id}: Association aborted during move requests, trying again"
        )
    else:
        logger.error(
            f"Query_id {query.query_id}: Move association not established, and not aborted!"
        )
    assoc = ae.associate(remote["host"], remote["port"], ae_title=remote["aet"])
    if assoc.is_established:
        move = _move_req(ae, assoc, d, study_no, series_no, query)
        if move:
            return True, None
    elif assoc.is_aborted:
        msg = "Move aborted twice in succession. Aborting move request."
        logger.warning(f"Query_id {query.query_id}: {msg}")
        return False, msg
    elif assoc.is_rejected:
        msg = "Association rejected after being aborted. Aborting move request."
        logger.warning(f"Query_id {query.query_id}: {msg}")
        return False, msg
    msg = "Move failed twice in succession. Aborting move request"
    logger.warning(f"Query_id {query.query_id}: {msg}")
    return False, msg


def movescu(query_id):
    """
    C-Move request element of query-retrieve service class user
    :param query_id: ID of query in the DicomQuery table
    :return: None
    """
    # debug_logger()

    logger.debug("Query_id {0}: Starting move request".format(query_id))
    try:
        query = DicomQuery.objects.get(query_id=query_id)
    except ObjectDoesNotExist:
        msg = "Move called with invalid query_id {0}. Move abandoned.".format(query_id)
        logger.warning(msg)
        record_task_error_exit(msg)
        return 0
    query.move_complete = False
    query.move_task = get_current_task()
    query.failed = False
    query.save()
    qr_scp = query.qr_scp_fk
    store_scp = query.store_scp_fk

    ae = AE()
    ae.add_requested_context(StudyRootQueryRetrieveInformationModelMove)
    ae.ae_title = store_scp.aetitle
    ae.dimse_timeout = 300

    logger.debug("Move AE my_ae {0} started".format(ae))

    # remote application entity
    remote = {"port": qr_scp.port, "aet": qr_scp.aetitle}
    if qr_scp.hostname:
        remote["host"] = qr_scp.hostname
    else:
        remote["host"] = qr_scp.ip

    logger.debug("Query_id {0}: Requesting move association".format(query_id))
    assoc = ae.associate(remote["host"], remote["port"], ae_title=remote["aet"])
    logger.debug("Query_id {0}: Move association requested".format(query_id))

    query.move_summary = "Preparing to start move request"
    query.save()
    logger.debug("Query_id {0}: Preparing to start move request".format(query_id))

    studies = query.dicomqrrspstudy_set.filter(deleted_flag=False).all()
    query.move_summary = "Requesting move of {0} studies".format(studies.count())
    query.save()
    logger.info(
        "Query_id {0}: Requesting move of {1} studies".format(query_id, studies.count())
    )

    if assoc.is_established:
        logger.debug(
            f"Mv {query.move_uuid} Association with {remote['aet']} is established."
        )
        study_no = 0
        move = True
        for study in studies:
            if not move:
                break
            study_no += 1
            logger.debug("Mv: study_no {0}".format(study_no))
            series_no = 0
            for series in study.dicomqrrspseries_set.filter(deleted_flag=False).all():
                if not move:
                    break
                series_no += 1
                logger.debug(
                    "Mv: study no {0} series no {1}".format(study_no, series_no)
                )
                d = Dataset()
                d.StudyInstanceUID = study.study_instance_uid
                d.QueryRetrieveLevel = "SERIES"
                d.SeriesInstanceUID = series.series_instance_uid
                if series.number_of_series_related_instances:
                    num_objects = " Series contains {0} objects".format(
                        series.number_of_series_related_instances
                    )
                else:
                    num_objects = ""
                query.move_summary = "Requesting move: modality {0}, study {1} (of {2}) series {3} (of {4}).{5}".format(
                    study.modality,
                    study_no,
                    studies.count(),
                    series_no,
                    study.dicomqrrspseries_set.filter(deleted_flag=False).count(),
                    num_objects,
                )
                logger.info(
                    "Requesting move: modality {0}, study {1} (of {2}) series {3} (of {4}).{5}".format(
                        study.modality,
                        study_no,
                        studies.count(),
                        series_no,
                        study.dicomqrrspseries_set.filter(deleted_flag=False).count(),
                        num_objects,
                    )
                )
                query.save()
                logger.debug("_move_req launched")
                if series.image_level_move:
                    d.QueryRetrieveLevel = "IMAGE"
                    _remove_duplicate_images(series)
                    for image in series.dicomqrrspimage_set.filter(
                        deleted_flag=False
                    ).all():
                        d.SOPInstanceUID = image.sop_instance_uid
                        logger.debug("Image-level move - d is: {0}".format(d))
                        move, msg = _move_if_established(
                            ae, assoc, d, study_no, series_no, query, remote
                        )
                        if not move:
                            break
                else:
                    logger.debug("Series-level move - d is: {0}".format(d))
                    move, msg = _move_if_established(
                        ae, assoc, d, study_no, series_no, query, remote
                    )
                    if not move:
                        break
        try:
            assoc.release()
            logger.info("Query_id {0}: Move association released".format(query_id))
        except AttributeError:
            logger.info(
                "Query_id {0}: Could not release Move association due to an AttributeError: perhaps no "
                "studies were present".format(query_id)
            )
        if move:
            query.move_complete = True
            msg = (
                f"Move complete. {studies.count()} studies.  "
                f"Cumulative sub-ops completed: {query.move_completed_sub_ops}, "
                f"failed: {query.move_failed_sub_ops}, warning: {query.move_warning_sub_ops}."
            )
            query.save()
            logger.debug(msg)

            logger.debug("Query_id {0}: Releasing move association".format(query_id))
        else:
            record_task_error_exit(
                "Something went wrong, cannot move further. "
                "Aborting. (Probably lost connection for a short time)"
            )
            return 0

    elif assoc.is_rejected:
        msg = "{0}: {1}".format(
            assoc.acceptor.primitive.result_str, assoc.acceptor.primitive.reason_str
        )
        logger.warning(
            "Association rejected from {0} {1} {2}. {3}".format(
                remote["host"], remote["port"], remote["aet"], msg
            )
        )
        record_task_error_exit(msg)
    elif assoc.is_aborted:
        msg = "Association aborted or never connected"
        logger.warning(
            "{3} to {0} {1} {2}".format(
                remote["host"], remote["port"], remote["aet"], msg
            )
        )
        record_task_error_exit(msg)
    else:
        msg = "Association Failed"
        logger.warning(
            "{3} with {0} {1} {2}".format(
                remote["host"], remote["port"], remote["aet"], msg
            )
        )
        record_task_error_exit(msg)
    query.move_summary = msg
    query.save()


def _create_parser():

    parser = argparse.ArgumentParser(
        description="Query remote server and retrieve to OpenREM"
    )
    parser.add_argument("qr_id", type=int, help="Database ID of the remote QR node")
    parser.add_argument(
        "store_id", type=int, help="Database ID of the local store node"
    )
    parser.add_argument(
        "-ct", action="store_true", help="Query for CT studies. Cannot be used with -sr"
    )
    parser.add_argument(
        "-mg",
        action="store_true",
        help="Query for mammography studies. Cannot be used with -sr",
    )
    parser.add_argument(
        "-fl",
        action="store_true",
        help="Query for fluoroscopy studies. Cannot be used with -sr",
    )
    parser.add_argument(
        "-dx",
        action="store_true",
        help="Query for planar X-ray studies (includes panoramic X-ray studies). Cannot be used with -sr",
    )
    parser.add_argument(
        "-nm",
        action="store_true",
        help="Query for nuclear medicine studies. Cannot be used with -sr",
    )
    parser.add_argument(
        "-f",
        "--dfrom",
        help="Date from, format yyyy-mm-dd. Cannot be used with --single_date",
        metavar="yyyy-mm-dd",
    )
    parser.add_argument(
        "-t",
        "--duntil",
        help="Date until, format yyyy-mm-dd. Cannot be used with --single_date",
        metavar="yyyy-mm-dd",
    )
    parser.add_argument(
        "-sd",
        "--single_date",
        help="Date, format yyy-mm-dd. Cannot be used with --dfrom or --duntil",
        metavar="yyyy-mm-dd",
    )
    parser.add_argument(
        "-tf",
        "--tfrom",
        help="Time from, format hhmm. Requires --single_date.",
        metavar="hhmm",
    )
    parser.add_argument(
        "-tt",
        "--tuntil",
        help="Time until, format hhmm. Requires --single_date.",
        metavar="hhmm",
    )
    parser.add_argument(
        "-e",
        "--desc_exclude",
        help="Terms to exclude in study description, comma separated, quote whole string",
        metavar="string",
    )
    parser.add_argument(
        "-i",
        "--desc_include",
        help="Terms that must be included in study description, comma separated, quote whole string",
        metavar="string",
    )
    parser.add_argument(
        "-sne",
        "--stationname_exclude",
        help="Terms to exclude in station name, comma separated, quote whole string",
        metavar="string",
    )
    parser.add_argument(
        "-sni",
        "--stationname_include",
        help="Terms to include in station name, comma separated, quote whole string",
        metavar="string",
    )
    parser.add_argument(
        "--stationname_study_level",
        help="Advanced: Filter station name at Study level, instead of at Series level",
        action="store_true",
    )
    parser.add_argument(
        "-toshiba",
        action="store_true",
        help="Advanced: Attempt to retrieve CT dose summary objects and one image from each series",
    )
    parser.add_argument(
        "-sr",
        action="store_true",
        help="Advanced: Use if store has RDSRs only, no images. Cannot be used with -ct, -mg, -fl, -dx",
    )
    parser.add_argument(
        "-dup",
        action="store_true",
        help="Advanced: Retrieve duplicates (objects that have been processed before)",
    )
    parser.add_argument(
        "-emptysr",
        action="store_true",
        help="Advanced: Get SR series that return nothing at image level query",
    )

    return parser


def _process_args(parser_args, parser):
    from .tools import (  # pylint: disable-import-outside-toplevel
        echoscu,
    )  # If I don't leave this here the patching doesn't work in test...

    logger.info("qrscu script called")

    modalities = []
    if parser_args.ct:
        modalities += ["CT"]
    if parser_args.mg:
        modalities += ["MG"]
    if parser_args.fl:
        modalities += ["FL"]
    if parser_args.dx:
        modalities += ["DX"]
    if parser_args.nm:
        modalities += ["NM"]
    if parser_args.sr:
        if modalities:
            parser.error("The sr option can not be combined with any other modalities")
        else:
            modalities += ["SR"]

    if not modalities:
        parser.error("At least one modality must be specified")
    else:
        logger.info("Modalities are {0}".format(modalities))

    # Check if dates are in the right format, but keep them as strings
    if (parser_args.single_date and parser_args.dfrom) or (
        parser_args.single_date and parser_args.duntil
    ):
        parser.error("--single_date cannot be used with --dfrom or --duntil")
    date_1 = None
    single_date = False
    try:
        if parser_args.dfrom:
            datetime.strptime(parser_args.dfrom, "%Y-%m-%d")
            logger.info("Date from: {0}".format(parser_args.dfrom))
            date_1 = parser_args.dfrom
        if parser_args.duntil:
            datetime.strptime(parser_args.duntil, "%Y-%m-%d")
            logger.info("Date until: {0}".format(parser_args.duntil))
        if parser_args.single_date:
            datetime.strptime(parser_args.single_date, "%Y-%m-%d")
            logger.info("Single date: {0}".format(parser_args.single_date))
            date_1 = parser_args.single_date
            single_date = True
    except ValueError:
        parser.error("Incorrect data format, should be YYYY-MM-DD")

    # Check if times are in the right format, but keep them as strings
    if (parser_args.tfrom or parser_args.tuntil) and not single_date:
        parser.error("--tfrom and --tuntil require --single_date to be used")
    try:
        # Need to check only one date provided - only accept if times within a day. Time range with date range will give
        # those times each day. Extended negotiation allows for time on date 1 to time on date 2, but we don't have that
        if parser_args.tfrom:
            datetime.strptime(parser_args.tfrom, "%H%M")
            logger.info("Time from: {0}".format(parser_args.tfrom))
        if parser_args.tuntil:
            datetime.strptime(parser_args.tuntil, "%H%M")
            logger.info("Time until: {0}".format(parser_args.tuntil))
    except ValueError:
        parser.error("Incorrect time format, should be HHMM")

    if parser_args.desc_exclude:
        study_desc_exc = [
            x.strip().lower() for x in parser_args.desc_exclude.split(",")
        ]
        logger.info("Study description exclude terms are {0}".format(study_desc_exc))
    else:
        study_desc_exc = None
    if parser_args.desc_include:
        study_desc_inc = [
            x.strip().lower() for x in parser_args.desc_include.split(",")
        ]
        logger.info("Study description include terms are {0}".format(study_desc_inc))
    else:
        study_desc_inc = None

    if parser_args.stationname_exclude:
        stationname_exc = [
            x.strip().lower() for x in parser_args.stationname_exclude.split(",")
        ]
        logger.info("Stationname exclude terms are {0}".format(stationname_exc))
    else:
        stationname_exc = None
    if parser_args.stationname_include:
        stationname_inc = [
            x.strip().lower() for x in parser_args.stationname_include.split(",")
        ]
        logger.info("Stationname include terms are {0}".format(stationname_inc))
    else:
        stationname_inc = None

    filters = {
        "stationname_inc": stationname_inc,
        "stationname_exc": stationname_exc,
        "study_desc_inc": study_desc_inc,
        "study_desc_exc": study_desc_exc,
        "stationname_study": parser_args.stationname_study_level,
    }

    remove_duplicates = not parser_args.dup  # if flag, duplicates will be retrieved.

    get_toshiba = parser_args.toshiba
    get_empty_sr = parser_args.emptysr

    qr_node_up = echoscu(parser_args.qr_id, qr_scp=True)
    store_node_up = echoscu(parser_args.store_id, store_scp=True)

    if qr_node_up != "Success" or store_node_up != "Success":
        logger.error(
            "Query-retrieve aborted: DICOM nodes not ready. QR SCP echo is {0}, Store SCP echo is {1}".format(
                qr_node_up, store_node_up
            )
        )
        record_task_error_exit(
            "Query-retrieve aborted: DICOM nodes not ready. QR SCP echo is {0}, Store SCP echo is {1}".format(
                qr_node_up, store_node_up
            )
        )
        sys.exit()

    return_args = {
        "qr_id": parser_args.qr_id,
        "store_id": parser_args.store_id,
        "modalities": modalities,
        "remove_duplicates": remove_duplicates,
        "dfrom": date_1,
        "duntil": parser_args.duntil,
        "single_date": single_date,
        "tfrom": parser_args.tfrom,
        "tuntil": parser_args.tuntil,
        "filters": filters,
        "get_toshiba": get_toshiba,
        "get_empty_sr": get_empty_sr,
    }

    return return_args


def qrscu_script():
    """Query-Retrieve function that can be called by the openrem_qr.py script. Always triggers a move.

    :param args: sys.argv from command line call
    :return:
    """

    parser = _create_parser()
    args = parser.parse_args()
    processed_args = _process_args(args, parser)
    b = run_in_background(
        qrscu,
        "query",
        qr_scp_pk=processed_args["qr_id"],
        store_scp_pk=processed_args["store_id"],
        move=True,
        modalities=processed_args["modalities"],
        remove_duplicates=processed_args["remove_duplicates"],
        date_from=processed_args["dfrom"],
        date_until=processed_args["duntil"],
        single_date=processed_args["single_date"],
        time_from=processed_args["tfrom"],
        time_until=processed_args["tuntil"],
        filters=processed_args["filters"],
        get_toshiba_images=processed_args["get_toshiba"],
        get_empty_sr=processed_args["get_empty_sr"],
    )
    print("Running Query")
    wait_task(b)
