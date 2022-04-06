# This Python file uses the following encoding: utf-8
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
..  module:: nm_image.
    :synopsis: Module to extract nm specific radiation information from PET/NM Images. Those methods 
        are 'careful' in the sense of not ever overwriting any data that is already present. If 
        a study with the same id as the one used here is already present it will only update 
        fields concerned with radiopharmaceutical administration, and also those only if they
        are not already present.
        The rrdsr reader on the other hand will overwrite data read by this module without futher
        notice (at least with data present in the rrdsr). The rationale for this is that rrdsr's
        are (assumed to be) more complete and therefore take precedence.

..  moduleauthor:: Jannis Widmer

"""
from datetime import datetime
from decimal import Decimal
import logging
import sys

from celery import shared_task
from django.db.models import Q
import pydicom

from remapp.models import (
    GeneralStudyModuleAttr,
    ObjectUIDsProcessed,
    RadiopharmaceuticalAdministrationEventData,
    RadiopharmaceuticalRadiationDose,
)

from ..tools.dcmdatetime import get_date_time, get_time
from ..tools.get_values import (
    get_value_kw,
    get_or_create_cid,
    get_seq_code_meaning,
    get_seq_code_value,
)
from .extract_common import (
    generalequipmentmoduleattributes,
    generalstudymoduleattributes,
    patientstudymoduleattributes,
    patient_module_attributes,
)

logger = logging.getLogger(
    "remapp.extractors.nm_image"
)  # Explicitly named so that it is still handled when using __main__


def _check_replace(study, attr, value, study_id, is_not_equal):
    """Check if a field on db should be overwritten (will say yes when it's empty)"""
    current = getattr(study, attr)
    if not current:
        return True
    elif is_not_equal(current, value):
        logger.warn(
            f"Found different values for {attr} on image and existing object for study {study_id}"
        )
    return False


def _try_set_value(
    where,
    kw_database,
    dataset,
    kw_dataset,
    study_id,
    get_with=get_value_kw,
    mod=lambda x: x,
    is_not_equal=lambda x, y: x != y,
):
    """Tries to get value from dataset and write onto db if it does not exist in db"""
    value = get_with(kw_dataset, dataset)
    if not value:
        return
    value = mod(value)
    if _check_replace(where, kw_database, value, study_id, is_not_equal):
        setattr(where, kw_database, value)


def _code_handler(value):
    return get_or_create_cid(value.code_value, value.code_meaning)


def _code_getter(kw, ds):
    a = type("", (), {})()
    setattr(a, "code_value", get_seq_code_value(kw, ds))
    setattr(a, "code_meaning", get_seq_code_meaning(kw, ds))
    if not a.code_value and not a.code_meaning:
        return None
    else:
        return a


def _isotope(study, dataset):
    """
    method should work for both NM and PET images. Some of the values are only present in PET, not in NM.
    Some of the information is named differently in both models - in such a case both names are tried.

    study should contain a Study Date when already, as this is used to populate Start/Stop DateTime if no date is stored.
    The dataset is expected to be at the top of the file (such that SOPClassUID can be read)
    """
    is_nm_img = dataset.SOPClassUID == "1.2.840.10008.5.1.4.1.1.20"
    dataset = dataset[0x54, 0x16].value[0]

    float_not_equal = lambda x, y: abs(x - y) > 10e-5
    float_convert = lambda x: Decimal(x)

    study_id = study.study_instance_uid
    radio = (
        study.radiopharmaceuticalradiationdose_set.get().radiopharmaceuticaladministrationeventdata_set.get()
    )
    _try_set_value(
        radio,
        "radionuclide",
        dataset,
        "RadionuclideCodeSequence",
        study_id,
        _code_getter,
        _code_handler,
    )
    _try_set_value(
        radio,
        "radionuclide_half_life",
        dataset,
        "RadionuclideHalfLife",
        study_id,
        mod=float_convert,
        is_not_equal=float_not_equal,
    )
    for ds_name in [
        "RadiopharmaceuticalCodeSequence",
        "RadiopharmaceuticalInformationSequence",
    ]:
        _try_set_value(
            radio,
            "radiopharmaceutical_agent",
            dataset,
            ds_name,
            study_id,
            _code_getter,
            _code_handler,
        )
    _try_set_value(
        radio,
        "radiopharmaceutical_agent_string",
        dataset,
        "Radiopharmaceutical",
        study_id,
    )
    _try_set_value(
        radio,
        "radiopharmaceutical_specific_activity",
        dataset,
        "RadiopharmaceuticalSpecificActivity",
        study_id,
        mod=float_convert,
        is_not_equal=float_not_equal,
    )

    # We don't load the Radiopharmaceutical Administration UID, even though it's present because:
    #   1: Sometimes it's wrong, for whatever reason
    #   2: By not reading it we will have None as our UID, which allows the rdsr import to recognise such studies
    #       and reimport them

    _try_set_value(
        radio,
        "radiopharmaceutical_start_datetime",
        dataset,
        "RadiopharmaceuticalStartDateTime",
        study_id,
        get_date_time,
    )
    _try_set_value(
        radio,
        "radiopharmaceutical_stop_datetime",
        dataset,
        "RadiopharmaceuticalStopDateTime",
        study_id,
        get_date_time,
    )

    def build_datetime(x):
        return datetime.combine(study.study_date, x.time())

    _try_set_value(
        radio,
        "radiopharmaceutical_start_datetime",
        dataset,
        "RadiopharmaceuticalStartTime",
        study_id,
        get_time,
        build_datetime,
    )
    _try_set_value(
        radio,
        "radiopharmaceutical_stop_datetime",
        dataset,
        "RadiopharmaceuticalStopTime",
        study_id,
        get_time,
        build_datetime,
    )

    if is_nm_img:
        conversion_func = float_convert
    else:
        conversion_func = lambda x: Decimal(x) / 10**6  # Convert to MBq from Bq
    _try_set_value(
        radio,
        "administered_activity",
        dataset,
        "RadionuclideTotalDose",
        study_id,
        mod=conversion_func,
        is_not_equal=float_not_equal,
    )
    _try_set_value(
        radio,
        "radiopharmaceutical_volume",
        dataset,
        "RadiopharmaceuticalVolume",
        study_id,
    )
    _try_set_value(
        radio,
        "route_of_administration",
        dataset,
        "AdministrationRouteCodeSequence",
        study_id,
        _code_getter,
        _code_handler,
    )

    radio.save()


def _record_object_imported(dataset, study):
    """
    Saves that this object has been imported.
    """
    o = ObjectUIDsProcessed.objects.create(general_study_module_attributes=study)
    o.sop_instance_uid = dataset.SOPInstanceUID
    o.save()


def _nm2db(dataset):
    if "StudyInstanceUID" in dataset:
        study = GeneralStudyModuleAttr.objects.filter(
            Q(study_instance_uid__exact=dataset.StudyInstanceUID)
            & Q(modality_type__exact="NM")
        ).first()
        if study is not None:
            processed_count = study.objectuidsprocessed_set.filter(
                sop_instance_uid__exact=dataset.SOPInstanceUID
            ).count()
            if processed_count > 0:
                logger.info(
                    f"The Image with {dataset.SOPInstanceUID} was already imported. Will not import."
                )
                return

            _record_object_imported(dataset, study)
            _isotope(study, dataset)
            return

    study = GeneralStudyModuleAttr.objects.create()
    generalequipmentmoduleattributes(dataset, study)
    study.modality_type = "NM"  # will be saved by generalstudymoduleattributes call
    generalstudymoduleattributes(dataset, study, logger)
    _record_object_imported(dataset, study)
    patientstudymoduleattributes(dataset, study)
    patient_module_attributes(dataset, study)
    t = RadiopharmaceuticalRadiationDose.objects.create(
        general_study_module_attributes=study
    )
    t.save()
    RadiopharmaceuticalAdministrationEventData.objects.create(
        radiopharmaceutical_radiation_dose=t
    ).save()
    _isotope(study, dataset)


@shared_task(name="remapp.extractors.nm_image.nm_image")
def nm_image(file: str):
    """Extract radiation dose related data from DICOM PET/NM-Image.

    :param file: relative or absolute path to PET/DICOM Image.
    """
    dataset = pydicom.dcmread(file)
    dataset.decode()

    if dataset.SOPClassUID in [
        "1.2.840.10008.5.1.4.1.1.128",
        "1.2.840.10008.5.1.4.1.1.20",
    ]:
        _nm2db(dataset)
    else:
        logger.error(f"{file} is not an NM or PET Image. Will not import.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Error: Supply exactly one argument - the NM or PET Image")

    sys.exit(nm_image(sys.argv[1]))
