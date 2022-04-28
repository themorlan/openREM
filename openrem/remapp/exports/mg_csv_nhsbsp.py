#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2014  The Royal Marsden NHS Foundation Trust and Jonathan Cole
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
..  module:: mg_csv_nhsbsp.
    :synopsis: Module to export mammography data to CSV files in the NHSBSP format.

..  moduleauthor:: Ed McDonagh and Jonathan Cole

"""
import datetime
import logging

from django.core.exceptions import ObjectDoesNotExist
from openrem.remapp.tools.background import get_or_generate_task_uuid

logger = logging.getLogger(__name__)


def mg_csv_nhsbsp(filterdict, user=None):
    """Export filtered mammography database data to a NHSBSP formatted single-sheet CSV file.

    :param filterdict: Dictionary of query parameters from the mammo filtered page URL.
    :type filterdict: dict
    :returns: None - file is saved to disk and location is stored in database

    """

    from remapp.models import GeneralStudyModuleAttr
    from ..interface.mod_filters import MGSummaryListFilter
    from .export_common import (
        create_csv,
        write_export,
        abort_if_zero_studies,
        create_export_task,
    )

    datestamp = datetime.datetime.now()
    task_id = get_or_generate_task_uuid()
    tsk = create_export_task(
        task_id=task_id,
        modality="MG",
        export_type="NHSBSP CSV export",
        date_stamp=datestamp,
        pid=False,
        user=user,
        filters_dict=filterdict,
    )

    tmpfile, writer = create_csv(tsk)
    if not tmpfile:
        exit()

    # Resetting the ordering key to avoid duplicates
    if isinstance(filterdict, dict):
        if (
            "o" in filterdict
            and filterdict["o"] == "-projectionxrayradiationdose__accumxraydose__"
            "accummammographyxraydose__accumulated_average_glandular_dose"
        ):
            logger.info("Replacing AGD ordering with study date to avoid duplication")
            filterdict["o"] = "-study_date"

    # Get the data!
    studies_qs = MGSummaryListFilter(
        filterdict,
        queryset=GeneralStudyModuleAttr.objects.filter(modality_type__exact="MG"),
    )
    s = studies_qs.qs

    tsk.progress = "Required study filter complete."
    tsk.save()

    tsk.num_records = s.count()
    if abort_if_zero_studies(tsk.num_records, tsk):
        return

    writer.writerow(
        [
            "Survey number",
            "Patient number",
            "View code",
            "kV",
            "Anode",
            "Filter",
            "Thickness",
            "mAs",
            "large cassette used",
            "auto/man",
            "Auto mode",
            "Density setting",
            "Age",
            "Comment",
            "AEC density mode",
        ]
    )

    for i, study in enumerate(s):
        tsk.progress = f"{i + 1} of {tsk.num_records}"
        tsk.save()
        unique_views = set()

        try:
            exposures = (
                study.projectionxrayradiationdose_set.get().irradeventxraydata_set.all()
            )
            for exp in exposures:
                try:
                    laterality = exp.laterality.code_meaning
                except AttributeError:
                    exp.nccpm_view = None
                    continue
                exp.nccpm_view = laterality[:1]

                views = {
                    "cranio-caudal": "CC",
                    "medio-lateral oblique": "OB",
                    "medio-lateral": "ML",
                    "latero-medial": "LM",
                    "latero-medial oblique": "LMO",
                    "caudo-cranial (from below)": "FB",
                    "superolateral to inferomedial oblique": "SIO",
                    "inferomedial to superolateral oblique": "ISO",
                    "cranio-caudal exaggerated laterally": "XCCL",
                    "cranio-caudal exaggerated medially": "XCCM",
                }  # See http://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_4014.html
                try:
                    if exp.image_view.code_meaning in views:
                        exp.nccpm_view += views[exp.image_view.code_meaning]
                    else:
                        exp.nccpm_view += exp.image_view.code_meaning
                except AttributeError:
                    exp.nccpm_view = None
                    continue  # Avoid exporting exposures with no image_view recorded

                if "specimen" in exp.image_view.code_meaning:
                    logger.debug(
                        "Exposure excluded due to image_view containing specimen: {0}".format(
                            exp.image_view.code_meaning
                        )
                    )
                    exp.nccpm_view = None
                    continue  # No point including these in the export

                bad_acq_words = [
                    "scout",
                    "postclip",
                    "prefire",
                    "biopsy",
                    "postfire",
                    "stereo",
                    "specimen",
                    "artefact",
                ]
                try:
                    if any(
                        word in exp.acquisition_protocol.lower()
                        for word in bad_acq_words
                    ):
                        logger.debug(
                            "Exposure excluded due to biopsy word in {0}".format(
                                exp.acquisition_protocol.lower()
                            )
                        )
                        exp.nccpm_view = None
                        continue  # Avoid exporting biopsy related exposures
                except AttributeError:
                    logger.debug("No protocol information. Carrying on.")

                try:
                    target = (
                        exp.irradeventxraysourcedata_set.get().anode_target_material.code_meaning
                    )
                except AttributeError:
                    logger.debug(
                        "Exposure excluded due to attribute error on target information"
                    )
                    exp.nccpm_view = None
                    continue  # Avoid exporting exposures with no anode material recorded
                if "TUNGSTEN" in target.upper():
                    target = "W"
                elif "MOLY" in target.upper():
                    target = "Mo"
                elif "RHOD" in target.upper():
                    target = "Rh"

                try:
                    filter_mat = (
                        exp.irradeventxraysourcedata_set.get()
                        .xrayfilters_set.get()
                        .xray_filter_material.code_meaning
                    )
                except AttributeError:
                    logger.debug(
                        "Exposure excluded due to attribute error on filter material"
                    )
                    exp.nccpm_view = None
                    continue  # Avoid exporting exposures with no filter material recorded
                if "ALUM" in filter_mat.upper():
                    filter_mat = "Al"
                elif "MOLY" in filter_mat.upper():
                    filter_mat = "Mo"
                elif "RHOD" in filter_mat.upper():
                    filter_mat = "Rh"
                elif "SILV" in filter_mat.upper():
                    filter_mat = "Ag"

                if exp.nccpm_view:
                    if exp.nccpm_view not in unique_views:
                        unique_views.add(exp.nccpm_view)
                    else:
                        for x in range(20):
                            if exp.nccpm_view + str(x + 2) not in unique_views:
                                exp.nccpm_view += str(x + 2)
                                unique_views.add(exp.nccpm_view)
                                break
                else:
                    logger.debug("Exposure excluded due to no generated nncp_view")
                    continue  # Avoid exporting exposures with no view code

                automan_short = None
                try:
                    automan = (
                        exp.irradeventxraysourcedata_set.get().exposure_control_mode
                    )
                    if "AUTO" in automan.upper():
                        automan_short = "AUTO"
                    elif "MAN" in automan.upper():
                        automan_short = "MANUAL"
                except AttributeError:
                    automan = None

                writer.writerow(
                    [
                        "1",
                        i + 1,
                        exp.nccpm_view,
                        exp.irradeventxraysourcedata_set.get().kvp_set.get().kvp,
                        target,
                        filter_mat,
                        exp.irradeventxraymechanicaldata_set.get().compression_thickness,
                        exp.irradeventxraysourcedata_set.get()
                        .exposure_set.get()
                        .exposure
                        / 1000,
                        "",  # not applicable to FFDM
                        automan_short,
                        automan,
                        "",  # no consistent behaviour for recording density setting on FFDM units
                        exp.projection_xray_radiation_dose.general_study_module_attributes.patientstudymoduleattr_set.get().patient_age_decimal,
                        "",  # not in DICOM headers
                        "",  # no consistent behaviour for recording density mode on FFDM units
                    ]
                )
        except ObjectDoesNotExist:
            error_message = (
                "DoesNotExist error whilst exporting study {0} of {1},  study UID {2}, accession number"
                " {3} - maybe database entry was deleted as part of importing later version of same"
                " study?".format(
                    i + 1,
                    tsk.num_records,
                    study.study_instance_uid,
                    study.accession_number,
                )
            )
            logger.error(error_message)
            writer.writerow([error_message])

    tsk.progress = "All study data written."
    tsk.save()

    csvfilename = "mg_nhsbsp_{0}.csv".format(datestamp.strftime("%Y%m%d-%H%M%S%f"))
    write_export(tsk, csvfilename, tmpfile, datestamp)
