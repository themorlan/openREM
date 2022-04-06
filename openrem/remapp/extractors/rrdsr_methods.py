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
..  module:: rdsr_methods
    :synopsis: methods used to read the radiopharmaceutical radiation
        part of rrdsr. Used by rdsr.py.

..  moduleauthor:: Ed McDonagh

"""

import logging

from remapp.models import (
    RadionuclideIdentifier,
    RadiopharmaceuticalAdministrationEventData,
    RadiopharmaceuticalAdministrationPatientCharacteristics,
    RadiopharmaceuticalLotIdentifier,
    RadiopharmaceuticalRadiationDose,
    ReagentVialIdentifier,
    PatientState,
    OrganDose,
    LanguageofContentItemandDescendants,
    IntravenousExtravasationSymptoms,
    GlomerularFiltrationRate,
    DrugProductIdentifier,
    BillingCode,
    ObserverContext,
)

from ..tools.get_values import (
    get_or_create_cid,
    test_numeric_value,
)
from ..tools.dcmdatetime import make_date_time
from .extract_common import (
    observercontext,
    person_participant,
)

logger = logging.getLogger("remapp.extractors.rdsr")


def _radiopharmaceutical_patient_state(dataset, rad_admin_pat_charac):
    patient_state: PatientState = PatientState.objects.create(
        radiopharmaceutical_administration_patient_characteristics=rad_admin_pat_charac
    )
    patient_state.patient_state = get_or_create_cid(
        dataset.ConceptCodeSequence[0].CodeValue,
        dataset.ConceptCodeSequence[0].CodeMeaning,
    )
    patient_state.save()


def _radiopharmaceutical_glomerular_filtration_rate(dataset, rad_admin_pat_charac):
    glomerular_filtration_rate: GlomerularFiltrationRate = GlomerularFiltrationRate.objects.create(
        radiopharmaceutical_administration_patient_characteristics=rad_admin_pat_charac
    )
    glomerular_filtration_rate.glomerular_filtration_rate = test_numeric_value(
        dataset.MeasuredValueSequence[0].NumericValue
    )
    if hasattr(dataset, "ContentSequence"):
        for cont in dataset.ContentSequence:
            if (
                cont.ConceptNameCodeSequence[0].CodeValue == "370129005"
                and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
            ) or (
                cont.ConceptNameCodeSequence[0].CodeValue == "G-C036"
                and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
            ):  # Measurement Method
                glomerular_filtration_rate.measurement_method = get_or_create_cid(
                    cont.ConceptCodeSequence[0].CodeValue,
                    cont.ConceptCodeSequence[0].CodeMeaning,
                )
            elif (
                cont.ConceptNameCodeSequence[0].CodeValue == "121050"
            ):  # Equivalent meaning of concept name
                glomerular_filtration_rate.equivalent_meaning_of_concept_name = (
                    get_or_create_cid(
                        cont.ConceptCodeSequence[0].CodeValue,
                        cont.ConceptCodeSequence[0].CodeMeaning,
                    )
                )

    glomerular_filtration_rate.save()


def _radiopharmaceutical_administration_patient_characteristics(
    dataset, radiopharmaceutical_dose
):
    patient_character = (
        RadiopharmaceuticalAdministrationPatientCharacteristics.objects.create(
            radiopharmaceutical_radiation_dose=radiopharmaceutical_dose
        )
    )

    def get_content_sequence(cont):
        return cont.ContentSequence if hasattr(cont, "ContentSequence") else []

    patient_character.save()
    for cont in dataset.ContentSequence:
        if cont.ConceptNameCodeSequence[0].CodeValue == "121033":  # Subject Age
            patient_character.subject_age = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "121032":  # Subject Sex
            patient_character.subject_sex = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "8302-2":  # Patient Height
            patient_character.patient_height = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "29463-7":  # Patient Weight
            patient_character.patient_weight = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "8277-6":  # Body Surface Area
            patient_character.body_surface_area = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
            for cont2 in get_content_sequence(cont):
                if (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "8278-4"
                ):  # Body Surface Area Formula
                    patient_character.body_surface_area_formula = get_or_create_cid(
                        cont2.ConceptCodeSequence[0].CodeValue,
                        cont2.ConceptCodeSequence[0].CodeMeaning,
                    )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "60621009"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
        ) or (
            cont.ConceptNameCodeSequence[0].CodeValue == "F-01860"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
        ):  # Body Mass Index
            patient_character.body_mass_index = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
            for cont2 in get_content_sequence(cont):
                if cont2.ConceptNameCodeSequence[0].CodeValue == "121420":  # Equation
                    patient_character.equation = get_or_create_cid(
                        cont2.ConceptCodeSequence[0].CodeValue,
                        cont2.ConceptCodeSequence[0].CodeMeaning,
                    )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "14749-6":  # Glucose
            patient_character.glucose = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "113550":  # Fasting Duration
            patient_character.fasting_duration = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "113551":  # Hydration Volume
            patient_character.hydration_volume = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113552"
        ):  # Recent Physical Activity
            patient_character.recent_physical_activity = cont.TextValue
        elif cont.ConceptNameCodeSequence[0].CodeValue == "2160-0":  # Serum Creatinine
            patient_character.serum_creatinine = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "109054":  # Patient state
            _radiopharmaceutical_patient_state(cont, patient_character)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "80274001"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
        ) or (
            cont.ConceptNameCodeSequence[0].CodeValue == "F-70210"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
        ):  # Glomerular Filtration Rate
            _radiopharmaceutical_glomerular_filtration_rate(cont, patient_character)

    patient_character.save()


def _intravenous_extravasation_symptoms(
    dataset, radiopharmaceutical_administration_event
):
    intravenous_extravasation_symptoms = IntravenousExtravasationSymptoms.objects.create(
        radiopharmaceutical_administration_event_data=radiopharmaceutical_administration_event
    )
    intravenous_extravasation_symptoms.intravenous_extravasation_symptoms = (
        get_or_create_cid(
            dataset.ConceptCodeSequence[0].CodeValue,
            dataset.ConceptCodeSequence[0].CodeMeaning,
        )
    )
    intravenous_extravasation_symptoms.save()


def _organ_dose(dataset, radiopharmaceutical_administration_event):
    organ_dose: OrganDose = OrganDose.objects.create(
        radiopharmaceutical_administration_event_data=radiopharmaceutical_administration_event
    )
    for cont in dataset.ContentSequence:
        if (
            cont.ConceptNameCodeSequence[0].CodeValue == "363698007"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
        ) or (
            cont.ConceptNameCodeSequence[0].CodeValue == "G-C0E3"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
        ):  # Finding Site
            organ_dose.finding_site = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "272741003"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
        ) or (
            cont.ConceptNameCodeSequence[0].CodeValue == "G-C171"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
        ):  # Laterality
            organ_dose.laterality = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "118538004"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
        ) or (
            cont.ConceptNameCodeSequence[0].CodeValue == "G-D701"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
        ):  # Mass
            organ_dose.mass = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
            for cont2 in cont.ContentSequence:
                if (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "370129005"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
                ) or (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "G-C036"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
                ):  # Measurement Method
                    organ_dose.measurement_method = cont2.TextValue
        elif cont.ConceptNameCodeSequence[0].CodeValue == "113518":  # Organ Dose
            organ_dose.organ_dose = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
            for cont2 in cont.ContentSequence:
                if (
                    cont2.ConceptNameCodeSequence[0].CodeValue
                    == "121406"  # Reference Authority
                    and cont2.ValueType == "CODE"
                ):
                    organ_dose.reference_authority_code = get_or_create_cid(
                        cont2.ConceptCodeSequence[0].CodeValue,
                        cont2.ConceptCodeSequence[0].CodeMeaning,
                    )
                elif (
                    cont2.ConceptNameCodeSequence[0].CodeValue
                    == "121406"  # Reference Authority
                    and cont2.ValueType == "TEXT"
                ):
                    organ_dose.reference_authority_text = cont2.TextValue
    organ_dose.save()


def _radiopharmaceutical_billing_code(
    dataset, radiopharmaceutical_administration_event
):
    billing_code: BillingCode = BillingCode.objects.create(
        radiopharmaceutical_administration_event_data=radiopharmaceutical_administration_event
    )
    billing_code.billing_code = get_or_create_cid(
        dataset.ConceptCodeSequence[0].CodeValue,
        dataset.ConceptCodeSequence[0].CodeMeaning,
    )
    billing_code.save()


def _drug_product_identifier(dataset, radiopharmaceutical_administration_event):
    drug_product_identifier: DrugProductIdentifier = DrugProductIdentifier.objects.create(
        radiopharmaceutical_administration_event_data=radiopharmaceutical_administration_event
    )
    drug_product_identifier.drug_product_identifier = get_or_create_cid(
        dataset.ConceptCodeSequence[0].CodeValue,
        dataset.ConceptCodeSequence[0].CodeMeaning,
    )
    drug_product_identifier.save()


def _radiopharmaceutical_lot_identifier(
    dataset, radiopharmaceutical_administration_event
):
    radiopharmaceutical_lot_identifier = RadiopharmaceuticalLotIdentifier.objects.create(
        radiopharmaceutical_administration_event_data=radiopharmaceutical_administration_event
    )
    radiopharmaceutical_lot_identifier.radiopharmaceutical_lot_identifier = (
        dataset.TextValue
    )
    radiopharmaceutical_lot_identifier.save()


def _reagent_vial_identifier(dataset, radiopharmaceutical_administration_event):
    reagent_vial_identifier: ReagentVialIdentifier = ReagentVialIdentifier.objects.create(
        radiopharmaceutical_administration_event_data=radiopharmaceutical_administration_event
    )
    reagent_vial_identifier.reagent_vial_identifier = dataset.TextValue
    reagent_vial_identifier.save()


def _radionuclide_identifier(dataset, radiopharmaceutical_administration_event):
    radionuclide_identifier: RadionuclideIdentifier = RadionuclideIdentifier.objects.create(
        radiopharmaceutical_administration_event_data=radiopharmaceutical_administration_event
    )
    radionuclide_identifier.radionuclide_identifier = dataset.TextValue
    radionuclide_identifier.save()


def _record_administered_activity(administered, is_pre_activity, rad_event):
    measured_activity = test_numeric_value(
        administered.MeasuredValueSequence[0].NumericValue
    )
    if is_pre_activity:
        rad_event.pre_administration_measured_activity = measured_activity
    else:
        rad_event.post_administration_measured_activity = measured_activity
    if hasattr(administered, "ContentSequence"):
        for cont2 in administered.ContentSequence:
            if (
                cont2.ConceptNameCodeSequence[0].CodeValue == "113540"
            ):  # Activity Measurement Device
                measurement_device = get_or_create_cid(
                    cont2.ConceptCodeSequence[0].CodeValue,
                    cont2.ConceptCodeSequence[0].CodeMeaning,
                )
                if is_pre_activity:
                    rad_event.pre_activity_measurement_device = measurement_device
                else:
                    rad_event.post_activity_measurement_device = measurement_device
            elif (
                cont2.ConceptNameCodeSequence[0].CodeValue == "121005"
            ):  # Observer Type
                observer: ObserverContext = ObserverContext.objects.create(
                    radiopharmaceutical_administration_event_data=rad_event
                )
                observer.radiopharmaceutical_administration_is_pre_observer = (
                    is_pre_activity
                )
                observercontext(administered, observer)


def _radiopharmaceutical_administration_event_data(dataset, radiopharmaceutical_dose):
    rad_event = RadiopharmaceuticalAdministrationEventData.objects.create(
        radiopharmaceutical_radiation_dose=radiopharmaceutical_dose
    )

    def get_content_sequence(cont):
        return cont.ContentSequence if hasattr(cont, "ContentSequence") else []

    rad_event.save()

    for cont in dataset.ContentSequence:
        if (
            cont.ConceptNameCodeSequence[0].CodeValue == "349358000"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
        ) or (
            cont.ConceptNameCodeSequence[0].CodeValue == "F-61FDB"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
        ):  # Radiopharmaceutical agent
            rad_event.radiopharmaceutical_agent = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
            for cont2 in get_content_sequence(cont):
                if (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "89457008"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
                ) or (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "C-10072"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
                ):  # Radionuclide
                    rad_event.radionuclide = get_or_create_cid(
                        cont2.ConceptCodeSequence[0].CodeValue,
                        cont2.ConceptCodeSequence[0].CodeMeaning,
                    )
                elif (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "304283002"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
                ) or (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "R-42806"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
                ):  # Radionuclide Half Life
                    rad_event.radionuclide_half_life = test_numeric_value(
                        cont2.MeasuredValueSequence[0].NumericValue
                    )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "123007"
        ):  # Radiopharmaceutical Specific Activity
            rad_event.radiopharmaceutical_specific_activity = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113503"
        ):  # Radiopharmaceutical Administration Event UID
            rad_event.radiopharmaceutical_administration_event_uid = cont.UID
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113505"
        ):  # Intravenous Extravasation Symptoms
            _intravenous_extravasation_symptoms(cont, rad_event)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113506"
        ):  # Estimated Extravasation Activity
            rad_event.estimated_extravasation_activity = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "123003"
        ):  # Radiopharmaceutical Start DateTime
            rad_event.radiopharmaceutical_start_datetime = make_date_time(cont.DateTime)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "123004"
        ):  # Radiopharmaceutical Stop DateTime
            rad_event.radiopharmaceutical_stop_datetime = make_date_time(cont.DateTime)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113507"
        ):  # Administered activity
            rad_event.administered_activity = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "123005"
        ):  # Radiopharmaceutical Volume
            rad_event.radiopharmaceutical_volume = test_numeric_value(
                cont.MeasuredValueSequence[0].NumericValue
            )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113508"
        ):  # Pre-Administration Measured Activity
            _record_administered_activity(cont, True, rad_event)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113509"
        ):  # Post-Administration Measured Activity
            _record_administered_activity(cont, False, rad_event)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "410675002"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
        ) or (
            cont.ConceptNameCodeSequence[0].CodeValue == "G-C340"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
        ):  # Route of administration
            rad_event.route_of_administration = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
            for cont2 in get_content_sequence(cont):
                if (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "272737002"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
                ) or (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "G-C581"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
                ):  # Site of
                    rad_event.site_of = get_or_create_cid(
                        cont2.ConceptCodeSequence[0].CodeValue,
                        cont2.ConceptCodeSequence[0].CodeMeaning,
                    )
                    for cont3 in get_content_sequence(cont2):
                        if (
                            cont3.ConceptNameCodeSequence[0].CodeValue == "272741003"
                            and cont3.ConceptNameCodeSequence[0].CodingSchemeDesignator
                            == "SCT"
                        ) or (
                            cont3.ConceptNameCodeSequence[0].CodeValue == "G-C171"
                            and cont3.ConceptNameCodeSequence[0].CodingSchemeDesignator
                            == "SRT"
                        ):  # Laterality
                            rad_event.laterality = get_or_create_cid(
                                cont3.ConceptCodeSequence[0].CodeValue,
                                cont3.ConceptCodeSequence[0].CodeMeaning,
                            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "111529":  # Brand Name
            rad_event.brand_name = cont.TextValue
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113511"
        ):  # Radiopharmaceutical Dispense Unit Identifier
            rad_event.radiopharmaceutical_dispense_unit_identifier = cont.TextValue
            for cont2 in get_content_sequence(cont):
                if (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "113512"
                ):  # Radiopharmaceutical Lot Identifier
                    _radiopharmaceutical_lot_identifier(cont2, rad_event)
                elif (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "113513"
                ):  # Reagent Vial Identifier
                    _reagent_vial_identifier(cont2, rad_event)
                elif (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "113514"
                ):  # Radionuclide Identifier
                    _radionuclide_identifier(cont2, rad_event)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113516"
        ):  # Prescription Identifier
            rad_event.prescription_identifier = cont.TextValue
        elif cont.ConceptNameCodeSequence[0].CodeValue == "121106":  # Comment
            rad_event.comment = cont.TextValue
        elif cont.ConceptNameCodeSequence[0].CodeValue == "113517":  # Organ dose
            _organ_dose(cont, rad_event)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "220001"
        ):  # Effective dose information
            for cont2 in get_content_sequence(cont):
                if (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "113839"
                ):  # Effective dose in mSv
                    rad_event.effective_dose = test_numeric_value(
                        cont2.MeasuredValueSequence[0].NumericValue
                    )
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113870"
        ):  # Person Participant
            person_participant(
                cont, "radiopharmaceutical_administration_event_data", rad_event, logger
            )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "121147":  # Billing Code(s)
            _radiopharmaceutical_billing_code(cont, rad_event)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113510"
        ):  # Drug Product Identifier
            _drug_product_identifier(cont, rad_event)

    rad_event.save()


def _language_of_content(dataset, radiopharmaceutical_dose):
    language: LanguageofContentItemandDescendants = (
        LanguageofContentItemandDescendants.objects.create(
            radiopharmaceutical_radiation_dose=radiopharmaceutical_dose
        )
    )
    language.language_of_contentitem_and_descendants = get_or_create_cid(
        dataset.ConceptCodeSequence[0].CodeValue,
        dataset.ConceptCodeSequence[0].CodeMeaning,
    )
    subcont = dataset.ContentSequence if hasattr(dataset, "ContentSequence") else []
    for cont in subcont:
        if cont.ConceptNameCodeSequence[0].CodeValue == "121046":  # Country of Language
            language.country_of_language = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
    language.save()


def _radiopharmaceuticalradiationdose(dataset, g):
    rdose: RadiopharmaceuticalRadiationDose = (
        RadiopharmaceuticalRadiationDose.objects.create(
            general_study_module_attributes=g
        )
    )
    rdose.general_study_module_attributes.modality_type = "NM"
    rdose.general_study_module_attributes.save()
    rdose.save()

    for cont in dataset.ContentSequence:
        if (
            cont.ConceptNameCodeSequence[0].CodeValue == "363589002"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
        ) or (
            cont.ConceptNameCodeSequence[0].CodeValue == "G-C2D0"
            and cont.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
        ):  # Associated Procedure
            rdose.associated_procedure = get_or_create_cid(
                cont.ConceptCodeSequence[0].CodeValue,
                cont.ConceptCodeSequence[0].CodeMeaning,
            )
            for cont2 in cont.ContentSequence:
                if (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "363703001"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SCT"
                ) or (
                    cont2.ConceptNameCodeSequence[0].CodeValue == "G-C0E8"
                    and cont2.ConceptNameCodeSequence[0].CodingSchemeDesignator == "SRT"
                ):  # Has Intent
                    rdose.has_intent = get_or_create_cid(
                        cont2.ConceptCodeSequence[0].CodeValue,
                        cont2.ConceptCodeSequence[0].CodeMeaning,
                    )
        elif cont.ConceptNameCodeSequence[0].CodeValue == "121106":  # Comment
            rdose.comment = cont.TextValue
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "121049"
        ):  # Language of Content Item and Descendants
            _language_of_content(cont, rdose)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "113502"
        ):  # Radiopharmaceutical Administration
            _radiopharmaceutical_administration_event_data(cont, rdose)
        elif (
            cont.ConceptNameCodeSequence[0].CodeValue == "121118"
        ):  # Patient Characteristics
            _radiopharmaceutical_administration_patient_characteristics(cont, rdose)

    rdose.save()
