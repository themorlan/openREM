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
..  module:: get_values.
    :synopsis: Module to return values from DICOM elements using pydicom.

..  moduleauthor:: Ed McDonagh

"""
from decimal import Decimal  # pylint: disable=redefined-builtin
from pydicom import charset
from pydicom.charset import default_encoding
from django.utils.encoding import smart_text
import logging

logger = logging.getLogger(__name__)


def get_value_kw(tag, dataset):
    """Get DICOM value by keyword reference.

    :param tag:         DICOM keyword, no spaces or plural as per dictionary.
    :type tag:          str.
    :param dataset:     The DICOM dataset containing the tag.
    :type dataset:      dataset
    :returns:           str. -- value
    """
    if tag in dataset:
        val = getattr(dataset, tag)
        if val != "":
            return val
    return None


def get_value_num(tag, dataset):
    """Get DICOM value by tag group and element number.

    Always use get_value_kw by preference for readability. This module can
    be required when reading private elements.

    :param tag:     DICOM group and element number as a single hexadecimal number (prefix 0x).
    :type tag:          hex
    :param dataset:     The DICOM dataset containing the tag.
    :type dataset:      dataset
    :returns:           str. -- value
    """
    if tag in dataset:
        val = dataset[tag].value
        try:
            val = val.decode(default_encoding)
        except AttributeError:
            pass
        if val != "":
            return val
    return None


def get_seq_code_value(sequence, dataset):
    """From a DICOM sequence, get the code value.

    :param sequence:    DICOM sequence name.
    :type sequence:     DICOM keyword, no spaces or plural as per dictionary.
    :param dataset:     The DICOM dataset containing the sequence.
    :type dataset:      DICOM dataset
    :returns:           int. -- code value
    """
    if sequence in dataset:
        seq = getattr(dataset, sequence)
        if seq and hasattr(seq[0], "CodeValue"):
            return seq[0].CodeValue
    return None


def get_seq_code_meaning(sequence, dataset):
    """From a DICOM sequence, get the code meaning.

    :param sequence:    DICOM sequence name.
    :type sequence:     DICOM keyword, no spaces or plural as per dictionary.
    :param dataset:     The DICOM dataset containing the sequence.
    :type dataset:      DICOM dataset
    :returns:           str. -- code meaning
    """
    if sequence in dataset:
        seq = getattr(dataset, sequence)
        if seq and hasattr(seq[0], "CodeMeaning"):
            meaning = seq[0].CodeMeaning
            if meaning != "":
                return meaning
    return None


def get_or_create_cid(codevalue, codemeaning):
    """Create a code_value code_meaning pair entry in the ContextID
    table if it doesn't already exist.

    :param codevalue:   Code value as defined in the DICOM standard part 16
    :type codevalue:    int.
    :param codemeaning: Code meaning as defined in the DICOM standard part 16
    :type codevalue:    int.
    :returns:           ContextID entry for code value passed
    """
    from remapp.models import ContextID

    if codevalue:
        if not ContextID.objects.all().filter(code_value=codevalue).exists():
            cid = ContextID(code_value=codevalue, code_meaning=codemeaning)
            cid.save()
        code = ContextID.objects.filter(code_value__exact=codevalue)
        if code.count() > 1:
            logger.warning(
                f"Duplicate entry in the ContextID table: {codevalue}/{codemeaning}, import continuing"
            )
        return code[0]
    return None


def return_for_export(model, field):
    """
    Prevent errors due to missing data in models
    :param model: database table
    :param field: database field
    :return: value or None
    """
    import datetime
    from django.core.exceptions import ObjectDoesNotExist

    try:
        val = getattr(model, field)
        if val:
            if isinstance(val, datetime.date):
                return val
            val = str(val)
        return val
    except ObjectDoesNotExist:
        return None


def test_numeric_value(string_number):
    """
    Tests if string can be converted to a float. If it can, return it
    :param string_number: string to test if is a number
    :return: string if number, nothing otherwise
    """
    try:
        float(string_number)
        return string_number
    except (ValueError, TypeError):
        return None


def to_decimal_value(string_number):
    """
    Tests if string can be converted to a float. If yes returns it
    as decimal.
    :param string_number: string to test if a number
    :return: Decimal if convertable, None otherwise
    """
    if test_numeric_value(string_number) is None:
        return None
    else:
        return Decimal(string_number)


def list_to_string(dicom_value):
    """
    Turn multivalue names into a single string for correct encoding and pretty reproduction
    :param dicom_value: returned DICOM value, usually a name field. Might be single (string) or multivalue (list)
    :return: string of name(s)
    """
    from pydicom.multival import MultiValue

    if dicom_value and isinstance(dicom_value, MultiValue):
        name_str = ""
        for name in dicom_value:
            if name.name_suffix:
                name_str += name.formatted(
                    "%(family_name)s^%(given_name)s^%(middle_name)s^%(name_prefix)s^%(name_suffix)s"
                )
            elif name.name_prefix:
                name_str += name.formatted(
                    "%(family_name)s^%(given_name)s^%(middle_name)s^%(name_prefix)s"
                )
            elif name.middle_name:
                name_str += name.formatted(
                    "%(family_name)s^%(given_name)s^%(middle_name)s"
                )
            elif name.given_name:
                name_str += name.formatted("%(family_name)s^%(given_name)s")
            elif name.family_name:
                name_str += name.formatted("%(family_name)s")
            name_str += " | "
        name_str = name_str[:-3]
        return name_str
    return dicom_value


def get_keys_by_value(dict_of_elements, value_to_find):
    """
    Get a list of keys from a dictionary which have the given value
    :param dict_of_elements: a dictionary of elements
    :param value_to_find: the value to look for in the dictionary
    :return: list of key names matching the given value
    """
    list_of_keys = list()
    list_of_items = list(dict_of_elements.items())
    for item in list_of_items:
        if item[1] == value_to_find:
            list_of_keys.append(item[0])
    return list_of_keys
