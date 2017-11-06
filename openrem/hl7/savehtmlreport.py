import pprint
# TODO: Look at this unicode stuff....doesn't seem to be OK.

class MyPrettyPrinter(pprint.PrettyPrinter):
    """
    Class for printing data structures nicely
    """
    def format(self, object, context, maxlevels, level):
        """
        Override format of PrettyPrinter, take care of character encoding.
        :param object: Text
        :param context: passed to base function
        :param maxlevels: passed to base function
        :param level: passed to base function
        :return: formatted text
        """
        from hl7settings import HL7_MESSAGE_ENCODING
        if isinstance(object, unicode):
            return object.encode(HL7_MESSAGE_ENCODING), True, False
        return pprint.PrettyPrinter.format(self, object, context, maxlevels, level)


def hl7_message_to_dict(m, use_long_name=True):
    """Convert an HL7 message to a dictionary
    :param m: The HL7 message as returned by :func:`hl7apy.parser.parse_message`
    :param use_long_name: Whether or not to user the long names
                          (e.g. "patient_name" instead of "pid_5")
    :returns: A dictionary representation of the HL7 message
    """
    from hl7settings import HL7_MESSAGE_ENCODING

    if m.children:
        d = {}
        for c in m.children:
            name = str(c.name).lower().decode(HL7_MESSAGE_ENCODING)
            if use_long_name:
                name = str(c.long_name).lower().decode(HL7_MESSAGE_ENCODING) if c.long_name else name
            dictified = hl7_message_to_dict(c, use_long_name=use_long_name)
            if name in d:
                if not isinstance(d[name], list):
                    d[name] = [d[name]]
                d[name].append(dictified)
            else:
                d[name] = dictified
        return d
    else:
        return m.to_er7()


def htmlescape(text):
    """
    Escape characters for html

    Only characters that really needs to be escaped are: &, < and >

    :param text: text to escape
    :return: escaped text
    """
    return text.replace(u'&', '&amp;').replace(u'<', '&lt;').replace(u'>', '%gt;')


def save_html_report(msg):
    """
    Save a html report of the parsing result of the hl7-message to disk

    :param msg: hl7-message
    :return: nothing
    """
    from hl7mapping import HL7Mapping
    from hl7settings import HL7_HTML_FILENAME, HL7_MESSAGE_ENCODING

    hl7_dict = hl7_message_to_dict(msg, False)
    hl7_mapping = HL7Mapping(msg)
    with open(HL7_HTML_FILENAME, 'w', 0) as filepointer:
        filepointer.write(u'<html><head><meta charset="' + HL7_MESSAGE_ENCODING + '"></head><body>')
        filepointer.write(u'<h2>Message read:</h2>')
        filepointer.write(u'{0}'.format(htmlescape(msg.to_er7().encode(HL7_MESSAGE_ENCODING)).replace(u'\r', '<br />')))

        filepointer.write(u'<h2>Patient information retrieved from message:</h2>')
        filepointer.write(u'<table>')
        filepointer.write(u'test')
        filepointer.write(u'<tr><td><b>Patient name</b></td><td>{0}</td></tr>'.format(str(hl7_mapping.patient_name).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'test2')
        filepointer.write(u'<tr><td><b>Patient id</b></td><td>{0}</td></tr>'.format(str(hl7_mapping.patient_id).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Patient birthdate</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.patient_birthdate))
        filepointer.write(u'<tr><td><b>Patient sex</b></td><td>{0}</td></tr>'.format(hl7_mapping.patient_sex))
        filepointer.write(u'<tr><td><b>Patient other ids</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.patient_other_ids))
        filepointer.write(u'<tr><td><b>Patient merge id</b></td><td>{0}</td></tr>'.format(str(hl7_mapping.patient_mrg_id).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'</table>')
        filepointer.write(u'<h2>Patient Study information retrieved from message:</h2>')
        filepointer.write(u'<table>')
        filepointer.write(u'<tr><td><b>patient weight (kg)</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_patient_weight))
        filepointer.write(u'<tr><td><b>Patient size (m)</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_patient_size))
        filepointer.write(u'</table>')
        filepointer.write(u'<h2>Study information retrieved from message:</h2>')
        filepointer.write(u'<table>')
        filepointer.write(u'<tr><td><b>Study instance UID</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_instance_uid))
        filepointer.write(u'<tr><td><b>Study date</b></td><td>{0}</td></tr>'.format(hl7_mapping.study_date))
        filepointer.write(u'<tr><td><b>Study time</b></td><td>{0}</td></tr>'.format(hl7_mapping.study_time))
        filepointer.write(u'<tr><td><b>Referring physician</b></td><td>{0} ({1})</td></tr>'.format(
            str(hl7_mapping.study_referring_physician_name).encode(HL7_MESSAGE_ENCODING),
            str(hl7_mapping.study_referring_physician_id).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Study id</b></td><td>{0}</td></tr>'.format(str(hl7_mapping.study_id.encode(HL7_MESSAGE_ENCODING))))
        filepointer.write(u'<tr><td><b>Study accession number</b></td><td>{0}</td></tr>'.format(
            str(hl7_mapping.study_accession_number).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Study description</b></td><td>{0}</td></tr>'.format(
            str(hl7_mapping.study_description).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Physician of record</b></td><td>{0}</td></tr>'.format(
            str(hl7_mapping.study_physician_of_record).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Name of physician reading study</b></td><td>{0}</td></tr>'.format(
            str(hl7_mapping.study_name_of_physician_reading_study).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Performing physician name</b></td><td>{0}</td></tr>'.format(
            str(hl7_mapping.study_performing_physician).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Operator</b></td><td>{0}</td></tr>'.format(str(hl7_mapping.study_operator).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Modality</b></td><td>{0}</td></tr>'.format(hl7_mapping.study_modality))
        filepointer.write(u'<tr><td><b>Procedure code value</b></td><td>{0}</td></tr>'.format(
            str(hl7_mapping.study_procedure_code_value).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Procedure code meaning</b></td><td>{0}</td></tr>'.format(
            str(hl7_mapping.study_procedure_code_meaning).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Requested procedure code value</b></td><td>{0}</td></tr>'.format(
            str(hl7_mapping.study_requested_procedure_code_value).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'<tr><td><b>Requested procedure code meaning</b></td><td>{0}</td></tr>'.format(
            str(hl7_mapping.study_requested_procedure_code_meaning).encode(HL7_MESSAGE_ENCODING)))
        filepointer.write(u'</table>')

        filepointer.write(u'<h2>Message in dictionary format:</h2>')
        hl7_dict_string = MyPrettyPrinter().pformat(hl7_dict)
        filepointer.write(htmlescape(hl7_dict_string).replace(u'\n', '<br />\n').replace(u'  ', '&nbsp;'))
        filepointer.write(u'<br />')

        filepointer.write(u'</body></html>')
