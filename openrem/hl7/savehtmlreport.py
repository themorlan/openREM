import pprint


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
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '%gt;')


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
    with open(HL7_HTML_FILENAME, 'w') as filepointer:
        filepointer.write('<html><head><meta charset="' + HL7_MESSAGE_ENCODING + '"></head><body>')
        filepointer.write('<h2>Message read:</h2>')
        filepointer.write('{0}'.format(htmlescape(msg.to_er7().encode(HL7_MESSAGE_ENCODING)).replace('\r', '<br />')))

        filepointer.write('<h2>Patient information retrieved from message:</h2>')
        filepointer.write('<table>')
        filepointer.write('<tr><td><b>Patient name</b></td><td>{0}</td></tr>'.format(hl7_mapping.patient_name))
        filepointer.write('<tr><td><b>Patient id</b></td><td>{0}</td></tr>'.format(hl7_mapping.patient_id))
        filepointer.write('<tr><td><b>Patient birthdate</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.patient_birthdate))
        filepointer.write('<tr><td><b>Patient sex</b></td><td>{0}</td></tr>'.format(hl7_mapping.patient_sex))
        filepointer.write('<tr><td><b>Patient other ids</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.patient_other_ids))
        filepointer.write('<tr><td><b>Patient merge id</b></td><td>{0}</td></tr>'.format(hl7_mapping.patient_mrg_id))
        filepointer.write('</table>')
        filepointer.write('<h2>Patient Study information retrieved from message:</h2>')
        filepointer.write('<table>')
        filepointer.write('<tr><td><b>patient weight (kg)</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_patient_weight))
        filepointer.write('<tr><td><b>Patient size (m)</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_patient_size))
        filepointer.write('</table>')
        filepointer.write('<h2>Study information retrieved from message:</h2>')
        filepointer.write('<table>')
        filepointer.write('<tr><td><b>Study instance UID</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_instance_uid))
        filepointer.write('<tr><td><b>Study date</b></td><td>{0}</td></tr>'.format(hl7_mapping.study_date))
        filepointer.write('<tr><td><b>Study time</b></td><td>{0}</td></tr>'.format(hl7_mapping.study_time))
        filepointer.write('<tr><td><b>Referring physician</b></td><td>{0} ({1})</td></tr>'.format(
            hl7_mapping.study_referring_physician_name, hl7_mapping.study_referring_physician_id))
        filepointer.write('<tr><td><b>Study id</b></td><td>{0}</td></tr>'.format(hl7_mapping.study_id))
        filepointer.write('<tr><td><b>Study accession number</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_accession_number))
        filepointer.write('<tr><td><b>Study description</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_description))
        filepointer.write('<tr><td><b>Physician of record</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_physician_of_record))
        filepointer.write('<tr><td><b>Name of physician reading study</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_name_of_physician_reading_study))
        filepointer.write('<tr><td><b>Performing physician name</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_performing_physician))
        filepointer.write('<tr><td><b>Operator</b></td><td>{0}</td></tr>'.format(hl7_mapping.study_operator))
        filepointer.write('<tr><td><b>Modality</b></td><td>{0}</td></tr>'.format(hl7_mapping.study_modality))
        filepointer.write('<tr><td><b>Procedure code value</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_procedure_code_value))
        filepointer.write('<tr><td><b>Procedure code meaning</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_procedure_code_meaning))
        filepointer.write('<tr><td><b>Requested procedure code value</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_requested_procedure_code_value))
        filepointer.write('<tr><td><b>Requested procedure code meaning</b></td><td>{0}</td></tr>'.format(
            hl7_mapping.study_requested_procedure_code_meaning))
        filepointer.write('</table>')

        filepointer.write('<h2>Message in dictionary format:</h2>')
        hl7_dict_string = MyPrettyPrinter().pformat(hl7_dict)
        filepointer.write(htmlescape(hl7_dict_string).replace('\n', '<br />\n').replace('  ', '&nbsp;'))
        filepointer.write('<br />')

        filepointer.write('</body></html>')
