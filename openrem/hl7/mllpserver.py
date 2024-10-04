from hl7apy.mllp import MLLPServer, AbstractHandler, AbstractErrorHandler, UnsupportedMessageType
from hl7updater import parse_hl7, apply_hl7
from hl7apy.core import Message
from hl7settings import *
import logging

logger = logging.getLogger('hl7.mllpserver')  # Explicitly named so that it is still handled when using __main__


def init_response(received_msg):
    """
    Return pre-filled response message based on received message

    :param received_msg: the message received
    :return: initial pre-filled response message
    """
    from hl7apy import exceptions
    from random import randint
    try:
        if received_msg:
            response = Message('ACK', version=received_msg.MSH.MSH_12.to_er7())
            response.MSH.MSH_3 = received_msg.MSH.MSH_5
            response.MSH.MSH_4 = received_msg.MSH.MSH_6
            response.MSH.MSH_5 = received_msg.MSH.MSH_3
            response.MSH.MSH_6 = received_msg.MSH.MSH_4
            # response.MSH.MSH_7 is filled by creation
            # response.MSH.MSH_8 remains empty
            response.MSH.MSH_9 = u'ACK^{0}'.format(received_msg.MSH.MSH_9.MSH_9_2.to_er7())
            response.MSH.MSH_10 = u'ret_{0}'.format(received_msg.MSH.MSH_10.to_er7())
            response.MSH.MSH_11 = u'P'  # production
            response.add_segment(u'MSA')
        else:
            response = Message('ACK', version=HL7_DEFAULT_VERSION)
            response.MSH.MSH_3 = u'OPENREM'
            response.MSH.MSH_4 = u'OPENREM'
            response.MSH.MSH_5 = u''
            response.MSH.MSH_6 = u''
            # response.MSH.MSH_7 is filled by creation
            # response.MSH.MSH_8 remains empty
            response.MSH.MSH_9 = u'ACK^A01'
            response.MSH.MSH_10 = u'OR_%06d' % (randint(1, 100000),)
            response.MSH.MSH_11 = u'P'  # production
    except exceptions.HL7apyException as e:
        logger.fatal('Error creating HL7 response-message: {0}'.format(e))
        return None

    return response


class Hl7Handler(AbstractHandler):
    """
    HL7-message handler

    The messages handled are defined in "handlers" (part of MLLPServer). The function reply takes care of parsing and
    creating a response message.
    """
    def reply(self):
        """
        Parse receiving message and create er7 response message (wrapped with mllp encoding characters).

        :return: er7 response message wrapped with mllp encoding characters
        """
        logger.debug('Handling incoming HL7-message.')
        logger.debug(u'Hl7-message: \r\n{0}'.format(self.incoming_message.replace('\r', '\r\n')))
        result, msg = parse_hl7(self.incoming_message)
        # Fill MSH from original message
        if result >= 0:
            result = apply_hl7(msg)
            if result >= 0:
                # message correctly read and applied, let's send an ack
                response = init_response(msg)
                response.MSA = u'MSA|AA|{0}'.format(msg.MSH.MSH_10.to_er7())
            elif HL7_RESPONSE_ALWAYS_ACCEPT:
                # we can't read hl7 message, so we can't response correctly as we can't fill MSH correctly,
                # but let's try to send an ack
                response = init_response(None)
                response.MSA = u'MSA|AA'
            else:
                # message correctly read, but could not be applied, let's send a nack
                response = init_response(msg)
                response.MSA = u'MSA|AE|{0}'.format(msg.MSH.MSH_10.to_er7())
        else:
            # we can't read hl7 message, so we can't response correctly as we can't fill MSH correctly,
            # But let's try to send a (n)ack
            # Hope we can read MSH_10 (or should we just leave it empty)
            # Anyway, we probably end up in the ErrorHandler and leave it empty over there.
            response = init_response(None)
            if HL7_RESPONSE_ALWAYS_ACCEPT:
                response.MSA = u'MSA|AA|{0}'.format(msg.MSH.MSH_10.to_er7())
            else:
                response.MSA = u'MSA|AE|{0}'.format(msg.MSH.MSH_10.to_er7())
        return response.to_mllp()


class ErrorHandler(AbstractErrorHandler):
    """
    HL7 error handler

    The function reply creates a response message in case of an (parsing) error.
    """

    def reply(self):
        """
        Create er7 response message wrapped with mllp encoding characters in case of an error

        :return: er7 response message wrapped with mllp encoding characters
        """
        if isinstance(self.exc, UnsupportedMessageType):
            logger.info('Incoming unsupported HL7 message.')
            result, msg = parse_hl7(self.incoming_message)
            if result >= 0:
                response = init_response(self.incoming_message)
                response.MSA = 'MSA|{0}|{1}'.format(HL7_RESPONSE_UNSUPPORTED_MESSAGE, msg.MSH.MSH_10)
            elif HL7_RESPONSE_ALWAYS_ACCEPT:
                # we can't read hl7 message, so we can't response correctly as we can't fill MSH correctly,
                # but let's try to send an ack
                response = init_response(None)
                response.MSA = 'MSA|AA'
            else:
                # we can't read hl7 message, so we can't response correctly as we can't fill MSH correctly,
                # but let's try to send a nack
                response = init_response(None)
                response.MSA = 'MSA|AE'
        else:
            logger.error('exception while processing incoming HL7-message.')
            logger.error('Hl7-message: \r\n{0}'.format(self.incoming_message.replace('\r', '\r\n')))
            # we can't read hl7 message, so we can't response correctly as we can't fill MSH correctly,
            # but let's try to return a response
            response = init_response(None)
            if HL7_RESPONSE_ALWAYS_ACCEPT:
                response.MSA = 'MSA|AA'
            else:
                response.MSA = 'MSA|AE'
        return response


# Define the supported HL7-messages. Messages are all handled by the same HL7 handler.
handlers = {
    'ADT^A01': (Hl7Handler,),
    'ADT^A01^ADT_A01': (Hl7Handler,),
    'ADT^A02': (Hl7Handler,),
    'ADT^A02^ADT_A02': (Hl7Handler,),
    'ADT^A03': (Hl7Handler,),
    'ADT^A03^ADT_A03': (Hl7Handler,),
    'ADT^A04': (Hl7Handler,),
    'ADT^A04^ADT_A04': (Hl7Handler,),
    'ADT^A08': (Hl7Handler,),
    'ADT^A08^ADT_A08': (Hl7Handler,),
    'ADT^A40': (Hl7Handler,),
    'ADT^A40^ADT_A40': (Hl7Handler,),
    'ORU^R01': (Hl7Handler,),
    'ORU^R01^ORU_R01': (Hl7Handler,),
    'ORM^O01': (Hl7Handler,),
    'ORM^O01^ORM_O01': (Hl7Handler,),
    'OMG^O19': (Hl7Handler,),
    'OMG^O19^OMG_O19': (Hl7Handler,),
    'OMI^O23': (Hl7Handler,),
    'OMI^O23^OMI_O23': (Hl7Handler,),
    'ERR': (ErrorHandler,),
}

# Create MLLP-server.
server = MLLPServer(HL7_SERVER_HOSTNAME, HL7_SERVER_PORT, handlers, timeout=None, char_encoding=HL7_MESSAGE_ENCODING,
                    keep_connection_open=HL7_KEEP_CONNECTION_ALIVE)


def start_server():
    """
    Start the MLLP-server

    :return: Nothing
    """
    logger.info('starting mllpserver on {0}:{1}'.format(HL7_SERVER_HOSTNAME, HL7_SERVER_PORT))
    server.serve_forever()


def stop_server():
    """
    Stop the MLLP-server

    :return: Nothing
    """
    logger.info('stopping mllpserver')
    server.shutdown()
    server.server_close()


if __name__ == '__main__':
    # This part is necessary if django is not started via "manage.py"
    # Also add basepath to python-path. It is excepted that this file is in ....\openrem\hl7\
    import django
    import os
    import sys
    import atexit
    basepath = os.path.dirname(__file__)
    basepath = os.path.join(basepath, '..')
    sys.path.append(basepath)
    os.environ['DJANGO_SETTINGS_MODULE'] = 'openremproject.settings'
    django.setup()

    atexit.register(stop_server)
    start_server()
