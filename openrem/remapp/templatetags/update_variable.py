# This Python file uses the following encoding: utf-8

"""
..  module:: update_variable
    :synopsis: template assignment tag to change the value of a template variable
"""

from django import template
register = template.Library()


@register.simple_tag()
def update_variable(value):
    """ Template assignment tag to return a supplied value to enable template variable values to be updated

    :param value: the value to return
    :return: the received value parameter
    """
    return value
