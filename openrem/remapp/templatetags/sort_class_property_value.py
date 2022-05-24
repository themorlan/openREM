"""
..  module:: sort_on_class_item
    :synoposis: Returns a list of class items sorted into ascending order

..  moduleauthor:: David Platten
"""

from django import template

register = template.Library()


@register.filter(name="sort_class_items")
def sort_class_items(value, arg):
    return sorted([getattr(item, arg) for item in value if getattr(item, arg)])
