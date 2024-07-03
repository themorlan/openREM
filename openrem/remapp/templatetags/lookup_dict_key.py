from django import template

register = template.Library()

@register.filter(name="lookup_dict_key")
def lookup_and_format(value_dict, key):
    value = value_dict.get(key)
    if value is None:
        return '-'
    return "{:.2f}".format(value)

