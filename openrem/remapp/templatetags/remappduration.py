# This Python file uses the following encoding: utf-8

from django import template

register = template.Library()


def naturalduration(seconds):
    if not seconds:
        return ""

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    hplural = ""
    mplural = ""
    splural = ""

    if not h == 1:
        hplural = "s"
    if not m == 1:
        mplural = "s"
    if not s == 1:
        splural = "s"

    if h:
        duration = f"{h:.0f} hour{hplural} and {m:.0f} minute{mplural}"
    elif m:
        duration = f"{m:.0f} minute{mplural} and {s:.0f} second{splural}"
    else:
        duration = f"{s:.1f} second{splural}"

    return duration


register.filter("naturalduration", naturalduration)
