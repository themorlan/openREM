"""
..  module:: proper_paginate
    :synoposis: Allows sensible list of pages to click on when paginating over large numbers of pages

..  moduleauthor:: see https://gist.github.com/sumitlni/4f308e5999d2d4d8cb284fea7bf0309c
"""

from django import template

register = template.Library()


@register.filter(name="proper_paginate")
def proper_paginate(paginator, current_page, neighbors=5):
    """Paginate over sensible number of page links

    :param paginator: paginator pages
    :param current_page: current page number
    :param neighbors: how many page links to display
    :return: range of page links to display
    """
    print(paginator)
    if paginator.num_pages > 2 * neighbors:
        start_index = max(1, current_page - neighbors)
        end_index = min(paginator.num_pages, current_page + neighbors)
        if end_index < start_index + 2 * neighbors:
            end_index = start_index + 2 * neighbors
        elif start_index > end_index - 2 * neighbors:
            start_index = end_index - 2 * neighbors
        if start_index < 1:
            end_index -= start_index
            start_index = 1
        elif end_index > paginator.num_pages:
            start_index -= end_index - paginator.num_pages
            end_index = paginator.num_pages
        page_list = [f for f in range(start_index, end_index + 1)]
        return page_list[: (2 * neighbors + 1)]
    return paginator.page_range
