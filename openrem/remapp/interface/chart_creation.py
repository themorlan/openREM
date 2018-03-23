import pandas as pd
import numpy as np

# todo define more colors to pick from
colors = {
          'orange': '#F79727',
          'darkBlue': '#006991',
          'lightBlue': '#40C3E5',
          'red': '#C1493C',
          'green': '#008040',
          'purple': '#800080',
          'yellow': '#FFFF00',
          'pink': '#FF00FF',
          'brown': '#FF4000',
         }


def get_outliers_boxplot(group, y_param, whisker_lower, whisker_upper):
    return group[(group[y_param] > whisker_upper.loc[group.name][y_param])
                 | (group[y_param] < whisker_lower.loc[group.name][y_param])]


def create_scatter_plot(request, data, id_parameter, grouping_parameter, grouping_name, x_axis_parameter, x_axis_label,
                        y_axis_parameter, y_axis_label):
    """
    :param request: original http-request (is used for clicking to specific study/studies)
    :param data: data used for calculations as pandas dataframe
    :param id_parameter: name of column indicating study row id
    :param grouping_parameter: column in data that is used for grouping
    :param grouping_name: column in data that is used for grouping name (for legend)
    :param y_axis_parameter: column in data that should be used to plot on for x-axis
    :param x_axis_label: label used for x-axis
    :param y_axis_parameter: column in data that should be used to plot on y-axis
    :param y_axis_label: label used for y-axis
    :return: scatter plot (figure, script, descriptive statistics)
    """
    from bokeh.plotting import figure
    from bokeh.embed import components
    from bokeh.resources import INLINE
    from bokeh.models import HoverTool, TapTool, OpenURL, ColumnDataSource
    from re import sub, MULTILINE

    if not isinstance(data, pd.DataFrame):
        raise ValueError('data is not a pandas DataFrame.')

    grouped_df = data.groupby(grouping_parameter)
    description = grouped_df[y_axis_parameter].describe()

    # draw the figure
    hover = HoverTool()
    chart = figure(tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'save', 'reset', 'tap'])

    # data
    color_nr = 0
    for group in grouped_df:
        source = ColumnDataSource(data=dict(
            x=group[1][x_axis_parameter],
            y=group[1][y_axis_parameter],
            legend= [group[0]] * len(group[1][x_axis_parameter]),
            id=group[1][id_parameter]
            ))
        chart.scatter('x', 'y', marker="diamond", size=15, line_color="#000000",
                      fill_color=colors.values()[color_nr], alpha=0.7, legend='legend', source=source)
        color_nr = color_nr + 1
        if color_nr == len(colors):
            color_nr = 0
    chart.select(dict(type=HoverTool)).tooltips = [('{0}'.format(grouping_name), '@legend'),
                                                   ('{0}'.format(x_axis_label), '@x'),
                                                   ('{0}'.format(y_axis_label), '@y')]
    url = request.path.rsplit('/', 2)[0] + '/@id'
    taptool = chart.select(type=TapTool)
    taptool.callback = OpenURL(url=url)

    # axis
    chart.xaxis.major_label_text_font_size = '12pt'
    chart.xaxis.axis_label = x_axis_label
    chart.yaxis.major_label_text_font_size = '12pt'
    chart.yaxis.axis_label = y_axis_label

    chart_script, chart_figure = components(chart, INLINE)

    html_description = description.to_html(classes='table table-striped table-bordered small')
    # remove second row with only grouping parameter
    html_description = sub(u'.*?<tr>\n.*?<th>{0}[\s\S]+?</tr>'.format(grouping_parameter), '',
                            html_description, MULTILINE)

    return chart_script, chart_figure, html_description


def create_histogram(request, data, grouping_parameter, grouping_name, x_axis_parameter, x_axis_label, nr_of_bins=50):
    """

    :param data:
    :param grouping_parameter:
    :param x_axis_label:
    :param y_axis_parameter:
    :param y_axis_label:
    :param nr_of_bins:
    :return:
    """
    from bokeh.plotting import figure
    from bokeh.embed import components
    from bokeh.resources import INLINE
    from bokeh.models import HoverTool, TapTool, OpenURL, ColumnDataSource
    from re import sub, MULTILINE

    if not isinstance(data, pd.DataFrame):
        raise ValueError('data is not a pandas DataFrame.')

    grouped_df = data.groupby(grouping_parameter)
    description = grouped_df[x_axis_parameter].describe()

    # draw the figure
    hover = HoverTool()
    chart = figure(tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'save', 'reset', 'tap'])

    # data
    # bars_ds = ColumnDataSource()
    color_nr = 0
    for group in grouped_df[x_axis_parameter]:
        counts, divisions = np.histogram(group[1].dropna(), bins=nr_of_bins)
        chart.quad(top=counts, bottom=0, left=divisions[:-1], right=divisions[1:],
                   fill_color=colors.values()[color_nr], line_color="#033649", fill_alpha=0.8, legend=group[0])
        # bars_ds.data.update(dict(legend=group[0],
        #                         left=divisions[:-1],
        #                         right=divisions[1:],
        #                         counts=counts,))
        color_nr = color_nr + 1
        if color_nr == len(colors):
            color_nr = 0

    hover.tooltips = [('{0}'.format(grouping_name), '@legend'),
                      ('{0}'.format(x_axis_label), '@left-@right'),
                      ('count', '@counts')]

    # axis
    chart.xaxis.major_label_text_font_size = '12pt'
    chart.xaxis.axis_label = x_axis_label
    chart.yaxis.major_label_text_font_size = '12pt'
    chart.yaxis.axis_label = 'Counts'

    chart_script, chart_figure = components(chart, INLINE)

    html_description = description.to_html(classes='table table-striped table-bordered small')
    # remove second row with only grouping parameter
    html_description = sub(u'.*?<tr>\n.*?<th>{0}[\s\S]+?</tr>'.format(grouping_parameter), '',
                           html_description, MULTILINE)

    return chart_script, chart_figure, html_description


def create_box_and_whisker(request, data, id_parameter, grouping_parameter, x_axis_label, y_axis_parameter,
                           y_axis_label, draw_mean=True, whisker_factor=1.5,
                           whisker_percentage=None):
    """
    return box and whisker plot.

    Based on: http://bokeh.pydata.org/en/latest/docs/gallery/boxplot.html

    :param request: original http-request (is used for clicking to specific study/studies)
    :param data: data used for calculations as pandas dataframe
    :param id_parameter: name of column indicating study row id
    :param grouping_parameter: column in data that is used for grouping
    :param x_axis_label: label used for x-axis
    :param y_axis_parameter: column in data that should be shown
    :param y_axis_label: label used for y-axis
    :param draw_mean: show mean value in box and whisker plot
    :param whisker_factor: length of whisker as factor of inter quartile distance
    :param whisker_percentage: length of whisker in percentile score (90; 95; ....)
    :param filter_query: string representing a valid pandas query string
    :return: box and whisker plot
    """
    from bokeh.plotting import figure
    from bokeh.embed import components
    from bokeh.resources import INLINE
    from bokeh.models import HoverTool, TapTool, OpenURL, ColumnDataSource, SaveTool
    from re import sub, MULTILINE

    if not isinstance(data, pd.DataFrame):
        raise ValueError('data is not a pandas DataFrame.')
    if ((whisker_factor is None) and (whisker_percentage is None)) or \
       ((whisker_factor is not None) and (whisker_percentage is not None)):
        raise ValueError('(Only) one of parameters whisker_factor or whisker_percentage should be given.')

    grouped_df = data.groupby(grouping_parameter)
    description = grouped_df[y_axis_parameter].describe()
    group_names = description.index.tolist()
    # make sure group_names are strings
    group_names = [str(name) for name in group_names]
    if ('25%' in description) and ('50%' in description) and ('75%' in description):
        q1 = description['25%'].to_frame(name=y_axis_parameter)
        median = description['50%'].to_frame(name=y_axis_parameter)
        q3 = description['75%'].to_frame(name=y_axis_parameter)
    else:
        # TODO: return error as data contains non-number data.
        q1 = 0
        median = 0
        q3 = 0
    if whisker_factor:
        iqr = q3 - q1
        whisker_upper = q3 + whisker_factor * iqr
        whisker_lower = q1 - whisker_factor * iqr
    else:
        whisker_upper = grouped_df.quantile(whisker_percentage/100)
        whisker_lower = grouped_df.quantile(1-whisker_percentage/100)

    # find all outliers
    outliers = grouped_df.apply(get_outliers_boxplot, y_param=y_axis_parameter,
                                whisker_lower=whisker_lower, whisker_upper=whisker_upper).dropna()
    outx = []
    outy = []
    outid = []
    if not outliers.empty:
        for group in grouped_df:
            if not outliers.loc[group[0]].empty:
                for _, row in outliers.loc[group[0]].iterrows():
                    outx.append(group[0])
                    outy.append(row[y_axis_parameter])
                    outid.append(str(row[id_parameter]))

    # draw the figure
    hover = HoverTool()
    chart = figure(x_range=group_names, tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'save', 'reset', 'tap'])

    # if no outliers, length of whiskers should be limited to minimum and/or maximum
    qmin = description['min'].to_frame(name=y_axis_parameter)
    qmax = description['max'].to_frame(name=y_axis_parameter)
    whisker_upper[y_axis_parameter] = [min([x, y]) for (x, y) in zip(list(qmax.loc[:, y_axis_parameter]), whisker_upper[y_axis_parameter])]
    whisker_lower[y_axis_parameter] = [max([x, y]) for (x, y) in zip(list(qmin.loc[:, y_axis_parameter]), whisker_lower[y_axis_parameter])]

    # whiskers
    chart.segment(group_names, whisker_upper[y_axis_parameter], group_names, q3[y_axis_parameter], line_color='black')
    chart.segment(group_names, whisker_lower[y_axis_parameter], group_names, q1[y_axis_parameter], line_color='black')
    chart.rect(group_names, whisker_lower[y_axis_parameter], 0.2, 0.01, line_color='black')
    chart.rect(group_names, whisker_upper[y_axis_parameter], 0.2, 0.01, line_color='black')

    # boxes
    chart.vbar(group_names, 0.7, median[y_axis_parameter], q3[y_axis_parameter], fill_color='white', line_color='black')
    chart.vbar(group_names, 0.7, q1[y_axis_parameter], median[y_axis_parameter], fill_color='white', line_color='black')

    # outliers
    outlier_ds = ColumnDataSource(data=dict(
        out_x=outx,
        out_y=outy,
        out_id=outid
    ))
    if not outliers.empty:
        chart.circle('out_x', 'out_y', size=4, color=colors['red'], fill_alpha=0.8, source=outlier_ds)
        url = request.path.rsplit('/', 2)[0] + '/@out_id'
        taptool = chart.select(type=TapTool)
        taptool.callback = OpenURL(url=url)
        hover.tooltips = [('{0}'.format(x_axis_label), '@out_x'), ('{0}'.format(y_axis_label), '@out_y{0.0}')]

    # mean
    if draw_mean:
        chart.asterisk(group_names, description['mean'], size=4, color=colors['darkBlue'])

    # axis
    chart.xaxis.major_label_text_font_size = '12pt'
    chart.xaxis.major_label_orientation = 3.1415 / 4
    chart.yaxis.major_label_text_font_size = '12pt'
    chart.yaxis.axis_label = y_axis_label

    chart_script, chart_figure = components(chart, INLINE)

    html_description = description.to_html(classes='table table-striped table-bordered small',
                                           float_format=lambda x: '%.3f' % x)
    # remove second row with only grouping parameter
    html_description = sub(u'.*?<tr>\n.*?<th>{0}[\s\S]+?</tr>'.format(grouping_parameter), '',
                           html_description, MULTILINE)

    return chart_script, chart_figure, html_description
