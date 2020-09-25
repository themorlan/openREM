# This Python file uses the following encoding: utf-8
#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2017  The Royal Marsden NHS Foundation Trust
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
..  module:: chart_functions
    :synopsis: Helper functions for calculating chart data

..  moduleauthor:: David Platten

"""

from django.conf import settings
from builtins import range  # pylint: disable=redefined-builtin
import pandas as pd


def global_config(filename, height_multiplier=1.0):
    return {
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": 1080 * height_multiplier,
            "width": 1920,
            "scale": 1,
        },
        "displaylogo": False,
        "scrollZoom": True,
    }


def create_dataframe(
    database_events,
    data_point_name_fields=None,
    data_point_value_fields=None,
    data_point_date_fields=None,
    data_point_time_fields=None,
    system_name_field=None,
    data_point_name_lowercase=None,
    data_point_value_multipliers=None,
    uid=None,
):

    fields_to_include = set()
    if uid:
        fields_to_include.add(uid)
    if data_point_name_fields:
        for field in data_point_name_fields:
            fields_to_include.add(field)
    if data_point_value_fields:
        for field in data_point_value_fields:
            fields_to_include.add(field)
    if data_point_date_fields:
        for field in data_point_date_fields:
            fields_to_include.add(field)
    if data_point_time_fields:
        for field in data_point_time_fields:
            fields_to_include.add(field)
    if system_name_field:
        fields_to_include.add(system_name_field)

    # NOTE: I am not excluding zero-value events from the calculations (zero DLP or zero CTDI)
    df = pd.DataFrame.from_records(database_events.values(*fields_to_include))

    dtype_conversion = {}
    for name_field in data_point_name_fields:
        dtype_conversion[name_field] = "category"

        # Replace any empty values with "Blank" (Plotly doesn't like empty values)
        df[name_field].fillna(value="Blank", inplace=True)
        # Make lowercase if required
        if data_point_name_lowercase:
            df[name_field] = df[name_field].str.lower()

    if system_name_field:
        df.rename(columns={system_name_field: "x_ray_system_name"}, inplace=True)
        df.sort_values(by="x_ray_system_name", inplace=True)
    else:
        df.insert(0, "x_ray_system_name", "All systems")
    dtype_conversion["x_ray_system_name"] = "category"

    if data_point_value_fields:
        for idx, value_field in enumerate(data_point_value_fields):
            df[value_field] = df[value_field].astype(float)
            if data_point_value_multipliers:
                df[value_field] *= data_point_value_multipliers[idx]

    if data_point_date_fields:
        for date_field in data_point_date_fields:
            df[date_field] = pd.to_datetime(df[date_field])

    df = df.astype(dtype_conversion)

    return df


def create_dataframe_time_series(
    df,
    df_name_col,
    df_value_col,
    df_date_col="study_date",
    time_period="M",
    average_choices=None,
):
    if average_choices is None:
        average_choices = ["mean"]

    df_time_series = (
        df.set_index(df_date_col)
        .groupby(["x_ray_system_name", df_name_col, pd.Grouper(freq=time_period)])
        .agg({df_value_col: average_choices})
    )
    df_time_series.columns = [s + df_value_col for s in average_choices]
    df_time_series = df_time_series.reset_index()
    return df_time_series


def create_dataframe_weekdays(
    df,
    df_name_col,
    df_date_col="study_date",
):
    df["weekday"] = pd.DatetimeIndex(df[df_date_col]).day_name()
    df["hour"] = [x.hour for x in df["study_time"]]

    df_time_series = (
        df.groupby(["x_ray_system_name", "weekday", "hour"])
        .agg({df_name_col: "count"})
        .reset_index()
    )
    return df_time_series


def create_dataframe_aggregates(df, df_name_col, df_agg_col, stats=None):
    # Make it possible to have multiple value cols (DLP, CTDI, for example)
    if stats is None:
        stats = ["count"]

    groupby_cols = ["x_ray_system_name", df_name_col]

    grouped_df = df.groupby(groupby_cols).agg({df_agg_col: stats})
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()

    return grouped_df


def plotly_set_default_theme(theme_name):
    import plotly.io as pio

    pio.templates.default = theme_name


def calculate_colour_sequence(scale_name="jet", n_colours=10):
    import matplotlib.cm
    import matplotlib.colors

    colour_seq = []
    cmap = matplotlib.cm.get_cmap(scale_name)
    if n_colours > 1:
        for i in range(n_colours):
            c = cmap(i / (n_colours - 1))
            colour_seq.append(matplotlib.colors.rgb2hex(c))
    else:
        c = cmap(0)
        colour_seq.append(matplotlib.colors.rgb2hex(c))

    return colour_seq


def plotly_boxplot(
    df,
    df_name_col,
    df_value_col,
    value_axis_title="",
    name_axis_title="",
    colourmap="RdYlBu",
    filename="OpenREM_boxplot_chart",
    sorted_category_list=None,
):
    from plotly.offline import plot
    import plotly.express as px

    n_colours = len(df.x_ray_system_name.unique())
    colour_sequence = calculate_colour_sequence(colourmap, n_colours)

    try:
        fig = px.box(
            df,
            x=df_name_col,
            y=df_value_col,
            color="x_ray_system_name",
            labels={
                df_value_col: value_axis_title,
                df_name_col: name_axis_title,
                "x_ray_system_name": "System",
            },
            color_discrete_sequence=colour_sequence,
            category_orders=sorted_category_list,
            height=750,
        )

        fig.update_traces(quartilemethod="exclusive")

        fig.update_xaxes(tickson="boundaries")

        return plot(
            fig,
            output_type="div",
            include_plotlyjs=False,
            config=global_config(filename),
        )

    except ValueError as e:
        msg = "<div class='alert alert-warning' role='alert'>"
        if settings.DEBUG:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of systems."
            msg += "<p>Error is:</p>"
            msg += "<pre>" + e.args[0].replace("\n", "<br>") + "</pre>"
        else:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of systems."

        msg += "</div>"

        return msg


def create_freq_sorted_category_list(df, df_name_col, sorting):
    category_sorting_df = df.groupby(df_name_col).count().reset_index()
    if sorting[1] == "name":
        sort_by = df_name_col
    else:
        sort_by = "x_ray_system_name"

    sorted_categories = {
        df_name_col: list(
            category_sorting_df.sort_values(by=sort_by, ascending=sorting[0])[
                df_name_col
            ]
        )
    }

    return sorted_categories


def create_sorted_category_list(df, df_name_col, df_value_col, sorting):
    # Calculate the required aggregates for creating a list of categories for sorting
    grouped_df = df.groupby(df_name_col).agg({df_value_col: ["mean", "count"]})
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()

    if sorting[1] == "name":
        sort_by = df_name_col
    elif sorting[1] == "frequency":
        sort_by = "count"
    else:
        sort_by = "mean"

    categories_sorted = {
        df_name_col: list(
            grouped_df.sort_values(by=sort_by, ascending=sorting[0])[df_name_col]
        )
    }

    return categories_sorted


def plotly_barchart(
    df,
    df_name_col,
    value_axis_title="",
    name_axis_title="",
    colourmap="RdYlBu",
    filename="OpenREM_bar_chart",
    sorted_category_list=None,
    average_choice="mean",
):
    from plotly.offline import plot
    import plotly.express as px

    n_colours = len(df.x_ray_system_name.unique())
    colour_sequence = calculate_colour_sequence(colourmap, n_colours)

    fig = px.bar(
        df,
        x=df_name_col,
        y=average_choice,
        color="x_ray_system_name",
        barmode="group",
        labels={
            average_choice: value_axis_title,
            df_name_col: name_axis_title,
            "x_ray_system_name": "System",
            "count": "Frequency",
        },
        category_orders=sorted_category_list,
        color_discrete_sequence=colour_sequence,
        hover_name="x_ray_system_name",
        hover_data={
            "x_ray_system_name": False,
            average_choice: ":.2f",
            "count": ":.0d",
        },
        height=750,
    )

    fig.update_xaxes(tickson="boundaries")

    return plot(
        fig, output_type="div", include_plotlyjs=False, config=global_config(filename)
    )


def plotly_barchart_mean_median(
    df,
    df_name_col,
    value_axis_title="",
    name_axis_title="",
    colourmap="RdYlBu",
    filename="OpenREM_bar_chart",
    sorted_category_list=None,
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from plotly.offline import plot

    n_colours = len(df.x_ray_system_name.unique())
    colour_sequence = calculate_colour_sequence(colourmap, n_colours)
    system_names = df.x_ray_system_name.unique()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True)

    i = 0
    for system, subset in df.groupby("x_ray_system_name"):

        trace = go.Bar(
            x=subset[df_name_col],
            y=subset["mean"],
            name=system,
            xaxis="x1",
            yaxis="y1",
            marker_color=colour_sequence[i],
            showlegend=False,
            legendgroup=i,
            text=subset["count"],
            hovertemplate=f"<b>{system_names[i]}</b><br>"
            + "%{x}<br>"
            + "Mean: %{y:.2f}<br>"
            + "Count: %{text:.0d}<br>"
            + "<extra></extra>",
        )

        fig.append_trace(trace, row=1, col=1)

        trace = go.Bar(
            x=subset[df_name_col],
            y=subset["median"],
            name=system,
            xaxis="x2",
            yaxis="y2",
            marker_color=colour_sequence[i],
            legendgroup=i,
            text=subset["count"],
            hovertemplate=f"<b>{system_names[i]}</b><br>"
            + "%{x}<br>"
            + "Median: %{y:.2f}<br>"
            + "Count: %{text:.0d}<br>"
            + "<extra></extra>",
        )

        fig.append_trace(trace, row=2, col=1)
        i += 1

    layout = go.Layout(
        height=750,
        xaxis={
            "categoryorder": "array",
            "categoryarray": sorted_category_list[df_name_col],
        },
    )

    fig.update_layout(layout)

    fig.update_xaxes(title_text=name_axis_title, row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=1, col=1)
    fig.update_yaxes(title_text="Mean " + value_axis_title, row=1, col=1)
    fig.update_yaxes(title_text="Median " + value_axis_title, row=2, col=1)

    return plot(
        fig, output_type="div", include_plotlyjs=False, config=global_config(filename)
    )


def plotly_histogram(
    df,
    df_facet_col,
    df_value_col,
    df_category_name_col="x_ray_system_name",
    value_axis_title="",
    legend_title="System",
    n_bins=10,
    colourmap="RdYlBu",
    filename="OpenREM_histogram_chart",
    facet_col_wrap=3,
    sorted_category_list=None,
):
    from plotly.offline import plot
    import plotly.express as px
    import math

    n_facet_rows = math.ceil(len(df[df_facet_col].unique()) / facet_col_wrap)
    chart_height = n_facet_rows * 250
    if chart_height < 750:
        chart_height = 750

    n_colours = len(df[df_category_name_col].unique())
    colour_sequence = calculate_colour_sequence(colourmap, n_colours)

    try:
        fig = px.histogram(
            df,
            x=df_value_col,
            nbins=n_bins,
            barmode="group",
            color=df_category_name_col,
            facet_col=df_facet_col,
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=0.40
            / n_facet_rows,  # default is 0.07 when facet_col_wrap is used
            labels={df_value_col: value_axis_title, df_category_name_col: legend_title},
            color_discrete_sequence=colour_sequence,
            category_orders=sorted_category_list,
            height=chart_height,
        )

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        fig.update_xaxes(rangemode="nonnegative")
        fig.update_yaxes(rangemode="nonnegative")

        return plot(
            fig,
            output_type="div",
            include_plotlyjs=False,
            config=global_config(filename, height_multiplier=chart_height / 750.0),
        )

    except ValueError as e:
        msg = "<div class='alert alert-warning' role='alert'>"
        if settings.DEBUG:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."
            msg += "<p>Error is:</p>"
            msg += "<pre>" + e.args[0].replace("\n", "<br>") + "</pre>"
        else:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."

        msg += "</div>"

        return msg


def plotly_histogram_barchart(
    df,
    df_facet_col,
    df_category_col,
    df_value_col,
    value_axis_title="",
    legend_title="System",
    n_bins=10,
    colourmap="RdYlBu",
    filename="OpenREM_histogram_chart",
    facet_col_wrap=3,
    df_facet_category_list=None,
    df_category_name_list=None,
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from plotly.offline import plot
    import math
    import numpy as np

    n_facets = len(df_facet_category_list)
    n_facet_rows = math.ceil(n_facets / facet_col_wrap)
    chart_height = n_facet_rows * 250
    if chart_height < 750:
        chart_height = 750

    if n_facets < facet_col_wrap:
        facet_col_wrap = n_facets

    n_colours = len(df[df_category_col].unique())
    colour_sequence = calculate_colour_sequence(colourmap, n_colours)

    try:
        fig = make_subplots(
            rows=n_facet_rows, cols=facet_col_wrap, vertical_spacing=0.40 / n_facet_rows
        )

        current_row = 1
        current_col = 1
        current_facet = 0
        category_names = []

        for facet_name in df_facet_category_list:
            facet_subset = df[df[df_facet_col] == facet_name]

            min_bin_value = facet_subset[df_value_col].min()
            max_bin_value = facet_subset[df_value_col].max()
            bins = np.linspace(min_bin_value, max_bin_value, n_bins + 1)
            mid_bins = 0.5 * (bins[:-1] + bins[1:])
            bin_labels = np.array(
                ["{:.2f} to {:.2f}".format(i, j) for i, j in zip(bins[:-1], bins[1:])]
            )

            for category_name in df_category_name_list:
                category_subset = facet_subset[
                    facet_subset[df_category_col] == category_name
                ]

                if category_name in category_names:
                    show_legend = False
                else:
                    show_legend = True
                    category_names.append(category_name)

                category_idx = category_names.index(category_name)

                counts, junk = np.histogram(
                    category_subset[df_value_col].values, bins=bins
                )

                trace = go.Bar(
                    x=mid_bins,
                    y=counts,
                    name=category_name,
                    marker_color=colour_sequence[category_idx],
                    legendgroup=category_idx,
                    showlegend=show_legend,
                    text=bin_labels,
                    hovertemplate=f"<b>{facet_name}</b><br>"
                    + f"{category_name}<br>"
                    + "Frequency: %{y:.0d}<br>"
                    + "Bin range: %{text}<br>"
                    + "Mid-bin: %{x:.2f}<br>"
                    + "<extra></extra>",
                )

                fig.append_trace(trace, row=current_row, col=current_col)

            fig.update_xaxes(
                title_text=facet_name + " " + value_axis_title,
                tickvals=bins,
                row=current_row,
                col=current_col,
            )

            if current_col == 1:
                fig.update_yaxes(
                    title_text="Frequency", row=current_row, col=current_col
                )

            current_facet += 1
            current_col += 1
            if current_col > facet_col_wrap:
                current_row += 1
                current_col = 1

        layout = go.Layout(height=chart_height)

        fig.update_layout(layout)
        fig.update_layout(legend_title_text=legend_title)

        return plot(
            fig,
            output_type="div",
            include_plotlyjs=False,
            config=global_config(filename, height_multiplier=chart_height / 750.0),
        )

    except ValueError as e:
        msg = "<div class='alert alert-warning' role='alert'>"
        if settings.DEBUG:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."
            msg += "<p>Error is:</p>"
            msg += "<pre>" + e.args[0].replace("\n", "<br>") + "</pre>"
        else:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."

        msg += "</div>"

        return msg


def plotly_binned_statistic_barchart(
    df,
    df_x_value_col,
    df_y_value_col,
    x_axis_title="",
    y_axis_title="",
    df_category_col=None,
    df_facet_col="x_ray_system_name",
    facet_title="System",
    user_bins=None,
    colour_map="RdYlBu",
    file_name="OpenREM_binned_statistic_chart",
    facet_col_wrap=3,
    df_facet_category_list=None,
    df_category_name_list=None,
    stat_name="mean",
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from plotly.offline import plot
    import math
    import numpy as np
    from scipy import stats

    n_facets = len(df_facet_category_list)
    n_facet_rows = math.ceil(n_facets / facet_col_wrap)
    chart_height = n_facet_rows * 250
    if chart_height < 750:
        chart_height = 750

    if n_facets < facet_col_wrap:
        facet_col_wrap = n_facets

    n_colours = len(df[df_category_col].unique())
    colour_sequence = calculate_colour_sequence(colour_map, n_colours)

    try:
        fig = make_subplots(
            rows=n_facet_rows, cols=facet_col_wrap, vertical_spacing=0.40 / n_facet_rows
        )

        current_row = 1
        current_col = 1
        current_facet = 0
        category_names = []

        bins = np.sort(np.array(user_bins))

        # Drop any rows with nan values in x or y
        df = df.dropna(subset=[df_x_value_col, df_y_value_col])

        for facet_name in df_facet_category_list:
            facet_subset = df[df[df_facet_col] == facet_name]

            facet_x_min = facet_subset[df_x_value_col].min()
            facet_x_max = facet_subset[df_x_value_col].max()

            if np.isfinite(facet_x_min):
                if facet_x_min < np.amin(bins):
                    bins = np.concatenate([[facet_x_min], bins])
            if np.isfinite(facet_x_max):
                if facet_x_max > np.amax(bins):
                    bins = np.concatenate([bins, [facet_x_max]])

            bin_labels = np.array(
                ["{:.1f} to {:.1f}".format(i, j) for i, j in zip(bins[:-1], bins[1:])]
            )

            for category_name in df_category_name_list:
                category_subset = facet_subset[
                    facet_subset[df_category_col] == category_name
                ]

                if len(category_subset.index) > 0:
                    if category_name in category_names:
                        show_legend = False
                    else:
                        show_legend = True
                        category_names.append(category_name)

                    category_idx = category_names.index(category_name)

                    statistic, junk, bin_numbers = stats.binned_statistic(
                        category_subset[df_x_value_col].values,
                        category_subset[df_y_value_col].values,
                        statistic=stat_name,
                        bins=bins,
                    )
                    bin_counts = np.bincount(bin_numbers)
                    trace_labels = np.array(
                        [
                            "Frequency: {}<br>Bin range: {}".format(i, j)
                            for i, j in zip(bin_counts, bin_labels)
                        ]
                    )

                    trace = go.Bar(
                        x=bin_labels,
                        y=statistic,
                        name=category_name,
                        marker_color=colour_sequence[category_idx],
                        legendgroup=category_idx,
                        showlegend=show_legend,
                        text=trace_labels,
                        hovertemplate=f"<b>{facet_name}</b><br>"
                        + f"{category_name}<br>"
                        + f"{stat_name.capitalize()}: "
                        + "%{y:.2f}<br>"
                        + "%{text}<br>"
                        + "<extra></extra>",
                    )

                    fig.append_trace(trace, row=current_row, col=current_col)

            fig.update_xaxes(
                title_text=facet_name + " " + x_axis_title,
                tickson="boundaries",
                row=current_row,
                col=current_col,
            )

            if current_col == 1:
                fig.update_yaxes(
                    title_text=stat_name.capitalize() + " " + y_axis_title,
                    row=current_row,
                    col=current_col,
                )

            current_facet += 1
            current_col += 1
            if current_col > facet_col_wrap:
                current_row += 1
                current_col = 1

        layout = go.Layout(height=chart_height)

        fig.update_layout(layout)
        fig.update_layout(legend_title_text=facet_title)

        return plot(
            fig,
            output_type="div",
            include_plotlyjs=False,
            config=global_config(file_name, height_multiplier=chart_height / 750.0),
        )

    except ValueError as e:
        msg = "<div class='alert alert-warning' role='alert'>"
        if settings.DEBUG:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."
            msg += "<p>Error is:</p>"
            msg += "<pre>" + e.args[0].replace("\n", "<br>") + "</pre>"
        else:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."

        msg += "</div>"

        return msg


def plotly_frequency_barchart(
    df,
    df_legend_col,
    legend_title="",
    df_x_axis_col="x_ray_system_name",
    x_axis_title="System",
    colourmap="RdYlBu",
    filename="OpenREM_bar_chart",
    sorted_category_list=None,
):
    from plotly.offline import plot
    import plotly.express as px

    n_colours = len(df[df_legend_col].unique())
    colour_sequence = calculate_colour_sequence(colourmap, n_colours)

    fig = px.bar(
        df,
        x=df_x_axis_col,
        y="count",
        color=df_legend_col,
        labels={
            "count": "Frequency",
            df_legend_col: legend_title,
            df_x_axis_col: x_axis_title,
        },
        category_orders=sorted_category_list,
        color_discrete_sequence=colour_sequence,
        height=750,
    )

    fig.update_xaxes(tickson="boundaries")

    return plot(
        fig, output_type="div", include_plotlyjs=False, config=global_config(filename)
    )


def plotly_timeseries_linechart(
    df,
    df_name_col,
    df_value_col,
    df_date_col,
    facet_col="x_ray_system_name",
    value_axis_title="",
    name_axis_title="",
    legend_title="",
    colourmap="RdYlBu",
    filename="OpenREM_over_time_chart",
    facet_col_wrap=3,
    sorted_category_list=None,
):
    from plotly.offline import plot
    import plotly.express as px
    import math

    n_facet_rows = math.ceil(len(df[facet_col].unique()) / facet_col_wrap)
    chart_height = n_facet_rows * 250
    if chart_height < 750:
        chart_height = 750

    n_colours = len(df[df_name_col].unique())
    colour_sequence = calculate_colour_sequence(colourmap, n_colours)

    try:
        fig = px.scatter(
            df,
            x=df_date_col,
            y=df_value_col,
            color=df_name_col,
            facet_col=facet_col,
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=0.40
            / n_facet_rows,  # default is 0.07 when facet_col_wrap is used
            labels={
                df_value_col: value_axis_title,
                df_name_col: legend_title,
                df_date_col: name_axis_title,
                "x_ray_system_name": "System",
            },
            color_discrete_sequence=colour_sequence,
            category_orders=sorted_category_list,
            height=chart_height,
        )

        for data_set in fig.data:
            data_set.update(mode="markers+lines")

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        fig.update_xaxes(showticklabels=True)
        fig.update_yaxes(showticklabels=True)

        return plot(
            fig,
            output_type="div",
            include_plotlyjs=False,
            config=global_config(filename),
        )

    except ValueError as e:
        msg = "<div class='alert alert-warning' role='alert'>"
        if settings.DEBUG:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."
            msg += "<p>Error is:</p>"
            msg += "<pre>" + e.args[0].replace("\n", "<br>") + "</pre>"
        else:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."

        msg += "</div>"

        return msg


def plotly_scatter(
    df,
    df_x_value_col,
    df_y_value_col,
    df_category_name_col,
    df_facet_col="x_ray_system_name",
    facet_title="System",
    x_axis_title="",
    y_axis_title="",
    legend_title="",
    colourmap="RdYlBu",
    filename="OpenREM_scatter_chart",
    facet_col_wrap=3,
    sorted_category_list=None,
):
    from plotly.offline import plot
    import plotly.express as px
    import math

    n_facet_rows = math.ceil(len(df[df_facet_col].unique()) / facet_col_wrap)
    chart_height = n_facet_rows * 250
    if chart_height < 750:
        chart_height = 750

    n_colours = len(df[df_category_name_col].unique())
    colour_sequence = calculate_colour_sequence(colourmap, n_colours)

    try:
        # Drop any rows with nan values in x or y
        df = df.dropna(subset=[df_x_value_col, df_y_value_col])

        fig = px.scatter(
            df,
            x=df_x_value_col,
            y=df_y_value_col,
            color=df_category_name_col,
            facet_col=df_facet_col,
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=0.40
            / n_facet_rows,  # default is 0.07 when facet_col_wrap is used
            labels={
                df_x_value_col: x_axis_title,
                df_y_value_col: y_axis_title,
                df_category_name_col: legend_title,
                df_facet_col: facet_title,
            },
            color_discrete_sequence=colour_sequence,
            category_orders=sorted_category_list,
            opacity=0.6,
            height=chart_height,
        )

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        fig.update_xaxes(showticklabels=True)
        fig.update_yaxes(showticklabels=True)

        return plot(
            fig,
            output_type="div",
            include_plotlyjs=False,
            config=global_config(filename, height_multiplier=chart_height / 750.0),
        )

    except ValueError as e:
        msg = "<div class='alert alert-warning' role='alert'>"
        if settings.DEBUG:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."
            msg += "<p>Error is:</p>"
            msg += "<pre>" + e.args[0].replace("\n", "<br>") + "</pre>"
        else:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of categories or systems."

        msg += "</div>"

        return msg


def plotly_barchart_weekdays(
    df,
    df_name_col,
    df_value_col,
    name_axis_title="",
    value_axis_title="",
    colourmap="RdYlBu",
    filename="OpenREM_worload_chart",
    facet_col_wrap=3,
):
    from plotly.offline import plot
    import plotly.express as px
    import math

    n_facet_rows = math.ceil(len(df.x_ray_system_name.unique()) / facet_col_wrap)
    chart_height = n_facet_rows * 250
    if chart_height < 750:
        chart_height = 750

    try:
        fig = px.bar(
            df,
            x=df_name_col,
            y=df_value_col,
            facet_col="x_ray_system_name",
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=0.40
            / n_facet_rows,  # default is 0.07 when facet_col_wrap is used
            color=df_value_col,
            labels={
                df_name_col: name_axis_title,
                df_value_col: value_axis_title,
                "x_ray_system_name": "System",
                "hour": "Hour",
            },
            color_continuous_scale=colourmap,
            hover_data={
                "x_ray_system_name": True,
                "weekday": True,
                "hour": ":.2f",
                df_value_col: True,
            },
            height=chart_height,
        )

        fig.update_xaxes(
            categoryarray=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            tickson="boundaries",
        )

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        return plot(
            fig,
            output_type="div",
            include_plotlyjs=False,
            config=global_config(filename, height_multiplier=chart_height / 750.0),
        )

    except ValueError as e:
        msg = "<div class='alert alert-warning' role='alert'>"
        if settings.DEBUG:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of systems."
            msg += "<p>Error is:</p>"
            msg += "<pre>" + e.args[0].replace("\n", "<br>") + "</pre>"
        else:
            msg += "Could not resolve chart. Try filtering the data to reduce the number of systems."

        msg += "</div>"

        return msg


def construct_frequency_chart(
    df=None,
    df_name_col=None,
    sorting_choice=None,
    legend_title=None,
    df_x_axis_col=None,
    x_axis_title=None,
    grouping_choice=None,
    colour_map=None,
    file_name=None,
    sorted_categories=None,
):

    df_aggregated = create_dataframe_aggregates(df, df_name_col, df_name_col, ["count"])

    if not sorted_categories:
        sorted_categories = create_freq_sorted_category_list(
            df, df_name_col, sorting_choice
        )

    df_legend_col = df_name_col
    if grouping_choice == "series":
        df_legend_col = "x_ray_system_name"
        x_axis_title = legend_title
        legend_title = "System"
        df_x_axis_col = df_name_col

    return plotly_frequency_barchart(
        df_aggregated,
        df_legend_col,
        legend_title=legend_title,
        df_x_axis_col=df_x_axis_col,
        x_axis_title=x_axis_title,
        colourmap=colour_map,
        filename=file_name,
        sorted_category_list=sorted_categories,
    )


def construct_scatter_chart(
    df=None,
    df_name_col=None,
    df_x_col=None,
    df_y_col=None,
    sorting=None,
    grouping_choice=None,
    legend_title=None,
    colour_map=None,
    facet_col_wrap=None,
    x_axis_title=None,
    y_axis_title=None,
    file_name=None,
):
    sorted_categories = create_sorted_category_list(df, df_name_col, df_y_col, sorting)

    df_legend_col = df_name_col
    df_group_col = "x_ray_system_name"
    if grouping_choice == "series":
        df_legend_col = "x_ray_system_name"
        legend_title = "System"
        df_group_col = df_name_col

    return plotly_scatter(
        df,
        df_x_col,
        df_y_col,
        df_legend_col,
        df_facet_col=df_group_col,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        legend_title=legend_title,
        colourmap=colour_map,
        filename=file_name,
        facet_col_wrap=facet_col_wrap,
        sorted_category_list=sorted_categories,
    )


def construct_over_time_charts(
    df=None,
    df_name_col=None,
    df_value_col=None,
    df_date_col=None,
    name_title=None,
    value_title=None,
    date_title=None,
    sorting=None,
    time_period=None,
    average_choices=None,
    grouping_choice=None,
    colour_map=None,
    facet_col_wrap=None,
    file_name=None,
):
    sorted_categories = create_sorted_category_list(
        df, df_name_col, df_value_col, sorting
    )

    df_time_series = create_dataframe_time_series(
        df,
        df_name_col,
        df_value_col,
        df_date_col=df_date_col,
        time_period=time_period,
        average_choices=average_choices,
    )

    category_names_col = df_name_col
    group_by_col = "x_ray_system_name"
    if grouping_choice == "series":
        category_names_col = "x_ray_system_name"
        group_by_col = df_name_col

    return_value = {}

    if "mean" in average_choices:
        return_value["mean"] = plotly_timeseries_linechart(
            df_time_series,
            category_names_col,
            "mean" + df_value_col,
            df_date_col,
            facet_col=group_by_col,
            value_axis_title=value_title,
            name_axis_title=date_title,
            legend_title=name_title,
            colourmap=colour_map,
            filename=file_name,
            facet_col_wrap=facet_col_wrap,
            sorted_category_list=sorted_categories,
        )

    if "median" in average_choices:
        return_value["median"] = plotly_timeseries_linechart(
            df_time_series,
            category_names_col,
            "median" + df_value_col,
            df_date_col,
            facet_col=group_by_col,
            value_axis_title=value_title,
            name_axis_title=date_title,
            legend_title=name_title,
            colourmap=colour_map,
            filename=file_name,
            facet_col_wrap=facet_col_wrap,
            sorted_category_list=sorted_categories,
        )

    return return_value
