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

from builtins import filter  # pylint: disable=redefined-builtin
from builtins import range  # pylint: disable=redefined-builtin
import pandas as pd


def create_dataframe(
        database_events,
        data_point_name_fields=None,
        data_point_value_fields=None,
        data_point_date_fields=None,
        system_name_field=None,
        data_point_name_lowercase=None,
        data_point_value_multiplier=None
):

    fields_to_include = set()
    if data_point_name_fields:
        for field in data_point_name_fields:
            fields_to_include.add(field)
    if data_point_value_fields:
        for field in data_point_value_fields:
            fields_to_include.add(field)
    if data_point_date_fields:
        for field in data_point_date_fields:
            fields_to_include.add(field)
    if system_name_field:
        fields_to_include.add(system_name_field)

    # NOTE: I am not excluding zero-value events from the calculations (zero DLP or zero CTDI)
    df = pd.DataFrame.from_records(database_events.values(*fields_to_include))

    if data_point_name_lowercase:
        for name_field in data_point_name_fields:
            df[name_field] = df[name_field].str.lower()

    if system_name_field:
        df.rename(columns={system_name_field: "x_ray_system_name"}, inplace=True)
    else:
        df.insert(0, "x_ray_system_name", "All systems")

    for value_field in data_point_value_fields:
        df[value_field] = df[value_field].astype(float)
        if data_point_value_multiplier:
            df[value_field] *= data_point_value_multiplier

    for date_field in data_point_date_fields:
        df[date_field] = pd.to_datetime(df[date_field])

    return df


def plotly_boxplot(
        df,
        df_name_col,
        df_value_col,
        value_axis_title="",
        name_axis_title=""
):
    from plotly.offline import plot
    import plotly.express as px

    fig = px.box(
        df,
        x=df_name_col,
        y=df_value_col,
        color="x_ray_system_name",
        labels={
            df_value_col:value_axis_title,
            df_name_col:name_axis_title,
            "x_ray_system_name": "System"
        }
    )
    fig.update_traces(quartilemethod="exclusive")
    return plot(fig, output_type="div")


def plotly_barchart(
        df,
        df_name_col,
        df_value_col,
        value_axis_title="",
        name_axis_title=""
):
    from plotly.offline import plot
    import plotly.express as px

    fig = px.histogram(
        df,
        x=df_name_col,
        y=df_value_col,
        color="x_ray_system_name",
        barmode="group",
        histfunc="avg",
        labels={
            df_value_col:value_axis_title,
            df_name_col:name_axis_title,
            "x_ray_system_name": "System"
        }
    )
    return plot(fig, output_type="div")


def altair_barchart_average(
    df,
    df_name_col,
    df_value_col,
    average_choice="mean",
    value_axis_title=""
):
    """ This function creates an Altair bar chart of average value vs. a category.

    Args:
        df: Pandas DataFrame containing the data to be charted
        df_name_col: the column containing data names
        df_value_col: the column containing data values
        average_choice: string, default="mean; set to either mean, median or both
        value_axis_title: string, default=""; to use for the value axis label

    Returns:
        An Altair chart object
    """
    import altair as alt

    # Disable maximum of 5000 rows in a DataFrame
    alt.data_transformers.disable_max_rows()

    # Create a plot with either the mean or median
    if average_choice in ["mean", "median"]:
        selection = alt.selection_multi(fields=["x_ray_system_name"], bind="legend")

        return alt.Chart(df).mark_bar().encode(
            row=alt.Row(df_name_col, title="", header=alt.Header(labelAngle=0, labelAlign="left")),
            y=alt.Y("x_ray_system_name", axis=alt.Axis(labels=False, title="")),
            x=alt.X(average_choice + "(" + df_value_col + ")", title=average_choice.capitalize() + " " + value_axis_title),
            color=alt.Color("x_ray_system_name", legend=alt.Legend(title="System")),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.05)),
            tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                     alt.Tooltip(df_name_col, title="Name"),
                     alt.Tooltip(average_choice + "(" + df_value_col + ")", format=".2f", title=average_choice.capitalize()),
                     alt.Tooltip("count(" + df_value_col + ")", format=".0f", title="Frequency")]
        ).add_selection(
            selection
        ).resolve_axis(
            x="independent"
        ).interactive()

    # Assume the user must have selected "both"
    else:
        selection = alt.selection_multi(fields=["aggregate"], bind="legend")

        return alt.Chart(df).transform_aggregate(
            mean="mean(" + df_value_col + ")",
            median="median(" + df_value_col + ")",
            groupby=[df_name_col, "x_ray_system_name"]
        ).transform_fold(
            ["mean", "median"],
            as_=["aggregate", "value"]
        ).mark_bar().encode(
            row=alt.Row("x_ray_system_name", title=""),
            x=alt.X("value:Q", title="", stack=None),
            y=alt.Y(df_name_col, axis=alt.Axis(title="")),
            color=alt.Color("aggregate:N", legend=alt.Legend(title="Average " + value_axis_title)),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.05)),
            tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                     alt.Tooltip(df_name_col, title="Name"),
                     alt.Tooltip("value:Q", format=".2f", title="Average")]
        ).add_selection(
            selection
        ).interactive()


def altair_barchart_histogram(
        df,
        df_name_col,
        df_value_col,
        n_bins=10,
        value_axis_title=""
):
    """ This function creates an Altair bar chart histogram of values.

    Args:
        df: Pandas DataFrame containing the data to be charted
        df_name_col: the column containing data names
        df_value_col: the column containing data values
        n_bins: integer, default=10; the maximum number of bins to use
        value_axis_title: string, default=""; to use for the value axis label

    Returns:
        An Altair chart object
    """
    import altair as alt

    selection = alt.selection_multi(fields=["x_ray_system_name"], bind="legend")

    return alt.Chart(df).mark_bar().encode(
        row=alt.Row(df_name_col, title="", header=alt.Header(labelAngle=0, labelAlign="left")),
        x=alt.X(df_value_col, bin=alt.Bin(maxbins=n_bins), title="Binned " + value_axis_title),
        y=alt.Y("count()", title="Frequency", stack=None),
        color=alt.Color("x_ray_system_name", legend=alt.Legend(title="System")),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.05)),
        tooltip=[alt.Tooltip("count()", title="Frequency"),
                 alt.Tooltip(df_name_col, bin=alt.Bin(maxbins=n_bins), title="Bin range")]
    ).add_selection(
        selection
    ).resolve_scale(
        y="independent",
        x="independent"
    ).interactive()


def altair_barchart_frequency(
    df,
    df_name_col,
    legend_title=""
):
    """ This function creates an Altair bar chart of category frequency.

    Args:
        df: Pandas DataFrame containing the data to be charted
        df_name_col: the column containing data names
        legend_title: string, default=""; to use for the legend title

    Returns:
        An Altair chart object
    """
    import altair as alt

    # Disable maximum of 5000 rows in a DataFrame
    alt.data_transformers.disable_max_rows()

    selection = alt.selection_multi(fields=[df_name_col], bind="legend")

    return alt.Chart(df).mark_bar().encode(
        x=alt.X("count(" + df_name_col + "):Q", title="Frequency"),
        y=alt.Y("x_ray_system_name", axis=alt.Axis(title="")),
        color=alt.Color(df_name_col,
                        sort=alt.EncodingSortField(df_name_col, op="count", order="descending"),
                        legend=alt.Legend(title=legend_title, symbolLimit=250)),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.05)),
        order=alt.Order("count(" + df_name_col + "):Q", sort="descending"),
        tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                 alt.Tooltip(df_name_col, title="Name"),
                 alt.Tooltip("count(" + df_name_col + "):Q", format=".0f", title="Frequency")]
    ).add_selection(
        selection
    ).interactive()


def altair_linechart_average(
    df,
    df_name_col,
    df_value_col,
    average_choice="mean",
    time_unit="yearmonth",
    value_axis_title="",
    legend_title=""
):
    """ This function creates an Altair line chart of average value over time.

    Args:
        df: Pandas DataFrame containing the data to be charted
        df_name_col: the column containing data names
        df_value_col: the column containing data values
        average_choice: string, default="mean; set to either mean, median or both
        time_unit: string, default="yearmonth"; set to determine the time increment
        value_axis_title: string, default=""; to use for the value axis label
        legend_title: string, default=""; to use for the legend title

    Returns:
        An Altair chart object
    """
    import altair as alt

    # Disable maximum of 5000 rows in a DataFrame
    alt.data_transformers.disable_max_rows()

    if average_choice == "both":
        averages = ["mean", "median"]
    else:
        averages = [average_choice]

    selection = alt.selection_multi(fields=[df_name_col], bind="legend")

    chart = alt.Chart(df).mark_line(point=True).encode(
        row=alt.Row("x_ray_system_name:N", title=""),
        x=alt.X(time_unit + "(study_date):T", title="Study date", axis=alt.Axis(labelAngle=90)),
        y=alt.Y(averages[0] + "(" + df_value_col + "):Q",
                title=averages[0].capitalize() + " " + value_axis_title),
        color=alt.Color(df_name_col, legend=alt.Legend(title=legend_title, symbolLimit=250)),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
        tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                 alt.Tooltip(df_name_col, title="Name"),
                 alt.Tooltip(averages[0] + "(" + df_value_col + "):Q", format=".2f", title=averages[0].capitalize())]
    ).add_selection(
        selection
    ).interactive()

    if average_choice == "both":
        other = chart.encode(
            y=alt.Y("median(" + df_value_col + "):Q",
                    title="Median " + value_axis_title),
            tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                     alt.Tooltip(df_name_col, title="Name"),
                     alt.Tooltip("median(" + df_value_col + "):Q", format=".2f", title="Median")]
        ).interactive()

        chart = alt.hconcat(chart, other)

    return chart


def altair_barchart_workload(
        df,
        value_axis_title=""
):
    """ This function creates an Altair bar chart of workload.

    Args:
        df: Pandas DataFrame containing the data to be charted
        value_axis_title: string, default=""; to use for the value axis label

    Returns:
        An Altair chart object
    """
    import altair as alt

    # Disable maximum of 5000 rows in a DataFrame
    alt.data_transformers.disable_max_rows()

    selection = alt.selection_single(encodings=["y"])

    return alt.Chart(df).mark_bar().encode(
        row=alt.Row("x_ray_system_name:N", title=""),
        y=alt.Y("day(study_date):O", title=""),
        x=alt.X("count(x_ray_system_name)", title=value_axis_title + " frequency"),
        color=alt.Color("x_ray_system_name", legend=alt.Legend(title="System", symbolLimit=250)),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
        tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                 alt.Tooltip("day(study_date)", title="Day"),
                 alt.Tooltip("count()", format=".0f", title="Frequency")]
    ).add_selection(
        selection
    ).interactive()


def average_chart_inc_histogram_data(
    database_events,
    db_display_name_relationship,
    db_series_names,
    db_value_name,
    value_multiplier=1.0,
    plot_average=False,
    plot_freq=False,
    plot_series_per_system=False,
    plot_average_choice="mean",
    num_hist_bins=10,
    exclude_constant_angle=False,
    calculate_histograms=False,
    case_insensitive_categories=False,
    chart_value_axis_title="",
    chart_category_name="",
    plot_average_over_time=False,
    time_period="yearmonth",
    plot_workload=False
):
    """ This function calculates the data for an OpenREM Highcharts plot of average value vs. a category, as well as a
    histogram of values for each category. It is also used for OpenREM Highcharts frequency plots.

    Args:
        database_events: database events to use for the plot
        db_display_name_relationship: database table and field of x-ray system display name, relative to database_events
        db_series_names: database field to use as categories
        db_value_name: database field to use as values
        value_multiplier: float value used to multiply all db_value_name values by
        plot_average: boolean to set whether average data is calculated
        plot_freq: boolean to set whether frequency data should be calculated
        plot_series_per_system: boolean to set whether to calculate a series for each value found in db_display_name_relationship
        plot_average_choice: string set to either mean, median or both
        plot_average_over_time: boolean to set whether to calculate an over-time chart
        time_period: string to set the TimeUnit to use for the over-time chart
        num_hist_bins: integer value to set how many histogram bins to calculate
        exclude_constant_angle: boolean used to set whether to exclude CT constant angle acquisitions
        calculate_histograms: boolean used to set whether to calculate histogram data
        case_insensitive_categories: boolean to set whether to make categories case-insensitive
        chart_value_axis_title: string to use for the value axis label
        chart_category_name: string to use for the category label


    Params:
        exclude_constant_angle: boolean, default=False; set to true to exclude Constant Angle Acquisition data
        calculate_histograms: boolean, default=False; set to true to calculate histogram data

    Returns:
        A structure containing the required average, histogram and frequency data. This structure can include:
        series_names: a list of unique names of the db_series_names field present in database_events
        system_list: if plot_series_per_system then this contains a list of unique names of the db_display_name_relationship field present in database_events
        if plot_series_per_system is false then this contains a single value of 'All systems'
        summary: a list of lists: the top list has one entry per item in system_list. Each of these then contains a list of series_names items with the average and frequency data for that name and system
        histogram_data: a list of lists: the top list has one entry per item in system_list_entry. Each of these then contains histogram data for each item in series_names for that system
    """
    from django.db.models import (
        Avg,
        Count,
        Min,
        Max,
        FloatField,
        When,
        Case,
        Sum,
        IntegerField,
    )
    import pandas as pd
    import altair as alt

    # Exclude all zero value events from the calculations
    database_events = database_events.exclude(**{db_value_name: 0})

    return_structure = {}

    fields_to_include = {db_series_names}
    if plot_average or plot_average_over_time:
        fields_to_include.add(db_value_name)
    if plot_series_per_system:
        fields_to_include.add(db_display_name_relationship)
    if plot_average_over_time or plot_workload:
        fields_to_include.add("study_date")

    df_test = pd.DataFrame.from_records(database_events.values(*fields_to_include))

    if case_insensitive_categories:
        df_test[db_series_names] = df_test[db_series_names].str.lower()

    if plot_average or plot_average_over_time:
        df_test[db_value_name] = df_test[db_value_name].astype(float)
        if value_multiplier != 1.0:
            df_test[db_value_name] *= value_multiplier

    if plot_series_per_system:
        df_test.rename(columns={db_display_name_relationship:"x_ray_system_name"}, inplace=True)
    else:
        df_test.insert(0, "x_ray_system_name", "All systems")

    df_test.rename(columns={db_series_names:"data_point_name"}, inplace=True)

    alt.data_transformers.disable_max_rows()

    if plot_average_over_time or plot_workload:
        df_test["study_date"] = pd.to_datetime(df_test["study_date"])

    if plot_average_over_time:
        selection = alt.selection_multi(fields=["data_point_name"], bind="legend")

        return_structure["averageOverTimeChart"] = alt.Chart(df_test).mark_line(point=True).encode(
            x=alt.X(time_period + "(study_date):T", title="Study date", axis=alt.Axis(labelAngle=90)),
            y=alt.Y(plot_average_choice + "(" + db_value_name + ")", title=plot_average_choice.capitalize() + " " + chart_value_axis_title),
            color=alt.Color("data_point_name", legend=alt.Legend(title=chart_category_name, symbolLimit=250)),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
            tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                     alt.Tooltip("data_point_name", title="Name"),
                     alt.Tooltip(plot_average_choice + "(" + db_value_name + ")", format=".2f", title=plot_average_choice.capitalize())]
        ).facet(
            row=alt.Row("x_ray_system_name:N", title="")
        ).add_selection(
            selection
        ).resolve_axis(
            x="independent"
#        ).properties(
#            width="container",
#            height="container"
        ).interactive()

    if plot_workload:
        selection = alt.selection_multi(fields=["x_ray_system_name"], bind="legend")

        return_structure["workloadChart"] = alt.Chart(df_test).mark_bar().encode(
            y=alt.Y("day(study_date):O", title=""),
            x=alt.X("count(x_ray_system_name)", title=chart_category_name + " frequency"),
            color=alt.Color("x_ray_system_name", legend=alt.Legend(title="System", symbolLimit=250)),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
            tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                     alt.Tooltip("day(study_date)", title="Day"),
                     alt.Tooltip("count()", format=".0f", title="Frequency")]
        ).properties(
            width="container",
            height="container"
        ).facet(
            row=alt.Row("x_ray_system_name:N", title="")
        ).add_selection(
            selection
        ).interactive()

    if plot_average:
        # Create a plot with either the mean or median
        if plot_average_choice == "mean" or "median":
            selection = alt.selection_multi(fields=["x_ray_system_name"], bind="legend")

            return_structure["averageChart"] = alt.Chart(df_test).mark_bar().encode(
                row=alt.Row("data_point_name",
                            title="",
                            header=alt.Header(labelAngle=0, labelAlign="left")),
                y=alt.Y("x_ray_system_name", axis=alt.Axis(labels=False, title="")),
                x=alt.X(plot_average_choice + "(" + db_value_name + ")", title=plot_average_choice.capitalize() + " " + chart_value_axis_title),
                color=alt.Color("x_ray_system_name", legend=alt.Legend(title="System")),
                opacity=alt.condition(selection, alt.value(1.0), alt.value(0.05)),
                tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                         alt.Tooltip("data_point_name", title="Name"),
                         alt.Tooltip(plot_average_choice + "(" + db_value_name + ")", format=".2f", title=plot_average_choice.capitalize()),
                         alt.Tooltip("count(" + db_value_name + ")", format=".0f", title="Frequency")]
            ).add_selection(
                selection
            ).resolve_axis(
                x="independent"
            ).properties(
                width="container",
                height="container"
            ).interactive()

            # Temporary boxplot code as proof-of-concept
            # return_structure["averageChart"] = alt.Chart(df_test).mark_boxplot().encode(
            #     row=alt.Row("data_point_name",
            #                 title="",
            #                 header=alt.Header(labelAngle=0, labelAlign="left")),
            #     y=alt.Y("x_ray_system_name", axis=alt.Axis(labels=False, title="")),
            #     x=alt.X(db_value_name, title=chart_value_axis_title),
            #     color=alt.Color("x_ray_system_name", legend=alt.Legend(title="System"))
            # ).resolve_axis(
            #     x="independent"
            # ).interactive()

        # Create a plot with both the mean and median
        if plot_average_choice == "both":
            selection = alt.selection_multi(fields=["aggregate"], bind="legend")

            return_structure["averageChart"] = alt.Chart(df_test).transform_aggregate(
                mean="mean(" + db_value_name + ")",
                median="median(" + db_value_name + ")",
                groupby=["data_point_name", "x_ray_system_name"]
            ).transform_fold(
                ["mean", "median"],
                as_=["aggregate", "value"]
            ).mark_bar().encode(
                row=alt.Row("x_ray_system_name", title=""),
                x=alt.X("value:Q", title="", stack=None),
                y=alt.Y("data_point_name", axis=alt.Axis(title="")),
                color=alt.Color("aggregate:N", legend=alt.Legend(title="Average " + chart_value_axis_title)),
                opacity=alt.condition(selection, alt.value(1.0), alt.value(0.05)),
                tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                         alt.Tooltip("data_point_name", title="Name"),
                         alt.Tooltip("value:Q", format=".2f", title="Average")]
            ).add_selection(
                selection
            ).properties(
                width="container",
                height="container"
            ).interactive()

        if calculate_histograms:
            # Calculate histogram for each category and system.
            selection = alt.selection_multi(fields=["x_ray_system_name"], bind="legend")

            return_structure["histogramChart"] = alt.Chart(df_test).mark_bar().encode(
                row=alt.Row("data_point_name",
                            title="",
                            header=alt.Header(labelAngle=0, labelAlign="left")),
                x=alt.X(db_value_name, bin=alt.Bin(maxbins=num_hist_bins), title="Binned " + chart_value_axis_title),
                y=alt.Y("count()", title="Frequency", stack=None),
                color=alt.Color("x_ray_system_name", legend=alt.Legend(title="System")),
                opacity=alt.condition(selection, alt.value(1.0), alt.value(0.05)),
                tooltip=[alt.Tooltip("count()", title="Frequency"),
                         alt.Tooltip(db_value_name, bin=alt.Bin(maxbins=num_hist_bins), title="Bin range")]
            ).add_selection(
                selection
            ).resolve_scale(
                y="independent",
                x="independent"
            ).properties(
                width="container",
                height="container"
            ).interactive()

    if plot_freq:
        # Create a plot that shows the frequencies - used to be a pie chart.
        selection = alt.selection_multi(fields=["data_point_name"], bind="legend")

        return_structure["frequencyChart"] = alt.Chart(df_test).mark_bar().encode(
            x=alt.X("count(data_point_name):Q", title="Frequency"),
            y=alt.Y("x_ray_system_name", axis=alt.Axis(title="")),
            color=alt.Color("data_point_name", sort=alt.EncodingSortField("data_point_name", op="count", order="descending"), legend=alt.Legend(title=chart_category_name, symbolLimit=250)),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.05)),
            order=alt.Order("count(data_point_name)", sort="descending"),
            tooltip=[alt.Tooltip("x_ray_system_name", title="System"),
                     alt.Tooltip("data_point_name", title="Name"),
                     alt.Tooltip("count(data_point_name):Q", format=".0f", title="Frequency")]
        ).add_selection(
            selection
        ).properties(
            width="container",
            height="container"
        ).interactive()

    return return_structure


def average_chart_over_time_data(
    database_events,
    db_series_names,
    db_value_name,
    db_date_field,
    db_date_time_field,
    median_available,
    plot_average_choice,
    value_multiplier,
    time_period,
    case_insensitive_categories=False,
):
    """ This function calculates the data for an OpenREM Highcharts plot of average value per category over time. It
    uses the time_series function of the qsstats package to do this.

    Args:
        database_events: database events to use for the plot
        db_display_name_relationship: database table and field of x-ray system display name, relative to database_events
        db_series_names: database field to use as categories
        db_value_name: database field to use as values
        db_date_field: database field containing the event date, used to determine the first data on which there is data
        db_date_time_field: database field containing the event datetime used by QuerySetStats to calculate the average over time
        median_available: boolean to set whether the database can calculate median values
        plot_average_choice: string set to either mean, median or both
        value_multiplier: float value used to multiply all db_value_name values by
        time_period: string containing either days, weeks, months or years
        case_insensitive_categories: boolean to set whether to make categories case-insensitive

    Returns:
        A structure containing the required average data over time. The structure contains two items:
        series_names: a list of unique names of the db_series_names field present in database_events
        mean_over_time: the average value of each item in series_names at a series of time intervals determined by
        time_period
    """
    import datetime
    import qsstats
    from django.db.models import Min, Avg
    from remapp.models import Median

    # Exclude all zero value events from the calculations
    database_events = database_events.exclude(
        **{db_value_name: 0, db_value_name: None,}
    )

    return_structure = dict()

    if case_insensitive_categories:
        from django.db.models.functions import Lower

        database_events = database_events.annotate(
            db_series_names_to_use=Lower(db_series_names)
        )
    else:
        from django.db.models.functions import Concat

        database_events = database_events.annotate(
            db_series_names_to_use=Concat(db_series_names, None)
        )

    return_structure["series_names"] = list(
        database_events.values_list("db_series_names_to_use", flat=True)
        .distinct()
        .order_by("db_series_names_to_use")
    )

    start_date = database_events.aggregate(Min(db_date_field)).get(
        db_date_field + "__min"
    )
    today = datetime.date.today()

    if median_available and (
        plot_average_choice == "median" or plot_average_choice == "both"
    ):
        return_structure["median_over_time"] = [None] * len(
            return_structure["series_names"]
        )
    if plot_average_choice == "mean" or plot_average_choice == "both":
        return_structure["mean_over_time"] = [None] * len(
            return_structure["series_names"]
        )

    for i, series_name in enumerate(return_structure["series_names"]):
        subqs = database_events.filter(**{"db_series_names_to_use": series_name})

        if plot_average_choice == "mean" or plot_average_choice == "both":
            qss = qsstats.QuerySetStats(
                subqs,
                db_date_time_field,
                aggregate=Avg(db_value_name) * value_multiplier,
            )
            return_structure["mean_over_time"][i] = qss.time_series(
                start_date, today, interval=time_period
            )
        if median_available and (
            plot_average_choice == "median" or plot_average_choice == "both"
        ):
            qss = qsstats.QuerySetStats(
                subqs,
                db_date_time_field,
                aggregate=Median(db_value_name) * value_multiplier,
            )
            return_structure["median_over_time"][i] = qss.time_series(
                start_date, today, interval=time_period
            )

    return return_structure


def workload_chart_data(database_events):
    """ This function calculates the data for an OpenREM Highcharts plot of number of studies per day of the week. It
    also breaks down the numbers into how many were carried out during each of the 24 hours in that day. It uses the
    time_series function of the qsstats package to do this together with the study_workload_chart_time database field.

    Args:
        database_events: database events to use for the plot

    Returns:
        A structure containing the required breakdown of events per day of the week and per 24 hours in each day. The
        structure contains a single item:
        workload: a two-dimensional list [7][24] containing the number of study_workload_chart_time events that fall
        within each hour of each day of the week.
    """
    import datetime
    import qsstats

    return_structure = dict()

    return_structure["workload"] = [[0 for x in range(24)] for x in range(7)]
    for day in range(7):
        study_times_on_this_weekday = database_events.filter(
            study_date__week_day=day + 1
        ).values("study_workload_chart_time")

        if study_times_on_this_weekday:
            qss = qsstats.QuerySetStats(
                study_times_on_this_weekday, "study_workload_chart_time"
            )
            hourly_breakdown = qss.time_series(
                datetime.datetime(1900, 1, 1, 0, 0),
                datetime.datetime(1900, 1, 1, 23, 59),
                interval="hours",
            )
            for hour in range(24):
                return_structure["workload"][day][hour] = hourly_breakdown[hour][1]

    return return_structure


def scatter_plot_data(
    database_events,
    x_field,
    y_field,
    y_value_multiplier,
    plot_series_per_system,
    db_display_name_relationship,
):
    """ This function calculates the data for an OpenREM Highcharts plot of average value vs. a category, as well as a
    histogram of values for each category. It is also used for OpenREM Highcharts frequency plots.

    Args:
        database_events: database events to use for the plot
        x_field: database field containing data for the x-axis
        y_field: database field containing data for the y-axis
        y_value_multiplier: float value used to multiply all y_field values by
        plot_series_per_system: boolean to set whether to calculate a series for each value found in db_display_name_relationship
        db_display_name_relationship: database table and field of x-ray system display name, relative to database_events

    Returns:
        A structure containing the x-y data.
    """
    from django.db.models import Q

    # Exclude all zero value events from the calculations
    database_events = database_events.exclude(Q(**{x_field: 0}) | Q(**{y_field: 0}))

    return_structure = dict()

    if plot_series_per_system:
        return_structure["system_list"] = list(
            database_events.values_list(db_display_name_relationship, flat=True)
            .distinct()
            .order_by(db_display_name_relationship)
        )
    else:
        return_structure["system_list"] = ["All systems"]

    return_structure["scatterData"] = []
    if plot_series_per_system:
        for system in return_structure["system_list"]:
            return_structure["scatterData"].append(
                database_events.filter(
                    **{db_display_name_relationship: system}
                ).values_list(x_field, y_field)
            )
    else:
        return_structure["scatterData"].append(
            database_events.values_list(x_field, y_field)
        )

    for index in range(len(return_structure["scatterData"])):
        return_structure["scatterData"][index] = [
            [floatIfValue(i[0]), floatIfValue(i[1]) * y_value_multiplier]
            for i in return_structure["scatterData"][index]
        ]

    import numpy as np

    max_data = [0, 0]
    for index in range(len(return_structure["scatterData"])):
        current_max = np.amax(return_structure["scatterData"][index], 0).tolist()
        if current_max[0] > max_data[0]:
            max_data[0] = current_max[0]
        if current_max[1] > max_data[1]:
            max_data[1] = current_max[1]
    return_structure["maxXandY"] = max_data

    return return_structure


def floatIfValue(val):
    """ This function returns the float() of a the passed value if that value is a number; otherwise it returns the
    value 0.0.

    Args:
        val: any variable, but hopefully one that is a number

    Returns:
        float(val) if val is a number; otherwise 0.0
    """
    import numbers

    return float(val) if isinstance(val, numbers.Number) else 0.0


def floatIfValueNone(val):
    """ This function returns the float() of a the passed value if that value is a number; otherwise it returns None.

    Args:
        val: any variable, but hopefully one that is a number

    Returns:
        float(val) if val is a number; otherwise None
    """
    import numbers

    return float(val) if isinstance(val, numbers.Number) else None


def stringIfNone(val):
    """ This function returns the passed parameter if it is a string; otherwise it returns ''.

    Args:
        val: any variable, but hopefully one that is a string

    Returns:
        str if it is a string; otherwise ''
    """
    return val if isinstance(val, str) else ""
