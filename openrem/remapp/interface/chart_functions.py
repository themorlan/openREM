# pylint: disable=too-many-lines
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

import os
import math
import base64
from builtins import range  # pylint: disable=redefined-builtin
from datetime import datetime
import textwrap
import warnings

from django.conf import settings
from django.utils.translation import gettext as _
import numpy as np
import pandas as pd
import matplotlib.cm
import matplotlib.colors
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from scipy import stats


def global_config(
    filename,
    height_multiplier=1.0,
    height=1080,
    width=1920,
):
    """
    Creates a Plotly global configuration dictionary. The parameters all relate
    to the chart bitmap that can be saved by the user.

    :param filename: string containing the file name to use if the user saves the chart as a graphic file
    :param height_multiplier: floating point value used to scale the chart height
    :param height: int value for the height of the chart graphic file
    :param width: int value for the width of the chart graphic file
    :return: a dictionary of Plotly options
    """
    return {
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": height * height_multiplier,
            "width": width,
            "scale": 1,
        },
        "displaylogo": False,
        "scrollZoom": True,
    }


def create_dataframe(
    database_events,
    field_dict,
    data_point_name_lowercase=None,
    data_point_name_remove_whitespace_padding=None,
    data_point_value_multipliers=None,
    char_wrap=500,
    uid=None,
):
    """
    Creates a Pandas DataFrame from the supplied database records.
    names fields are made categorical to save system memory
    Any missing (na) values in names fields are set to Blank

    :param database_events: the database events
    :param field_dict: a dictionary of lists, each containing database field names to include in the DataFrame. The
                       dictionary should include "names", "values", "dates", "times" and optionally "system" items
    :param data_point_name_lowercase: boolean flag to determine whether to make all "names" field values lower case
    :param data_point_name_remove_whitespace_padding: boolean flag to determine whether to strip whitespace
    :param data_point_value_multipliers: list of float valuse to multiply each "values" field value by
    :param uid: string containing database field name which contains a unique identifier for each record
    :return: a Pandas DataFrame with a column per required field
    """
    start = None
    if settings.DEBUG:
        start = datetime.now()

    fields_to_include = set()
    if uid:
        fields_to_include.add(uid)

    fields_to_include.update(field_dict["names"])
    fields_to_include.update(field_dict["values"])
    fields_to_include.update(field_dict["dates"])
    fields_to_include.update(field_dict["times"])
    fields_to_include.update(field_dict["system"])

    start = None
    if settings.DEBUG:
        start = datetime.now()

    # NOTE: I am not excluding zero-value events from the calculations (zero DLP or zero CTDI)

    # The "order_by()" in the command below removes the custom ordering on the query set that is used to order things
    # correctly on the filtered page tables. This ordering isn't required for the DataFrame; removing it speeds up
    # the DataFrame.from_records command.
    df = pd.DataFrame.from_records(
        data=database_events.order_by().values_list(
            *fields_to_include
        ),  # values_list uses less memory than values
        columns=fields_to_include,  # need to specify the column names as we're now using values_list
        coerce_float=True,  # force Decimal to float - saves doing a type conversion later
    )

    if settings.DEBUG:
        print(f"Initial Dataframe created from records in {datetime.now() - start}")
        start = datetime.now()
        print("Initial DataFrame info, including memory use, is:")
        df.info()

    if uid:
        df[uid] = df[uid].astype("UInt32")

    # Replace any NaN values in the names columns with "Blank"
    df[field_dict["names"]] = df[field_dict["names"]].apply(lambda x: x.fillna("Blank"))

    # Make names column values lowercase if required
    if data_point_name_lowercase:
        df[field_dict["names"]] = df[field_dict["names"]].apply(lambda x: x.str.lower())

    # Strip whitespace from the beginning and end of any names column values
    # Also replace multiple spaces with a single space
    if data_point_name_remove_whitespace_padding:
        df[field_dict["names"]] = df[field_dict["names"]].apply(
            lambda x: x.str.strip().replace("\s+", " ", regex=True)
        )

    # Make the names columns all "category" type - this saves memory. Must be done after the above, as the string
    # replacement lines revert the columns back to "object"
    df[field_dict["names"]] = df[field_dict["names"]].astype("category")

    # Rename the "system" column to "x_ray_system_name" if it is present
    if field_dict["system"]:
        df.rename(columns={field_dict["system"][0]: "x_ray_system_name"}, inplace=True)
        df["x_ray_system_name"] = df["x_ray_system_name"].astype("category")
        df.sort_values(by="x_ray_system_name", inplace=True)
    # Else create the "x_ray_system_name" column populated with a single "All systems" category
    else:
        df["x_ray_system_name"] = pd.Categorical(np.full(len(df.index), "All systems"))

    # Loop through each value field, multiplying the values by the corresponding multiplier
    for idx, value_field in enumerate(field_dict["values"]):
        if data_point_value_multipliers:
            df[value_field] *= data_point_value_multipliers[idx]
            df[value_field] = df[value_field].astype("float32")

    # Convert each date field to a pd datetime using a specific date format
    for date_field in field_dict["dates"]:
        df[date_field] = pd.to_datetime(df[date_field], format="%Y-%m-%d")

    # Character wrap the system and name fields
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        df.update(
            df["x_ray_system_name"].apply(
                lambda x: (textwrap.fill(x, char_wrap)).replace("\n", "<br>")
            )
        )

    df["x_ray_system_name"] = df["x_ray_system_name"].astype("category")
    for field in field_dict["names"]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            df.update(
                df[field].apply(
                    lambda x: (textwrap.fill(x, char_wrap)).replace("\n", "<br>")
                )
            )
        df[field] = df[field].astype("category")

    if settings.DEBUG:
        print(
            f"Dataframe fillna, lower case, whitespace stripping etc took {datetime.now() - start}"
        )
        print("DataFrame info after processing, including memory use, is:")
        df.info()

    return df


def create_dataframe_time_series(
    df,
    df_name_col,
    df_value_col,
    df_date_col="study_date",
    time_period="M",
    average_choices=None,
    group_by_physician=None,
):
    """
    Creates a Pandas DataFrame time series of average values grouped by x_ray_system_name and df_name_col

    :param df: the Pandas DataFrame containing the raw data
    :param df_name_col: string containing the DataFrame columnn name used to group the data
    :param df_value_col: string containing the DataFrame column containing the values to be averaged
    :param df_date_col: string containing the DataFrame column containing the dates
    :param time_period: string containing the time period to average over; "A" (years), "Q" (quarters), "M" (months),
                        "W" (weeks), "D" (days)
    :param average_choices: list of strings containing one or both of "mean" and "median"
    :param group_by_physician: boolean flag to set whether to group by physician
    :return: Pandas DataFrame containing the time series of average values grouped by system and name
    """
    if average_choices is None:
        average_choices = ["mean"]

    group_by_column = "x_ray_system_name"
    if group_by_physician:
        group_by_column = "performing_physician_name"

    df_time_series = (
        df.set_index(df_date_col)
        .groupby([group_by_column, df_name_col, pd.Grouper(freq=time_period)])
        .agg({df_value_col: average_choices})
    )
    df_time_series.columns = [s + df_value_col for s in average_choices]
    df_time_series = df_time_series.reset_index()
    return df_time_series


def create_dataframe_weekdays(df, df_name_col, df_date_col="study_date"):
    """
    Creates a Pandas DataFrame of the number of events in each day of the
    week, and in hour of that day.

    :param df: Pandas DataFrame containing the raw data; it must have a "study_time" and "x_ray_system_name" column
    :param df_name_col: string containing the df column name to group the results by
    :param df_date_col: string containing the df column name containing dates
    :return: Pandas DataFrame containing the number of studies per day and hour grouped by name
    """
    start = None
    if settings.DEBUG:
        start = datetime.now()

    df["weekday"] = pd.Categorical(pd.DatetimeIndex(df[df_date_col]).day_name())
    df["hour"] = df["study_time"].apply(lambda row: row.hour).astype("int8")

    df_time_series = (
        df.groupby(["x_ray_system_name", "weekday", "hour"])
        .agg({df_name_col: "count"})
        .reset_index()
    )

    if settings.DEBUG:
        print(f"Weekday and hour dataframe created in {datetime.now() - start}")

    return df_time_series


def create_dataframe_aggregates(df, df_name_cols, df_agg_col, stats_to_use=None):
    """
    Creates a Pandas DataFrame with the specified statistics (mean, median, count, for example) grouped by
    x-ray system name and by the list of provided df_name_cols.

    :param df: Pandas DataFrame containing the raw data; it must have an "x_ray_system_name" column
    :param df_name_cols: list of strings representing the DataFrame column names to group by
    :param df_agg_col: string containing the DataFrame column over which to calculate the statistics
    :param stats_to_use: list of strings containing the statistics to calculate, such as "mean", "median", "count"
    :return: Pandas DataFrame containing the grouped aggregate data
    """
    start = None
    if settings.DEBUG:
        start = datetime.now()

    # Make it possible to have multiple value cols (DLP, CTDI, for example)
    if stats_to_use is None:
        stats_to_use = ["count"]

    groupby_cols = ["x_ray_system_name"] + df_name_cols
    grouped_df = df.groupby(groupby_cols).agg({df_agg_col: stats_to_use})
    grouped_df.columns = grouped_df.columns.droplevel(level=0)
    grouped_df = grouped_df.reset_index()

    if settings.DEBUG:
        print(f"Aggregated dataframe created in {datetime.now() - start}")

    return grouped_df


def plotly_set_default_theme(theme_name):
    """
    A short method to set the plotly chart theme

    :param theme_name: the name of the theme
    :return:
    """
    pio.templates.default = theme_name


def calculate_colour_sequence(scale_name="RdYlBu", n_colours=10):
    """
    Calculates a sequence of n_colours from the matplotlib colourmap scale_name

    :param scale_name: string containing the name of the matplotlib colour scale to use
    :param n_colours: int representing the number of colours required
    :return: list of hexadecimal colours from a matplotlib colormap
    """
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


def empty_dataframe_msg(params=None):
    """
    Returns a string containing an HTML DIV with a message warning that the DataFrame is empty

    :param params: parameters which may contain a custom_msg_line
    :return: string containing an html div with the empty DataFrame message
    """
    msg_line = ""
    if params:
        if "custom_msg_line" in params:
            msg_line = params["custom_msg_line"]

    msg = "<div class='alert alert-warning' role='alert'>"
    msg += _("No data left after excluding missing values.")
    msg += msg_line
    msg += "</div>"

    return msg


def failed_chart_message_div(custom_msg_line, e):
    """
    Returns a string containing an HTML DIV with a failed chart message

    :param custom_msg_line: string containing a custom line to add to the message
    :param e: Python error object
    :return: string containing the message in an HTML DIV
    """
    msg = "<div class='alert alert-warning' role='alert'>"
    if settings.DEBUG:
        msg += custom_msg_line
        msg += "<p>Error is:</p>"
        msg += "<pre>" + e.args[0].replace("\n", "<br>") + "</pre>"
    else:
        msg += custom_msg_line
    msg += "</div>"
    return msg


def csv_data_barchart(fig, params):
    """
    Calculates a Pandas DataFrame containing chart data to be used for csv download

    :param fig: Plotly figure containing the data to extract
    :param params: a dictionary of parameters
    :param params["df_name_col"]: (string) DataFrame column containing categories
    :param params["name_axis_title"]: (string) title for the name data
    :param params["value_axis_title"]: (string) title for the value data
    :param params["facet_col"]: (string) DataFrame column used to split data into subgroups
    :return: DataFrame containing the data for download
    """
    fig_data_dict = fig.to_dict()["data"]

    if params["df_name_col"] != "performing_physician_name":
        df = pd.DataFrame(
            data=fig_data_dict[0]["x"], columns=[params["name_axis_title"]]
        )
        df = df.replace("<br>", " ", regex=True)

        for data_set in fig_data_dict:
            new_col_df = pd.DataFrame(
                data=list(zip(data_set["y"], [x[1] for x in data_set["customdata"]])),
                columns=[
                    data_set["name"]
                    + " "
                    + params["value_axis_title"]
                    .replace("<sup>2</sup>", "^2")
                    .replace("<sub>vol</sub>", "vol"),
                    "Frequency",
                ],  # pylint: disable=line-too-long
            )
            df = pd.concat([df, new_col_df], axis=1)

        return df

    else:
        df = pd.DataFrame(
            data=fig_data_dict[0]["x"], columns=[params["name_axis_title"]]
        )
        df = df.replace("<br>", " ", regex=True)

        for data_set in fig_data_dict:
            series_name = (
                data_set["hovertemplate"]
                .split(params["facet_col"] + "=")[1]
                .split("<br>Performing")[0]
            ).replace("<br>", " ")
            new_col_df = pd.DataFrame(
                data=list(zip(data_set["y"], [x[1] for x in data_set["customdata"]])),
                columns=[
                    data_set["name"]
                    + " "
                    + series_name
                    + " "
                    + params["value_axis_title"]
                    .replace("<sup>2</sup>", "^2")
                    .replace("<sub>vol</sub>", "vol"),
                    "Frequency",
                ],  # pylint: disable=line-too-long
            )
            df = pd.concat([df, new_col_df], axis=1)
        return df


def csv_data_frequency(fig, params):
    """
    Calculates a Pandas DataFrame containing chart data to be used for csv download

    :param fig: Plotly figure containing the data to extract
    :param params: a dictionary of parameters; must include "x_axis_title"
    :return: DataFrame containing the data for download
    """
    fig_data_dict = fig.to_dict()["data"]

    if params["df_name_col"] != "performing_physician_name":
        df = pd.DataFrame(data=fig_data_dict[0]["x"], columns=[params["x_axis_title"]])
        for data_set in fig_data_dict:
            df = pd.concat(
                [df, pd.DataFrame(data=data_set["y"], columns=[data_set["name"]])],
                axis=1,
            )

        return df

    else:
        df = pd.DataFrame(data=fig_data_dict[0]["x"], columns=[params["x_axis_title"]])
        for data_set in fig_data_dict:
            series_name = data_set["customdata"][0][1]
            new_col_df = pd.DataFrame(
                data=data_set["y"],  # pylint: disable=line-too-long
                columns=[
                    data_set["name"] + " " + series_name + " frequency",
                ],  # pylint: disable=line-too-long
            )
            df = pd.concat([df, new_col_df], axis=1)
        return df


def calc_facet_rows_and_height(df, facet_col_name, facet_col_wrap):
    """
    Calculates the required total chart height and the number of facet rows. Each row has a hard-coded height
    of 500 pixels.

    :param df: Pandas DataFrame containing the data
    :param facet_col_name: string containing the DataFrame column name containing the facet names
    :param facet_col_wrap: int representing the number of subplots to have on each row
    :return: two-element list containing the chart height in pixels (int) and the number of facet rows (int)
    """
    n_facet_rows = math.ceil(len(df[facet_col_name].unique()) / facet_col_wrap)
    chart_height = n_facet_rows * 750
    if chart_height < 750:
        chart_height = 750
    return chart_height, n_facet_rows


def save_fig_as_html_div(fig, filename, active=settings.SAVE_CHARTS_AS_HTML):
    """
    Saves the Plotly figure as an HTML file containing a single DIV. The file is saved on the OpenREM server in
    MEDIA_ROOT\charts\yyyy\mm\dd\. Viewing the saved file requires an active internet connection as the Plotly
    JavaScript library is not included in the file.

    This method is not currently accessible to an OpenREM user or administrator - it is present to assist developers
    when producing example charts for the OpenREM documentation. It must be manually activated by setting active=True
    in the method definition.

    Args:
        fig: a Plotly figure
        filename: (string )the filename to use
        active: (boolean) to set whether to save the figure
    """
    if active:
        datestamp = datetime.now()
        path = os.path.join(
            settings.MEDIA_ROOT, "charts", datestamp.strftime("%Y/%m/%d")
        )
        os.makedirs(
            os.path.join(settings.MEDIA_ROOT, "charts", datestamp.strftime("%Y/%m/%d")),
            exist_ok=True,
        )
        fig.write_html(
            os.path.join(path, filename + ".html"),
            include_plotlyjs="cdn",
            full_html=False,
        )


def plotly_boxplot(
    df,
    params,
):
    """
    Produce a plotly boxplot

    :param df: Pandas DataFrame containing the data
    :param params: a dictionary of parameters
    :param params["df_value_col"]: (string) DataFrame column containing values
    :param params["value_axis_title"]: (string) x-axis title
    :param params["df_name_col"]: (string) DataFrame column containing categories
    :param params["name_axis_title"]: (string) y-axis title
    :param params["df_facet_col"]: (string) DataFrame column used to create subplots
    :param params["df_facet_col_wrap"]: (int) number of subplots per row
    :param params["sorting_choice"]: 2-element list. [0] sets sort direction, [1] used to determine which field to sort
    :param params["colourmap"]: (string) colourmap to use
    :param params["return_as_dict"]: (boolean) flag to trigger return as a dictionary rather than a HTML DIV
    :return: Plotly figure embedded in an HTML DIV; or Plotly figure as a dictionary (if params["return_as_dict"] is
             True); or an error message embedded in an HTML DIV if there was a ValueError when calculating the figure
    """
    chart_height = 500
    n_facet_rows = 1

    try:
        # Drop any rows with nan values in the df_value column
        df = df.dropna(subset=[params["df_value_col"]])
        if df.empty:
            return empty_dataframe_msg(params)

        if params["facet_col"]:
            chart_height, n_facet_rows = calc_facet_rows_and_height(
                df, params["facet_col"], params["facet_col_wrap"]
            )

        sort_ascending = True
        if params["sorting_choice"][0] == 0:
            sort_ascending = False

        sorting_categories = None
        if params["sorting_choice"][1].lower() == "name":
            sorting_categories = {
                params["df_name_col"]: (
                    df.sort_values(by=params["df_name_col"], ascending=sort_ascending)[
                        params["df_name_col"]
                    ]
                )
                .unique()
                .tolist()
            }
            sorting_categories["x_ray_system_name"] = (
                (
                    df.sort_values(by="x_ray_system_name", ascending=sort_ascending)[
                        "x_ray_system_name"
                    ]
                )
                .unique()
                .tolist()
            )
            if params["facet_col"]:
                sorting_categories[params["facet_col"]] = (
                    (
                        df.sort_values(
                            by=params["facet_col"], ascending=sort_ascending
                        )[params["facet_col"]]
                    )
                    .unique()
                    .tolist()
                )
        elif params["sorting_choice"][1].lower() == "frequency":
            sorting_categories = {
                params["df_name_col"]: df.groupby(params["df_name_col"])
                .agg(freq=(params["df_name_col"], "count"))
                .sort_values(by="freq", ascending=sort_ascending)
                .reset_index()[params["df_name_col"]]
                .tolist()
            }
            sorting_categories["x_ray_system_name"] = (
                df.groupby("x_ray_system_name")
                .agg(freq=("x_ray_system_name", "count"))
                .sort_values(by="freq", ascending=sort_ascending)
                .reset_index()["x_ray_system_name"]
                .tolist()
            )
            if params["facet_col"]:
                sorting_categories[params["facet_col"]] = (
                    df.groupby(params["facet_col"])
                    .agg(freq=(params["facet_col"], "count"))
                    .sort_values(by="freq", ascending=sort_ascending)
                    .reset_index()[params["facet_col"]]
                    .tolist()
                )
        else:
            sorting_categories = {
                params["df_name_col"]: df.groupby(params["df_name_col"])
                .agg(mean=(params["df_value_col"], "mean"))
                .sort_values(by="mean", ascending=sort_ascending)
                .reset_index()[params["df_name_col"]]
                .tolist()
            }
            sorting_categories["x_ray_system_name"] = (
                df.groupby("x_ray_system_name")
                .agg(mean=(params["df_value_col"], "mean"))
                .sort_values(by="mean", ascending=sort_ascending)
                .reset_index()["x_ray_system_name"]
                .tolist()
            )
            if params["facet_col"]:
                sorting_categories[params["facet_col"]] = (
                    df.groupby(params["facet_col"])
                    .agg(mean=(params["df_value_col"], "mean"))
                    .sort_values(by="mean", ascending=sort_ascending)
                    .reset_index()[params["facet_col"]]
                    .tolist()
                )

        n_colours = len(df.x_ray_system_name.unique())
        colour_sequence = calculate_colour_sequence(params["colourmap"], n_colours)

        fig = px.box(
            df,
            x=params["df_name_col"],
            y=params["df_value_col"],
            facet_col=params["facet_col"],
            facet_col_wrap=params["facet_col_wrap"],
            facet_row_spacing=0.50 / n_facet_rows,
            color=df["x_ray_system_name"],
            labels={
                params["df_value_col"]: params["value_axis_title"],
                params["df_name_col"]: params["name_axis_title"],
                "x_ray_system_name": "System",
            },
            color_discrete_sequence=colour_sequence,
            category_orders=sorting_categories,
            height=chart_height,
        )

        fig.update_traces(quartilemethod="exclusive")

        fig.update_xaxes(
            tickson="boundaries",
            ticks="outside",
            ticklen=5,
            showticklabels=True,
            title=params["name_axis_title"],
        )
        fig.update_yaxes(showticklabels=True, matches=None)

        fig.update_layout(legend_title_text="System")

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        save_fig_as_html_div(fig, params["filename"])

        if params["return_as_dict"]:
            return fig.to_dict()
        else:
            return plot(
                fig,
                output_type="div",
                include_plotlyjs=False,
                config=global_config(
                    params["filename"], height_multiplier=chart_height / 500.0
                ),
            )

    except ValueError as e:
        return failed_chart_message_div(
            "Could not resolve chart. Try filtering the data to reduce the number of systems.",
            e,
        )


def create_freq_sorted_category_list(df, df_name_col, sorting):
    """
    Create a sorted list of categories for frequency charts. Makes use of  Pandas DataFrame sort_values
    (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html).

    sorting[0] sets sort direction

    sorting[1] used to determine field to sort on: "name" sorts by df_name_col; otherwise sorted by "x_ray_system_name"

    :param df: Pandas DataFrame containing the data
    :param df_name_col: DataFrame column containing the category names
    :param sorting: 2-element list. [0] sets sort direction, [1] used to determine which field to sort on
    :return: dictionary with key df_name_col and a list of sorted categories as the value
    """
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
    """
    Create a sorted list of categories for scatter and over-time charts. The data is grouped by df_name_col and the
    mean and count calculated for each. The grouped DataFrame is then sorted according to the provided sorting.
    Makes use of  Pandas DataFrame sort_values
    (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html).

    sorting[0] sets sort direction

    sorting[1] used to determine sort order: "name" sorts by df_name_col; otherwise sorted by "x_ray_system_name"

    :param df: Pandas DataFrame containing the data
    :param df_name_col: DataFrame column containing the category names. Used to group the data
    :param df_value_col: DataFrame column containing values to count and calculate the mean
    :param sorting: 2-element list. [0] sets sort direction, [1] used to determine which field to sort on
    :return: dictionary with key df_name_col and a list of sorted categories as the value
    """
    # Calculate the required aggregates for creating a list of categories for sorting
    grouped_df = df.groupby(df_name_col)

    if sorting[1].lower() == "name":
        sort_by = df_name_col
        grouped_df = df
    elif sorting[1].lower() == "frequency":
        sort_by = "count"
    elif sorting[1].lower() == "mean":
        sort_by = "mean"
    else:
        sort_by = "median"

    if sort_by in ["count", "mean", "median"]:
        grouped_df = grouped_df.agg({df_value_col: [sort_by]})
        grouped_df.columns = grouped_df.columns.droplevel(level=0)

    grouped_df = grouped_df.reset_index()

    ascending_order = True
    if sorting[0] == 0:
        ascending_order = False

    categories_sorted = {
        df_name_col: list(
            grouped_df.sort_values(by=sort_by, ascending=ascending_order)[df_name_col]
        )
    }

    return categories_sorted


def plotly_barchart(
    df,
    params,
    csv_name="OpenREM chart data.csv",
):
    """
    Create a plotly bar chart

    :param df: Pandas DataFrame containing the data
    :param params: a dictionary of parameters
    :param params["average_choice"]: (string) DataFrame column containing values ("mean" or "median")
    :param params["value_axis_title"]: (string) y-axis title
    :param params["df_name_col"]: (string) DataFrame column containing categories
    :param params["name_axis_title"]: (string) x-axis title
    :param params["facet_col"]: (string) DataFrame column used to create subplots
    :param params["facet_col_wrap"]: (int) number of subplots per row
    :param params["sorting_choice"]: 2-element list. [0] sets sort direction, [1] used to determine which field to sort
    :param params["colourmap"]: (string) colourmap to use
    :param params["return_as_dict"]: (boolean) flag to trigger return as a dictionary rather than a HTML DIV
    :param params["filename"]: (string) default filename to use for plot bitmap export
    :param csv_name: (string) default filename to use for plot csv export
    :return: Plotly figure embedded in an HTML DIV; or Plotly figure as a dictionary (if params["return_as_dict"] is
             True); or an error message embedded in an HTML DIV if there was a ValueError when calculating the figure
    """
    if df.empty:
        return empty_dataframe_msg(params), None

    chart_height = 500
    n_facet_rows = 1

    if params["facet_col"]:
        chart_height, n_facet_rows = calc_facet_rows_and_height(
            df, params["facet_col"], params["facet_col_wrap"]
        )

    sort_ascending = True
    if params["sorting_choice"][0] == 0:
        sort_ascending = False

    sorting_categories = None
    if params["sorting_choice"][1].lower() == "name":
        sorting_categories = {
            params["df_name_col"]: (
                df.sort_values(by=params["df_name_col"], ascending=sort_ascending)[
                    params["df_name_col"]
                ]
            )
            .unique()
            .tolist()
        }
        sorting_categories["x_ray_system_name"] = (
            (
                df.sort_values(by="x_ray_system_name", ascending=sort_ascending)[
                    "x_ray_system_name"
                ]
            )
            .unique()
            .tolist()
        )
        if params["facet_col"]:
            sorting_categories[params["facet_col"]] = (
                (
                    df.sort_values(by=params["facet_col"], ascending=sort_ascending)[
                        params["facet_col"]
                    ]
                )
                .unique()
                .tolist()
            )
    elif params["sorting_choice"][1].lower() == "frequency":
        sorting_categories = {
            params["df_name_col"]: df.groupby(params["df_name_col"])
            .agg({"count": "sum"})
            .sort_values(by="count", ascending=sort_ascending)
            .reset_index()[params["df_name_col"]]
            .tolist()
        }
        sorting_categories["x_ray_system_name"] = (
            df.groupby("x_ray_system_name")
            .agg({"count": "sum"})
            .sort_values(by="count", ascending=sort_ascending)
            .reset_index()["x_ray_system_name"]
            .tolist()
        )
        if params["facet_col"]:
            sorting_categories[params["facet_col"]] = (
                df.groupby(params["facet_col"])
                .agg({"count": "sum"})
                .sort_values(by="count", ascending=sort_ascending)
                .reset_index()[params["facet_col"]]
                .tolist()
            )
    else:
        sorting_categories = {
            params["df_name_col"]: df.groupby(params["df_name_col"])
            .agg({params["average_choice"]: "mean"})
            .sort_values(by=params["average_choice"], ascending=sort_ascending)
            .reset_index()[params["df_name_col"]]
            .tolist()
        }
        sorting_categories["x_ray_system_name"] = (
            df.groupby("x_ray_system_name")
            .agg({params["average_choice"]: "mean"})
            .sort_values(by=params["average_choice"], ascending=sort_ascending)
            .reset_index()["x_ray_system_name"]
            .tolist()
        )
        if params["facet_col"]:
            sorting_categories[params["facet_col"]] = (
                df.groupby(params["facet_col"])
                .agg({params["average_choice"]: "mean"})
                .sort_values(by=params["average_choice"], ascending=sort_ascending)
                .reset_index()[params["facet_col"]]
                .tolist()
            )

    n_colours = len(df.x_ray_system_name.unique())
    colour_sequence = calculate_colour_sequence(params["colourmap"], n_colours)

    fig = px.bar(
        df,
        x=params["df_name_col"],
        y=params["average_choice"],
        color="x_ray_system_name",
        barmode="group",
        facet_col=params["facet_col"],
        facet_col_wrap=params["facet_col_wrap"],
        facet_row_spacing=0.50 / n_facet_rows,
        labels={
            params["average_choice"]: params["value_axis_title"],
            params["df_name_col"]: params["name_axis_title"],
            "x_ray_system_name": "System",
            "count": "Frequency",
        },
        category_orders=sorting_categories,
        color_discrete_sequence=colour_sequence,
        hover_name="x_ray_system_name",
        hover_data={
            "x_ray_system_name": False,
            params["average_choice"]: ":.2f",
            "count": ":.0d",
        },
        height=chart_height,
    )

    fig.update_xaxes(
        tickson="boundaries", ticks="outside", ticklen=5, showticklabels=True
    )
    fig.update_yaxes(showticklabels=True, matches=None)

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    save_fig_as_html_div(fig, params["filename"])

    if params["return_as_dict"]:
        return fig.to_dict(), None
    else:
        csv_data = download_link(
            csv_data_barchart(fig, params),
            csv_name,
        )

        return (
            plot(
                fig,
                output_type="div",
                include_plotlyjs=False,
                config=global_config(
                    params["filename"], height_multiplier=chart_height / 500.0
                ),
            ),
            csv_data,
        )


def plotly_histogram_barchart(
    df,
    params,
):
    """
    Create a plotly histogram bar chart

    :param df: Pandas DataFrame containing the data
    :param params: a dictionary of parameters
    :param  params["df_value_col"]: (string) DataFrame column containing values
    :param  params["value_axis_title"]: (string) y-axis title
    :param  params["df_facet_col"]: (string) DataFrame column used to create subplots
    :param  params["df_category_name_list"]: string list of each category name
    :param  params["df_facet_col_wrap"]: (int) number of subplots per row
    :param  params["n_bins"]: (int) number of hisgogram bins to use
    :param  params["colourmap"]: (string) colourmap to use
    :param params["sorting_choice"]: 2-element list. [0] sets sort direction, [1] used to determine which field to sort
    :param  params["global_max_min"]: (boolean) flag to calculate global max and min or per-subplot max and min
    :param  params["legend_title"]: (string) legend title
    :param  params["return_as_dict"]: (boolean) flag to trigger return as a dictionary rather than a HTML DIV
    :param  params["filename"]: (string) default filename to use for plot bitmap export
    :return: Plotly figure embedded in an HTML DIV; or Plotly figure as a dictionary (if params["return_as_dict"] is
             True); or an error message embedded in an HTML DIV if there was a ValueError when calculating the figure
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    if df.empty:
        return empty_dataframe_msg(params)

    sort_ascending = True
    if params["sorting_choice"][0] == 0:
        sort_ascending = False

    df_facet_list = None
    df_category_list = None
    if params["sorting_choice"][1].lower() == "name":
        df_facet_list = (
            (
                df.sort_values(by=params["df_facet_col"], ascending=sort_ascending)[
                    params["df_facet_col"]
                ]
            )
            .unique()
            .tolist()
        )
        df_category_list = (
            (
                df.sort_values(by=params["df_category_col"], ascending=sort_ascending)[
                    params["df_category_col"]
                ]
            )
            .unique()
            .tolist()
        )
    elif params["sorting_choice"][1].lower() == "frequency":
        df_facet_list = (
            df.groupby(params["df_facet_col"])
            .agg(freq=(params["df_facet_col"], "count"))
            .sort_values(by="freq", ascending=sort_ascending)
            .reset_index()[params["df_facet_col"]]
            .tolist()
        )
        df_category_list = (
            df.groupby(params["df_category_col"])
            .agg(freq=(params["df_category_col"], "count"))
            .sort_values(by="freq", ascending=sort_ascending)
            .reset_index()[params["df_category_col"]]
            .tolist()
        )
    else:
        df_facet_list = (
            df.groupby(params["df_facet_col"])
            .agg(mean=(params["df_value_col"], "mean"))
            .sort_values(by="mean", ascending=sort_ascending)
            .reset_index()[params["df_facet_col"]]
            .tolist()
        )
        df_category_list = (
            df.groupby(params["df_category_col"])
            .agg(mean=(params["df_value_col"], "mean"))
            .sort_values(by="mean", ascending=sort_ascending)
            .reset_index()[params["df_category_col"]]
            .tolist()
        )

    chart_height, n_facet_rows = calc_facet_rows_and_height(
        df, params["df_facet_col"], params["facet_col_wrap"]
    )

    n_colours = len(df[params["df_category_col"]].unique())
    colour_sequence = calculate_colour_sequence(params["colourmap"], n_colours)

    bins = None
    mid_bins = None
    bin_labels = None
    if params["global_max_min"]:
        bin_labels, bins, mid_bins = calc_histogram_bin_data(
            df, params["df_value_col"], n_bins=params["n_bins"]
        )

    try:
        n_cols = params["facet_col_wrap"]
        if len(df_facet_list) < n_cols:
            n_cols = len(df_facet_list)

        fig = make_subplots(
            rows=n_facet_rows,
            cols=n_cols,
            vertical_spacing=0.40 / n_facet_rows,
        )

        current_row = 1
        current_col = 1
        current_facet = 0
        category_names = []

        for facet_name in df_facet_list:
            facet_subset = df[df[params["df_facet_col"]] == facet_name].dropna(
                subset=[params["df_value_col"]]
            )

            # If the subset is empty then skip to the next facet
            if facet_subset.empty:
                continue

            if not params["global_max_min"]:
                bin_labels, bins, mid_bins = calc_histogram_bin_data(
                    facet_subset, params["df_value_col"], n_bins=params["n_bins"]
                )

            for category_name in df_category_list:
                category_subset = facet_subset[
                    facet_subset[params["df_category_col"]] == category_name
                ].dropna(subset=[params["df_value_col"]])

                # If the subset is empty then skip to the next category
                if category_subset.empty:
                    continue

                if category_name in category_names:
                    show_legend = False
                else:
                    show_legend = True
                    category_names.append(category_name)

                category_idx = category_names.index(category_name)

                histogram_data = np.histogram(
                    category_subset[params["df_value_col"]].values, bins=bins
                )

                trace = go.Bar(
                    x=mid_bins,
                    y=histogram_data[0],
                    name=category_name,
                    marker_color=colour_sequence[category_idx],
                    legendgroup=category_idx,
                    showlegend=show_legend,
                    customdata=bin_labels,
                    hovertemplate=f"<b>{facet_name}</b><br>"
                    + f"{category_name}<br>"
                    + "Frequency: %{y:.0d}<br>"
                    + "Bin range: %{customdata}<br>"
                    + "Mid-bin: %{x:.2f}<br>"
                    + "<extra></extra>",
                )

                fig.append_trace(trace, row=current_row, col=current_col)

            fig.update_xaxes(
                title_text=facet_name + " " + params["value_axis_title"],
                tickvals=bins,
                ticks="outside",
                ticklen=5,
                row=current_row,
                col=current_col,
            )

            if current_col == 1:
                fig.update_yaxes(
                    title_text="Frequency", row=current_row, col=current_col
                )

            current_facet += 1
            current_col += 1
            if current_col > n_cols:
                current_row += 1
                current_col = 1

        layout = go.Layout(height=chart_height)

        fig.update_layout(layout)
        fig.update_layout(legend_title_text=params["legend_title"])

        save_fig_as_html_div(fig, params["filename"])

        if params["return_as_dict"]:
            return fig.to_dict()
        else:
            return plot(
                fig,
                output_type="div",
                include_plotlyjs=False,
                config=global_config(
                    params["filename"], height_multiplier=chart_height / 500.0
                ),
            )

    except ValueError as e:
        return failed_chart_message_div(
            "Could not resolve chart. Try filtering the data to reduce the number of categories or systems.",
            e,
        )


def calc_histogram_bin_data(df, value_col_name, n_bins=10):
    """
    Calculates histogram bin label text, bin boundaries and bin mid-points

    :param df: the Pandas DataFrame containing the data
    :param value_col_name: (string )name of the DataFrame column that contains the values
    :param n_bins: (int) the number of bins to use
    :return: a three element list containing the bin labels, bin boundaries and bin mid-points
    """
    min_bin_value, max_bin_value = df[value_col_name].agg([min, max])
    bins = np.linspace(min_bin_value, max_bin_value, n_bins + 1)
    mid_bins = 0.5 * (bins[:-1] + bins[1:])
    bin_labels = np.array(
        ["{:.2f}≤x<{:.2f}".format(i, j) for i, j in zip(bins[:-1], bins[1:])]
    )
    return bin_labels, bins, mid_bins


def plotly_binned_statistic_barchart(
    df,
    params,
):
    """
    Create a plotly binned statistic bar chart

    :param df: Pandas DataFrame containing the data
    :param params: a dictionary of parameters
    :param params["df_category_col"]: (string) DataFrame column containing categories
    :param params["df_facet_col"]: (string) DataFrame column used to create subplots
    :param params["facet_title"]: (string) Subplot title
    :param params["facet_col_wrap"]: (int) number of subplots per row
    :param params["user_bins"]: list of ints containing bin edges for binning
    :param params["df_category_col"]: (string) DataFrame column containing categories
    :param params["df_x_value_col"]: (string) DataFrame column containing x data
    :param params["df_y_value_col"]: (string) DataFrame column containing y data
    :param params["x_axis_title"]: (string) Title for x-axis
    :param params["y_axis_title"]: (string) Title for y-axis
    :param params["stat_name"]: (string) "mean" or "median"
    :param params["sorting_choice"]: 2-element list. [0] sets sort direction, [1] used to determine which field to sort
    :param params["colourmap"]: (string) colourmap to use
    :param params["return_as_dict"]: (boolean) flag to trigger return as a dictionary rather than a HTML DIV
    :param params["filename"]: (string) default filename to use for plot bitmap export
    :return: Plotly figure embedded in an HTML DIV; or Plotly figure as a dictionary (if params["return_as_dict"] is
             True); or an error message embedded in an HTML DIV if there was a ValueError when calculating the figure
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    if df.empty:
        return empty_dataframe_msg(params)

    chart_height, n_facet_rows = calc_facet_rows_and_height(
        df, params["df_facet_col"], params["facet_col_wrap"]
    )

    n_colours = len(df[params["df_category_col"]].unique())
    colour_sequence = calculate_colour_sequence(params["colourmap"], n_colours)

    sort_ascending = True
    if params["sorting_choice"][0] == 0:
        sort_ascending = False

    if params["sorting_choice"][1].lower() == "name":
        df_category_name_list = (
            (
                df.sort_values(by=params["df_category_col"], ascending=sort_ascending)[
                    params["df_category_col"]
                ]
            )
            .unique()
            .tolist()
        )
        df_facet_category_list = (
            (
                df.sort_values(by=params["df_facet_col"], ascending=sort_ascending)[
                    params["df_facet_col"]
                ]
            )
            .unique()
            .tolist()
        )
    elif params["sorting_choice"][1].lower() == "frequency":
        df_category_name_list = (
            df.groupby(params["df_category_col"])
            .agg(freq=(params["df_category_col"], "count"))
            .sort_values(by="freq", ascending=sort_ascending)
            .reset_index()[params["df_category_col"]]
            .tolist()
        )
        df_facet_category_list = (
            df.groupby(params["df_facet_col"])
            .agg(freq=(params["df_facet_col"], "count"))
            .sort_values(by="freq", ascending=sort_ascending)
            .reset_index()[params["df_facet_col"]]
            .tolist()
        )
    else:
        df_category_name_list = (
            df.groupby(params["df_category_col"])
            .agg(avg=(params["df_y_value_col"], params["stat_name"]))
            .sort_values(by="avg", ascending=sort_ascending)
            .reset_index()[params["df_category_col"]]
            .tolist()
        )
        df_facet_category_list = (
            df.groupby(params["df_facet_col"])
            .agg(avg=(params["df_y_value_col"], params["stat_name"]))
            .sort_values(by="avg", ascending=sort_ascending)
            .reset_index()[params["df_facet_col"]]
            .tolist()
        )

    try:
        n_cols = params["facet_col_wrap"]
        if len(df_facet_category_list) < n_cols:
            n_cols = len(df_facet_category_list)

        fig = make_subplots(
            rows=n_facet_rows,
            cols=n_cols,
            vertical_spacing=0.40 / n_facet_rows,
        )

        current_row = 1
        current_col = 1
        current_facet = 0
        category_names = []

        bins = np.sort(np.array(params["user_bins"]))

        for facet_name in df_facet_category_list:
            facet_subset = df[df[params["df_facet_col"]] == facet_name].dropna(
                subset=[params["df_x_value_col"], params["df_y_value_col"]]
            )

            # Skip to the next facet if the subset is empty
            if facet_subset.empty:
                continue

            facet_x_min = facet_subset[params["df_x_value_col"]].min()
            facet_x_max = facet_subset[params["df_x_value_col"]].max()

            if np.isfinite(facet_x_min):
                if facet_x_min < np.amin(bins):
                    bins = np.concatenate([[facet_x_min], bins])
            if np.isfinite(facet_x_max):
                if facet_x_max > np.amax(bins):
                    bins = np.concatenate([bins, [facet_x_max]])

            bin_labels = np.array(
                ["{:.0f}≤x<{:.0f}".format(i, j) for i, j in zip(bins[:-1], bins[1:])]
            )

            for category_name in df_category_name_list:
                category_subset = facet_subset[
                    facet_subset[params["df_category_col"]] == category_name
                ].dropna(subset=[params["df_x_value_col"], params["df_y_value_col"]])

                # Skip to the next category name if the subset is empty
                if category_subset.empty:
                    continue

                if len(category_subset.index) > 0:
                    if category_name in category_names:
                        show_legend = False
                    else:
                        show_legend = True
                        category_names.append(category_name)

                    category_idx = category_names.index(category_name)

                    binned_stats = stats.binned_statistic(
                        category_subset[params["df_x_value_col"]].values,
                        category_subset[params["df_y_value_col"]].values,
                        statistic=params["stat_name"],
                        bins=bins,
                    )
                    bin_counts = np.bincount(binned_stats[2])
                    trace_labels = np.array(
                        [
                            "Frequency: {}<br>Bin range: {}".format(i, j)
                            for i, j in zip(bin_counts[1:], bin_labels)
                        ]
                    )

                    trace = go.Bar(
                        x=bin_labels,
                        y=binned_stats[0],
                        name=category_name,
                        marker_color=colour_sequence[category_idx],
                        legendgroup=category_idx,
                        showlegend=show_legend,
                        customdata=trace_labels,
                        hovertemplate=f"<b>{facet_name}</b><br>"
                        + f"{category_name}<br>"
                        + f"{params['stat_name'].capitalize()}: "
                        + "%{y:.2f}<br>"
                        + "%{customdata}<br>"
                        + "<extra></extra>",
                    )

                    fig.append_trace(trace, row=current_row, col=current_col)

            fig.update_xaxes(
                title_text=facet_name + " " + params["x_axis_title"],
                tickson="boundaries",
                ticks="outside",
                ticklen=5,
                row=current_row,
                col=current_col,
            )

            if current_col == 1:
                fig.update_yaxes(
                    title_text=params["stat_name"].capitalize()
                    + " "
                    + params["y_axis_title"],
                    row=current_row,
                    col=current_col,
                )

            current_facet += 1
            current_col += 1
            if current_col > n_cols:
                current_row += 1
                current_col = 1

        layout = go.Layout(height=chart_height)

        fig.update_layout(layout)
        fig.update_layout(legend_title_text=params["facet_title"])

        save_fig_as_html_div(fig, params["filename"])

        if params["return_as_dict"]:
            return fig.to_dict()
        else:
            return plot(
                fig,
                output_type="div",
                include_plotlyjs=False,
                config=global_config(
                    params["filename"], height_multiplier=chart_height / 500.0
                ),
            )

    except ValueError as e:
        return failed_chart_message_div(
            "Could not resolve chart. Try filtering the data to reduce the number of categories or systems.",
            e,
        )


def plotly_timeseries_linechart(
    df,
    params,
):
    """
    Create a plotly line chart of data over time

    :param df: Pandas DataFrame containing the data
    :param params: a dictionary of parameters
    :param  params["df_facet_col"]: (string) DataFrame column used to create subplots
    :param  params["df_facet_col_wrap"]: (int) number of subplots per row
    :param  params["facet_title"]: (string) subplot title
    :param  params["df_value_col"]: (string) DataFrame column containing values
    :param  params["value_axis_title"]: (string) y-axis title
    :param  params["colourmap"]: (string) colourmap to use
    :param  params["colourmap"]: (string) colourmap to use
    :param  params["df_date_col"]: (string) DataFrame column containing dates
    :param  params["df_count_col"]: (string) DataFrame column containing frequency data
    :param  params["df_name_col"]: (string) DataFrame column containing categories
    :param  params["legend_title"]: (string) legend title
    :param  params["name_axis_title"]: (string) x-axis title
    :param params["return_as_dict"]: (boolean) flag to trigger return as a dictionary rather than a HTML DIV
    :param params["filename"]: (string) default filename to use for plot bitmap export
    :return: Plotly figure embedded in an HTML DIV; or Plotly figure as a dictionary (if "return_as_dict" is True);
             or an error message embedded in an HTML DIV if there was a ValueError when calculating the figure
    """
    if df.empty:
        return empty_dataframe_msg(params)

    chart_height, n_facet_rows = calc_facet_rows_and_height(
        df, params["facet_col"], params["facet_col_wrap"]
    )

    n_colours = len(df[params["df_name_col"]].unique())
    colour_sequence = calculate_colour_sequence(params["colourmap"], n_colours)

    try:
        fig = px.scatter(
            df,
            x=params["df_date_col"],
            y=params["df_value_col"],
            color=params["df_name_col"],
            facet_col=params["facet_col"],
            facet_col_wrap=params["facet_col_wrap"],
            facet_row_spacing=0.40 / n_facet_rows,
            labels={
                params["facet_col"]: params["facet_title"],
                params["df_value_col"]: params["value_axis_title"],
                params["df_count_col"]: "Frequency",
                params["df_name_col"]: params["legend_title"],
                params["df_date_col"]: params["name_axis_title"],
                "x_ray_system_name": "System",
            },
            hover_name=params["df_name_col"],
            hover_data={
                params["df_name_col"]: False,
                params["df_value_col"]: ":.2f",
                params["df_count_col"]: ":.0f",
            },
            color_discrete_sequence=colour_sequence,
            category_orders=params["sorted_category_list"],
            height=chart_height,
            render_mode="svg",
        )

        for data_set in fig.data:
            data_set.update(mode="markers+lines")

        fig.update_xaxes(
            showticklabels=True,
            ticks="outside",
            ticklen=5,
        )
        fig.update_yaxes(showticklabels=True, matches=None)

        fig.update_layout(legend_title_text=params["legend_title"])

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        save_fig_as_html_div(fig, params["filename"])

        if params["return_as_dict"]:
            return fig.to_dict()
        else:
            return plot(
                fig,
                output_type="div",
                include_plotlyjs=False,
                config=global_config(
                    params["filename"], height_multiplier=chart_height / 500.0
                ),
            )

    except ValueError as e:
        return failed_chart_message_div(
            "Could not resolve chart. Try filtering the data to reduce the number of categories or systems.",
            e,
        )


def plotly_scatter(
    df,
    params,
):
    """
    Create a plotly scatter chart

    :param df: Pandas DataFrame containing the data
    :param params: a dictionary of parameters
    :param params["df_name_col"]: (string) DataFrame column containing categories
    :param params["df_x_col"]: (string) DataFrame column containing x values
    :param params["df_y_col"]: (string) DataFrame column containing y values
    :param params["sorting_choice"]: 2-element list. [0] sets sort direction, [1] used to determine which field to sort
    :param params["grouping_choice"]: (string) "series" or "system"
    :param params["legend_title"]: (string) legend title
    :param params["facet_col_wrap"]: (int) number of subplots per row
    :param params["colourmap"]: (string) colourmap to use
    :param params["x_axis_title"]: (string) x-axis title
    :param params["y_axis_title"]: (string) y-axis title
    :param params["filename"]: (string) default filename to use for plot bitmap export
    :param params["return_as_dict"]: (boolean) flag to trigger return as a dictionary rather than a HTML DIV
    :return: Plotly figure embedded in an HTML DIV; or Plotly figure as a dictionary (if "return_as_dict" is True);
             or an error message embedded in an HTML DIV if there was a ValueError when calculating the figure
    """
    if df.empty:
        return empty_dataframe_msg(params)

    params["df_category_name_col"] = params["df_name_col"]
    params["df_group_col"] = "x_ray_system_name"
    if params["grouping_choice"] == "series":
        params["df_category_name_col"] = "x_ray_system_name"
        params["df_group_col"] = params["df_name_col"]
        params["legend_title"] = "System"

    sort_ascending = True
    if params["sorting_choice"][0] == 0:
        sort_ascending = False

    sorting_categories = None
    if params["sorting_choice"][1].lower() == "name":
        sorting_categories = {
            params["df_category_name_col"]: (
                df.sort_values(
                    by=params["df_category_name_col"], ascending=sort_ascending
                )[params["df_category_name_col"]]
            )
            .unique()
            .tolist()
        }
        sorting_categories["x_ray_system_name"] = (
            (
                df.sort_values(by="x_ray_system_name", ascending=sort_ascending)[
                    "x_ray_system_name"
                ]
            )
            .unique()
            .tolist()
        )
        sorting_categories[params["df_group_col"]] = (
            (
                df.sort_values(by=params["df_group_col"], ascending=sort_ascending)[
                    params["df_group_col"]
                ]
            )
            .unique()
            .tolist()
        )
    elif params["sorting_choice"][1].lower() == "frequency":
        sorting_categories = {
            params["df_category_name_col"]: df.groupby(params["df_category_name_col"])
            .agg(freq=(params["df_category_name_col"], "count"))
            .sort_values(by="freq", ascending=sort_ascending)
            .reset_index()[params["df_category_name_col"]]
            .tolist()
        }
        sorting_categories["x_ray_system_name"] = (
            df.groupby("x_ray_system_name")
            .agg(freq=("x_ray_system_name", "count"))
            .sort_values(by="freq", ascending=sort_ascending)
            .reset_index()["x_ray_system_name"]
            .tolist()
        )
        sorting_categories[params["df_group_col"]] = (
            df.groupby(params["df_group_col"])
            .agg(freq=(params["df_group_col"], "count"))
            .sort_values(by="freq", ascending=sort_ascending)
            .reset_index()[params["df_group_col"]]
            .tolist()
        )
    else:
        sorting_categories = {
            params["df_category_name_col"]: df.groupby(params["df_category_name_col"])
            .agg(mean=(params["df_y_col"], "mean"))
            .sort_values(by="mean", ascending=sort_ascending)
            .reset_index()[params["df_category_name_col"]]
            .tolist()
        }
        sorting_categories["x_ray_system_name"] = (
            df.groupby("x_ray_system_name")
            .agg(mean=(params["df_y_col"], "mean"))
            .sort_values(by="mean", ascending=sort_ascending)
            .reset_index()["x_ray_system_name"]
            .tolist()
        )
        sorting_categories[params["df_group_col"]] = (
            df.groupby(params["df_group_col"])
            .agg(mean=(params["df_y_col"], "mean"))
            .sort_values(by="mean", ascending=sort_ascending)
            .reset_index()[params["df_group_col"]]
            .tolist()
        )

    try:
        # Drop any rows with nan values in x or y
        df = df.dropna(subset=[params["df_x_col"], params["df_y_col"]])
        if df.empty:
            return empty_dataframe_msg(params)

        chart_height, n_facet_rows = calc_facet_rows_and_height(
            df, params["df_group_col"], params["facet_col_wrap"]
        )

        n_colours = len(df[params["df_category_name_col"]].unique())
        colour_sequence = calculate_colour_sequence(params["colourmap"], n_colours)

        fig = px.scatter(
            df,
            x=params["df_x_col"],
            y=params["df_y_col"],
            color=params["df_category_name_col"],
            facet_col=params["df_group_col"],
            facet_col_wrap=params["facet_col_wrap"],
            facet_row_spacing=0.40 / n_facet_rows,
            labels={
                params["df_x_col"]: params["x_axis_title"],
                params["df_y_col"]: params["y_axis_title"],
                params["df_category_name_col"]: params["legend_title"],
            },
            color_discrete_sequence=colour_sequence,
            category_orders=sorting_categories,
            opacity=0.6,
            height=chart_height,
            render_mode="svg",  # "webgl",
        )

        fig.update_traces(marker_line=dict(width=1, color="LightSlateGray"))

        fig.update_xaxes(showticklabels=True, matches=None)
        fig.update_yaxes(showticklabels=True, matches=None)

        fig.update_layout(legend_title_text=params["legend_title"])

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        save_fig_as_html_div(fig, params["filename"])

        if params["return_as_dict"]:
            return fig.to_dict()
        else:
            return plot(
                fig,
                output_type="div",
                include_plotlyjs=False,
                config=global_config(
                    params["filename"], height_multiplier=chart_height / 500.0
                ),
            )

    except ValueError as e:
        return failed_chart_message_div(
            "Could not resolve chart. Try filtering the data to reduce the number of categories or systems.",
            e,
        )


def plotly_barchart_weekdays(
    df,
    df_name_col,
    df_value_col,
    name_axis_title="",
    value_axis_title="",
    colourmap="RdYlBu",
    filename="OpenREM_workload_chart",
    facet_col_wrap=3,
    sorting_choice=None,
    return_as_dict=False,
):
    """
    Create a plotly bar chart of event workload

    :param df: Pandas DataFrame containing the data
    :param df_name_col: (string) DataFrame column containing categories
    :param df_value_col: (string) DataFrame column containing values
    :param name_axis_title: (string) x-axis title
    :param value_axis_title: (string) y-axis title
    :param colourmap: (string) colourmap to use
    :param filename: (string) default filename to use for plot bitmap export
    :param facet_col_wrap: (int) number of subplots per row
    :param sorting_choice: 2-element list. [0] sets sort direction, [1] used to determine which field to sort
    :param return_as_dict: (boolean) flag to trigger return as a dictionary rather than a HTML DIV
    :return: Plotly figure embedded in an HTML DIV; or Plotly figure as a dictionary (if "return_as_dict" is True);
             or an error message embedded in an HTML DIV if there was a ValueError when calculating the figure
    """
    # pylint: disable=too-many-locals
    if df.empty:
        return empty_dataframe_msg()

    if sorting_choice is None:
        # Set default sorting to be ascending name order
        sorting_choice = [1, "name"]

    sort_ascending = True
    if sorting_choice[0] == 0:
        sort_ascending = False

    if sorting_choice[1].lower() == "name":
        sorting_categories = {
            "x_ray_system_name": (
                df.sort_values(by="x_ray_system_name", ascending=sort_ascending)[
                    "x_ray_system_name"
                ]
            )
            .unique()
            .tolist()
        }
    else:
        sorting_categories = {
            "x_ray_system_name": df.groupby("x_ray_system_name")
            .agg(freq=(df_value_col, "sum"))
            .sort_values(by="freq", ascending=sort_ascending)
            .reset_index()["x_ray_system_name"]
            .tolist()
        }

    chart_height, n_facet_rows = calc_facet_rows_and_height(
        df, "x_ray_system_name", facet_col_wrap
    )

    try:
        fig = px.bar(
            df,
            x=df_name_col,
            y=df_value_col,
            facet_col="x_ray_system_name",
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=0.40 / n_facet_rows,
            color=df_value_col,
            labels={
                df_name_col: name_axis_title,
                df_value_col: value_axis_title,
                "x_ray_system_name": "System",
                "hour": "Hour",
            },
            color_continuous_scale=colourmap,
            category_orders=sorting_categories,
            hover_name="x_ray_system_name",
            hover_data={
                "x_ray_system_name": False,
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
            showticklabels=True,
        )

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        save_fig_as_html_div(fig, filename)

        if return_as_dict:
            return fig.to_dict()
        else:
            return plot(
                fig,
                output_type="div",
                include_plotlyjs=False,
                config=global_config(filename, height_multiplier=chart_height / 500.0),
            )

    except ValueError as e:
        return failed_chart_message_div(
            "Could not resolve chart. Try filtering the data to reduce the number of systems.",
            e,
        )


def plotly_frequency_barchart(
    df,
    params,
    csv_name="OpenREM chart data.csv",
):
    """
    Create a plotly bar chart of event frequency

    :param df: Pandas DataFrame containing the data
    :param params: a dictionary of parameters
    :param params["df_x_axis_col"]: (string) DataFrame column containing categories
    :param params["x_axis_title"]: (string) x-axis title
    :param params["groupby_cols"]: list of strings with DataFrame columns to group data by
    :param params["grouping_choice"]: (string) "series" or "system"
    :param params["sorting_choice"]: 2-element list. [0] sets sort direction, [1] used to determine which field to sort
    :param params["legend_title"]: (string) legend title
    :param params["facet_col"]: (string) DataFrame column used to create subplots
    :param params["facet_col_wrap"]: (int) number of subplots per row
    :param params["return_as_dict"]: (boolean) flag to trigger return as a dictionary rather than a HTML DIV
    :param params["colourmap"]: (string) colourmap to use
    :param params["filename"]: (string) default filename to use for plot bitmap export
    :param csv_name: (string) default filename to use for plot csv export
    :return: Plotly figure embedded in an HTML DIV; or Plotly figure as a dictionary (if "return_as_dict" is True);
             or an error message embedded in an HTML DIV if there was a ValueError when calculating the figure
    """
    if df.empty:
        return empty_dataframe_msg(params), None

    if params["groupby_cols"] is None:
        params["groupby_cols"] = [params["df_name_col"]]

    df_aggregated = create_dataframe_aggregates(
        df, params["groupby_cols"], params["df_name_col"], ["count"]
    )

    df_legend_col = params["df_name_col"]
    if params["grouping_choice"] == "series":
        df_legend_col = "x_ray_system_name"
        params["x_axis_title"] = params["legend_title"]
        params["legend_title"] = "System"
        params["df_x_axis_col"] = params["df_name_col"]

    chart_height = 500
    n_facet_rows = 1

    if params["facet_col"]:
        chart_height, n_facet_rows = calc_facet_rows_and_height(
            df, params["facet_col"], params["facet_col_wrap"]
        )

    sort_ascending = True
    if params["sorting_choice"][0] == 0:
        sort_ascending = False

    sorting_categories = None
    if params["sorting_choice"][1].lower() == "name":
        sorting_categories = {
            params["df_x_axis_col"]: (
                df_aggregated.sort_values(
                    by=params["df_x_axis_col"], ascending=sort_ascending
                )[params["df_x_axis_col"]]
            )
            .unique()
            .tolist()
        }
        sorting_categories[df_legend_col] = (
            (
                df_aggregated.sort_values(by=df_legend_col, ascending=sort_ascending)[
                    df_legend_col
                ]
            )
            .unique()
            .tolist()
        )
        if params["facet_col"]:
            sorting_categories[params["facet_col"]] = (
                (
                    df_aggregated.sort_values(
                        by=params["facet_col"], ascending=sort_ascending
                    )[params["facet_col"]]
                )
                .unique()
                .tolist()
            )
    else:
        sorting_categories = {
            params["df_x_axis_col"]: df_aggregated.groupby(params["df_x_axis_col"])
            .agg({"count": "sum"})
            .sort_values(by="count", ascending=sort_ascending)
            .reset_index()[params["df_x_axis_col"]]
            .tolist()
        }
        sorting_categories[df_legend_col] = (
            df_aggregated.groupby(df_legend_col)
            .agg({"count": "sum"})
            .sort_values(by="count", ascending=sort_ascending)
            .reset_index()[df_legend_col]
            .tolist()
        )
        if params["facet_col"]:
            sorting_categories[params["facet_col"]] = (
                df_aggregated.groupby(params["facet_col"])
                .agg({"count": "sum"})
                .sort_values(by="count", ascending=sort_ascending)
                .reset_index()[params["facet_col"]]
                .tolist()
            )

    n_colours = len(df_aggregated[df_legend_col].unique())
    colour_sequence = calculate_colour_sequence(params["colourmap"], n_colours)

    custom_data_fields = [df_legend_col]
    if params["facet_col"] is not None:
        custom_data_fields.append(params["facet_col"])

    fig = px.bar(
        df_aggregated,
        x=params["df_x_axis_col"],
        y="count",
        color=df_legend_col,
        facet_col=params["facet_col"],
        facet_col_wrap=params["facet_col_wrap"],
        facet_row_spacing=0.50 / n_facet_rows,
        color_discrete_sequence=colour_sequence,
        height=chart_height,
        custom_data=custom_data_fields,
        category_orders=sorting_categories,
    )

    fig.update_xaxes(
        tickson="boundaries",
        ticks="outside",
        ticklen=5,
        showticklabels=True,
        title=params["x_axis_title"],
    )
    fig.update_yaxes(showticklabels=True, matches=None)

    fig.update_layout(
        legend_title_text=params["legend_title"],
    )

    fig.update_traces(
        hovertemplate="<b>"
        + params["legend_title"]
        + ": %{customdata[0]}</b>"
        + "<br>"
        + params["x_axis_title"]
        + ": %{x}"
        + "<br>Frequency: %{y:.0d}"
        + "<extra></extra>",
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    save_fig_as_html_div(fig, params["filename"])

    if params["return_as_dict"]:
        return fig.to_dict(), None
    else:
        csv_data = download_link(
            csv_data_frequency(fig, params),
            csv_name,
        )

        return (
            plot(
                fig,
                output_type="div",
                include_plotlyjs=False,
                config=global_config(
                    params["filename"], height_multiplier=chart_height / 500.0
                ),
            ),
            csv_data,
        )


def construct_over_time_charts(
    df,
    params,
    group_by_physician=None,
):
    """
    Construct a Plotly line chart of average values over time, optionally grouped by performing physician name.
    For "boxplot" a plotly boxplot of values over time is returned instead of an plotly line chart.

    :param df: the Pandas DataFrame containing the data
    :param params: a dictionary of processing parameters

    :param params["df_name_col"]: (string) DataFrame column containing categories
    :param params["name_title"]: (string) name title
    :param params["df_value_col"]: (string) DataFrame column containing values
    :param params["value_title"]: (string) y-axis title
    :param params["df_date_col"]: (string) DataFrame column containing dates
    :param params["date_title"]: (string) date title
    :param params["facet_title"]: (string) subplot title
    :param params["sorting_choice"]: 2-element list. [0] sets sort direction, [1] used to determine which field to sort
    :param params["average_choices"]: list of strings containing requred averages ("mean", "median", "boxplot")
    :param params["time_period"]: string containing the time period to average over; "A" (years), "Q" (quarters),
                                  "M" (months), "W" (weeks), "D" (days)
    :param params["grouping_choice"]: (string) "series" or "system"
    :param params["colourmap"]: (string) colourmap to use
    :param params["filename"]: (string) default filename to use for plot bitmap export
    :param params["facet_col_wrap"]: (int) number of subplots per row
    :param params["return_as_dict"]: (boolean) flag to trigger return as a dictionary rather than a HTML DIV
    :param group_by_physician: boolean flag to set whether to group by physician name
    :return: a dictionary containing a combination of ["mean"], ["median"] and ["boxplot"] entries,
        each of which contains a Plotly figure embedded in an HTML DIV; or Plotly figure as a
        dictionary (if params["return_as_dict"] is True); or an error message embedded in an HTML DIV
        if there was a ValueError when calculating the figure
    """
    sorted_categories = create_sorted_category_list(
        df, params["df_name_col"], params["df_value_col"], params["sorting_choice"]
    )

    df = df.dropna(subset=[params["df_value_col"]])
    if df.empty:
        return_value = {}
        if "mean" in params["average_choices"]:
            return_value["mean"] = empty_dataframe_msg(params)
        if "median" in params["average_choices"]:
            return_value["median"] = empty_dataframe_msg(params)
        if "boxplot" in params["average_choices"]:
            return_value["boxplot"] = empty_dataframe_msg(params)
        return return_value

    if "mean" in params["average_choices"] or "median" in params["average_choices"]:
        df_time_series = create_dataframe_time_series(
            df,
            params["df_name_col"],
            params["df_value_col"],
            df_date_col=params["df_date_col"],
            time_period=params["time_period"],
            average_choices=list(
                set(params["average_choices"]).intersection(["mean", "median", "count"])
            ),
            group_by_physician=group_by_physician,
        )

    category_names_col = params["df_name_col"]
    group_by_col = "x_ray_system_name"
    if group_by_physician:
        group_by_col = "performing_physician_name"

    if params["grouping_choice"] == "series":
        category_names_col = "x_ray_system_name"
        group_by_col = params["df_name_col"]
        if group_by_physician:
            category_names_col = "performing_physician_name"
            params["name_title"] = "Physician"

    return_value = {}

    parameter_dict = {
        "df_count_col": "count" + params["df_value_col"],
        "df_name_col": category_names_col,
        "df_date_col": params["df_date_col"],
        "facet_col": group_by_col,
        "facet_title": params["facet_title"],
        "value_axis_title": params["value_title"],
        "name_axis_title": params["date_title"],
        "legend_title": params["name_title"],
        "colourmap": params["colourmap"],
        "filename": params["filename"],
        "facet_col_wrap": params["facet_col_wrap"],
        "sorted_category_list": sorted_categories,
        "return_as_dict": params["return_as_dict"],
    }
    if "mean" in params["average_choices"]:
        parameter_dict["df_value_col"] = "mean" + params["df_value_col"]
        return_value["mean"] = plotly_timeseries_linechart(
            df_time_series,
            parameter_dict,
        )

    if "median" in params["average_choices"]:
        parameter_dict["df_value_col"] = "median" + params["df_value_col"]
        return_value["median"] = plotly_timeseries_linechart(
            df_time_series,
            parameter_dict,
        )

    if "boxplot" in params["average_choices"]:
        df.set_index(params["df_date_col"], inplace=True)
        df = df.to_period(freq=params["time_period"], copy=False)
        df.reset_index(inplace=True)
        df[params["df_date_col"]] = df[params["df_date_col"]].map(
            lambda x: x.start_time
        )
        df.sort_values(params["df_date_col"], inplace=True)

        parameter_dict["df_name_col"] = params["df_date_col"]
        parameter_dict["df_value_col"] = params["df_value_col"]
        parameter_dict["sorting_choice"] = params["sorting_choice"]

        return_value["boxplot"] = plotly_boxplot(
            df,
            parameter_dict,
        )

    return return_value


def download_link(
    object_to_download, download_filename, download_link_text="Download csv"
):
    """
    Adapted from:
    https://discuss.streamlit.io/t/heres-a-download-function-that-works-for-dataframes-and-txt/4052

    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:

    ``download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')``

    ``download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')``

    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download.columns = object_to_download.columns.str.replace(
            "<br>", " ", regex=True
        )
        object_to_download = object_to_download.replace("<br>", " ", regex=True).to_csv(
            index=False
        )

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a class="btn btn-default btn-sm" role="button" href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'  # pylint: disable=line-too-long


def generate_average_chart_group(
    average_choices,
    chart_message,
    df,
    modality_text,
    name_field,
    name_text,
    return_as_dict,
    return_structure,
    units_text,
    user_profile,
    value_field,
    value_text,
    variable_name_start,
    variable_value_name,
    sorting_choice,
):
    # pylint: disable=too-many-locals
    if user_profile.plotBoxplots and "median" not in average_choices:
        average_choices = average_choices + ["median"]

    if user_profile.plotMean or user_profile.plotMedian:

        df_aggregated = create_dataframe_aggregates(
            df,
            [name_field],
            value_field,
            stats_to_use=average_choices + ["count"],
        )

        parameter_dict = {
            "df_name_col": name_field,
            "name_axis_title": name_text,
            "colourmap": user_profile.plotColourMapChoice,
            "facet_col": None,
            "facet_col_wrap": user_profile.plotFacetColWrapVal,
            "return_as_dict": return_as_dict,
            "sorting_choice": [
                user_profile.plotInitialSortingDirection,
                sorting_choice,
            ],
            "custom_msg_line": chart_message,
        }

        if user_profile.plotMean:
            parameter_dict["value_axis_title"] = "Mean " + value_text + " " + units_text
            parameter_dict["filename"] = (
                "OpenREM "
                + modality_text
                + " "
                + name_text
                + " "
                + value_text
                + " mean"
            )
            parameter_dict["average_choice"] = "mean"
            (
                return_structure[
                    variable_name_start + "Mean" + variable_value_name + "Data"
                ],
                return_structure[
                    variable_name_start + "Mean" + variable_value_name + "DataCSV"
                ],
            ) = plotly_barchart(  # pylint: disable=line-too-long
                df_aggregated,
                parameter_dict,
                csv_name=variable_name_start + "Mean" + value_text + "Data.csv",
            )

            # Create a data frame to use to display the data to the user in an html table
            table_df = df_aggregated[
                ["x_ray_system_name", name_field, "mean", "count"]
            ].round({"mean": 2})

            # Rename the data frame columns to have user-friendly names
            table_df.columns = [
                "X-ray system name",
                name_text,
                "Mean " + value_text + " " + units_text,
                "Count",
            ]

            # Pivot the table so that there is a column per system for the median and count
            table_df = table_df.pivot(index=name_text, columns="X-ray system name")
            table_df.columns = [
                "<br>".join((col[1], str(col[0]))) for col in table_df.columns
            ]
            table_df = table_df.reset_index()

            # Add a html table version of the data frame to the return structure
            tableName = variable_name_start + "Mean" + variable_value_name + "DataTable"
            return_structure[tableName] = table_df.to_html(
                classes="table table-bordered table-sm small sortable chart-data-table-contents",
                table_id=tableName,
                index=False,
                na_rep="-",
                escape=False,
            )

        if user_profile.plotMedian:
            parameter_dict["value_axis_title"] = (
                "Median " + value_text + " " + units_text
            )
            parameter_dict["filename"] = (
                "OpenREM "
                + modality_text
                + " "
                + name_text
                + " "
                + value_text
                + " median"
            )
            parameter_dict["average_choice"] = "median"
            (
                return_structure[
                    variable_name_start + "Median" + variable_value_name + "Data"
                ],
                return_structure[
                    variable_name_start + "Median" + variable_value_name + "DataCSV"
                ],
            ) = plotly_barchart(  # pylint: disable=line-too-long
                df_aggregated,
                parameter_dict,
                csv_name=variable_name_start + "Median" + value_text + "Data.csv",
            )

            # Create a data frame to use to display the data to the user in an html table
            table_df = df_aggregated[
                ["x_ray_system_name", name_field, "median", "count"]
            ].round({"median": 2})

            # Rename the data frame columns to have user-friendly names
            table_df.columns = [
                "X-ray system name",
                name_text,
                "Median " + value_text + " " + units_text,
                "Count",
            ]

            # Pivot the table so that there is a column per system for the median and count
            table_df = table_df.pivot(index=name_text, columns="X-ray system name")
            table_df.columns = [
                "<br>".join((col[1], str(col[0]))) for col in table_df.columns
            ]
            table_df = table_df.reset_index()

            # Add a html table version of the data frame to the return structure
            tableName = (
                variable_name_start + "Median" + variable_value_name + "DataTable"
            )
            return_structure[tableName] = table_df.to_html(
                classes="table table-bordered table-sm small sortable chart-data-table-contents",
                table_id=tableName,
                index=False,
                na_rep="-",
                escape=False,
            )

    if user_profile.plotBoxplots:
        parameter_dict = {
            "df_name_col": name_field,
            "df_value_col": value_field,
            "value_axis_title": value_text + " " + units_text,
            "name_axis_title": name_text,
            "colourmap": user_profile.plotColourMapChoice,
            "filename": "OpenREM "
            + modality_text
            + " "
            + name_text
            + " "
            + variable_value_name
            + " boxplot",
            "facet_col": None,
            "sorting_choice": [
                user_profile.plotInitialSortingDirection,
                sorting_choice,
            ],
            "facet_col_wrap": user_profile.plotFacetColWrapVal,
            "return_as_dict": return_as_dict,
            "custom_msg_line": chart_message,
        }

        return_structure[
            variable_name_start + "Boxplot" + variable_value_name + "Data"
        ] = plotly_boxplot(
            df,
            parameter_dict,
        )

    if user_profile.plotHistograms:
        category_names_col = name_field
        group_by_col = "x_ray_system_name"
        legend_title = name_text

        if user_profile.plotGroupingChoice == "series":
            category_names_col = "x_ray_system_name"
            group_by_col = name_field
            legend_title = "System"

        parameter_dict = {
            "df_facet_col": group_by_col,
            "df_category_col": category_names_col,
            "df_value_col": value_field,
            "value_axis_title": value_text + " " + units_text,
            "legend_title": legend_title,
            "n_bins": user_profile.plotHistogramBins,
            "colourmap": user_profile.plotColourMapChoice,
            "filename": "OpenREM "
            + modality_text
            + " "
            + name_text
            + " "
            + variable_value_name
            + " histogram",
            "facet_col_wrap": user_profile.plotFacetColWrapVal,
            "sorting_choice": [
                user_profile.plotInitialSortingDirection,
                sorting_choice,
            ],
            "global_max_min": user_profile.plotHistogramGlobalBins,
            "return_as_dict": return_as_dict,
            "custom_msg_line": chart_message,
        }
        return_structure[
            variable_name_start + "Histogram" + variable_value_name + "Data"
        ] = plotly_histogram_barchart(
            df,
            parameter_dict,
        )
    return return_structure
