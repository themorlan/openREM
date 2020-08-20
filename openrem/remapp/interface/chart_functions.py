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


def average_chart_inc_histogram_data(
    database_events,
    db_display_name_relationship,
    db_series_names,
    db_value_name,
    value_multiplier,
    plot_average,
    plot_freq,
    plot_series_per_system,
    plot_average_choice,
    median_available,
    num_hist_bins,
    exclude_constant_angle=False,
    calculate_histograms=False,
    case_insensitive_categories=False,
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
        median_available: boolean to set whether the database can calculate median values
        num_hist_bins: integer value to set how many histogram bins to calculate
        exclude_constant_angle: boolean used to set whether to exclude CT constant angle acquisitions
        calculate_histograms: boolean used to set whether to calculate histogram data
        case_insensitive_categories: boolean to set whether to make categories case-insensitive


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
    from remapp.models import Median
    import numpy as np
    import pandas as pd
    import altair as alt

    # Exclude all zero value events from the calculations
    database_events = database_events.exclude(**{db_value_name: 0})

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

    return_structure = {}

    summary_annotations = {}

    if plot_average or plot_freq:
        # Determine the mean, median and frequency annotations to use
        if exclude_constant_angle:
            if plot_average:
                if plot_average_choice == "both" or plot_average_choice == "mean":
                    summary_annotations["mean"] = (
                        Avg(
                            Case(
                                When(
                                    ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning__exact="Constant Angle Acquisition",
                                    then=None,
                                ),
                                default=db_value_name,
                                output_field=FloatField(),
                            )
                        )
                        * value_multiplier
                    )
                if plot_average_choice == "both" or plot_average_choice == "median":
                    summary_annotations["median"] = (
                        Median(
                            Case(
                                When(
                                    ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning__exact="Constant Angle Acquisition",
                                    then=None,
                                ),
                                default=db_value_name,
                                output_field=FloatField(),
                            )
                        )
                        * value_multiplier
                    )
            if plot_average or plot_freq:
                summary_annotations["num"] = Sum(
                    Case(
                        When(
                            ctradiationdose__ctirradiationeventdata__ct_acquisition_type__code_meaning__exact="Constant Angle Acquisition",
                            then=0,
                        ),
                        default=1,
                        output_field=IntegerField(),
                    )
                )
        else:
            # Don't exclude "Constant Angle Acquisitions" from the calculations
            if plot_average:
                if plot_average_choice == "both" or plot_average_choice == "mean":
                    summary_annotations["mean"] = (
                        Avg(db_value_name) * value_multiplier
                    )
                if plot_average_choice == "both" or plot_average_choice == "median":
                    summary_annotations["median"] = (
                        Median(db_value_name) * value_multiplier
                    )
            if plot_average or plot_freq:
                summary_annotations["num"] = Count(db_value_name)

        # Create a Pandas DataFrame from database_events including the annotations determined above
        if plot_series_per_system:
            # df_test includes all of the data rows - no calculation of mean, median or frequency required by the server
            df_test = pd.DataFrame.from_records(database_events.values(db_display_name_relationship, "db_series_names_to_use", db_value_name))
            df_test.rename(columns={db_display_name_relationship:"x_ray_system_name"}, inplace=True)

            df = pd.DataFrame.from_records(database_events.values(db_display_name_relationship, "db_series_names_to_use").annotate(**summary_annotations).order_by("db_series_names_to_use"))
            df.rename(columns={db_display_name_relationship:"x_ray_system_name"}, inplace=True)
        else:
            df = pd.DataFrame.from_records(database_events.values("db_series_names_to_use").annotate(**summary_annotations).order_by("db_series_names_to_use"))
            df.insert(0, "system_name", "All systems")

        df.rename(columns={"db_series_names_to_use":"data_point_name"}, inplace=True)

        df_test.rename(columns={"db_series_names_to_use":"data_point_name"}, inplace=True)
        df_test[db_value_name] = df_test[db_value_name].astype(float)

        # Change Decimal values to float so that to_json() works (Decimal values can't be JSON serialised)
        if plot_average_choice == "both" or plot_average_choice == "mean":
            df["mean"] = df["mean"].astype(float)

        # Change Decimal values to float so that to_json() works (Decimal values can't be JSON serialised)
        if plot_average_choice == "both" or plot_average_choice == "median":
            df["median"] = df["median"].astype(float)

        # Create a plot with either the mean, median or both values
        if plot_average_choice == "mean":
            chart = alt.Chart(df).mark_bar().encode(
                column=alt.Column("data_point_name"),
                x=alt.X("x_ray_system_name"),
                y=alt.Y("mean"),
                color="x_ray_system_name"
            ).interactive()

            # Create a plot using the raw data, getting the browser to calculate the mean
            alt.data_transformers.disable_max_rows()
            chart = alt.Chart(df_test).mark_bar().encode(
                column=alt.Column("data_point_name"),
                x=alt.X("x_ray_system_name"),
                y=alt.Y(db_value_name, "mean"),
                color="x_ray_system_name"
            ).interactive()

        elif plot_average_choice == "median":
            chart = alt.Chart(df).mark_bar().encode(
                column=alt.Column("data_point_name"),
                x=alt.X("x_ray_system_name"),
                y=alt.Y("median"),
                color="x_ray_system_name"
            ).interactive()

            # Create a plot using the raw data, getting the browser to calculate the median
            alt.data_transformers.disable_max_rows()
            chart = alt.Chart(df_test).mark_bar().encode(
                column=alt.Column("data_point_name"),
                x=alt.X("x_ray_system_name"),
                y=alt.Y(db_value_name, "median"),
                color="x_ray_system_name"
            ).interactive()

        else:
            # This doesn't produce what is needed at the moment - the mean and median values are stacked
            # on top of one another, resulting in a bar height that is the sum of the two values.
            data = pd.melt(df, id_vars=["x_ray_system_name", "data_point_name"], value_vars=["mean", "median"])
            chart = alt.Chart(data).mark_bar().encode(
                column=alt.Column("data_point_name"),
                x=alt.X("x_ray_system_name"),
                y=alt.Y("value"),
                color="variable"
            ).interactive()

            # Not sure how to calculate a mean and median plot using the raw data

        return chart


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
