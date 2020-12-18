######
Charts
######

From OpenREM version 1.0.0+ charts use the Plotly_ open source Python library.

***********
Chart types
***********

=============================================
Bar chart of average values across categories
=============================================

Example bar chart.

These show either mean or median data.


===============================================
Histogram bar chart of values across categories
===============================================

Example histogram bar chart.


===============================================
Boxplot of values across a number of categories
===============================================

Example boxplot.

These show either mean or median data.


==============================
Scatter chart of x vs y values
==============================

Example scatter chart.


===============================
Bar chart of category frequency
===============================

Example frequency bar chart.


=====================================
Line chart of average value over time
=====================================

Example line chart of average value over time.

These show either mean or median data.

Each datapoint represents the average over a user-specified time period. This can be a day, week, month, quarter or
year.


=====================
Bar chart of workload
=====================

Example bar chart showing workload.

These show the number of events taking place per weekday, sub-divided into hours of the day.


====================================================================
Bar chart of average value binned by another value across categories
====================================================================

Example bar chart showing mammography AGD vs compressed breast thickness bins.

These show either mean or median data over a number of binned criteria.

This is currently used to display the average glandular dose binned into compressed breast thickness bands.



***********************************
Chart options on the modality pages
***********************************

========================== ============================== ===================================================
Name                       Configuration options          Notes
========================== ============================== ===================================================
Average plots              | Any combination of **mean**,
                           | **median** or **boxplot**
-------------------------- ------------------------------ ---------------------------------------------------
Time period                | One of **day**, **week**,
                           | **month**, **quarter**,
                           | or **year**
-------------------------- ------------------------------ ---------------------------------------------------
Grouping choice            | **System names**             | **System names** groups by x-ray system
                           | **Series item names**        | **Series item names** groups by each category
-------------------------- ------------------------------ ---------------------------------------------------
Plot a series per system   **On** or **off**              **On** splits the data by x-ray system
-------------------------- ------------------------------ ---------------------------------------------------
Calculate histogram data   **On** or **off**              **On** calculate histograms for average bar charts
-------------------------- ------------------------------ ---------------------------------------------------
Chart sorting              | One of **name**,             | Sort the chart data according to the
                           | **frequency**, or **value**  | selected choice
-------------------------- ------------------------------ ---------------------------------------------------
Sorting direction          | One of **ascending**         Sets the sort direction
                           | or **descending**
-------------------------- ------------------------------ ---------------------------------------------------
Split plots by physician   **On** or **off**              | Calculate a series per physician
                                                          | (*fluoroscopy only*)
========================== ============================== ===================================================



*******************************************
Additional chart options on ``Config`` page
*******************************************

==================================== ========================= ==================================================
Name                                 Configuration options     Notes
==================================== ========================= ==================================================
Number of histogram bins             Value in the range 2 - 40 Default is 10
------------------------------------ ------------------------- --------------------------------------------------
Fixed histogram bins across subplots **On** or **off**         **On** forces all histograms to use the same bins
------------------------------------ ------------------------- --------------------------------------------------
Case-insensitive categories          **On** or **off**         **On** all category names forced to lowercase
------------------------------------ ------------------------- --------------------------------------------------
Chart theme                          | One of **Plotly**,      | Set the Plotly theme to use for the charts. Some
                                     | **Plotly white**,       | examples are provided on the Plotly themes_ page
                                     | **Plotly dark**,        | (an external link).
                                     | **presentation**,
                                     | **ggplot2**,
                                     | **Seaborn** or
                                     | **simple white**
------------------------------------ ------------------------- --------------------------------------------------
Colour map choice                    One of the available
                                     matplotlib colour maps
------------------------------------ ------------------------- --------------------------------------------------
Number of charts per row             Value in the range 1 - 10 Sets the number of sub-plots in each row
==================================== ========================= ==================================================



*******************
Available CT charts
*******************

====================================== =================================================================
Chart name                             Chart type
====================================== =================================================================
Acquisition DLP                        | Bar chart of average DLP per acquisition protocol
                                       | Boxplot with data point per acquisition protocol
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
Acquisition CTDI\ :sub:`vol`           | Bar chart of average DLP per
                                       | Boxplot with data point per acquisition protocol
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
Acquisition frequency                  | Bar chart of frequency of each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition CTDI\ :sub:`vol` vs mass   Scatter chart of CTDI\ :sub:`vol` vs mass for each acquisition
                                       protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition DLP vs mass	               Scatter chart of DLP vs mass for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition CTDI\ :sub:`vol` over time | Line chart of average CTDI\ :sub:`vol` over time
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition DLP over time              | Line chart of average DLP over time
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Study DLP                              | Bar chart of average DLP per study description
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
Study CTDI\ :sub:`vol`	               | Bar chart of average CTDI\ :sub:`vol` per study description
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
Study frequency	                       | Bar chart of frequency of each study description
-------------------------------------- -----------------------------------------------------------------
Study events                           | Bar chart of average number of radiation events per study description
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
Study workload                         Bar chart of number of studies carried out on each day of the week,
                                       with each bar sub-divided into hours of the day
-------------------------------------- -----------------------------------------------------------------
Study DLP over time	                   | Line chart of average DLP over time
                                       | for each study description
-------------------------------------- -----------------------------------------------------------------
Requested procedure DLP                | Bar chart of average DLP per requested procedure name
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
Requested procedure frequency	       | Bar chart of frequency of each requested procedure name
-------------------------------------- -----------------------------------------------------------------
Requested procedure events             | Bar chart of average number of radiation events per requested procedure name
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
Requested procedure DLP over time      | Line chart of average DLP over time
                                       | for each study description
====================================== =================================================================


*****************************
Available radiographic charts
*****************************

====================================== =================================================================
Chart name                             Chart type
====================================== =================================================================
Dummy entry
====================================== =================================================================


****************************
Available fluoroscopy charts
****************************

====================================== =================================================================
Chart name                             Chart type
====================================== =================================================================
Dummy entry
====================================== =================================================================


****************************
Available mammography charts
****************************

====================================== =================================================================
Chart name                             Chart type
====================================== =================================================================
Dummy entry
====================================== =================================================================


.. _Plotly: https://plotly.com/python/

.. _Pandas: https://pandas.pydata.org/

.. _themes: https://plotly.com/python/templates/