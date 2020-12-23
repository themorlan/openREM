######
Charts
######

From OpenREM version 1.0.0+ charts use the Plotly_ open source Python library.

***********
Chart types
***********

The charts below are examples of the types of chart included in OpenREM version 1.0.0+. The examples are fully
interactive in the same way as the charts included in a live OpenREM system. The data contained in the example charts
is synthetic.

Single-clicking on a legend entry toggles the display of that series. Double-clicking on a legend entry hides all but
that series; double-click again to show all series.

Hovering the cursor over a chart causes the chart menu to appear along the top right corner. From the menu you can:

* save a bitmap version of the chart
* set zoom, pan and selection options
* autoscale the chart
* reset the axes, and also reset the  regions
* toggle spike lines to graphically illustrate x- and y-axis data values on hover
* choose whether to show the closest data when hovering, or to compare data on hover

=============================================
Bar chart of average values across categories
=============================================

These can be configured to show mean or median data. The example below shows the median DAP for each requested procedure
name containing the word "knee" across eight x-ray rooms.

Hovering the cursor over a bar shows the:

* x-ray room name
* requested procedure name
* median DAP for that room and procedure
* number of requests for that room and procedure

.. raw:: html
   :file: charts/barchart.html


===============================================
Boxplot of values across a number of categories
===============================================

The example below shows the same data as for the bar chart above, but presented as a box plot.

Hovering the cursor over an outlier shows the:

* x-ray room name
* requested procedure name
* DAP of the data point

Hovering the cursor over the box shows the:

* maximum value
* minimum value
* median
* 1st and 3rd quartiles
* lower fence: 1rd quartile - (1.5 x interquartile range)
* upper fence: 3rd quartile + (1.5 x interquartile range)

.. raw:: html
   :file: charts/boxplot.html


===============================================
Histogram bar chart of values across categories
===============================================

The example below shows the distribution of DAP values for the knee data used in the box and bar plots above. The
number of bins used in the histograms can be configured in the `Additional chart options on the Config page`_.

Hovering the cursor over a bar shows the:

* requested procedure name
* x-ray room name
* bin DAP range
* bin DAP mid-point value
* bin frequency

.. raw:: html
   :file: charts/histogram.html


===============================
Bar chart of category frequency
===============================

The example below shows the frequency of the "knee" requested procedures for each x-ray room. The height of each bar is
the total frequency for that requested procedure. Each bar is sub-divided into sections representing the number of
requests for each x-ray room.

Hovering the cursor over a bar section shows the:

* x-ray room name
* requested procedure name
* requested procedure frequency

.. raw:: html
   :file: charts/frequency.html

Setting *Grouping choice* to **System names** in the `Chart options on the modality pages`_ groups the data by x-ray
system name rather than requested procedure name, as shown below:

.. raw:: html
   :file: charts/frequency_group_by_system.html


==============================
Scatter chart of x vs y values
==============================

The example below shows the average glandular dose plotted against compressed breast thickness for "MAMMOGRAM"
acquisitions made on two x-ray systems.

Hovering the cursor over a bar section shows the:

* x-ray room name
* acquisition protocol name
* compressed breast thickness
* average glandular dose

.. raw:: html
   :file: charts/scatter.html


=====================
Bar chart of workload
=====================

These show the number of studies taking place per weekday, sub-divided into hours of the day.

There is a bar per weekday. The total height of this bar is the number of studies carried out on that weekday. Each
bar is sub-divided into sections representing the number of studies carried out in each of the 24 hours of that day.
Each section is colour-coded according to how many studies it represents.

Hovering the cursor over a section shows you the:

* x-ray room name
* day of the week that the section represents
* hour of the day that the section represents
* number of studies that took place in that hour on that weekday in that x-ray room

.. raw:: html
   :file: charts/workload.html


=====================================
Line chart of average value over time
=====================================

These can be configured to show mean or median data. Each datapoint represents the average over a user-specified time
period. This can be a day, week, month, quarter or year.

The example below shows the median DAP for "Head" requests made in four CT scanners over the course of five years.

With *Grouping choice* set to **Series item names** in the `Chart options on the modality pages`_ a sub-plot is created
for each requested procedure name, each with a series per x-ray system as shown below. The *Number of charts per row* in
the `Additional chart options on the Config page`_ was set to 2 for these example charts.

Hovering the cursor over a section shows you the:

* scanner name
* requested procedure name
* date
* average DLP value
* number of requests included in the sample

.. raw:: html
   :file: charts/overtime.html

Setting *Grouping choice* to **System names** in the `Chart options on the modality pages`_ changes the grouping. Now
a sub-plot is created for each x-ray system, each with a series per requested procedure name, as shown below:

.. raw:: html
   :file: charts/overtime_group_by_system.html


=========================================================================
Bar chart of average value against another binned value across categories
=========================================================================

These can be configured to show mean or median data. The example below shows the median average glandular dose from
"MAMMOGRAM" protocol exposures plotted against compressed breast thickness bins. The data is from two x-ray systems.

Hovering the cursor over a section shows you the:

* x-ray room name
* acquisition protocol
* average AGD value
* number of acquisitions included in the sample
* compressed breast thickness bin range

.. raw:: html
   :file: charts/binned_statistic.html


***********************************
Chart options on the modality pages
***********************************

========================== ========================================= ==================================================
Name                       Configuration options                     Notes
========================== ========================================= ==================================================
Average plots              Any combination of **mean**, |br|
                           **median** or **boxplot**
-------------------------- ----------------------------------------- --------------------------------------------------
Time period                One of **day**, **week**, **month**, |br| Applies to over-time charts
                           **quarter** or **year**
-------------------------- ----------------------------------------- --------------------------------------------------
Grouping choice            **System names** |br|                     **System names** groups by x-ray system |br|
                           **Series item names**                     **Series item names** groups by each category
-------------------------- ----------------------------------------- --------------------------------------------------
Plot a series per system   **On** or **off**                         **On** splits the data by x-ray system
-------------------------- ----------------------------------------- --------------------------------------------------
Calculate histogram data   **On** or **off**                         **On** calculate histograms for average bar charts
-------------------------- ----------------------------------------- --------------------------------------------------
Chart sorting              One of **name**, **frequency** |br|       Sort the chart data according to the |br|
                           or **value**                              selected choice
-------------------------- ----------------------------------------- --------------------------------------------------
Sorting direction          One of **ascending** |br|                 Sets the sort direction
                           or **descending**
-------------------------- ----------------------------------------- --------------------------------------------------
Split plots by physician   **On** or **off**                         Calculate a series per physician |br|
                                                                     (*some fluoroscopy charts only*)
========================== ========================================= ==================================================



*******************************************
Additional chart options on the Config page
*******************************************

========================================== ========================= ==================================================
Name                                       Configuration options     Notes
========================================== ========================= ==================================================
Number of histogram bins                   Value in the range 2 - 40 Default is 10
------------------------------------------ ------------------------- --------------------------------------------------
Fixed histogram bins across subplots       **On** or **off**         **On** forces all histograms to use the same bins
------------------------------------------ ------------------------- --------------------------------------------------
Case-insensitive categories                **On** or **off**         **On** all category names forced to lowercase |br|
                                                                     For example, "Chest PA" becomes "chest pa"
------------------------------------------ ------------------------- --------------------------------------------------
Remove trailing whitespace from categories **On** or **off**         **On** strips whitespace from the end of category names |br|
                                                                     For example, "Chest PA " becomes "Chest PA"
------------------------------------------ ------------------------- --------------------------------------------------
Remove multiple whitespace from categories **On** or **off**         **On** removes multiple whitespace from category names |br|
                                                                     For example, "Chest   PA" becomes "Chest PA"
------------------------------------------ ------------------------- --------------------------------------------------
Colour map choice                          One of the available |br| See the `Available colourmaps`_ section
                                           matplotlib colour maps
------------------------------------------ ------------------------- --------------------------------------------------
Chart theme                                One of **Plotly**,        Set the Plotly theme to use for the charts. |br|
                                           **Plotly white**, |br|    `Some available themes`_ are provided below. |br|
                                           **Plotly dark**,          Examples of all themes on the Plotly themes_ |br|
                                           **presentation**, |br|    page (external link).
                                           **ggplot2**,
                                           **Seaborn** or |br|
                                           **simple white**
------------------------------------------ ------------------------- --------------------------------------------------
Number of charts per row                   Value in the range 1 - 10 Sets the number of sub-plots in each row
========================================== ========================= ==================================================


====================
Available colourmaps
====================

=================== ===========================
Name                Swatch
=================== ===========================
Red yellow blue     .. image:: img/RdYlBu.png
------------------- ---------------------------
Spectral            .. image:: img/Spectral.png
------------------- ---------------------------
Rainbow             .. image:: img/rainbow.png
------------------- ---------------------------
Jet                 .. image:: img/jet.png
------------------- ---------------------------
Pink yellow green   .. image:: img/PiYG.png
------------------- ---------------------------
Purple green        .. image:: img/PRGn.png
------------------- ---------------------------
Brown green         .. image:: img/BrBG.png
------------------- ---------------------------
Purple orange       .. image:: img/PuOr.png
------------------- ---------------------------
Red blue            .. image:: img/RdBu.png
------------------- ---------------------------
Red grey            .. image:: img/RdGy.png
------------------- ---------------------------
Yellow green blue   .. image:: img/YlGnBu.png
------------------- ---------------------------
Yellow orange brown .. image:: img/YlOrBr.png
------------------- ---------------------------
Hot                 .. image:: img/hot.png
------------------- ---------------------------
Inferno             .. image:: img/inferno.png
------------------- ---------------------------
Magma               .. image:: img/magma.png
------------------- ---------------------------
Plasma              .. image:: img/plasma.png
------------------- ---------------------------
Viridis             .. image:: img/viridis.png
------------------- ---------------------------
Cividis             .. image:: img/Spectral.png
=================== ===========================



=====================
Some available themes
=====================

The example `Chart types`_ at the top of this document use the default Plotly theme. Below are some examples of other
available themes.

+++++++++++
Plotly dark
+++++++++++

.. raw:: html
   :file: charts/barchart_plotly_dark.html

++++++++++++
Presentation
++++++++++++

.. raw:: html
   :file: charts/barchart_presentation.html

++++++++++++
Simple white
++++++++++++

.. raw:: html
   :file: charts/barchart_simple_white.html


*******************
Available CT charts
*******************

====================================== ==============================================================================
Chart name                             Chart type
====================================== ==============================================================================
Acquisition frequency                  Bar chart of acquisition protocol frequency
-------------------------------------- ------------------------------------------------------------------------------
Acquisition DLP                        Bar chart of average DLP per acquisition protocol |br|
                                       Boxplot with data point per acquisition protocol |br|
                                       Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- ------------------------------------------------------------------------------
Acquisition CTDI\ :sub:`vol`           Bar chart of average CTDI\ :sub:`vol` per acquisition protocol |br|
                                       Boxplot with data point per acquisition protocol |br|
                                       Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- ------------------------------------------------------------------------------
Acquisition DLP over time              Line chart of average DLP over time |br|
                                       for each acquisition protocol
-------------------------------------- ------------------------------------------------------------------------------
Acquisition CTDI\ :sub:`vol` over time Line chart of average CTDI\ :sub:`vol` over time |br|
                                       for each acquisition protocol
-------------------------------------- ------------------------------------------------------------------------------
Acquisition DLP vs mass                Scatter chart of DLP vs patient mass for each acquisition protocol
-------------------------------------- ------------------------------------------------------------------------------
Acquisition CTDI\ :sub:`vol` vs mass   Scatter chart of CTDI\ :sub:`vol` vs patient mass for each |br|
                                       acquisition protocol
-------------------------------------- ------------------------------------------------------------------------------
Study frequency                        Bar chart of study description frequency
-------------------------------------- ------------------------------------------------------------------------------
Study DLP                              Bar chart of average DLP per study description |br|
                                       Boxplot with data point per study description |br|
                                       Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- ------------------------------------------------------------------------------
Study CTDI\ :sub:`vol`                 Bar chart of average CTDI\ :sub:`vol` per study description |br|
                                       Boxplot with data point per study description |br|
                                       Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- ------------------------------------------------------------------------------
Study events                           Bar chart of average number of radiation events per study description |br|
                                       Boxplot with data point per study description |br|
                                       Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- ------------------------------------------------------------------------------
Study DLP over time                    Line chart of average DLP over time |br|
                                       for each study description
-------------------------------------- ------------------------------------------------------------------------------
Study workload                         Bar chart of number of studies carried out on each day of the |br|
                                       week, with each bar sub-divided into hours of the day
-------------------------------------- ------------------------------------------------------------------------------
Requested procedure frequency          Bar chart of requested procedure name frequency
-------------------------------------- ------------------------------------------------------------------------------
Requested procedure DLP                Bar chart of average DLP per requested procedure name |br|
                                       Boxplot with data point per study description |br|
                                       Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- ------------------------------------------------------------------------------
Requested procedure events             Bar chart of average number of radiation events per requested procedure name |br|
                                       Boxplot with data point per study description |br|
                                       Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- ------------------------------------------------------------------------------
Requested procedure DLP over time      Line chart of average DLP over time |br|
                                       for each study description
====================================== ==============================================================================


*****************************
Available radiographic charts
*****************************

=============================== ==================================================================
Chart name                      Chart type
=============================== ==================================================================
Acquisition frequency           Bar chart of acquisition protocol frequency
------------------------------- ------------------------------------------------------------------
Acquisition DAP                 Bar chart of average DAP per acquisition protocol |br|
                                Boxplot with data point per acquisition protocol |br|
                                Histograms also plotted if *Calculate histogram data* **on**
------------------------------- ------------------------------------------------------------------
Acquisition mAs                 Bar chart of average mAs per acquisition protocol |br|
                                Boxplot with data point per acquisition protocol |br|
                                Histograms also plotted if *Calculate histogram data* **on**
------------------------------- ------------------------------------------------------------------
Acquisition kVp                 Bar chart of average kVp per acquisition protocol |br|
                                Boxplot with data point per acquisition protocol |br|
                                Histograms also plotted if *Calculate histogram data* **on**
------------------------------- ------------------------------------------------------------------
Acquisition DAP over time       Line chart of average DAP over time |br|
                                for each acquisition protocol
------------------------------- ------------------------------------------------------------------
Acquisition mAs over time       Line chart of average mAs over time |br|
                                for each acquisition protocol
------------------------------- ------------------------------------------------------------------
Acquisition kVp over time       Line chart of average kVp over time |br|
                                for each acquisition protocol
------------------------------- ------------------------------------------------------------------
Acquisition DAP vs mass         Scatter chart of DAP vs patient mass |br|
                                for each acquisition protocol
------------------------------- ------------------------------------------------------------------
Study frequency                 Bar chart of study description frequency
------------------------------- ------------------------------------------------------------------
Study DAP                       Bar chart of average DAP per study description |br|
                                Boxplot with data point per study description |br|
                                Histograms also plotted if *Calculate histogram data* **on**
------------------------------- ------------------------------------------------------------------
Study DAP vs mass               Scatter chart of DAP vs patient mass for each study description
------------------------------- ------------------------------------------------------------------
Study workload                  Bar chart of number of studies carried out on each day of the |br|
                                week, with each bar sub-divided into hours of the day
------------------------------- ------------------------------------------------------------------
Requested procedure frequency   Bar chart of requested procedure name frequency
------------------------------- ------------------------------------------------------------------
Requested procedure DAP         Bar chart of average DAP per requested procedure name |br|
                                Boxplot with data point per study description |br|
                                Histograms also plotted if *Calculate histogram data* **on**
------------------------------- ------------------------------------------------------------------
Requested procedure DAP vs mass Scatter chart of DAP vs patient mass |br|
                                for each requested procedure name
=============================== ==================================================================


****************************
Available fluoroscopy charts
****************************

================================= ===============================================================
Chart name                        Chart type
================================= ===============================================================
Study frequency                   Bar chart of study description frequency
--------------------------------- ---------------------------------------------------------------
Study DAP                         Bar chart of average DAP per study description |br|
                                  Boxplot with data point per study description |br|
                                  Histograms also plotted if *Calculate histogram data* **on**
--------------------------------- ---------------------------------------------------------------
Study DAP over time               Line chart of average DAP over time |br|
                                  for each study description
--------------------------------- ---------------------------------------------------------------
Study workload                    Bar chart of number of studies carried out on each day of the |br|
                                  week, with each bar sub-divided into hours of the day
--------------------------------- ---------------------------------------------------------------
Requested procedure frequency     Bar chart of requested procedure name frequency
--------------------------------- ---------------------------------------------------------------
Requested procedure DAP           Bar chart of average DAP per requested procedure name |br|
                                  Boxplot with data point per study description |br|
                                  Histograms also plotted if *Calculate histogram data* **on**
--------------------------------- ---------------------------------------------------------------
Requested procedure DAP over time Line chart of average DAP over time |br|
                                  for each study description
================================= ===============================================================


****************************
Available mammography charts
****************************

==================================== ===================================================================
Chart name                           Chart type
==================================== ===================================================================
Acquisition frequency                Bar chart of acquisition protocol frequency
------------------------------------ -------------------------------------------------------------------
Acquisition AGD                      Bar chart of average AGDP per acquisition protocol |br|
                                     Boxplot with data point per acquisition protocol |br|
                                     Histograms also plotted if *Calculate histogram data* **on**
------------------------------------ -------------------------------------------------------------------
Acquisition average AGD vs thickness Bar chart of average AGD for each of the following nine compressed |br|
                                     breast thickness bands: |br|
                                     min ≤ x < 20; 20 ≤ x < 30; 30 ≤ x < 40; 40 ≤ x < 50; 50 ≤ x < 60; |br|
                                     60 ≤ x < 70; 70 ≤ x < 80; 80 ≤ x < 90; 90 ≤ x < max
------------------------------------ -------------------------------------------------------------------
Acquisition AGD over time            Line chart of average AGD over time |br|
                                     for each acquisition protocol
------------------------------------ -------------------------------------------------------------------
Acquisition AGD vs thickness         Scatter chart of AGD vs compressed breast thickness |br|
                                     for each acquisition protocol
------------------------------------ -------------------------------------------------------------------
Acquisition mAs vs thickness         Scatter chart of mAs vs compressed breast thickness |br|
                                     for each acquisition protocol
------------------------------------ -------------------------------------------------------------------
Acquisition kVp vs thickness         Scatter chart of kVp vs compressed breast thickness |br|
                                     for each acquisition protocol
------------------------------------ -------------------------------------------------------------------
Study workload                       Bar chart of number of studies carried out on each day of the |br|
                                     week, with each bar sub-divided into hours of the day
==================================== ===================================================================

.. _Plotly: https://plotly.com/python/

.. _Pandas: https://pandas.pydata.org/

.. _themes: https://plotly.com/python/templates/

.. |br| raw:: html

    <br>
