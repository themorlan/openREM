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

=============================================
Bar chart of average values across categories
=============================================

These can be configured to show mean or median data.

.. raw:: html
   :file: charts/barchart.html


===============================================
Boxplot of values across a number of categories
===============================================

.. raw:: html
   :file: charts/boxplot.html


===============================================
Histogram bar chart of values across categories
===============================================

.. raw:: html
   :file: charts/histogram.html


===============================
Bar chart of category frequency
===============================

.. raw:: html
   :file: charts/frequency.html


==============================
Scatter chart of x vs y values
==============================

.. raw:: html
   :file: charts/scatter.html


=====================
Bar chart of workload
=====================

These show the number of events taking place per weekday, sub-divided into hours of the day.

.. raw:: html
   :file: charts/workload.html


=====================================
Line chart of average value over time
=====================================

These show either mean or median data.

Each datapoint represents the average over a user-specified time period. This can be a day, week, month, quarter or
year.

.. raw:: html
   :file: charts/overtime.html


====================================================================
Bar chart of average value binned by another value across categories
====================================================================

These show either mean or median data over a number of binned criteria.

This is currently used to display the average glandular dose binned into compressed breast thickness bands.

.. raw:: html
   :file: charts/binned_statistic.html


***********************************
Chart options on the modality pages
***********************************

========================== ============================== ===================================================
Name                       Configuration options          Notes
========================== ============================== ===================================================
| Average plots            | Any combination of **mean**,
                           | **median** or **boxplot**
-------------------------- ------------------------------ ---------------------------------------------------
| Time period              | One of **day**, **week**,    | Applies to over-time charts
                           | **month**, **quarter**,
                           | or **year**
-------------------------- ------------------------------ ---------------------------------------------------
| Grouping choice          | **System names**             | **System names** groups by x-ray system
                           | **Series item names**        | **Series item names** groups by each category
-------------------------- ------------------------------ ---------------------------------------------------
Plot a series per system   **On** or **off**              **On** splits the data by x-ray system
-------------------------- ------------------------------ ---------------------------------------------------
Calculate histogram data   **On** or **off**              **On** calculate histograms for average bar charts
-------------------------- ------------------------------ ---------------------------------------------------
| Chart sorting            | One of **name**,             | Sort the chart data according to the
                           | **frequency**, or **value**  | selected choice
-------------------------- ------------------------------ ---------------------------------------------------
| Sorting direction        | One of **ascending**         | Sets the sort direction
                           | or **descending**
-------------------------- ------------------------------ ---------------------------------------------------
| Split plots by physician | **On** or **off**            | Calculate a series per physician
                                                          | (*some fluoroscopy charts only*)
========================== ============================== ===================================================



***************************************
Additional chart options on Config page
***************************************

==================================== ========================= ==================================================
Name                                 Configuration options     Notes
==================================== ========================= ==================================================
Number of histogram bins             Value in the range 2 - 40 Default is 10
------------------------------------ ------------------------- --------------------------------------------------
Fixed histogram bins across subplots **On** or **off**         **On** forces all histograms to use the same bins
------------------------------------ ------------------------- --------------------------------------------------
Case-insensitive categories          **On** or **off**         **On** all category names forced to lowercase
------------------------------------ ------------------------- --------------------------------------------------
| Remove trailing whitespace from    | **On** or **off**       | **On** strips whitespace from the end of category names
  categories                                                   | For example, "Chest PA" and "Chest PA " will be combined
                                                               | into a single category rather than displayed separately
------------------------------------ ------------------------- --------------------------------------------------
Colour map choice                    One of the available      See the `Available colourmaps`_ section
                                     matplotlib colour maps
------------------------------------ ------------------------- --------------------------------------------------
| Chart theme                        | One of **Plotly**,      | Set the Plotly theme to use for the charts.
                                       **Plotly white**,       | `Some available themes`_ are provided below. Examples of all themes
                                     | **Plotly dark**,        | on the Plotly themes_ page (external link).
                                       **presentation**,
                                     | **ggplot2**,
                                       **Seaborn** or
                                     | **simple white**
------------------------------------ ------------------------- --------------------------------------------------
Number of charts per row             Value in the range 1 - 10 Sets the number of sub-plots in each row
==================================== ========================= ==================================================



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

====================================== =================================================================
Chart name                             Chart type
====================================== =================================================================
Acquisition frequency                  Bar chart of acquisition protocol frequency
-------------------------------------- -----------------------------------------------------------------
| Acquisition DLP                      | Bar chart of average DLP per acquisition protocol
                                       | Boxplot with data point per acquisition protocol
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Acquisition CTDI\ :sub:`vol`         | Bar chart of average CTDI\ :sub:`vol` per acquisition protocol
                                       | Boxplot with data point per acquisition protocol
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Acquisition DLP over time            | Line chart of average DLP over time
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
| Acquisition CTDI\ :sub:`vol`         | Line chart of average CTDI\ :sub:`vol` over time
  over time                            | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition DLP vs mass                Scatter chart of DLP vs patient mass for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition CTDI\ :sub:`vol` vs mass   Scatter chart of CTDI\ :sub:`vol` vs patient mass for each
                                       acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Study frequency                        Bar chart of study description frequency
-------------------------------------- -----------------------------------------------------------------
| Study DLP                            | Bar chart of average DLP per study description
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Study CTDI\ :sub:`vol`               | Bar chart of average CTDI\ :sub:`vol` per study description
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Study events                         | Bar chart of average number of radiation events per study description
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Study DLP over time                  | Line chart of average DLP over time
                                       | for each study description
-------------------------------------- -----------------------------------------------------------------
| Study workload                       | Bar chart of number of studies carried out on each day of the
                                       | week, with each bar sub-divided into hours of the day
-------------------------------------- -----------------------------------------------------------------
Requested procedure frequency          Bar chart of requested procedure name frequency
-------------------------------------- -----------------------------------------------------------------
| Requested procedure DLP              | Bar chart of average DLP per requested procedure name
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Requested procedure events           | Bar chart of average number of radiation events per requested procedure name
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Requested procedure DLP over time    | Line chart of average DLP over time
                                       | for each study description
====================================== =================================================================


*****************************
Available radiographic charts
*****************************

====================================== =================================================================
Chart name                             Chart type
====================================== =================================================================
Acquisition frequency                  Bar chart of acquisition protocol frequency
-------------------------------------- -----------------------------------------------------------------
| Acquisition DAP                      | Bar chart of average DAP per acquisition protocol
                                       | Boxplot with data point per acquisition protocol
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Acquisition mAs                      | Bar chart of average mAs per acquisition protocol
                                       | Boxplot with data point per acquisition protocol
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Acquisition kVp                      | Bar chart of average kVp per acquisition protocol
                                       | Boxplot with data point per acquisition protocol
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Acquisition DAP over time            | Line chart of average DAP over time
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
| Acquisition mAs over time            | Line chart of average mAs over time
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
| Acquisition kVp over time            | Line chart of average kVp over time
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition DAP vs mass                Scatter chart of DAP vs patient mass for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Study frequency                        Bar chart of study description frequency
-------------------------------------- -----------------------------------------------------------------
| Study DAP                            | Bar chart of average DAP per study description
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
Study DAP vs mass                      Scatter chart of DAP vs patient mass for each study description
-------------------------------------- -----------------------------------------------------------------
| Study workload                       | Bar chart of number of studies carried out on each day of the
                                       | week, with each bar sub-divided into hours of the day
-------------------------------------- -----------------------------------------------------------------
Requested procedure frequency          Bar chart of requested procedure name frequency
-------------------------------------- -----------------------------------------------------------------
| Requested procedure DAP              | Bar chart of average DAP per requested procedure name
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
Requested procedure DAP vs mass        Scatter chart of DAP vs patient mass for each requested procedure name
====================================== =================================================================


****************************
Available fluoroscopy charts
****************************

====================================== =================================================================
Chart name                             Chart type
====================================== =================================================================
Study frequency                        Bar chart of study description frequency
-------------------------------------- -----------------------------------------------------------------
| Study DAP                            | Bar chart of average DAP per study description
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Study DAP over time                  | Line chart of average DAP over time
                                       | for each study description
-------------------------------------- -----------------------------------------------------------------
| Study workload                       | Bar chart of number of studies carried out on each day of the
                                       | week, with each bar sub-divided into hours of the day
-------------------------------------- -----------------------------------------------------------------
Requested procedure frequency          Bar chart of requested procedure name frequency
-------------------------------------- -----------------------------------------------------------------
| Requested procedure DAP              | Bar chart of average DAP per requested procedure name
                                       | Boxplot with data point per study description
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Requested procedure DAP over time    | Line chart of average DAP over time
                                       | for each study description
====================================== =================================================================


****************************
Available mammography charts
****************************

====================================== =================================================================
Chart name                             Chart type
====================================== =================================================================
Acquisition frequency                  Bar chart of acquisition protocol frequency
-------------------------------------- -----------------------------------------------------------------
| Acquisition AGD                      | Bar chart of average AGDP per acquisition protocol
                                       | Boxplot with data point per acquisition protocol
                                       | Histograms also plotted if *Calculate histogram data* **on**
-------------------------------------- -----------------------------------------------------------------
| Acquisition average AGD vs thickness | Bar chart of average AGD for each of the following 9 compressed
                                       | breast thickness bands:
                                       | min ≤ x < 20; 20 ≤ x < 30; 30 ≤ x < 40; 40 ≤ x < 50; 50 ≤ x < 60;
                                       | 60 ≤ x < 70; 70 ≤ x < 80; 80 ≤ x < 90; 90 ≤ x < max
-------------------------------------- -----------------------------------------------------------------
| Acquisition AGD over time            | Line chart of average AGD over time
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition AGD vs thickness           | Scatter chart of AGD vs compressed breast thickness
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition mAs vs thickness           | Scatter chart of mAs vs compressed breast thickness
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
Acquisition kVp vs thickness           | Scatter chart of kVp vs compressed breast thickness
                                       | for each acquisition protocol
-------------------------------------- -----------------------------------------------------------------
| Study workload                       | Bar chart of number of studies carried out on each day of the
                                       | week, with each bar sub-divided into hours of the day
====================================== =================================================================

.. _Plotly: https://plotly.com/python/

.. _Pandas: https://pandas.pydata.org/

.. _themes: https://plotly.com/python/templates/