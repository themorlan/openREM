/*global arrayToURL, urlToArray, chroma, updateAverageChart, sortChartDataToDefault, hideButtonsIfOneSeries,
updateFrequencyChart, sortByY, sortByName, plotAverageChoice, updateWorkloadChart, chartSorting, urlStartAcq,
urlStartReq, urlStartStudy, updateOverTimeChart, chartSortingDirection*/
/*eslint no-undef: "error"*/
/*eslint security/detect-object-injection: "off" */
/*eslint object-shorthand: "off" */

function updateBarCharts(namePrefix, json) {
    var parentDiv = $("#" + namePrefix + "ChartParentDiv");

    // Add the Plotly chart to the ChartDiv
    $("#"+namePrefix+"ChartDiv").html(json[namePrefix+"Data"]);

    // Add the "Download csv" button
    parentDiv.append(json[namePrefix+"DataCSV"]);

    // Add a show and hide data table button
    parentDiv.append("<a class='btn btn-default btn-sm' role='button' id='"+namePrefix+"DataTableBtnShow'>Show data table</a>");
    parentDiv.append("<a class='btn btn-default btn-sm' role='button' id='"+namePrefix+"DataTableBtnHide'>Hide data table</a>");

    // Hide the "Hide data table" button
    $("#"+namePrefix+"DataTableBtnHide").hide();

    // Add a function to both buttons that toggles the visiblity of the buttons and the data table
    $("a[id^="+namePrefix+"DataTableBtn").click(function () {
        $("#"+namePrefix+"DataTableDiv").toggle();
        $("#"+namePrefix+"DataTableBtnHide").toggle();
        $("#"+namePrefix+"DataTableBtnShow").toggle();
    })

    // Add the data table to a div and make the table sortable
    parentDiv.append("<div id='"+namePrefix+"DataTableDiv' class='chart-data-table'>"+json[namePrefix+"DataTable"]+"</div>");
    $("#"+namePrefix+"DataTableDiv").hide();
    var table = document.getElementById(namePrefix+"DataTable");
    sorttable.makeSortable(table);
}

// Code to update the page and chart data on initial page load.
$(document).ready(function() {
    var requestData = arrayToURL(urlToArray(this.URL));

    $(".ajax-progress").show();

    $.ajax({
        type: "GET",
        url: Urls.ct_summary_chart_data(),
        data: requestData,
        dataType: "json",
        success: function( json ) {

            // DLP per acquisition chart data
            if(typeof json.acquisitionMeanDLPData !== "undefined") {
                updateBarCharts("acquisitionMeanDLP", json)
            }
            if(typeof json.acquisitionMedianDLPData !== "undefined") {
                updateBarCharts("acquisitionMedianDLP", json)
            }
            if(typeof json.acquisitionBoxplotDLPData !=="undefined") {
                $("#acquisitionBoxplotDLPChartDiv").html(json.acquisitionBoxplotDLPData);
            }
            if(typeof json.acquisitionHistogramDLPData !=="undefined") {
                $("#acquisitionHistogramDLPChartDiv").html(json.acquisitionHistogramDLPData);
            }

            // DLP per standard acquisition name chart data
            if(typeof json.standardAcquisitionMeanDLPData !== "undefined") {
                updateBarCharts("standardAcquisitionMeanDLP", json);
            }
            if(typeof json.standardAcquisitionMedianDLPData !== "undefined") {
                updateBarCharts("standardAcquisitionMedianDLP", json);
            }
            if(typeof json.standardAcquisitionBoxplotDLPData !=="undefined") {
                $("#standardAcquisitionBoxplotDLPChartDiv").html(json.standardAcquisitionBoxplotDLPData);
            }
            if(typeof json.standardAcquisitionHistogramDLPData !=="undefined") {
                $("#standardAcquisitionHistogramDLPChartDiv").html(json.standardAcquisitionHistogramDLPData);
            }

            // CTDI per acquisition chart data
            if(typeof json.acquisitionMeanCTDIData !== "undefined") {
                updateBarCharts("acquisitionMeanCTDI", json);
            }
            if(typeof json.acquisitionMedianCTDIData !== "undefined") {
                updateBarCharts("acquisitionMedianCTDI", json);
            }
            if(typeof json.acquisitionBoxplotCTDIData !=="undefined") {
                $("#acquisitionBoxplotCTDIChartDiv").html(json.acquisitionBoxplotCTDIData);
            }
            if(typeof json.acquisitionHistogramCTDIData !=="undefined") {
                $("#acquisitionHistogramCTDIChartDiv").html(json.acquisitionHistogramCTDIData);
            }

            // CTDI per standard acquisition name chart data
            if(typeof json.standardAcquisitionMeanCTDIData !== "undefined") {
                updateBarCharts("standardAcquisitionMeanCTDI", json);
            }
            if(typeof json.standardAcquisitionMedianCTDIData !== "undefined") {
                updateBarCharts("standardAcquisitionMedianCTDI", json);
            }
            if(typeof json.standardAcquisitionBoxplotCTDIData !=="undefined") {
                $("#standardAcquisitionBoxplotCTDIChartDiv").html(json.standardAcquisitionBoxplotCTDIData);
            }
            if(typeof json.standardAcquisitionHistogramCTDIData !=="undefined") {
                $("#standardAcquisitionHistogramCTDIChartDiv").html(json.standardAcquisitionHistogramCTDIData);
            }

            // Acquisition CTDI over time chart data
            if(typeof json.acquisitionMeanCTDIOverTime !== "undefined") {
                $("#acquisitionMeanCTDIOverTimeChartDiv").html(json.acquisitionMeanCTDIOverTime);
            }
            if(typeof json.acquisitionMedianCTDIOverTime !== "undefined") {
                $("#acquisitionMedianCTDIOverTimeChartDiv").html(json.acquisitionMedianCTDIOverTime);
            }

            // Standard acquisition name CTDI over time chart data
            if(typeof json.standardAcquisitionMeanCTDIOverTime !== "undefined") {
                $("#standardAcquisitionMeanCTDIOverTimeChartDiv").html(json.standardAcquisitionMeanCTDIOverTime);
            }
            if(typeof json.standardAcquisitionMedianCTDIOverTime !== "undefined") {
                $("#standardAcquisitionMedianCTDIOverTimeChartDiv").html(json.standardAcquisitionMedianCTDIOverTime);
            }

            // Acquisition DLP over time chart data
            if(typeof json.acquisitionMeanDLPOverTime !== "undefined") {
                $("#acquisitionMeanDLPOverTimeChartDiv").html(json.acquisitionMeanDLPOverTime);
            }
            if(typeof json.acquisitionMedianDLPOverTime !== "undefined") {
                $("#acquisitionMedianDLPOverTimeChartDiv").html(json.acquisitionMedianDLPOverTime);
            }

            // Standard acquisition name DLP over time chart data
            if(typeof json.standardAcquisitionMeanDLPOverTime !== "undefined") {
                $("#standardAcquisitionMeanDLPOverTimeChartDiv").html(json.standardAcquisitionMeanDLPOverTime);
            }
            if(typeof json.standardAcquisitionMedianDLPOverTime !== "undefined") {
                $("#standardAcquisitionMedianDLPOverTimeChartDiv").html(json.standardAcquisitionMedianDLPOverTime);
            }

            // DLP per study chart data
            if(typeof json.studyMeanDLPData !== "undefined") {
                updateBarCharts("studyMeanDLP", json);
            }
            if(typeof json.studyMedianDLPData !== "undefined") {
                updateBarCharts("studyMedianDLP", json);
            }
            if(typeof json.studyBoxplotDLPData !=="undefined") {
                $("#studyBoxplotDLPChartDiv").html(json.studyBoxplotDLPData);
            }
            if(typeof json.studyHistogramDLPData !=="undefined") {
                $("#studyHistogramDLPChartDiv").html(json.studyHistogramDLPData);
            }

            // DLP per standard study chart data
            if(typeof json.standardStudyMeanDLPData !== "undefined") {
                updateBarCharts("standardStudyMeanDLP", json);
            }
            if(typeof json.standardStudyMedianDLPData !== "undefined") {
                updateBarCharts("standardStudyMedianDLP", json);
            }
            if(typeof json.standardStudyBoxplotDLPData !=="undefined") {
                $("#standardStudyBoxplotDLPChartDiv").html(json.standardStudyBoxplotDLPData);
            }
            if(typeof json.standardStudyHistogramDLPData !=="undefined") {
                $("#standardStudyHistogramDLPChartDiv").html(json.standardStudyHistogramDLPData);
            }

            // CTDI per study chart data
            if(typeof json.studyMeanCTDIData !== "undefined") {
                updateBarCharts("studyMeanCTDI", json);
            }
            if(typeof json.studyMedianCTDIData !== "undefined") {
                updateBarCharts("studyMedianCTDI", json);
            }
            if(typeof json.studyBoxplotCTDIData !=="undefined") {
                $("#studyBoxplotCTDIChartDiv").html(json.studyBoxplotCTDIData);
            }
            if(typeof json.studyHistogramCTDIData !=="undefined") {
                $("#studyHistogramCTDIChartDiv").html(json.studyHistogramCTDIData);
            }

            // DLP per request chart data start
            if(typeof json.requestMeanDLPData !== "undefined") {
                updateBarCharts("requestMeanDLP", json);
            }
            if(typeof json.requestMedianDLPData !== "undefined") {
                updateBarCharts("requestMedianDLP", json);
            }
            if(typeof json.requestBoxplotDLPData !=="undefined") {
                $("#requestBoxplotDLPChartDiv").html(json.requestBoxplotDLPData);
            }
            if(typeof json.requestHistogramDLPData !=="undefined") {
                $("#requestHistogramDLPChartDiv").html(json.requestHistogramDLPData);
            }

            // Number of events per study chart data
            if(typeof json.studyMeanNumEventsData !== "undefined") {
                updateBarCharts("studyMeanNumEvents", json);
            }
            if(typeof json.studyMedianNumEventsData !== "undefined") {
                updateBarCharts("studyMedianNumEvents", json);
            }
            if(typeof json.studyBoxplotNumEventsData !=="undefined") {
                $("#studyBoxplotNumEventsChartDiv").html(json.studyBoxplotNumEventsData);
            }
            if(typeof json.studyHistogramNumEventsData !=="undefined") {
                $("#studyHistogramNumEventsChartDiv").html(json.studyHistogramNumEventsData);
            }

            // Number of events per standard study name chart data
            if(typeof json.standardStudyMeanNumEventsData !== "undefined") {
                updateBarCharts("standardStudyMeanNumEvents", json);
            }
            if(typeof json.standardStudyMedianNumEventsData !== "undefined") {
                updateBarCharts("standardStudyMedianNumEvents", json);
            }
            if(typeof json.standardStudyBoxplotNumEventsData !=="undefined") {
                $("#standardStudyBoxplotNumEventsChartDiv").html(json.standardStudyBoxplotNumEventsData);
            }
            if(typeof json.standardStudyHistogramNumEventsData !=="undefined") {
                $("#standardStudyHistogramNumEventsChartDiv").html(json.standardStudyHistogramNumEventsData);
            }

            // Number of events per request chart data
            if(typeof json.requestMeanNumEventsData !== "undefined") {
                updateBarCharts("requestMeanNumEvents", json);
            }
            if(typeof json.requestMedianNumEventsData !== "undefined") {
                updateBarCharts("requestMedianNumEvents", json);
            }
            if(typeof json.requestBoxplotNumEventsData !=="undefined") {
                $("#requestBoxplotNumEventsChartDiv").html(json.requestBoxplotNumEventsData);
            }
            if(typeof json.requestHistogramNumEventsData !=="undefined") {
                $("#requestHistogramNumEventsChartDiv").html(json.requestHistogramNumEventsData);
            }

            // Requested procedure DLP over time chart data
            if(typeof json.requestMeanDLPOverTime !== "undefined") {
                $("#requestMeanDLPOverTimeChartDiv").html(json.requestMeanDLPOverTime);
            }
            if(typeof json.requestMedianDLPOverTime !== "undefined") {
                $("#requestMedianDLPOverTimeChartDiv").html(json.requestMedianDLPOverTime);
            }

            // Acquisition frequency chart data start
            if(typeof json.acquisitionFrequencyData !== "undefined") {
                $("#acquisitionFrequencyChartDiv").html(json.acquisitionFrequencyData);
                $("#acquisitionFrequencyChartParentDiv").append(json.acquisitionFrequencyDataCSV);
            }

            // Standard acquisition name frequency chart data start
            if(typeof json.standardAcquisitionFrequencyData !== "undefined") {
                $("#standardAcquisitionFrequencyChartDiv").html(json.standardAcquisitionFrequencyData);
                $("#standardAcquisitionFrequencyChartParentDiv").append(json.standardAcquisitionFrequencyDataCSV);
            }

            // Acqusition scatter of CTDI vs patient mass
            if(typeof json.acquisitionScatterCTDIvsMass !== "undefined") {
                $("#acquisitionScatterCTDIvsMassChartDiv").html(json.acquisitionScatterCTDIvsMass);
            }

            // Standard acqusition name scatter of CTDI vs patient mass
            if(typeof json.standardAcquisitionScatterCTDIvsMass !== "undefined") {
                $("#standardAcquisitionScatterCTDIvsMassChartDiv").html(json.standardAcquisitionScatterCTDIvsMass);
            }

            // Acqusition scatter of DLP vs patient mass
            if(typeof json.acquisitionScatterDLPvsMass !== "undefined") {
                $("#acquisitionScatterDLPvsMassChartDiv").html(json.acquisitionScatterDLPvsMass);
            }

            // Standard acqusition name scatter of DLP vs patient mass
            if(typeof json.standardAcquisitionScatterDLPvsMass !== "undefined") {
                $("#standardAcquisitionScatterDLPvsMassChartDiv").html(json.standardAcquisitionScatterDLPvsMass);
            }

            // Study frequency chart data start
            if(typeof json.studyFrequencyData !== "undefined") {
                $("#studyFrequencyChartDiv").html(json.studyFrequencyData);
                $("#studyFrequencyChartParentDiv").append(json.studyFrequencyDataCSV);
            }

            // Standard study name frequency chart data start
            if(typeof json.standardStudyFrequencyData !== "undefined") {
                $("#standardStudyFrequencyChartDiv").html(json.standardStudyFrequencyData);
                $("#standardStudyFrequencyChartParentDiv").append(json.standardStudyFrequencyDataCSV);
            }

            // Request frequency chart data start
            if(typeof json.requestFrequencyData !== "undefined") {
                $("#requestFrequencyChartDiv").html(json.requestFrequencyData);
                $("#requestFrequencyChartParentDiv").append(json.requestFrequencyDataCSV);
            }

            // DLP over time chart data
            if(typeof json.studyMeanDLPOverTime !== "undefined") {
                $("#studyMeanDLPOverTimeChartDiv").html(json.studyMeanDLPOverTime);
            }
            if(typeof json.studyMedianDLPOverTime !== "undefined") {
                $("#studyMedianDLPOverTimeChartDiv").html(json.studyMedianDLPOverTime);
            }

            // DLP over time chart data
            if(typeof json.standardStudyMeanDLPOverTime !== "undefined") {
                $("#standardStudyMeanDLPOverTimeChartDiv").html(json.standardStudyMeanDLPOverTime);
            }
            if(typeof json.standardStudyMedianDLPOverTime !== "undefined") {
                $("#standardStudyMedianDLPOverTimeChartDiv").html(json.standardStudyMedianDLPOverTime);
            }

            // Study workload chart data
            if(typeof json.studyWorkloadData !== "undefined") {
                $("#studyWorkloadChartDiv").html(json.studyWorkloadData);
            }

            // Standard study workload chart data
            if(typeof json.standardStudyWorkloadData !== "undefined") {
                $("#standardStudyWorkloadChartDiv").html(json.standardStudyWorkloadData);
            }

            $(".ajax-progress").hide();
        },
        error: function( xhr, status, errorThrown ) {
            $(".ajax-progress").hide();
            $(".ajax-error").show();
            console.log( "Error: " + errorThrown );
            console.log( "Status: " + status );
            console.dir( xhr );
        }
    });
    return false;
});