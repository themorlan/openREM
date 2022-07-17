/* global arrayToURL, urlToArray, chroma, updateAverageChart, sortChartDataToDefault, hideButtonsIfOneSeries, updateFrequencyChart, plotAverageChoice, updateWorkloadChart, urlStartStudy, urlStartRequest, chartSortingDirection */
/*eslint no-undef: "error"*/
/*eslint object-shorthand: "off" */

// Code to update the page and chart data on initial page load.
$(document).ready(function() {
    var requestData = arrayToURL(urlToArray(this.URL));

    $(".ajax-progress").show();

    $.ajax({
        type: "GET",
        url: Urls.rf_summary_chart_data(),
        data: requestData,
        dataType: "json",
        success: function( json ) {

            // DLP per study chart data
            if(typeof json.studyMeanData !== "undefined") {
                $("#studyMeanDAPChartDiv").html(json.studyMeanData);
                $("#studyMeanDAPChartParentDiv").append(json.studyMeanDataCSV);
            }
            if(typeof json.studyMedianData !=="undefined") {
                $("#studyMedianDAPChartDiv").html(json.studyMedianData);
                $("#studyMedianDAPChartParentDiv").append(json.studyMedianDataCSV);
            }
            if(typeof json.studyBoxplotData !=="undefined") {
                $("#studyBoxplotDAPChartDiv").html(json.studyBoxplotData);
            }
            if(typeof json.studyHistogramData !=="undefined") {
                $("#studyHistogramDAPChartDiv").html(json.studyHistogramData);
            }

            // DLP per request chart data
            if(typeof json.requestMeanData !== "undefined") {
                $("#requestMeanDAPChartDiv").html(json.requestMeanData);
                $("#requestMeanDAPChartParentDiv").append(json.requestMeanDataCSV);
            }
            if(typeof json.requestMedianData !== "undefined") {
                $("#requestMedianDAPChartDiv").html(json.requestMedianData);
                $("#requestMedianDAPChartParentDiv").append(json.requestMedianDataCSV);
            }
            if(typeof json.requestBoxplotData !=="undefined") {
                $("#requestBoxplotDAPChartDiv").html(json.requestBoxplotData);
            }
            if(typeof json.requestHistogramData !=="undefined") {
                $("#requestHistogramDAPChartDiv").html(json.requestHistogramData);
            }

            // Requested procedure DAP over time chart data
            if(typeof json.requestMeanDAPOverTime !== "undefined") {
                $("#requestMeanDAPOverTimeChartDiv").html(json.requestMeanDAPOverTime);
            }
            if(typeof json.requestMedianDAPOverTime !== "undefined") {
                $("#requestMedianDAPOverTimeChartDiv").html(json.requestMedianDAPOverTime);
            }

            // Study DAP over time chart data
            if(typeof json.studyMeanDAPOverTime !== "undefined") {
                $("#studyMeanDAPOverTimeChartDiv").html(json.studyMeanDAPOverTime);
            }
            if(typeof json.studyMedianDAPOverTime !== "undefined") {
                $("#studyMedianDAPOverTimeChartDiv").html(json.studyMedianDAPOverTime);
            }

            // Study frequency chart data start
            if(typeof json.studyFrequencyData !== "undefined") {
                $("#studyFrequencyChartDiv").html(json.studyFrequencyData);
                $("#studyFrequencyChartParentDiv").append(json.studyFrequencyDataCSV);
            }

            // Request frequency chart data start
            if(typeof json.requestFrequencyData !== "undefined") {
                $("#requestFrequencyChartDiv").html(json.requestFrequencyData);
                $("#requestFrequencyChartParentDiv").append(json.requestFrequencyDataCSV);
            }

            // Study workload chart data
            if(typeof json.studyWorkloadData !== "undefined") {
                $("#studyWorkloadChartDiv").html(json.studyWorkloadData);
            }

            // DLP per standard study namechart data
            if(typeof json.standardStudyMeanData !== "undefined") {
                $("#standardStudyMeanDAPChartDiv").html(json.standardStudyMeanData);
                $("#standardStudyMeanDAPChartParentDiv").append(json.standardStudyMeanDataCSV);
            }
            if(typeof json.standardStudyMedianData !=="undefined") {
                $("#standardStudyMedianDAPChartDiv").html(json.standardStudyMedianData);
                $("#standardStudyMedianDAPChartParentDiv").append(json.standardStudyMedianDataCSV);
            }
            if(typeof json.standardStudyBoxplotData !=="undefined") {
                $("#standardStudyBoxplotDAPChartDiv").html(json.standardStudyBoxplotData);
            }
            if(typeof json.standardStudyHistogramData !=="undefined") {
                $("#standardStudyHistogramDAPChartDiv").html(json.standardStudyHistogramData);
            }

            // Standard study nameDAP over time chart data
            if(typeof json.standardStudyMeanDAPOverTime !== "undefined") {
                $("#standardStudyMeanDAPOverTimeChartDiv").html(json.standardStudyMeanDAPOverTime);
            }
            if(typeof json.standardStudyMedianDAPOverTime !== "undefined") {
                $("#standardStudyMedianDAPOverTimeChartDiv").html(json.standardStudyMedianDAPOverTime);
            }

            // Standard study namefrequency chart data start
            if(typeof json.standardStudyFrequencyData !== "undefined") {
                $("#standardStudyFrequencyChartDiv").html(json.standardStudyFrequencyData);
                $("#standardStudyFrequencyChartParentDiv").append(json.standardStudyFrequencyDataCSV);
            }

            // Standard study nameworkload chart data
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