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
            }
            if(typeof json.studyMedianData !=="undefined") {
                $("#studyMedianDAPChartDiv").html(json.studyMedianData);
            }
            if(typeof json.studyMeanMedianData !=="undefined") {
                $("#studyMeanMedianDAPChartDiv").html(json.studyMeanMedianData);
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
            }
            if(typeof json.requestMedianData !== "undefined") {
                $("#requestMedianDAPChartDiv").html(json.requestMedianData);
            }
            if(typeof json.requestMeanMedianData !=="undefined") {
                $("#requestMeanMedianDAPChartDiv").html(json.requestMeanMedianData);
            }
            if(typeof json.requestBoxplotData !=="undefined") {
                $("#requestBoxplotDAPChartDiv").html(json.requestBoxplotData);
            }
            if(typeof json.requestHistogramData !=="undefined") {
                $("#requestHistogramDAPChartDiv").html(json.requestHistogramData);
            }

            // Study frequency chart data start
            if(typeof json.studyFrequencyData !== "undefined") {
                $("#studyFrequencyChartDiv").html(json.studyFrequencyData);
            }

            // Request frequency chart data start
            if(typeof json.requestFrequencyData !== "undefined") {
                $("#requestFrequencyChartDiv").html(json.requestFrequencyData);
            }

            // Study workload chart data
            if(typeof json.studyWorkloadData !== "undefined") {
                $("#studyWorkloadChartDiv").html(json.studyWorkloadData);
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