/*global arrayToURL, urlToArray, chroma, hideButtonsIfOneSeries, updateWorkloadChart, updateScatterChart*/
/*eslint no-undef: "error"*/
/*eslint object-shorthand: "off" */

// Code to update the page and chart data on initial page load.
$(document).ready(function() {
    var requestData = arrayToURL(urlToArray(this.URL));

    $(".ajax-progress").show();

    $.ajax({
        type: "GET",
        url: Urls.mg_summary_chart_data(),
        data: requestData,
        dataType: "json",
        success: function( json ) {

            // AGD per acquisition chart data
            if(typeof json.acquisitionMeanAGDData !== "undefined") {
                $("#acquisitionMeanAGDChartDiv").html(json.acquisitionMeanAGDData);
            }
            if(typeof json.acquisitionMedianAGDData !== "undefined") {
                $("#acquisitionMedianAGDChartDiv").html(json.acquisitionMedianAGDData);
            }
            if(typeof json.acquisitionBoxplotAGDData !=="undefined") {
                $("#acquisitionBoxplotAGDChartDiv").html(json.acquisitionBoxplotAGDData);
            }
            if(typeof json.acquisitionHistogramAGDData !=="undefined") {
                $("#acquisitionHistogramAGDChartDiv").html(json.acquisitionHistogramAGDData);
            }

            // AGD per acquisition grouped into compressed breast thickness bands
            if(typeof json.meanAGDvsThickness !== "undefined") {
                $("#acquisitionMeanAGDvsThickChartDiv").html(json.meanAGDvsThickness);
            }
            if(typeof json.medianAGDvsThickness !== "undefined") {
                $("#acquisitionMedianAGDvsThickChartDiv").html(json.medianAGDvsThickness);
            }

            // Study workload chart data
            if(typeof json.studyWorkloadData !== "undefined") {
                $("#studyWorkloadChartDiv").html(json.studyWorkloadData);
            }

            // AGD vs compressed thickness scatter plot
            if(typeof json.AGDvsThickness !== "undefined") {
                $("#acquisitionScatterAGDvsThickChartDiv").html(json.AGDvsThickness);
            }

            // kVp vs compressed thickness scatter plot
            if(typeof json.kVpvsThickness !== "undefined") {
                $("#acquisitionScatterkVpvsThickChartDiv").html(json.kVpvsThickness);
            }

            // mAs vs compressed thickness scatter plot
            if(typeof json.mAsvsThickness !== "undefined") {
                $("#acquisitionScattermAsvsThickChartDiv").html(json.mAsvsThickness);
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