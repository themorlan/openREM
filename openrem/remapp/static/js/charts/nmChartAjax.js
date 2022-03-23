/*global arrayToURL, urlToArray, chroma, updateAverageChart, sortChartDataToDefault, hideButtonsIfOneSeries,
updateFrequencyChart, sortByY, sortByName, plotAverageChoice, updateWorkloadChart, chartSorting, urlStartAcq,
urlStartReq, urlStartStudy, updateOverTimeChart, chartSortingDirection*/
/*eslint no-undef: "error"*/
/*eslint security/detect-object-injection: "off" */
/*eslint object-shorthand: "off" */

// Code to update the page and chart data on initial page load.
$(document).ready(function() {
    var requestData = arrayToURL(urlToArray(this.URL));

    $(".ajax-progress").show();

    $.ajax({
        type: "GET",
        url: Urls.nm_summary_chart_data(),
        data: requestData,
        dataType: "json",
        success: function( json ) {

            if(typeof json.studyFrequencyData !== "undefined") {
                $("#studyFrequencyChartDiv").html(json.studyFrequencyData)
                $("#studyFrequencyChartParentDiv").append(json.studyFrequencyDataCSV)
            }
            if(typeof json.studyWorkloadData !== "undefined") {
                $("#studyWorkloadChartDiv").html(json.studyWorkloadData)
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