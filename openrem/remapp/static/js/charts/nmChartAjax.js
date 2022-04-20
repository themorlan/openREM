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
                $("#studyFrequencyChartDiv").html(json.studyFrequencyData);
                $("#studyFrequencyChartParentDiv").append(json.studyFrequencyDataCSV);
            }
            if(typeof json.studyWorkloadData !== "undefined") {
                $("#studyWorkloadChartDiv").html(json.studyWorkloadData);
            }
            if(typeof json.studyInjectedDoseOverWeightData !== "undefined") {
                $("#studyInjectedDoseOverWeightChartDiv").html(json.studyInjectedDoseOverWeightData);
            }
            if(typeof json.studyMeanInjectedDoseData !== "undefined") {
                $("#studyInjectedDoseMeanChartDiv").html(json.studyMeanInjectedDoseData)
                $("#studyInjectedDoseMeanParentDiv").append(json.studyMeanInjectedDoseDataCSV)
            }
            if(typeof json.studyMedianInjectedDoseData !== "undefined") {
                $("#studyInjectedDoseMedianChartDiv").html(json.studyMedianInjectedDoseData)
                $("#studyInjectedDoseMedianParentDiv").append(json.studyMedianInjectedDoseDataCSV)
            }
            if(typeof json.studyBoxplotInjectedDoseData !== "undefined") {
                $("#studyInjectedDoseBoxplotChartDiv").html(json.studyBoxplotInjectedDoseData)
            }
            if(typeof json.studyHistogramInjectedDoseData !== "undefined") {
                $("#studyInjectedDoseHistogramChartDiv").html(json.studyHistogramInjectedDoseData)
            }
            if(typeof json.studyInjectedDoseOverTimeMeanData !== "undefined") {
                $("#studyInjectedDoseOverTimeMeanChartDiv").html(json.studyInjectedDoseOverTimeMeanData)
            }
            if(typeof json.studyInjectedDoseOverTimeMedianData !== "undefined") {
                $("#studyInjectedDoseOverTimeMedianChartDiv").html(json.studyInjectedDoseOverTimeMedianData)
            }
            if(typeof json.studyInjectedDoseOverTimeBoxplotData !== "undefined") {
                $("#studyInjectedDoseOverTimeBoxplotChartDiv").html(json.studyInjectedDoseOverTimeBoxplotData)
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