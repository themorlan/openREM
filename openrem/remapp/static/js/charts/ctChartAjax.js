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
        url: Urls.ct_summary_chart_data(),
        data: requestData,
        dataType: "json",
        success: function( json ) {

            // DLP per acquisition chart data
            if(typeof json.acquisitionMeanDLPData !== "undefined") {
                $("#acquisitionMeanDLPChartDiv").html(json.acquisitionMeanDLPData);
            }
            if(typeof json.acquisitionBoxplotDLPData !=="undefined") {
                $("#acquisitionMedianDLPChartDiv").html(json.acquisitionBoxplotDLPData);
            }
            if(typeof json.acquisitionHistDLPData !=="undefined") {
                $("#acquisitionHistogramDLPChartDiv").html(json.acquisitionHistDLPData);
            }

            // CTDI per acquisition chart data
            if(typeof json.acquisitionMeanCTDIData !== "undefined") {
                $("#acquisitionMeanCTDIChartDiv").html(json.acquisitionMeanCTDIData);
            }
            if(typeof json.acquisitionBoxplotCTDIData !=="undefined") {
                $("#acquisitionMedianCTDIChartDiv").html(json.acquisitionBoxplotCTDIData);
            }
            if(typeof json.acquisitionHistCTDIData !=="undefined") {
                $("#acquisitionHistogramCTDIChartDiv").html(json.acquisitionHistCTDIData);
            }

            // DLP per study chart data
            if(typeof json.studyMeanDLPData !== "undefined") {
                $("#studyMeanDLPChartDiv").html(json.studyMeanDLPData);
            }
            if(typeof json.studyBoxplotDLPData !=="undefined") {
                $("#studyMedianDLPChartDiv").html(json.studyBoxplotDLPData);
            }
            if(typeof json.studyHistDLPData !=="undefined") {
                $("#studyHistogramDLPChartDiv").html(json.studyHistDLPData);
            }

            // CTDI per study chart data
            if(typeof json.studyMeanCTDIData !== "undefined") {
                $("#studyMeanCTDIChartDiv").html(json.studyMeanCTDIData);
            }
            if(typeof json.studyBoxplotCTDIData !=="undefined") {
                $("#studyMedianCTDIChartDiv").html(json.studyBoxplotCTDIData);
            }
            if(typeof json.studyHistCTDIData !=="undefined") {
                $("#studyHistogramCTDIChartDiv").html(json.studyHistCTDIData);
            }

            // DLP per request chart data start
            if(typeof json.requestMeanData !== "undefined") {
                $("#requestMeanDLPChartDiv").html(json.requestMeanData);
            }
            if(typeof json.requestBoxplotData !=="undefined") {
                $("#requestMedianDLPChartDiv").html(json.requestBoxplotData);
            }
            if(typeof json.requestHistData !=="undefined") {
                $("#requestHistogramDLPChartDiv").html(json.requestHistData);
            }

            // Number of events per study chart data
            if(typeof json.studyMeanNumEventsData !== "undefined") {
                $("#studyMeanNumEventsChartDiv").html(json.studyMeanNumEventsData);
            }
            if(typeof json.studyBoxplotNumEventsData !=="undefined") {
                $("#studyMedianNumEventsChartDiv").html(json.studyBoxplotNumEventsData);
            }
            if(typeof json.studyHistNumEventsData !=="undefined") {
                $("#studyHistogramNumEventsChartDiv").html(json.studyHistNumEventsData);
            }

            // Number of events per request chart data
            if(typeof json.requestMeanNumEventsData !== "undefined") {
                $("#requestMeanNumEventsChartDiv").html(json.requestMeanNumEventsData);
            }
            if(typeof json.requestBoxplotNumEventsData !=="undefined") {
                $("#requestMedianNumEventsChartDiv").html(json.requestBoxplotNumEventsData);
            }
            if(typeof json.requestHistNumEventsData !=="undefined") {
                $("#requestHistogramNumEventsChartDiv").html(json.requestHistNumEventsData);
            }

            // Acquisition frequency chart data start
            if(typeof json.acquisitionFreqData !== "undefined") {
                $("#acquisitionFreqChartDiv").html(json.acquisitionFreqData);
            }

            // Study frequency chart data start
            if(typeof json.studyFreqData !== "undefined") {
                $("#studyFreqChartDiv").html(json.studyFreqData);
            }

            // Request frequency chart data start
            if(typeof json.requestFreqData !== "undefined") {
                $("#requestFreqChartDiv").html(json.requestFreqData);
            }

            // DLP over time chart data
            if(typeof json.studyDLPoverTime !== "undefined") {
                vegaEmbed('#studyAverageDLPOverTimeChartDiv',  JSON.parse(json.studyDLPoverTime)).catch(console.error);
            }

            // Study workload chart data
            if(typeof json.studyWorkloadData !== "undefined") {
                vegaEmbed('#studyWorkloadChartDiv',  JSON.parse(json.studyWorkloadData)).catch(console.error);
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