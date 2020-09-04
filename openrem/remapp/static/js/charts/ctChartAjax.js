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
            if(typeof json.acquisitionDLPData !== "undefined") {
                vegaEmbed('#acquisitionAverageDLPChartDiv',  JSON.parse(json.acquisitionDLPData)).catch(console.error);
            }

            // CTDI per acquisition chart data
            if(typeof json.acquisitionCTDIData !== "undefined") {
                vegaEmbed('#acquisitionAverageCTDIChartDiv',  JSON.parse(json.acquisitionCTDIData)).catch(console.error);
            }

            // DLP per study chart data
            if(typeof json.studyDLPData !== "undefined") {
                vegaEmbed('#studyAverageDLPChartDiv',  JSON.parse(json.studyDLPData)).catch(console.error);
            }

            // CTDI per study chart data
            if(typeof json.studyCTDIData !== "undefined") {
                vegaEmbed('#studyAverageCTDIChartDiv',  JSON.parse(json.studyCTDIData)).catch(console.error);
            }

            // DLP per request chart data start
            if(typeof json.requestData !== "undefined") {
                vegaEmbed('#requestAverageDLPChartDiv',  JSON.parse(json.requestData)).catch(console.error);
                if(typeof json.requestHistData !== "undefined") {
                    vegaEmbed('#requestHistogramDLPChartDiv', JSON.parse(json.requestHistData)).catch(console.error);
                }
            }

            // Number of events per study chart data
            if(typeof json.studyNumEventsData !== "undefined") {
                vegaEmbed('#studyAverageNumEventsChartDiv',  JSON.parse(json.studyNumEventsData)).catch(console.error);
            }

            // Number of events per request chart data
            if(typeof json.requestNumEventsData !== "undefined") {
                vegaEmbed('#requestAverageNumEventsChartDiv',  JSON.parse(json.requestNumEventsData)).catch(console.error);
            }

            // Acquisition frequency chart data start
            if(typeof json.acquisitionFreqData !== "undefined") {
                vegaEmbed('#acquisitionFreqChartDiv',  JSON.parse(json.acquisitionFreqData)).catch(console.error);
            }

            // Study frequency chart data start
            if(typeof json.studyFreqData !== "undefined") {
                vegaEmbed('#studyFreqChartDiv',  JSON.parse(json.studyFreqData)).catch(console.error);
            }

            // Request frequency chart data start
            if(typeof json.requestFreqData !== "undefined") {
                vegaEmbed('#requestFreqChartDiv',  JSON.parse(json.requestFreqData)).catch(console.error);
            }

            // DLP over time chart data
            if(typeof json.studyDLPoverTime !== "undefined") {
                vegaEmbed('#studyAverageDLPOverTimeChartDiv',  JSON.parse(json.studyDLPoverTime)).catch(console.error);
            }

            // Study workload chart data
            if(typeof json.studiesPerHourInWeekdays !== "undefined") {
                vegaEmbed('#piechartStudyWorkloadDIV',  JSON.parse(json.studiesPerHourInWeekdays)).catch(console.error);
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