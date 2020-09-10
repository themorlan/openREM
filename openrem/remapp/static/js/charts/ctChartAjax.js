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
            if(typeof json.studyDLPData !== "undefined") {
                vegaEmbed('#studyAverageDLPChartDiv',  JSON.parse(json.studyDLPData)).catch(console.error);
                if(typeof json.studyHistDLPData !== "undefined") {
                    vegaEmbed('#studyHistogramDLPChartDiv',  JSON.parse(json.studyHistDLPData)).catch(console.error);
                }
            }

            // CTDI per study chart data
            if(typeof json.studyCTDIData !== "undefined") {
                vegaEmbed('#studyAverageCTDIChartDiv',  JSON.parse(json.studyCTDIData)).catch(console.error);
                if(typeof json.studyHistCTDIData !== "undefined") {
                    vegaEmbed('#studyHistogramCTDIChartDiv',  JSON.parse(json.studyHistCTDIData)).catch(console.error);
                }
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
            if(typeof json.studyNumEventsData !== "undefined") {
                vegaEmbed('#studyAverageNumEventsChartDiv',  JSON.parse(json.studyNumEventsData)).catch(console.error);
                if(typeof json.studyHistNumEventsData !== "undefined") {
                    vegaEmbed('#studyHistogramNumEventsChartDiv',  JSON.parse(json.studyHistNumEventsData)).catch(console.error);
                }
            }

            // Number of events per request chart data
            if(typeof json.requestNumEventsData !== "undefined") {
                vegaEmbed('#requestAverageNumEventsChartDiv',  JSON.parse(json.requestNumEventsData)).catch(console.error);
                if(typeof json.requestNumEventsHistData !== "undefined") {
                    vegaEmbed('#requestHistogramNumEventsChartDiv', JSON.parse(json.requestNumEventsHistData)).catch(console.error);
                }
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