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
            if(typeof json.acquisitionHistogramDLPData !=="undefined") {
                $("#acquisitionHistogramDLPChartDiv").html(json.acquisitionHistogramDLPData);
            }

            // CTDI per acquisition chart data
            if(typeof json.acquisitionMeanCTDIData !== "undefined") {
                $("#acquisitionMeanCTDIChartDiv").html(json.acquisitionMeanCTDIData);
            }
            if(typeof json.acquisitionBoxplotCTDIData !=="undefined") {
                $("#acquisitionMedianCTDIChartDiv").html(json.acquisitionBoxplotCTDIData);
            }
            if(typeof json.acquisitionHistogramCTDIData !=="undefined") {
                $("#acquisitionHistogramCTDIChartDiv").html(json.acquisitionHistogramCTDIData);
            }

            // Acquisition CTDI over time chart data
            if(typeof json.acquisitionMeanCTDIOverTime !== "undefined") {
                $("#acquisitionMeanCTDIOverTimeChartDiv").html(json.acquisitionMeanCTDIOverTime);
            }
            if(typeof json.acquisitionMedianCTDIOverTime !== "undefined") {
                $("#acquisitionMedianCTDIOverTimeChartDiv").html(json.acquisitionMedianCTDIOverTime);
            }

            // Acquisition DLP over time chart data
            if(typeof json.acquisitionMeanDLPOverTime !== "undefined") {
                $("#acquisitionMeanDLPOverTimeChartDiv").html(json.acquisitionMeanDLPOverTime);
            }
            if(typeof json.acquisitionMedianDLPOverTime !== "undefined") {
                $("#acquisitionMedianDLPOverTimeChartDiv").html(json.acquisitionMedianDLPOverTime);
            }


            // DLP per study chart data
            if(typeof json.studyMeanDLPData !== "undefined") {
                $("#studyMeanDLPChartDiv").html(json.studyMeanDLPData);
            }
            if(typeof json.studyBoxplotDLPData !=="undefined") {
                $("#studyMedianDLPChartDiv").html(json.studyBoxplotDLPData);
            }
            if(typeof json.studyHistogramDLPData !=="undefined") {
                $("#studyHistogramDLPChartDiv").html(json.studyHistogramDLPData);
            }

            // CTDI per study chart data
            if(typeof json.studyMeanCTDIData !== "undefined") {
                $("#studyMeanCTDIChartDiv").html(json.studyMeanCTDIData);
            }
            if(typeof json.studyBoxplotCTDIData !=="undefined") {
                $("#studyMedianCTDIChartDiv").html(json.studyBoxplotCTDIData);
            }
            if(typeof json.studyHistogramCTDIData !=="undefined") {
                $("#studyHistogramCTDIChartDiv").html(json.studyHistogramCTDIData);
            }

            // DLP per request chart data start
            if(typeof json.requestMeanData !== "undefined") {
                $("#requestMeanDLPChartDiv").html(json.requestMeanData);
            }
            if(typeof json.requestBoxplotData !=="undefined") {
                $("#requestMedianDLPChartDiv").html(json.requestBoxplotData);
            }
            if(typeof json.requestHistogramData !=="undefined") {
                $("#requestHistogramDLPChartDiv").html(json.requestHistogramData);
            }

            // Number of events per study chart data
            if(typeof json.studyMeanNumEventsData !== "undefined") {
                $("#studyMeanNumEventsChartDiv").html(json.studyMeanNumEventsData);
            }
            if(typeof json.studyBoxplotNumEventsData !=="undefined") {
                $("#studyMedianNumEventsChartDiv").html(json.studyBoxplotNumEventsData);
            }
            if(typeof json.studyHistogramNumEventsData !=="undefined") {
                $("#studyHistogramNumEventsChartDiv").html(json.studyHistogramNumEventsData);
            }

            // Number of events per request chart data
            if(typeof json.requestMeanNumEventsData !== "undefined") {
                $("#requestMeanNumEventsChartDiv").html(json.requestMeanNumEventsData);
            }
            if(typeof json.requestBoxplotNumEventsData !=="undefined") {
                $("#requestMedianNumEventsChartDiv").html(json.requestBoxplotNumEventsData);
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
            }

            // Acqusition scatter of CTDI vs patient mass
            if(typeof json.acquisitionScatterCTDIvsMass !== "undefined") {
                $("#acquisitionScatterCTDIvsMassChartDiv").html(json.acquisitionScatterCTDIvsMass);
            }

            // Acqusition scatter of DLP vs patient mass
            if(typeof json.acquisitionScatterDLPvsMass !== "undefined") {
                $("#acquisitionScatterDLPvsMassChartDiv").html(json.acquisitionScatterDLPvsMass);
            }

            // Study frequency chart data start
            if(typeof json.studyFrequencyData !== "undefined") {
                $("#studyFrequencyChartDiv").html(json.studyFrequencyData);
            }

            // Request frequency chart data start
            if(typeof json.requestFrequencyData !== "undefined") {
                $("#requestFrequencyChartDiv").html(json.requestFrequencyData);
            }

            // DLP over time chart data
            if(typeof json.studyMeanDLPOverTime !== "undefined") {
                $("#studyMeanDLPOverTimeChartDiv").html(json.studyMeanDLPOverTime);
            }
            if(typeof json.studyMedianDLPOverTime !== "undefined") {
                $("#studyMedianDLPOverTimeChartDiv").html(json.studyMedianDLPOverTime);
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