/*global arrayToURL, urlToArray, chroma, updateAverageChart, sortChartDataToDefault, hideButtonsIfOneSeries,
updateFrequencyChart, sortByY, sortByName, plotAverageChoice, updateWorkloadChart, chartSorting, urlStartAcq,
urlStartReq, urlStartStudy, updateOverTimeChart, chartSortingDirection*/
/*eslint no-undef: "error"*/
/*eslint security/detect-object-injection: "off" */
/*eslint object-shorthand: "off" */

// Code to update the page and chart data on initial page load.
$(document).ready(function() {
    var requestData = arrayToURL(urlToArray(this.URL));
    var i;

    $(".ajax-progress").show();

    $.ajax({
        type: "GET",
        url: Urls.ct_summary_chart_data(),
        data: requestData,
        dataType: "json",
        success: function( json ) {
            // Initialise some colours to use for plotting
            var colourScale = chroma.scale("RdYlBu");

            // DLP per acquisition chart data
            if(typeof plotCTAcquisitionMeanDLP !== "undefined") {
                vegaEmbed('#acquisitionAverageDLPChartDiv',  JSON.parse(json.acquisitionDLPData)).catch(console.error);
            }

            // CTDI per acquisition chart data
            if(typeof plotCTAcquisitionMeanCTDI !== "undefined") {
                vegaEmbed('#acquisitionAverageCTDIChartDiv',  JSON.parse(json.acquisitionCTDIData)).catch(console.error);
            }

            // DLP per study chart data
            if(typeof plotCTStudyMeanDLP !== "undefined") {
                vegaEmbed('#studyAverageDLPChartDiv',  JSON.parse(json.studyDLPData)).catch(console.error);
            }

            // CTDI per study chart data
            if(typeof plotCTStudyMeanCTDI !== "undefined") {
                vegaEmbed('#studyAverageCTDIChartDiv',  JSON.parse(json.studyCTDIData)).catch(console.error);
            }

            // DLP per request chart data start
            if(typeof plotCTRequestMeanDLP !== "undefined") {
                vegaEmbed('#requestAverageDLPChartDiv',  JSON.parse(json.requestData)).catch(console.error);
            }

            // Number of events per study chart data
            if(typeof plotCTStudyNumEvents !== "undefined") {
                updateAverageChart(json.studyNameList, json.studySystemList, json.studySummaryNumEvents, json.studyHistogramDataNumEvents, plotAverageChoice, "studyNumEventsDIV", colourScale);
                sortChartDataToDefault(chartSorting, chartSortingDirection, "studyNumEventsDIV");
            }

            // Number of events per request chart data
            if(typeof plotCTRequestNumEvents !== "undefined") {
                updateAverageChart(json.requestNameList, json.requestSystemList, json.requestSummaryNumEvents, json.requestHistogramDataNumEvents, plotAverageChoice, "requestPlotNumEventsDIV", colourScale);
                sortChartDataToDefault(chartSorting, chartSortingDirection, "requestPlotNumEventsDIV");
            }

            // Acquisition frequency chart data start
            if(typeof plotCTAcquisitionFreq !== "undefined" || typeof plotCTAcquisitionMeanCTDI !== "undefined") {
                vegaEmbed('#acquisitionFreqChartDiv',  JSON.parse(json.acquisitionFreqData)).catch(console.error);
            }

            // Study frequency chart data start
            if(typeof plotCTStudyFreq !== "undefined") {
                vegaEmbed('#studyFreqChartDiv',  JSON.parse(json.studyFreqData)).catch(console.error);
            }

            // Request frequency chart data start
            if(typeof plotCTRequestFreq !== "undefined") {
                vegaEmbed('#requestFreqChartDiv',  JSON.parse(json.requestFreqData)).catch(console.error);
            }

            // DLP over time chart data
            if(typeof plotCTStudyMeanDLPOverTime !== "undefined") {
                var studyLineColours = new Array(json.studyNameList.length);
                if (typeof plotCTStudyFreq !== "undefined") {
                    studyLineColours = [];
                    var piechartStudyDiv = $("#piechartStudyDIV");
                    for (i = 0; i < piechartStudyDiv.highcharts().series[0].data.length; i++) {
                        studyLineColours.push(piechartStudyDiv.highcharts().series[0].data.sort(sortByName)[i].color);
                    }
                    piechartStudyDiv.highcharts().series[0].data.sort(sortByY);
                }
                else {studyLineColours = colourScale.colors(json.studyNameList.length);}

                var studyDlpOverTime = (plotAverageChoice === "mean") ? json.studyMeanDLPoverTime : json.studyMedianDLPoverTime;
                updateOverTimeChart(json.studyNameList, studyDlpOverTime, studyLineColours, urlStartStudyOverTime, "studyMeanDLPOverTimeDIV");
                hideButtonsIfOneSeries("studyMeanDLPOverTimeDIV", "study_dlp_time_series_");
            }

            // Study workload chart data
            if(typeof plotCTStudyPerDayAndHour !== "undefined") {
                updateWorkloadChart(json.studiesPerHourInWeekdays, "piechartStudyWorkloadDIV", colourScale);
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