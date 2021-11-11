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
                $("#acquisitionMeanDLPChartParentDiv").append(json.acquisitionMeanDLPDataCSV);
            }
            if(typeof json.acquisitionMedianDLPData !== "undefined") {
                $("#acquisitionMedianDLPChartDiv").html(json.acquisitionMedianDLPData);
                $("#acquisitionMedianDLPChartParentDiv").append(json.acquisitionMedianDLPDataCSV);
            }
            if(typeof json.acquisitionBoxplotDLPData !=="undefined") {
                $("#acquisitionBoxplotDLPChartDiv").html(json.acquisitionBoxplotDLPData);
            }
            if(typeof json.acquisitionHistogramDLPData !=="undefined") {
                $("#acquisitionHistogramDLPChartDiv").html(json.acquisitionHistogramDLPData);
            }

            // DLP per standard acquisition name chart data
            if(typeof json.standardAcquisitionMeanDLPData !== "undefined") {
                $("#standardAcquisitionMeanDLPChartDiv").html(json.standardAcquisitionMeanDLPData);
                $("#standardAcquisitionMeanDLPChartParentDiv").append(json.standardAcquisitionMeanDLPDataCSV);
            }
            if(typeof json.standardAcquisitionMedianDLPData !== "undefined") {
                $("#standardAcquisitionMedianDLPChartDiv").html(json.standardAcquisitionMedianDLPData);
                $("#standardAcquisitionMedianDLPChartParentDiv").append(json.standardAcquisitionMedianDLPDataCSV);
            }
            if(typeof json.standardAcquisitionBoxplotDLPData !=="undefined") {
                $("#standardAcquisitionBoxplotDLPChartDiv").html(json.standardAcquisitionBoxplotDLPData);
            }
            if(typeof json.standardAcquisitionHistogramDLPData !=="undefined") {
                $("#standardAcquisitionHistogramDLPChartDiv").html(json.standardAcquisitionHistogramDLPData);
            }

            // CTDI per acquisition chart data
            if(typeof json.acquisitionMeanCTDIData !== "undefined") {
                $("#acquisitionMeanCTDIChartDiv").html(json.acquisitionMeanCTDIData);
                $("#acquisitionMeanCTDIChartParentDiv").append(json.acquisitionMeanCTDIDataCSV);
            }
            if(typeof json.acquisitionMedianCTDIData !== "undefined") {
                $("#acquisitionMedianCTDIChartDiv").html(json.acquisitionMedianCTDIData);
                $("#acquisitionMedianCTDIChartParentDiv").append(json.acquisitionMedianCTDIDataCSV);
            }
            if(typeof json.acquisitionBoxplotCTDIData !=="undefined") {
                $("#acquisitionBoxplotCTDIChartDiv").html(json.acquisitionBoxplotCTDIData);
            }
            if(typeof json.acquisitionHistogramCTDIData !=="undefined") {
                $("#acquisitionHistogramCTDIChartDiv").html(json.acquisitionHistogramCTDIData);
            }

            // CTDI per standard acquisition name chart data
            if(typeof json.standardAcquisitionMeanCTDIData !== "undefined") {
                $("#standardAcquisitionMeanCTDIChartDiv").html(json.standardAcquisitionMeanCTDIData);
                $("#standardAcquisitionMeanCTDIChartParentDiv").append(json.standardAcquisitionMeanCTDIDataCSV);
            }
            if(typeof json.standardAcquisitionMedianCTDIData !== "undefined") {
                $("#standardAcquisitionMedianCTDIChartDiv").html(json.standardAcquisitionMedianCTDIData);
                $("#standardAcquisitionMedianCTDIChartParentDiv").append(json.standardAcquisitionMedianCTDIDataCSV);
            }
            if(typeof json.standardAcquisitionBoxplotCTDIData !=="undefined") {
                $("#standardAcquisitionBoxplotCTDIChartDiv").html(json.standardAcquisitionBoxplotCTDIData);
            }
            if(typeof json.standardAcquisitionHistogramCTDIData !=="undefined") {
                $("#standardAcquisitionHistogramCTDIChartDiv").html(json.standardAcquisitionHistogramCTDIData);
            }

            // Acquisition CTDI over time chart data
            if(typeof json.acquisitionMeanCTDIOverTime !== "undefined") {
                $("#acquisitionMeanCTDIOverTimeChartDiv").html(json.acquisitionMeanCTDIOverTime);
            }
            if(typeof json.acquisitionMedianCTDIOverTime !== "undefined") {
                $("#acquisitionMedianCTDIOverTimeChartDiv").html(json.acquisitionMedianCTDIOverTime);
            }

            // Standard acquisition name CTDI over time chart data
            if(typeof json.standardAcquisitionMeanCTDIOverTime !== "undefined") {
                $("#standardAcquisitionMeanCTDIOverTimeChartDiv").html(json.standardAcquisitionMeanCTDIOverTime);
            }
            if(typeof json.standardAcquisitionMedianCTDIOverTime !== "undefined") {
                $("#standardAcquisitionMedianCTDIOverTimeChartDiv").html(json.standardAcquisitionMedianCTDIOverTime);
            }

            // Acquisition DLP over time chart data
            if(typeof json.acquisitionMeanDLPOverTime !== "undefined") {
                $("#acquisitionMeanDLPOverTimeChartDiv").html(json.acquisitionMeanDLPOverTime);
            }
            if(typeof json.acquisitionMedianDLPOverTime !== "undefined") {
                $("#acquisitionMedianDLPOverTimeChartDiv").html(json.acquisitionMedianDLPOverTime);
            }

            // Standard acquisition name DLP over time chart data
            if(typeof json.standardAcquisitionMeanDLPOverTime !== "undefined") {
                $("#standardAcquisitionMeanDLPOverTimeChartDiv").html(json.standardAcquisitionMeanDLPOverTime);
            }
            if(typeof json.standardAcquisitionMedianDLPOverTime !== "undefined") {
                $("#standardAcquisitionMedianDLPOverTimeChartDiv").html(json.standardAcquisitionMedianDLPOverTime);
            }

            // DLP per study chart data
            if(typeof json.studyMeanDLPData !== "undefined") {
                $("#studyMeanDLPChartDiv").html(json.studyMeanDLPData);
                $("#studyMeanDLPChartParentDiv").append(json.studyMeanDLPDataCSV);
            }
            if(typeof json.studyMedianDLPData !== "undefined") {
                $("#studyMedianDLPChartDiv").html(json.studyMedianDLPData);
                $("#studyMedianDLPChartParentDiv").append(json.studyMedianDLPDataCSV);
            }
            if(typeof json.studyBoxplotDLPData !=="undefined") {
                $("#studyBoxplotDLPChartDiv").html(json.studyBoxplotDLPData);
            }
            if(typeof json.studyHistogramDLPData !=="undefined") {
                $("#studyHistogramDLPChartDiv").html(json.studyHistogramDLPData);
            }

            // DLP per standard study chart data
            if(typeof json.standardStudyMeanDLPData !== "undefined") {
                $("#standardStudyMeanDLPChartDiv").html(json.standardStudyMeanDLPData);
                $("#standardStudyMeanDLPChartParentDiv").append(json.standardStudyMeanDLPDataCSV);
            }
            if(typeof json.standardStudyMedianDLPData !== "undefined") {
                $("#standardStudyMedianDLPChartDiv").html(json.standardStudyMedianDLPData);
                $("#standardStudyMedianDLPChartParentDiv").append(json.standardStudyMedianDLPDataCSV);
            }
            if(typeof json.standardStudyBoxplotDLPData !=="undefined") {
                $("#standardStudyBoxplotDLPChartDiv").html(json.standardStudyBoxplotDLPData);
            }
            if(typeof json.standardStudyHistogramDLPData !=="undefined") {
                $("#standardStudyHistogramDLPChartDiv").html(json.standardStudyHistogramDLPData);
            }

            // CTDI per study chart data
            if(typeof json.studyMeanCTDIData !== "undefined") {
                $("#studyMeanCTDIChartDiv").html(json.studyMeanCTDIData);
                $("#studyMeanCTDIChartParentDiv").append(json.studyMeanCTDIDataCSV);
            }
            if(typeof json.studyMedianCTDIData !== "undefined") {
                $("#studyMedianCTDIChartDiv").html(json.studyMedianCTDIData);
                $("#studyMedianCTDIChartParentDiv").append(json.studyMedianCTDIDataCSV);
            }
            if(typeof json.studyBoxplotCTDIData !=="undefined") {
                $("#studyBoxplotCTDIChartDiv").html(json.studyBoxplotCTDIData);
            }
            if(typeof json.studyHistogramCTDIData !=="undefined") {
                $("#studyHistogramCTDIChartDiv").html(json.studyHistogramCTDIData);
            }

            // DLP per request chart data start
            if(typeof json.requestMeanDLPData !== "undefined") {
                $("#requestMeanDLPChartDiv").html(json.requestMeanDLPData);
                $("#requestMeanDLPChartParentDiv").append(json.requestMeanDLPDataCSV);
            }
            if(typeof json.requestMedianDLPData !== "undefined") {
                $("#requestMedianDLPChartDiv").html(json.requestMedianDLPData);
                $("#requestMedianDLPChartParentDiv").append(json.requestMedianDLPDataCSV);
            }
            if(typeof json.requestBoxplotDLPData !=="undefined") {
                $("#requestBoxplotDLPChartDiv").html(json.requestBoxplotDLPData);
            }
            if(typeof json.requestHistogramDLPData !=="undefined") {
                $("#requestHistogramDLPChartDiv").html(json.requestHistogramDLPData);
            }

            // Number of events per study chart data
            if(typeof json.studyMeanNumEventsData !== "undefined") {
                $("#studyMeanNumEventsChartDiv").html(json.studyMeanNumEventsData);
                $("#studyMeanNumEventsChartParentDiv").append(json.studyMeanNumEventsDataCSV);
            }
            if(typeof json.studyMedianNumEventsData !== "undefined") {
                $("#studyMedianNumEventsChartDiv").html(json.studyMedianNumEventsData);
                $("#studyMedianNumEventsChartParentDiv").append(json.studyMedianNumEventsDataCSV);
            }
            if(typeof json.studyBoxplotNumEventsData !=="undefined") {
                $("#studyBoxplotNumEventsChartDiv").html(json.studyBoxplotNumEventsData);
            }
            if(typeof json.studyHistogramNumEventsData !=="undefined") {
                $("#studyHistogramNumEventsChartDiv").html(json.studyHistogramNumEventsData);
            }


            // Number of events per standard study name chart data
            if(typeof json.standardStudyMeanNumEventsData !== "undefined") {
                $("#standardStudyMeanNumEventsChartDiv").html(json.standardStudyMeanNumEventsData);
                $("#standardStudyMeanNumEventsChartParentDiv").append(json.standardStudyMeanNumEventsDataCSV);
            }
            if(typeof json.standardStudyMedianNumEventsData !== "undefined") {
                $("#standardStudyMedianNumEventsChartDiv").html(json.standardStudyMedianNumEventsData);
                $("#standardStudyMedianNumEventsChartParentDiv").append(json.standardStudyMedianNumEventsDataCSV);
            }
            if(typeof json.standardStudyBoxplotNumEventsData !=="undefined") {
                $("#standardStudyBoxplotNumEventsChartDiv").html(json.standardStudyBoxplotNumEventsData);
            }
            if(typeof json.standardStudyHistogramNumEventsData !=="undefined") {
                $("#standardStudyHistogramNumEventsChartDiv").html(json.standardStudyHistogramNumEventsData);
            }


            // Number of events per request chart data
            if(typeof json.requestMeanNumEventsData !== "undefined") {
                $("#requestMeanNumEventsChartDiv").html(json.requestMeanNumEventsData);
                $("#requestMeanNumEventsChartParentDiv").append(json.requestMeanNumEventsDataCSV);
            }
            if(typeof json.requestMedianNumEventsData !== "undefined") {
                $("#requestMedianNumEventsChartDiv").html(json.requestMedianNumEventsData);
                $("#requestMedianNumEventsChartParentDiv").append(json.requestMedianNumEventsDataCSV);
            }
            if(typeof json.requestBoxplotNumEventsData !=="undefined") {
                $("#requestBoxplotNumEventsChartDiv").html(json.requestBoxplotNumEventsData);
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
                $("#acquisitionFrequencyChartParentDiv").append(json.acquisitionFrequencyDataCSV);
            }

            // Standard acquisition name frequency chart data start
            if(typeof json.standardAcquisitionFrequencyData !== "undefined") {
                $("#standardAcquisitionFrequencyChartDiv").html(json.standardAcquisitionFrequencyData);
                $("#standardAcquisitionFrequencyChartParentDiv").append(json.standardAcquisitionFrequencyDataCSV);
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
                $("#studyFrequencyChartParentDiv").append(json.studyFrequencyDataCSV);
            }

            // Standard study name frequency chart data start
            if(typeof json.standardStudyFrequencyData !== "undefined") {
                $("#standardStudyFrequencyChartDiv").html(json.standardStudyFrequencyData);
                $("#standardStudyFrequencyChartParentDiv").append(json.standardStudyFrequencyDataCSV);
            }

            // Request frequency chart data start
            if(typeof json.requestFrequencyData !== "undefined") {
                $("#requestFrequencyChartDiv").html(json.requestFrequencyData);
                $("#requestFrequencyChartParentDiv").append(json.requestFrequencyDataCSV);
            }

            // DLP over time chart data
            if(typeof json.studyMeanDLPOverTime !== "undefined") {
                $("#studyMeanDLPOverTimeChartDiv").html(json.studyMeanDLPOverTime);
            }
            if(typeof json.studyMedianDLPOverTime !== "undefined") {
                $("#studyMedianDLPOverTimeChartDiv").html(json.studyMedianDLPOverTime);
            }

            // DLP over time chart data
            if(typeof json.standardStudyMeanDLPOverTime !== "undefined") {
                $("#standardStudyMeanDLPOverTimeChartDiv").html(json.standardStudyMeanDLPOverTime);
            }
            if(typeof json.standardStudyMedianDLPOverTime !== "undefined") {
                $("#standardStudyMedianDLPOverTimeChartDiv").html(json.standardStudyMedianDLPOverTime);
            }

            // Study workload chart data
            if(typeof json.studyWorkloadData !== "undefined") {
                $("#studyWorkloadChartDiv").html(json.studyWorkloadData);
            }

            // Standard study workload chart data
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