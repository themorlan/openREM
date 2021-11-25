/*global arrayToURL, urlToArray, chroma, updateAverageChart, sortChartDataToDefault, hideButtonsIfOneSeries,
updateFrequencyChart, sortByY, sortByName, plotAverageChoice, updateWorkloadChart, updateOverTimeChart, urlStartAcq,
urlStartReq, urlStartStudy*/
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
        url: Urls.dx_summary_chart_data(),
        data: requestData,
        dataType: "json",
        success: function( json ) {

            // DAP per acquisition chart data
            if(typeof json.acquisitionMeanDAPData !== "undefined") {
                $("#acquisitionMeanDAPChartDiv").html(json.acquisitionMeanDAPData);
                $("#acquisitionMeanDAPChartParentDiv").append(json.acquisitionMeanDAPDataCSV);
            }
            if(typeof json.acquisitionMedianDAPData !== "undefined") {
                $("#acquisitionMedianDAPChartDiv").html(json.acquisitionMedianDAPData);
                $("#acquisitionMedianDAPChartParentDiv").append(json.acquisitionMedianDAPDataCSV);
            }
            if(typeof json.acquisitionBoxplotDAPData !=="undefined") {
                $("#acquisitionBoxplotDAPChartDiv").html(json.acquisitionBoxplotDAPData);
            }
            if(typeof json.acquisitionHistogramDAPData !=="undefined") {
                $("#acquisitionHistogramDAPChartDiv").html(json.acquisitionHistogramDAPData);
            }

            // Acquisition frequency chart data start
            if(typeof json.acquisitionFrequencyData !== "undefined") {
                $("#acquisitionFrequencyChartDiv").html(json.acquisitionFrequencyData);
                $("#acquisitionFrequencyChartParentDiv").append(json.acquisitionFrequencyDataCSV);
            }

            // kVp per acquisition chart data
            if(typeof json.acquisitionMeankVpData !== "undefined") {
                $("#acquisitionMeankVpChartDiv").html(json.acquisitionMeankVpData);
                $("#acquisitionMeankVpChartParentDiv").append(json.acquisitionMeankVpDataCSV);
            }
            if(typeof json.acquisitionMediankVpData !== "undefined") {
                $("#acquisitionMediankVpChartDiv").html(json.acquisitionMediankVpData);
                $("#acquisitionMediankVpChartParentDiv").append(json.acquisitionMediankVpDataCSV);
            }
            if(typeof json.acquisitionBoxplotkVpData !=="undefined") {
                $("#acquisitionBoxplotkVpChartDiv").html(json.acquisitionBoxplotkVpData);
            }
            if(typeof json.acquisitionHistogramkVpData !=="undefined") {
                $("#acquisitionHistogramkVpChartDiv").html(json.acquisitionHistogramkVpData);
            }

            // mAs per acquisition chart data
            if(typeof json.acquisitionMeanmAsData !== "undefined") {
                $("#acquisitionMeanmAsChartDiv").html(json.acquisitionMeanmAsData);
                $("#acquisitionMeanmAsChartParentDiv").append(json.acquisitionMeanmAsDataCSV);
            }
            if(typeof json.acquisitionMedianmAsData !== "undefined") {
                $("#acquisitionMedianmAsChartDiv").html(json.acquisitionMedianmAsData);
                $("#acquisitionMedianmAsChartParentDiv").append(json.acquisitionMedianmAsDataCSV);
            }
            if(typeof json.acquisitionBoxplotmAsData !=="undefined") {
                $("#acquisitionBoxplotmAsChartDiv").html(json.acquisitionBoxplotmAsData);
            }
            if(typeof json.acquisitionHistogrammAsData !=="undefined") {
                $("#acquisitionHistogrammAsChartDiv").html(json.acquisitionHistogrammAsData);
            }

            // DAP per study chart data
            if(typeof json.studyMeanDAPData !== "undefined") {
                $("#studyMeanDAPChartDiv").html(json.studyMeanDAPData);
                $("#studyMeanDAPChartParentDiv").append(json.studyMeanDAPDataCSV);
            }
            if(typeof json.studyMedianDAPData !== "undefined") {
                $("#studyMedianDAPChartDiv").html(json.studyMedianDAPData);
                $("#studyMedianDAPChartParentDiv").append(json.studyMedianDAPDataCSV);
            }
            if(typeof json.studyBoxplotDAPData !=="undefined") {
                $("#studyBoxplotDAPChartDiv").html(json.studyBoxplotDAPData);
            }
            if(typeof json.studyHistogramDAPData !=="undefined") {
                $("#studyHistogramDAPChartDiv").html(json.studyHistogramDAPData);
            }

            // Study frequency chart data start
            if(typeof json.studyFrequencyData !== "undefined") {
                $("#studyFrequencyChartDiv").html(json.studyFrequencyData);
                $("#studyFrequencyChartParentDiv").append(json.studyFrequencyDataCSV);
            }

            // DAP per request chart data
            if(typeof json.requestMeanDAPData !== "undefined") {
                $("#requestMeanDAPChartDiv").html(json.requestMeanDAPData);
                $("#requestMeanDAPChartParentDiv").append(json.requestMeanDAPDataCSV);
            }
            if(typeof json.requestMedianDAPData !== "undefined") {
                $("#requestMedianDAPChartDiv").html(json.requestMedianDAPData);
                $("#requestMedianDAPChartParentDiv").append(json.requestMedianDAPDataCSV);
            }
            if(typeof json.requestBoxplotDAPData !=="undefined") {
                $("#requestBoxplotDAPChartDiv").html(json.requestBoxplotDAPData);
            }
            if(typeof json.requestHistogramDAPData !=="undefined") {
                $("#requestHistogramDAPChartDiv").html(json.requestHistogramDAPData);
            }

            // Request frequency chart data start
            if(typeof json.requestFrequencyData !== "undefined") {
                $("#requestFrequencyChartDiv").html(json.requestFrequencyData);
                $("#requestFrequencyChartParentDiv").append(json.requestFrequencyDataCSV);
            }

            // Acquisition DAP over time chart data
            if(typeof json.acquisitionMeanDAPOverTime !== "undefined") {
                $("#acquisitionMeanDAPOverTimeChartDiv").html(json.acquisitionMeanDAPOverTime);
            }
            if(typeof json.acquisitionMedianDAPOverTime !== "undefined") {
                $("#acquisitionMedianDAPOverTimeChartDiv").html(json.acquisitionMedianDAPOverTime);
            }

            // Acquisition kVp over time chart data
            if(typeof json.acquisitionMeankVpOverTime !== "undefined") {
                $("#acquisitionMeankVpOverTimeChartDiv").html(json.acquisitionMeankVpOverTime);
            }
            if(typeof json.acquisitionMediankVpOverTime !== "undefined") {
                $("#acquisitionMediankVpOverTimeChartDiv").html(json.acquisitionMediankVpOverTime);
            }

            // Acquisition mAs over time chart data
            if(typeof json.acquisitionMeanmAsOverTime !== "undefined") {
                $("#acquisitionMeanmAsOverTimeChartDiv").html(json.acquisitionMeanmAsOverTime);
            }
            if(typeof json.acquisitionMedianmAsOverTime !== "undefined") {
                $("#acquisitionMedianmAsOverTimeChartDiv").html(json.acquisitionMedianmAsOverTime);
            }

            // Study workload chart data
            if(typeof json.studyWorkloadData !== "undefined") {
                $("#studyWorkloadChartDiv").html(json.studyWorkloadData);
            }

            // Acquisition DAP vs mass
            if(typeof json.acquisitionDAPvsMass !== "undefined") {
                $("#acquisitionDAPvsMassChartDiv").html(json.acquisitionDAPvsMass);
            }

            // Study DAP vs mass
            if(typeof json.studyDAPvsMass !== "undefined") {
                $("#studyDAPvsMassChartDiv").html(json.studyDAPvsMass);
            }

            // Request DAP vs mass
            if(typeof json.requestDAPvsMass !== "undefined") {
                $("#requestDAPvsMassChartDiv").html(json.requestDAPvsMass);
            }

            // DAP per standard acquisition name chart data
            if(typeof json.standardAcquisitionMeanDAPData !== "undefined") {
                $("#standardAcquisitionMeanDAPChartDiv").html(json.standardAcquisitionMeanDAPData);
                $("#standardAcquisitionMeanDAPChartParentDiv").append(json.standardAcquisitionMeanDAPDataCSV);
            }
            if(typeof json.standardAcquisitionMedianDAPData !== "undefined") {
                $("#standardAcquisitionMedianDAPChartDiv").html(json.standardAcquisitionMedianDAPData);
                $("#standardAcquisitionMedianDAPChartParentDiv").append(json.standardAcquisitionMedianDAPDataCSV);
            }
            if(typeof json.standardAcquisitionBoxplotDAPData !=="undefined") {
                $("#standardAcquisitionBoxplotDAPChartDiv").html(json.standardAcquisitionBoxplotDAPData);
            }
            if(typeof json.standardAcquisitionHistogramDAPData !=="undefined") {
                $("#standardAcquisitionHistogramDAPChartDiv").html(json.standardAcquisitionHistogramDAPData);
            }

            // Standard acquisition name frequency chart data start
            if(typeof json.standardAcquisitionFrequencyData !== "undefined") {
                $("#standardAcquisitionFrequencyChartDiv").html(json.standardAcquisitionFrequencyData);
                $("#standardAcquisitionFrequencyChartParentDiv").append(json.standardAcquisitionFrequencyDataCSV);
            }

            // kVp per standard acquisition name chart data
            if(typeof json.standardAcquisitionMeankVpData !== "undefined") {
                $("#standardAcquisitionMeankVpChartDiv").html(json.standardAcquisitionMeankVpData);
                $("#standardAcquisitionMeankVpChartParentDiv").append(json.standardAcquisitionMeankVpDataCSV);
            }
            if(typeof json.standardAcquisitionMediankVpData !== "undefined") {
                $("#standardAcquisitionMediankVpChartDiv").html(json.standardAcquisitionMediankVpData);
                $("#standardAcquisitionMediankVpChartParentDiv").append(json.standardAcquisitionMediankVpDataCSV);
            }
            if(typeof json.standardAcquisitionBoxplotkVpData !=="undefined") {
                $("#standardAcquisitionBoxplotkVpChartDiv").html(json.standardAcquisitionBoxplotkVpData);
            }
            if(typeof json.standardAcquisitionHistogramkVpData !=="undefined") {
                $("#standardAcquisitionHistogramkVpChartDiv").html(json.standardAcquisitionHistogramkVpData);
            }

            // mAs per standard acquisition name chart data
            if(typeof json.standardAcquisitionMeanmAsData !== "undefined") {
                $("#standardAcquisitionMeanmAsChartDiv").html(json.standardAcquisitionMeanmAsData);
                $("#standardAcquisitionMeanmAsChartParentDiv").append(json.standardAcquisitionMeanmAsDataCSV);
            }
            if(typeof json.standardAcquisitionMedianmAsData !== "undefined") {
                $("#standardAcquisitionMedianmAsChartDiv").html(json.standardAcquisitionMedianmAsData);
                $("#standardAcquisitionMedianmAsChartParentDiv").append(json.standardAcquisitionMedianmAsDataCSV);
            }
            if(typeof json.standardAcquisitionBoxplotmAsData !=="undefined") {
                $("#standardAcquisitionBoxplotmAsChartDiv").html(json.standardAcquisitionBoxplotmAsData);
            }
            if(typeof json.standardAcquisitionHistogrammAsData !=="undefined") {
                $("#standardAcquisitionHistogrammAsChartDiv").html(json.standardAcquisitionHistogrammAsData);
            }

            // Standard acquisition name DAP over time chart data
            if(typeof json.standardAcquisitionMeanDAPOverTime !== "undefined") {
                $("#standardAcquisitionMeanDAPOverTimeChartDiv").html(json.standardAcquisitionMeanDAPOverTime);
            }
            if(typeof json.standardAcquisitionMedianDAPOverTime !== "undefined") {
                $("#standardAcquisitionMedianDAPOverTimeChartDiv").html(json.standardAcquisitionMedianDAPOverTime);
            }

            // Standard acquisition name kVp over time chart data
            if(typeof json.standardAcquisitionMeankVpOverTime !== "undefined") {
                $("#standardAcquisitionMeankVpOverTimeChartDiv").html(json.standardAcquisitionMeankVpOverTime);
            }
            if(typeof json.standardAcquisitionMediankVpOverTime !== "undefined") {
                $("#standardAcquisitionMediankVpOverTimeChartDiv").html(json.standardAcquisitionMediankVpOverTime);
            }

            // Standard acquisition name mAs over time chart data
            if(typeof json.standardAcquisitionMeanmAsOverTime !== "undefined") {
                $("#standardAcquisitionMeanmAsOverTimeChartDiv").html(json.standardAcquisitionMeanmAsOverTime);
            }
            if(typeof json.standardAcquisitionMedianmAsOverTime !== "undefined") {
                $("#standardAcquisitionMedianmAsOverTimeChartDiv").html(json.standardAcquisitionMedianmAsOverTime);
            }

            // Standard acquisition name DAP vs mass
            if(typeof json.standardAcquisitionDAPvsMass !== "undefined") {
                $("#standardAcquisitionDAPvsMassChartDiv").html(json.standardAcquisitionDAPvsMass);
            }

            // DAP per standard study name chart data
            if(typeof json.standardStudyMeanDAPData !== "undefined") {
                $("#standardStudyMeanDAPChartDiv").html(json.standardStudyMeanDAPData);
                $("#standardStudyMeanDAPChartParentDiv").append(json.standardStudyMeanDAPDataCSV);
            }
            if(typeof json.standardStudyMedianDAPData !== "undefined") {
                $("#standardStudyMedianDAPChartDiv").html(json.standardStudyMedianDAPData);
                $("#standardStudyMedianDAPChartParentDiv").append(json.standardStudyMedianDAPDataCSV);
            }
            if(typeof json.standardStudyBoxplotDAPData !=="undefined") {
                $("#standardStudyBoxplotDAPChartDiv").html(json.standardStudyBoxplotDAPData);
            }
            if(typeof json.standardStudyHistogramDAPData !=="undefined") {
                $("#standardStudyHistogramDAPChartDiv").html(json.standardStudyHistogramDAPData);
            }

            // Standard study name frequency chart data start
            if(typeof json.standardStudyFrequencyData !== "undefined") {
                $("#standardStudyFrequencyChartDiv").html(json.standardStudyFrequencyData);
                $("#standardStudyFrequencyChartParentDiv").append(json.standardStudyFrequencyDataCSV);
            }

            // Standard study name workload chart data
            if(typeof json.standardStudyWorkloadData !== "undefined") {
                $("#standardStudyWorkloadChartDiv").html(json.standardStudyWorkloadData);
            }

            // Standard study name DAP vs mass
            if(typeof json.standardStudyDAPvsMass !== "undefined") {
                $("#standardStudyDAPvsMassChartDiv").html(json.standardStudyDAPvsMass);
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
