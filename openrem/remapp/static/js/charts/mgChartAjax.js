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

            // Acquisition frequency chart data
            if(typeof json.acquisitionFrequencyData !== "undefined") {
                $("#acquisitionFrequencyChartDiv").html(json.acquisitionFrequencyData);
                $("#acquisitionFrequencyChartParentDiv").append(json.acquisitionFrequencyDataCSV);
            }

            // AGD per acquisition chart data
            if(typeof json.acquisitionMeanAGDData !== "undefined") {
                $("#acquisitionMeanAGDChartDiv").html(json.acquisitionMeanAGDData);
                $("#acquisitionMeanAGDChartParentDiv").append(json.acquisitionMeanAGDDataCSV);
            }
            if(typeof json.acquisitionMedianAGDData !== "undefined") {
                $("#acquisitionMedianAGDChartDiv").html(json.acquisitionMedianAGDData);
                $("#acquisitionMedianAGDChartParentDiv").append(json.acquisitionMedianAGDDataCSV);
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

            // AGD over time chart data
            if(typeof json.acquisitionMeanAGDOverTime !== "undefined") {
                $("#acquisitionMeanAGDOverTimeChartDiv").html(json.acquisitionMeanAGDOverTime);
            }
            if(typeof json.acquisitionMedianAGDOverTime !== "undefined") {
                $("#acquisitionMedianAGDOverTimeChartDiv").html(json.acquisitionMedianAGDOverTime);
            }

            // Acquisition frequency chart data
            if(typeof json.standardAcquisitionFrequencyData !== "undefined") {
                $("#standardAcquisitionFrequencyChartDiv").html(json.standardAcquisitionFrequencyData);
                $("#standardAcquisitionFrequencyChartParentDiv").append(json.standardAcquisitionFrequencyDataCSV);
            }

            // AGD per acquisition chart data
            if(typeof json.standardAcquisitionMeanAGDData !== "undefined") {
                $("#standardAcquisitionMeanAGDChartDiv").html(json.standardAcquisitionMeanAGDData);
                $("#standardAcquisitionMeanAGDChartParentDiv").append(json.standardAcquisitionMeanAGDDataCSV);
            }
            if(typeof json.standardAcquisitionMedianAGDData !== "undefined") {
                $("#standardAcquisitionMedianAGDChartDiv").html(json.standardAcquisitionMedianAGDData);
                $("#standardAcquisitionMedianAGDChartParentDiv").append(json.standardAcquisitionMedianAGDDataCSV);
            }
            if(typeof json.standardAcquisitionBoxplotAGDData !=="undefined") {
                $("#standardAcquisitionBoxplotAGDChartDiv").html(json.standardAcquisitionBoxplotAGDData);
            }
            if(typeof json.standardAcquisitionHistogramAGDData !=="undefined") {
                $("#standardAcquisitionHistogramAGDChartDiv").html(json.standardAcquisitionHistogramAGDData);
            }

            // AGD per acquisition grouped into compressed breast thickness bands
            if(typeof json.standardMeanAGDvsThickness !== "undefined") {
                $("#standardAcquisitionMeanAGDvsThickChartDiv").html(json.standardMeanAGDvsThickness);
            }
            if(typeof json.standardMedianAGDvsThickness !== "undefined") {
                $("#standardAcquisitionMedianAGDvsThickChartDiv").html(json.standardMedianAGDvsThickness);
            }

            // AGD vs compressed thickness scatter plot
            if(typeof json.standardAGDvsThickness !== "undefined") {
                $("#standardAcquisitionScatterAGDvsThickChartDiv").html(json.standardAGDvsThickness);
            }

            // kVp vs compressed thickness scatter plot
            if(typeof json.standardkVpvsThickness !== "undefined") {
                $("#standardAcquisitionScatterkVpvsThickChartDiv").html(json.standardkVpvsThickness);
            }

            // mAs vs compressed thickness scatter plot
            if(typeof json.standardmAsvsThickness !== "undefined") {
                $("#standardAcquisitionScattermAsvsThickChartDiv").html(json.standardmAsvsThickness);
            }

            // AGD over time chart data
            if(typeof json.standardAcquisitionMeanAGDOverTime !== "undefined") {
                $("#standardAcquisitionMeanAGDOverTimeChartDiv").html(json.standardAcquisitionMeanAGDOverTime);
            }
            if(typeof json.standardAcquisitionMedianAGDOverTime !== "undefined") {
                $("#standardAcquisitionMedianAGDOverTimeChartDiv").html(json.standardAcquisitionMedianAGDOverTime);
            }

            // Standard study name workload chart data
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