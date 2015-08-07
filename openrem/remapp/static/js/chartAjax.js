function URLToArray(url) {
    var request = {};
    var pairs = url.substring(url.indexOf('?') + 1).split('&');
    for (var i = 0; i < pairs.length; i++) {
        if(!pairs[i])
            continue;
        var pair = pairs[i].split('=');
        request[decodeURIComponent(pair[0])] = decodeURIComponent(pair[1]);
    }
    return request;
}

function ArrayToURL(array) {
    var pairs = [];
    for (var key in array)
        if (array.hasOwnProperty(key))

            pairs.push(encodeURIComponent(key) + '=' + encodeURIComponent(array[key]));
    return pairs.join('&');
}


// Code to update the page and chart data on initial page load.
$(document).ready(function() {
    var request_data = ArrayToURL(URLToArray(this.URL));

    $.ajax({
        type: "GET",
        url: "/openrem/dx/chart/",
        data: request_data,
        dataType: "json",
        success: function( json ) {
            // A [num acq protocols] list of acquisition protocol names
            var names = $.map(json.acquisition_names, function(el) { return el.acquisition_protocol; });

            // [num acq protocols] lists of 2-element arrays.
            // Element 0 contains a 20-element array of counts
            // Element 1 contains a 21-element array of bin boundaries
            var acq_histogram_data = json.acquisitionHistogramData;
            var acq_histogram_kVp_data = json.acquisitionHistogramkVpData;
            var acq_histogram_mAs_data = json.acquisitionHistogrammAsData;

            // A [num acq protocols] list of objects with acquisition_protocol, mean_dap, median_dap and num_acq properties
            var acq_summary = $.map(json.acquisitionSummary, function(el) { return el; });
            var acq_kVp_summary = $.map(json.acquisitionkVpSummary, function(el) { return el; });
            var acq_mAs_summary = $.map(json.acquisitionmAsSummary, function(el) { return el; });

            // A [7][24] list of integer values
            var studies_per_hour_in_weekdays = json.studiesPerHourInWeekdays;

            // A [num acq protocols][num time periods] list of 2-element arrays containing datetime and average DAP values
            var acq_mean_dap_over_time = json.acquisitionMeanDAPoverTime;

            // Construct JavaScript arrays needed by the plot of acquisition protocol mean DAP
            protocolNames = names;
            seriesDataN = [];
            protocolCounts = [];
            protocolBins = [];
            seriesData = [];
            for (i = 0; i < names.length; i++) {
                seriesDataN.push(acq_summary[i].num_acq);
                protocolCounts.push(acq_histogram_data[i][0]);
                protocolBins.push(acq_histogram_data[i][1]);
                seriesData.push({name:protocolNames[i], y:acq_summary[i].mean_dap, freq:acq_summary[i].num_acq, bins:protocolBins[i], tooltip:protocolNames[i]+'<br>'+acq_summary[i].mean_dap.toFixed(1)+' mean<br>(n='+acq_summary[i].num_acq+')',drilldown:protocolNames[i]});
            }

            var temp = [];
            seriesDrilldown = [];
            for (i = 0; i < names.length; i++) {
                temp = [];
                for (j = 0; j < protocolCounts[0].length; j++) {
                    temp.push([protocolBins[i][j].toFixed(1).toString() + ' \u2264 x < ' + protocolBins[i][j+1].toFixed(1).toString(), protocolCounts[i][j]]);
                }
                seriesDrilldown.push({id: protocolNames[1-1], name: protocolNames[1-1], useHTML: true, data: temp});
            }

            var chart = $('#container').highcharts();
            chart.series[0].setData(seriesData);
            chart.xAxis[0].setCategories(protocolNames);
            chart.redraw({ duration: 1000 });
            // The drilldown data isn't replaced at the moment.
        },
        error: function( xhr, status, errorThrown ) {
            alert( "Sorry, there was a problem getting the chart data for initial page view" );
            console.log( "Error: " + errorThrown );
            console.log( "Status: " + status );
            console.dir( xhr );
        }
    });
    return false;
});


// Code to update the page and chart data when the user clicks on the submit button.
$(document).ready(function() {
    var form = $('form#examFilterForm');
    form.submit(function(event) {
        event.preventDefault();
        var serialized_form = form.serialize();

        // Send the form data to the usual view
        $.ajax({ type: "GET",
            url: $(this).attr('action'),
            data: serialized_form,
            dataType: "html",
            success: function(html_response) {
                // Extract the new HTML that is contained within the DJP tags
                var new_DJP_div = html_response.replace(/^[\S\s]*DJPstart><\/a>/i, "").replace(/<a href=DJPend[\S\s]*$/i, "");
                // Replace the html in the DJP div with the new HTML. Note I also need to replace the number of studies here too.
                $('#DJP').html(new_DJP_div);
            },
            error: function( xhr, status, errorThrown ) {
                alert( "Sorry, there was a problem getting the data for the updated tabular form" );
                console.log( "Error: " + errorThrown );
                console.log( "Status: " + status );
                console.dir( xhr );
            }
        });

        // Submit the form data again, but to the chart data view
        $.ajax({
            type: "GET",
            url: "/openrem/dx/chart/",
            data: serialized_form,
            dataType: "json",
            success: function( json ) {
                //$('#examFilterForm').submit();

                // A [num acq protocols] list of acquisition protocol names
                var names = $.map(json.acquisition_names, function(el) { return el.acquisition_protocol; });

                // [num acq protocols] lists of 2-element arrays.
                // Element 0 contains a 20-element array of counts
                // Element 1 contains a 21-element array of bin boundaries
                var acq_histogram_data = json.acquisitionHistogramData;
                var acq_histogram_kVp_data = json.acquisitionHistogramkVpData;
                var acq_histogram_mAs_data = json.acquisitionHistogrammAsData;

                // A [num acq protocols] list of objects with acquisition_protocol, mean_dap, median_dap and num_acq properties
                var acq_summary = $.map(json.acquisitionSummary, function(el) { return el; });
                var acq_kVp_summary = $.map(json.acquisitionkVpSummary, function(el) { return el; });
                var acq_mAs_summary = $.map(json.acquisitionmAsSummary, function(el) { return el; });

                // A [7][24] list of integer values
                var studies_per_hour_in_weekdays = json.studiesPerHourInWeekdays;

                // A [num acq protocols][num time periods] list of 2-element arrays containing datetime and average DAP values
                var acq_mean_dap_over_time = json.acquisitionMeanDAPoverTime;

                // Construct JavaScript arrays needed by the plot of acquisition protocol mean DAP
                protocolNames = names;
                seriesDataN = [];
                protocolCounts = [];
                protocolBins = [];
                seriesData = [];
                for (i = 0; i < names.length; i++) {
                    seriesDataN.push(acq_summary[i].num_acq);
                    protocolCounts.push(acq_histogram_data[i][0]);
                    protocolBins.push(acq_histogram_data[i][1]);
                    seriesData.push({name:protocolNames[i], y:acq_summary[i].mean_dap, freq:acq_summary[i].num_acq, bins:protocolBins[i], tooltip:protocolNames[i]+'<br>'+acq_summary[i].mean_dap.toFixed(1)+' mean<br>(n='+acq_summary[i].num_acq+')',drilldown:protocolNames[i]});
                }

                var temp = [];
                seriesDrilldown = [];
                for (i = 0; i < names.length; i++) {
                    temp = [];
                    for (j = 0; j < protocolCounts[0].length; j++) {
                        temp.push([protocolBins[i][j].toFixed(1).toString() + ' \u2264 x < ' + protocolBins[i][j+1].toFixed(1).toString(), protocolCounts[i][j]]);
                    }
                    seriesDrilldown.push({id: protocolNames[1-1], name: protocolNames[1-1], useHTML: true, data: temp});
                }

                var chart = $('#container').highcharts();
                chart.series[0].setData(seriesData);
                chart.xAxis[0].setCategories(protocolNames);
                chart.redraw({ duration: 1000 });
                // The drilldown data isn't replaced at the moment.
            },
            error: function( xhr, status, errorThrown ) {
                alert( "Sorry, there was a problem getting the data for the updated chart data" );
                console.log( "Error: " + errorThrown );
                console.log( "Status: " + status );
                console.dir( xhr );
            }
        });
        return false;
    })
});