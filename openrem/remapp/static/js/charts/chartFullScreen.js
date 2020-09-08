var chartStartHeight = 0;
var chartStartFooterHeight = 0;
var chartFullScreen = false;

function enterFullScreen(divId, chartDiv) {
    if (chartFullScreen === false) {
        chartStartHeight = document.getElementById(chartDiv).clientHeight;
        chartStartFooterHeight = document.getElementById(divId).clientHeight - chartStartHeight;
        document.getElementById(chartDiv).style.height = $(window).height() - chartStartFooterHeight + "px";
        chartFullScreen = true;
    }
    else {
        document.getElementById(chartDiv).style.height = chartStartHeight + "px";
        chartFullScreen = false;
    }

    $("#"+divId).toggleClass("fullscreen");

    var chartDivElement = $("#"+chartDiv);
    var chart = chartDivElement.highcharts();
    chart.setSize(chartDivElement.width(), chartDivElement.height());
}


function fitChartToDiv(chartDiv) {
    var chartDivElement = $("#"+chartDiv);
    if (chartDivElement.width() && chartDivElement.height()) {
        var chart = chartDivElement.highcharts();
        chart.setSize(chartDivElement.width(), chartDivElement.height());
    }
}


function fitPlotlyChartToDiv(chartGrandParentDiv) {
    var chartGrandParentDivElement = $("#" + chartGrandParentDiv);

    if (chartGrandParentDivElement.width() && chartGrandParentDivElement.height()) {

        var num_divs = chartGrandParentDivElement.children().length;

        for (i = 0; i < num_divs; i++) {
            var chart_div_id = $((chartGrandParentDivElement.children()[i]).innerHTML).attr("id")
            var update = {
                width: chartGrandParentDivElement.width(),
                height: chartGrandParentDivElement.height() / num_divs
            };
            Plotly.relayout(chart_div_id, update);
        }
    }
}