var chartStartHeight = 0;
var chartStartWidth = 0;
var chartParentStartHeight = 0;
var chartParentStartWidth = 0;
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


function fitPlotlyChartToDiv() {
    window.dispatchEvent(new Event("resize"));
}


function enterFullScreenPlotly(chartDivId, chartParentDivId) {

    var plotly_div_id = $("#"+chartDivId+ " :first-child :first-child").attr("id");

    var plotlyDiv = $("#"+plotly_div_id);
    var parentDiv = $("#"+chartParentDivId);

    if (chartFullScreen === false) {
        chartStartHeight = plotlyDiv.height();
        chartStartWidth = plotlyDiv.width();
        chartParentStartHeight = parentDiv.height();
        chartParentStartWidth = parentDiv.width();

        chartStartFooterHeight = chartParentStartHeight - chartStartHeight;

        parentDiv.width($(window).width());
        parentDiv.height($(window).height());

        plotlyDiv.width($(window).width());
        plotlyDiv.height($(window).height() - chartStartFooterHeight);

        chartFullScreen = true;
    }
    else {
        parentDiv.width(chartParentStartWidth);
        parentDiv.height(chartParentStartHeight);

        plotlyDiv.width(chartStartWidth);
        plotlyDiv.height(chartStartHeight);

        chartFullScreen = false;
    }

    parentDiv.toggleClass("fullscreen");
    fitPlotlyChartToDiv();
}
