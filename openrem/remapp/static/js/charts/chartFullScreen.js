var chartStartHeight = 0;
var chartStartWidth = 0;
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


function triggerResizeEvent() {
    var evt = window.document.createEvent('UIEvents');
    evt.initUIEvent('resize', true, false, window, 0);
    window.dispatchEvent(evt);
}


function enterFullScreenPlotly(chartDivId, chartParentDivId) {

    var plotlyDiv = $("#"+chartDivId);
    var parentDiv = $("#"+chartParentDivId);

    if (chartFullScreen === false) {
        chartStartHeight = plotlyDiv.height();
        chartStartWidth = plotlyDiv.width();
        chartStartFooterHeight = parentDiv.height() - chartStartHeight;

        plotlyDiv.width($(window).width());
        plotlyDiv.height($(window).height() - chartStartFooterHeight);
        $("#"+chartDivId+" :first-child").height("100%");

        chartFullScreen = true;
    }
    else {
        plotlyDiv.width(chartStartWidth);
        plotlyDiv.height(chartStartHeight);

        plotlyDiv.css("width","auto");

        chartFullScreen = false;
    }

    parentDiv.toggleClass("fullscreen");
    triggerResizeEvent();
}
