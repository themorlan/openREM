var jsonData = "";
var staticUrl = ".";

function remSearchBuilderLine(pos) {
    var $searchBuilder = $("#search_builder");
    $searchBuilder.find("#line_" + pos).remove();
    $searchBuilder.find(".add_line").last()[0].style.visibility = "visible";
    updateSearchString();
}

function addSearchBuilderLine(pos, first) {
    pos = parseInt(pos);
    var strPos = "" + (pos + 1);
    var visibility = "hidden";
    if (first === false)
        visibility = "visible";

    $("#search_builder").append("<div id='line_" + strPos + "' class='builder_line'>" +
      "    <label for='concatoperator_" + strPos + "' class='labels' style='visibility: " + visibility + ";'>Select operator</label>" +
      "    <select id='concatoperator_" + strPos + "' class='concatoperator' style='visibility: " + visibility + ";'>" +
      "        <option value='AND'>AND</option>" +
      "        <option value='OR'>OR</option>" +
      "        <option value='AND_NOT'>AND NOT</option>" +
      "        <option value='OR_NOT'>OR NOT</option>" +
      "    </select>" +
      "    <input id='prebracket_" + strPos + "' class='prebrackets' size=1/>" +
      "    <label for='parameter_" + strPos + "' class='labels'>Select parameter</label>" +
      "    <select id='parameter_" + strPos + "' class='parameter'>" +
      "    </select>" +
      "    <label for='operator_" + strPos + "' class='labels'>Select operator</label>" +
      "    <select id='operator_" + strPos + "' class='operator'>" +
      "    </select>" +
      "    <div id='search_term_" + strPos + "' class='inline_div'>" +
      "        <label for='term_" + strPos + "' class='labels'>parameter value</label>" +
      "        <input id='term_" + strPos + "' class='searchinput' size=40/>" +
      "    </div>" +
      "    <input id='postbracket_" + strPos + "' class='postbrackets' size=1/>" +
      "    <a href='javascript:;' class='rem_line' style='visibility: " + visibility + ";'><img id='remline_" + strPos + "' src='" + staticUrl + "img/delete.png' alt='remove line'/></a>" +
      "    <a href='javascript:;' class='add_line'><img id='addline_" + strPos + "' src='" + staticUrl + "img/add.png' alt='add line'/></a>" +
      "</div>");
    if (first === false)
        $("#line_" + pos).find(".add_line")[0].style.visibility = "hidden";
    var cBox = document.getElementById("parameter_" + strPos);
    addHandlers();
    fillComboBox(cBox);
    updateOperatorCB(cBox);
}

function updateOperatorCB(cb) {
    var id=cb.id.substring(cb.id.indexOf("_")+1);
    var operatorCb = document.getElementById("operator_" + id);
    var operators = jsonData.filter(function (x) { return x.label === cb.value; })[0].comparison;
    $(operatorCb).empty();
    $.each(operators, function(index, value) {
        $(operatorCb).append("<option>" + value + "</option>");
    });
}

function fillComboBox(cb) {
    var j;
    $(cb).empty();
    $.each(jsonData.map(j => j.label), function(index, value) {
        $(cb).append("<option>" + value + "</option>");
    });
}

function checkBrackets() {
    var bracketSaldo = 0;
    $(".builder_line").each(function() {
        var prebracket = $(this).find(".prebrackets")[0].value;
        var postbracket = $(this).find(".postbrackets")[0].value;
        var regexpBracketOpen = new RegExp(/\(/, "g");
        var regexpBracketClose = new RegExp(/\)/, "g");
        if (prebracket.match(regexpBracketOpen) !== null)
            bracketSaldo += prebracket.match(regexpBracketOpen).length;
        if (postbracket.match(regexpBracketClose) !== null)
            bracketSaldo -= postbracket.match(regexpBracketClose).length;
    });
    if (bracketSaldo > 0) {
        $(".prebrackets").filter(function() {
            return this.value.length > 0
        }).css("border", "2px solid orange");
    }
    else if (bracketSaldo < 0) {
        $(".postbrackets").filter(function() {
            return this.value.length > 0
        }).css("border", "2px solid orange");
    }
    else {
        $(".prebrackets").css("border", "");
        $(".postbrackets").css("border", "");
    }
}

function updateSearchString() {
    var builtSearchString = "";
    $(".builder_line").each(function() {
        var searchterm = $(this).find(".searchinput")[0].value;
        if (searchterm !== "")
        {
            var prebracket = $(this).find(".prebrackets")[0].value;
            var postbracket = $(this).find(".postbrackets")[0].value;
            var concatOperatorElement = $(this).find(".concatoperator")[0];
            var concatOperator = concatOperatorElement.options[concatOperatorElement.selectedIndex].text;
            var parameterElement = $(this).find(".parameter")[0];
            var parameter = parameterElement.options[parameterElement.selectedIndex].text;
            var operatorElement = $(this).find(".operator")[0];
            var operator = operatorElement.options[operatorElement.selectedIndex].text;
            if ((concatOperatorElement.style.visibility === "visible") && (builtSearchString !== "")) {
                builtSearchString += " " + concatOperator;
            }
            builtSearchString += " " + prebracket + "{[" + parameter + "] " + operator + " '" + searchterm + "'}" + postbracket;
        }
    });
    document.getElementById("id_advancedSearchString").value = builtSearchString;
}

function initializeJSON(filterJSON, staticPath) {
    jsonData = filterJSON;
    staticUrl = staticPath;
}

function addHandlers()
{
    var $searchInput = $(".searchinput");
    $searchInput.off();
    $searchInput.on("keyup", function() {
        updateSearchString();
    });
    $searchInput.on("paste", function() {
        //setTimeout needed, otherwise paste hasn't been applied yet (then update is applied before "paste" is applied)
        setTimeout(updateSearchString, 100);
    });

    var $preBrackets = $(".prebrackets");
    var $postBrackets = $(".postbrackets");
    var $preAndPostBrackets = $(".prebrackets, .postbrackets");
    $preAndPostBrackets.off();
    $preAndPostBrackets.on("keyup", function() {
        checkBrackets();
        updateSearchString();
    });
    $preAndPostBrackets.on("paste", function(event) {
        //don't allow paste
        event.preventDefault();
    });
    $preBrackets.keydown(function (event) {
      // allow: backspace, tab, enter, shift, escape, delete, delete
      // second line is key 9/(
      if (!(($.inArray(event.originalEvent.keyCode, [8, 9, 13, 16, 27, 46, 110]) !== -1) ||
          (event.originalEvent.shiftKey === true && event.originalEvent.keyCode === 57)))
        event.preventDefault();
    });
    $postBrackets.keydown(function (event) {
      // allow: backspace, tab, enter, shift, escape, delete, delete
      // second line is key  0/)
      if (!(($.inArray(event.originalEvent.keyCode, [8, 9, 13, 16, 27, 46, 110]) !== -1) ||
          (event.originalEvent.shiftKey === true && event.originalEvent.keyCode === 48)))
        event.preventDefault();
    });

    $(".parameter").change(function(event) {
        updateOperatorCB(event.target);
        updateSearchString();
    });

    $(".operator").change(function() {
        updateSearchString();
    });

    $(".concatoperator").change(function() {
        updateSearchString();
    });

    var $addLine = $(".add_line");
    $addLine.off();
    $addLine.on("click", function(event) {
        var id = "";
        if (event.target.className === "add_Line") {
            //pressed enter with focus on hyperlink instead of clicking "image"
            //still click-event is triggered
            id = event.target.firstChild.id.substring(event.target.id.indexOf("_")+1);
        }
        else {
            id = event.target.id.substring(event.target.id.indexOf("_")+1);
        }
        addSearchBuilderLine(parseInt(id), false);
    });

    var $remLine = $(".rem_line");
    $remLine.off();
    $remLine.on("click", function(event) {
        var id = "";
        if (event.target.className === "rem_Line") {
            //pressed enter with focus on hyperlink instead of clicking "image"
            //still click-event is triggered
            id = event.target.firstChild.id.substring(event.target.id.indexOf("_")+1);
        }
        else {
            id = event.target.id.substring(event.target.id.indexOf("_")+1);
        }
        remSearchBuilderLine(parseInt(id));
    });
}