let jsonData = "";
let staticUrl = ".";

function remSearchBuilderLine(pos) {
    let $searchBuilder = $("#search_builder");
    $searchBuilder.find("#line_" + pos).remove();
    $searchBuilder.find(".add_line").last()[0].style.visibility = "visible";
    updateSearchString();
}

function addSearchBuilderLine(pos, first) {
    pos = parseInt(pos);
    let strNewPos = "" + (pos + 1);
    let concatOptions = "<option value='AND'>AND</option>" +
                        "<option value='OR'>OR</option>" +
                        "<option value='AND_NOT'>AND NOT</option>" +
                        "<option value='OR_NOT'>OR NOT</option>";
    if (first) {
        concatOptions = "<option value=''></option>" +
                        "<option value='NOT'>NOT</option>";
    }

    $('#search_builder_table tr:last').after("<tr id='line_" + strNewPos + "' class='builder_line'>" +
      "    <td><select id='concatoperator_" + strNewPos + "' class='concatoperator'>" + concatOptions +
      "        </select></td>" +
      "    <td style='text-align:right'>" +
      "      <a href='javascript:;' class='remove_prebracket'>" +
      "        <img id='removeprebracket_" + strNewPos + "' src='" + staticUrl + "img/delete.png' alt='remove bracket'/>" +
      "      </a>" +
      "      <a href='javascript:;' class='add_prebracket'>" +
      "        <img id='addprebracket_" + strNewPos + "' src='" + staticUrl + "img/add.png' alt='add bracket'/>" +
      "      </a>" +
      "    </td>" +
      "    <td><input style='width:100%' id='prebracket_" + strNewPos + "' class='prebrackets' " +
      "               readonly/></td>" +
      "    <td><select style='width:100%' id='parameter_" + strNewPos + "' class='parameter'></select></td>" +
      "    <td><select style='width:100%' id='operator_" + strNewPos + "' class='operator'></select></td>" +
      "    <td><div id='search_term_" + strNewPos + "' class='search_inline_div'>" +
      "        <input id='term_" + strNewPos + "' class='searchinput' size=40/>" +
      "    </div></td>" +
      "    <td style='text-align:right'>" +
      "      <a href='javascript:;' class='remove_postbracket'>" +
      "        <img id='removepostbracket_" + strNewPos + "' src='" + staticUrl + "img/delete.png' alt='remove bracket'/>" +
      "      </a>" +
      "      <a href='javascript:;' class='add_postbracket'>" +
      "        <img id='addpostbracket_" + strNewPos + "' src='" + staticUrl + "img/add.png' alt='add bracket'/>" +
      "      </a>" +
      "    </td>" +
      "    <td><input style='width:100%' id='postbracket_" + strNewPos + "' class='postbrackets' " +
      "               readonly/></td>" +
      "    <td style='text-align:right'>" +
      "        <a href='javascript:;' class='rem_line' style='visibility: " + (first ? "hidden" : "visible") + ";'>" +
      "        <img id='remline_" + strNewPos + "' src='" + staticUrl + "img/delete.png' alt='remove line'/></a>" +
      "    <a href='javascript:;' class='add_line'><img id='addline_" + strNewPos + "' src='" + staticUrl + "img/add.png' alt='add line'/></a></td>" +
      "</tr>");
    if (first === false)
        $("#line_" + pos).find(".add_line")[0].style.visibility = "hidden";
    let cBox = document.getElementById("parameter_" + strNewPos);
    addHandlers();
    fillComboBox(cBox);
    updateOperatorCB(cBox);
}

function updateOperatorCB(cb) {
    let id=cb.id.substring(cb.id.indexOf("_")+1);
    let operatorCb = document.getElementById("operator_" + id);
    let operators = jsonData.filter(function (x) { return x.label === cb.value; })[0].comparison;
    $(operatorCb).empty();
    $.each(operators, function(index, value) {
        $(operatorCb).append("<option>" + value + "</option>");
    });
}

function sort_by_key(array, key)
{
 return array.sort(function(a, b)
 {
  let x = a[key];
  let y = b[key];
  return ((x < y) ? -1 : ((x > y) ? 1 : 0));
 });
}

function fillComboBox(cb) {
    $(cb).empty();
    let sorted_json = sort_by_key(jsonData, 'label')
    $.each(sorted_json.map(j => j.label), function(index, value) {
        $(cb).append("<option>" + value + "</option>");
    });
}

function checkBrackets() {
    let bracketSaldo = 0;
    let lineNumber = 0;
    while (document.getElementById("prebracket_" + lineNumber.toString()) !== null) {
        let prebracketNr = document.getElementById("prebracket_" + lineNumber.toString()).value.length;
        let postbracketNr = document.getElementById("postbracket_" + lineNumber.toString()).value.length;
        bracketSaldo = bracketSaldo + prebracketNr - postbracketNr;
        if (bracketSaldo < 0)
            // if bracketSaldo < 0 there are more closing brackets than opening brackets at this point.
            break;
        lineNumber++;
    }
    if (bracketSaldo > 0) {
        $(".prebrackets").filter(function() {
            return this.value.length > 0
        }).css("border", "2px solid orange");
        // disable submit buttons
        $("input[name='submit']").prop('disabled', true)
    }
    else if (bracketSaldo < 0) {
        $(".postbrackets").filter(function() {
            return this.value.length > 0
        }).css("border", "2px solid orange");
        // disable submit buttons
        $("input[name='submit']").prop('disabled', true)
    }
    else {
        $(".prebrackets").css("border", "");
        $(".postbrackets").css("border", "");
        // enable submit buttons
        $("input[name='submit']").prop('disabled', false)
    }
}

function updateSearchString() {
    let builtSearchString = "";
    $(".builder_line").each(function() {
        let searchterm = $(this).find(".searchinput")[0].value;
        if (searchterm !== "")
        {
            let prebracket = $(this).find(".prebrackets")[0].value;
            let postbracket = $(this).find(".postbrackets")[0].value;
            let concatOperatorElement = $(this).find(".concatoperator")[0];
            let concatOperator = "";
            if (concatOperatorElement.selectedIndex > -1) {
                concatOperator = concatOperatorElement.options[concatOperatorElement.selectedIndex].text;
            }
            let parameterElement = $(this).find(".parameter")[0];
            let parameter = parameterElement.options[parameterElement.selectedIndex].text;
            let operatorElement = $(this).find(".operator")[0];
            let operator = operatorElement.options[operatorElement.selectedIndex].text;
            if (builtSearchString !== "") {
                builtSearchString += " " + concatOperator;
            } else {
                builtSearchString = concatOperator;
            }
            builtSearchString += " " + prebracket + "{[" + parameter + "] " + operator + " '" + searchterm + "'}" + postbracket;
        }
    });
    document.getElementById("id_advanced_search_string").value = builtSearchString.trim();
}

function createBuilderFromSearchString(searchString) {
    //delete all builderlines
    let $searchBuilder = $("#search_builder");
    $searchBuilder.find("#line_").remove();
    let indexOfStartBrace = searchString.indexOf("{");
    let pos = -1;
    let concatOperator = searchString.substring(0,3) === "NOT" ? "NOT" : "";
    while (indexOfStartBrace > -1)
    {
        let indexOfEndBrace = searchString.indexOf("}", indexOfStartBrace);
        if (indexOfEndBrace === -1)
            return;
        let indexOfStartBracket = searchString.indexOf("[", indexOfStartBrace);
        let indexOfEndBracket = searchString.indexOf("]", indexOfStartBracket);
        if ((indexOfStartBracket >= indexOfEndBracket) || (indexOfEndBracket > indexOfEndBrace))
            return;
        let parameter = searchString.substring(indexOfStartBracket+1, indexOfEndBracket);
        let indexOfStartSingleQuote = searchString.indexOf("'", indexOfStartBrace);
        let indexOfEndSingleQuote = searchString.indexOf("'",indexOfStartSingleQuote+1);
        if ((indexOfStartSingleQuote >= indexOfEndSingleQuote) || (indexOfStartSingleQuote < indexOfStartBracket) ||
            (indexOfEndSingleQuote > indexOfEndBrace))
            return;
        let value = searchString.substring(indexOfStartSingleQuote+1, indexOfEndSingleQuote);
        let comparison = searchString.substring(indexOfEndBracket+1, indexOfStartSingleQuote).trim();
        let startRoundBracket = "";
        if (indexOfStartBrace > 0) {
            let indexOfLastBrace = searchString.lastIndexOf("}", indexOfStartBrace);
            let indexOfStartConcatOperator = searchString.indexOf(" ", indexOfLastBrace);
            let indexOfStartRoundBracket = searchString.indexOf("(", indexOfLastBrace);
            if ((indexOfStartRoundBracket > -1) && (indexOfStartRoundBracket < indexOfStartBrace)) {
                startRoundBracket = searchString.substring(indexOfStartRoundBracket, indexOfStartBrace).trim();
                if (concatOperator !== "NOT") {
                    concatOperator = searchString.substring(indexOfStartConcatOperator, indexOfStartRoundBracket).trim().replace(" ", "_");
                }
            }
            else if (concatOperator !== "NOT")
                concatOperator = searchString.substring(indexOfStartConcatOperator, indexOfStartBrace).trim().replace(" ", "_");
        }
        let endRoundBracket = "";
        if (indexOfEndBrace !== searchString.trim().length)
        {
            let indexOfEndRoundBracket = searchString.indexOf(")", indexOfEndBrace);
            let indexOfNextStartBrace = searchString.indexOf("{", indexOfEndBrace);
            if (indexOfEndRoundBracket === -1)
                endRoundBracket = "";
            else if (indexOfNextStartBrace === -1)
                endRoundBracket = searchString.substring(indexOfEndRoundBracket).trim();
            else if (indexOfNextStartBrace > indexOfEndRoundBracket)
                endRoundBracket = searchString.substring(indexOfEndRoundBracket, searchString.indexOf(" ", indexOfEndRoundBracket));
        }
        addSearchBuilderLine(pos, pos === -1);
        pos++;
        $searchBuilder.find("#concatoperator_" + pos).val(concatOperator);
        $searchBuilder.find("#prebracket_" + pos).val(startRoundBracket);
        $searchBuilder.find("#parameter_" + pos).val(parameter);
        updateOperatorCB(document.getElementById("parameter_" + pos));
        $searchBuilder.find("#operator_" + pos).val(comparison);
        $searchBuilder.find("#term_" + pos).val(value);
        $searchBuilder.find("#postbracket_" + pos).val(endRoundBracket);

        indexOfStartBrace = searchString.indexOf("{", indexOfEndBrace);
        concatOperator = "";
    }

}

function initializeJSON(filterJSON, staticPath) {
    jsonData = filterJSON;
    staticUrl = staticPath;
}

function addHandlers()
{
    let $searchInput = $(".searchinput");
    $searchInput.off();
    $searchInput.on("keyup", function() {
        updateSearchString();
    });
    $searchInput.on("paste", function() {
        //setTimeout needed, otherwise paste hasn't been applied yet (then update is applied before "paste" is applied)
        setTimeout(updateSearchString, 100);
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

    let $addPrebracket = $(".add_prebracket");
    $addPrebracket.off();
    $addPrebracket.on("click", function(event) {
        let id;
        if (event.target.className === "add_prebracket") {
            //pressed enter with focus on hyperlink instead of clicking "image"
            //still click-event is triggered
            id = event.target.firstChild.id.substring(event.target.id.indexOf("_")+1);
        }
        else {
            id = event.target.id.substring(event.target.id.indexOf("_")+1);
        }
        document.getElementById('prebracket_'+id).value =
            document.getElementById('prebracket_'+id).value + "("
        checkBrackets();
        updateSearchString();
    });

    let $addPostbracket = $(".add_postbracket");
    $addPostbracket.off();
    $addPostbracket.on("click", function(event) {
        let id;
        if (event.target.className === "add_postbracket") {
            //pressed enter with focus on hyperlink instead of clicking "image"
            //still click-event is triggered
            id = event.target.firstChild.id.substring(event.target.id.indexOf("_")+1);
        }
        else {
            id = event.target.id.substring(event.target.id.indexOf("_")+1);
        }
        document.getElementById('postbracket_'+id).value =
            document.getElementById('postbracket_'+id).value + ")"
        checkBrackets();
        updateSearchString();
    });

    let $removePrebracket = $(".remove_prebracket");
    $removePrebracket.off();
    $removePrebracket.on("click", function(event) {
        let id;
        if (event.target.className === "remove_prebracket") {
            //pressed enter with focus on hyperlink instead of clicking "image"
            //still click-event is triggered
            id = event.target.firstChild.id.substring(event.target.id.indexOf("_")+1);
        }
        else {
            id = event.target.id.substring(event.target.id.indexOf("_")+1);
        }
        let inputprebracket = document.getElementById('prebracket_'+id).value;
        if (inputprebracket.length > 0)
            document.getElementById('prebracket_'+id).value =
                inputprebracket.substr(0, inputprebracket.length-1);
        checkBrackets();
        updateSearchString();
    });

    let $removePostbracket = $(".remove_postbracket");
    $removePostbracket.off();
    $removePostbracket.on("click", function(event) {
        let id;
        if (event.target.className === "remove_postbracket") {
            //pressed enter with focus on hyperlink instead of clicking "image"
            //still click-event is triggered
            id = event.target.firstChild.id.substring(event.target.id.indexOf("_")+1);
        }
        else {
            id = event.target.id.substring(event.target.id.indexOf("_")+1);
        }
        let inputpostbracket = document.getElementById('postbracket_'+id).value;
        if (inputpostbracket.length > 0)
            document.getElementById('postbracket_'+id).value =
                inputpostbracket.substr(0, inputpostbracket.length-1);
        checkBrackets();
        updateSearchString();
    });


    let $addLine = $(".add_line");
    $addLine.off();
    $addLine.on("click", function(event) {
        let id;
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

    let $remLine = $(".rem_line");
    $remLine.off();
    $remLine.on("click", function(event) {
        let id;
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