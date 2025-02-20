/*global skinDoseMapObj:true, skinDoseMapColourScaleObj, skinDoseMap3dPersonObj, skinDoseMap3dObj:true,
skinDoseMap3dHUDObj:true, skinDoseMapGroupOrigHeight:true, skinDoseMapGroupOrigWidth:true, show3dSkinDoseMap, render,
isCanvasSupported, Urls*/
/*eslint no-undef: "error"*/
/*eslint security/detect-object-injection: "off" */
/*eslint object-shorthand: "off" */

/**
 * Returns the logarithm of y with base x (ie. logxy). This is used as an alternative to Math.log10 which is not
 * supported in Internet Explorer
 * @param x - the base to use
 * @param y - the number for which the logarithm is required
 * @returns {number} - the base x logarithm of y
 */
function getBaseLog(x, y) {
    return Math.log(y) / Math.log(x);
}


/**
 * Function to convert a url to an array of key pairs
 * @param url
 * @returns {{}}
 * @constructor
 */
function urlToArray(url) {
    var request = {};
    var pairs = url.substring(url.indexOf("?") + 1).split("&");
    for (var i = 0; i < pairs.length; i++) {
        if(!pairs[i]) {continue;}
        var pair = pairs[i].split("=");
        request[decodeURIComponent(pair[0])] = decodeURIComponent(pair[1]).replace(/\+/g, " ");
    }
    return request;
}


/**
 * Function to convert an array of key pairs to a url
 * @param array
 * @returns {string}
 * @constructor
 */
function arrayToURL(array) {
    var pairs = [];
    for (var key in array) {
        if (array.hasOwnProperty(key)) {
            pairs.push(encodeURIComponent(key) + "=" + encodeURIComponent(array[key]));
        }
    }
    return pairs.join("&");
}


// Code to update the page and chart data on initial page load.
$(document).ready(function() {
    var requestData = arrayToURL(urlToArray(this.URL));
    var errorMessage = "";

    if (window.location.pathname.includes("force_recalculation/")) {
        var currentUrl = window.location.origin + window.location.pathname;
        var newUrl = currentUrl.replace("force_recalculation/", "");

        // Replace the url after a 2 second delay to give the new task a chance to start running
        setTimeout(function(){
          window.location.replace(newUrl);
        }, 2000);
    }

    $(".ajax-progress-skin-dose").show();

    $.ajax({
        type: "GET",
        url: this.URL + "skin_map/",
        data: requestData,
        dataType: "json",
        success: function( json ) {
            var skinDoseMapContainer = $("#skinDoseMapContainer");

            if (isCanvasSupported()) {

                skinDoseMapObj.initialise(json.skin_map, json.width, json.height + json.phantom_head_height, json.phantom_flat_dist,
                json.phantom_curved_dist);

                if (skinDoseMapObj.maxDose !== 0 && isFinite(skinDoseMapObj.maxDose)) {

                    var skinDoseMapGroup = $("#skinDoseMapGroup");
                    var openskinInfo = $("#openskin_info");

                    var decimalPlaces = Math.abs(Math.ceil(getBaseLog(10, skinDoseMapObj.maxDose))) + 2;
                    if (!isFinite(decimalPlaces)) {decimalPlaces = 0;}

                    skinDoseMapGroup.width(skinDoseMapObj.skinDoseMapCanvas.width + 80).height(skinDoseMapObj.skinDoseMapCanvas.height);
                    openskinInfo.width(skinDoseMapGroup.width());

                    skinDoseMapObj.draw();

                    skinDoseMapColourScaleObj.initialise(skinDoseMapObj.minDose, skinDoseMapObj.maxDose, 70, skinDoseMapObj.skinDoseMapCanvas.height, decimalPlaces);
                    skinDoseMapColourScaleObj.draw();

                    skinDoseMapGroupOrigWidth = skinDoseMapGroup.width();
                    skinDoseMapGroupOrigHeight = skinDoseMapGroup.height();

                    skinDoseMapObj.maxDoseLabel = skinDoseMapObj.maxDose.toFixed(decimalPlaces);
                    skinDoseMapObj.phantomDimensionsLabel = (json.phantom_height + json.phantom_head_height).toFixed(1) + "x" + json.phantom_width.toFixed(1) + "x" + json.phantom_depth.toFixed(1);
                    skinDoseMapObj.patientHeight = (json.patient_height / 100).toFixed(2);
                    skinDoseMapObj.patientMass = json.patient_mass.toFixed(1);
                    skinDoseMapObj.patientOrientation = json.patient_orientation;
                    skinDoseMapObj.fractionDAP = json.fraction_DAP;


                    if (json.patient_height_source.indexOf("extracted") >= 0) {skinDoseMapObj.patientHeightSource = "Extracted";}
                    if (json.patient_mass_source.indexOf("extracted") >= 0) {skinDoseMapObj.patientMassSource = "Extracted";}
                    if (json.patient_orientation_source.indexOf("extracted") >= 0) {skinDoseMapObj.patientOrientationSource = "Extracted";}

                    skinDoseMapObj.writeInformation();

                    $("input[name=windowWidthSlider]").prop({
                        "max": skinDoseMapObj.windowWidth,
                        "step": Math.pow(10, -decimalPlaces),
                        "value": skinDoseMapObj.windowWidth
                    });
                    $("input[name=currentWindowWidth]").val(skinDoseMapObj.windowWidth.toFixed(decimalPlaces));

                    $("input[name=windowLevelSlider]").prop({
                        "max": skinDoseMapObj.windowWidth,
                        "step": Math.pow(10, -decimalPlaces),
                        "value": skinDoseMapObj.windowLevel
                    });
                    $("input[name=currentWindowLevel]").val(skinDoseMapObj.windowLevel.toFixed(decimalPlaces));

                    $("input[name=minDoseSlider]").prop({
                        "min": skinDoseMapObj.minDose,
                        "max": skinDoseMapObj.maxDose,
                        "step": Math.pow(10, -decimalPlaces),
                        "value": skinDoseMapObj.minDose
                    });
                    $("input[name=currentMinDisplayedDose]").val(skinDoseMapObj.minDose.toFixed(decimalPlaces));

                    $("input[name=maxDoseSlider]").prop({
                        "min": skinDoseMapObj.minDose,
                        "max": skinDoseMapObj.maxDose,
                        "step": Math.pow(10, -decimalPlaces),
                        "value": skinDoseMapObj.maxDose
                    });
                    $("input[name=currentMaxDisplayedDose]").val(skinDoseMapObj.maxDose.toFixed(decimalPlaces));

                    if (show3dSkinDoseMap) {
                        skinDoseMap3dObj.canvas.width = skinDoseMapObj.skinDoseMapCanvas.width;
                        skinDoseMap3dObj.canvas.height = skinDoseMapObj.skinDoseMapCanvas.height;
                        skinDoseMap3dObj.windowWidth = skinDoseMapObj.windowWidth;
                        skinDoseMap3dObj.windowLevel = skinDoseMapObj.windowLevel;
                        skinDoseMap3dObj.initialise(json.skin_map, json.phantom_flat_dist, json.phantom_curved_dist,
                        json.phantom_height, json.phantom_depth / 2, json.phantom_head_height, json.phantom_head_radius);
                        skinDoseMap3dObj.draw();
                        skinDoseMap3dPersonObj.initialise(json.phantom_height + json.phantom_head_height);

                        skinDoseMap3dHUDObj.maxDoseLabel = skinDoseMapObj.maxDose.toFixed(decimalPlaces);
                        skinDoseMap3dHUDObj.phantomDimensionsLabel = (json.phantom_height + json.phantom_head_height).toFixed(1) + "x"
                            + json.phantom_width.toFixed(1) + "x" + json.phantom_depth.toFixed(1);
                        skinDoseMap3dHUDObj.patientHeight = (json.patient_height / 100).toFixed(2);
                        skinDoseMap3dHUDObj.patientMass = json.patient_mass.toFixed(1);
                        skinDoseMap3dHUDObj.patientOrientation = json.patient_orientation;
                        skinDoseMap3dHUDObj.fractionDAP = json.fraction_DAP;
                        if (json.patient_height_source.indexOf("extracted") >= 0) {skinDoseMap3dHUDObj.patientHeightSource = "Extracted";}
                        if (json.patient_mass_source.indexOf("extracted") >= 0) {skinDoseMap3dHUDObj.patientMassSource = "Extracted";}
                        if (json.patient_orientation_source.indexOf("extracted") >= 0) {skinDoseMap3dHUDObj.patientOrientationSource = "Extracted";}
                        skinDoseMap3dHUDObj.initialise(skinDoseMap3dObj.canvas.width, skinDoseMap3dObj.canvas.height);

                        render();
                    }
                    $(".ajax-progress-skin-dose").hide();

                    skinDoseMapGroup.show();
                    $("#skin_map_maxmin_controls").show();
                    openskinInfo.show();

                    // Show the "Hide overlay" button; hide the "Show overlay" button
                    $("#skinDoseMapOverlayHide").show();
                    $("#skinDoseMapOverlayShow").hide();

                    // Display the 2d skin map overlay
                    skinDoseMapObj.toggleOverlay();
                }

                else if (json.disabled_skin_maps) {
                    $(".ajax-progress-skin-dose").hide();

                    errorMessage = "<h2>OpenSkin radiation exposure incidence map</h2>" +
                        "<p>Skin maps are disabled for this system: check the settings in <a href='" + Urls.display_names_view() + "'>Display names &amp; modality</a>.</p>";

                    errorMessage += "<p>Create <a href='" + Urls.rfopenskin({pk: json.primary_key}) + "'>openSkin export</a>. (Only available if you have export permissions.)</p>";

                    skinDoseMapContainer.html(errorMessage);
                }

                else if (json.in_progress) {
                    $(".ajax-progress-skin-dose").hide();

                    var taskAdminText = "<a href='" + Urls.task_admin() + "'>task admin page</a>";

                    errorMessage = "<h2>OpenSkin radiation exposure incidence map</h2>" +
                        "<p>Calculation of this skin dose map is in progress. See the " +
                        taskAdminText + " for details.</p>";

                    if (json.skin_map_progress) {
                        errorMessage += "<p>Working on irradiation " + json.skin_map_progress + "</p>";
                    }

                    skinDoseMapContainer.html(errorMessage);
                }

                else if (json.skin_map_calculation_failed) {
                    $(".ajax-progress-skin-dose").hide();

                    errorMessage = "<h2>OpenSkin radiation exposure incidence map</h2>" +
                        "<p>The last time a skin map calculation was attempted for this study it failed.</p>" +
                        "<p>Click <a href='" + window.location.origin + window.location.pathname + "force_recalculation/'>here</a> to force the calculation " +
                        "to try again, or disable calculation for this system in " +
                        "<a href='" + Urls.display_names_view() + "'>Display names &amp; modality</a>.</p>";

                    errorMessage += "<p>Please consider feeding this back to the <a href='http://bitbucket.org/openskin/openskin/'>openSkin BitBucket project</a> " +
                        "or <a href='http://groups.google.com/forum/#!forum/openrem'>OpenREM discussion group</a> so that the issue can be addressed.</p>";

                    errorMessage += "<p>Create <a href='" + Urls.rfopenskin({pk: json.primary_key}) + "'>openSkin export</a>. (Only available if you have export permissions.)</p>";

                    skinDoseMapContainer.html(errorMessage);
                }

                else {
                    $(".ajax-progress-skin-dose").hide();

                    errorMessage = "<h2>OpenSkin radiation exposure incidence map</h2>" +
                        "<p>Sorry, the skin dose map could not be calculated for this study. Possible reasons for this are shown below:</p>" +
                        "<ul>";

                    errorMessage += "<li>Skin maps are disabled for this system: check the settings in Config &rarr; Display names & modality.</li>";
                    errorMessage += "<li>The openSkin code currently only works for Siemens equipment.</li>";
                    if (skinDoseMapObj.maxDose === 0) {errorMessage += "<li>The maximum calculated dose was zero: it may be that every exposure has missed the phantom. This may be due to the way in which this x-ray system has defined the table and x-ray beam geometry.</li>";}
                    if (!isFinite(skinDoseMapObj.maxDose)) {errorMessage +=  "<li>There is no data in skin dose map: the x-ray source to isocentre distance or dose at reference point are not present.</li>";}

                    errorMessage += "</ul>" +
                        "<p>Please consider feeding this back to the <a href='http://bitbucket.org/openskin/openskin/'>openSkin BitBucket project</a> " +
                        "or <a href='http://groups.google.com/forum/#!forum/openrem'>OpenREM discussion group</a> so that the issue can be addressed.</p>";

                    errorMessage += "<p>Create <a href='" + Urls.rfopenskin({pk: json.primary_key}) + "'>openSkin export</a>. (Only available if you have export permissions.)</p>";

                    skinDoseMapContainer.html(errorMessage);
                }
            }

            else {
                $(".ajax-progress-skin-dose").hide();
                skinDoseMapContainer.html("<h2>OpenSkin radiation exposure incidence map</h2>" +
                    "<p>The skin dose map cannot be shown: your browser does not support the HTML &lt;canvas&gt; element.</p>");
            }
        },
        error: function( xhr, status, errorThrown ) {
            $(".ajax-progress-skin-dose").hide();
            $(".ajax-error").show();
            console.log( "Error: " + errorThrown );
            console.log( "Status: " + status );
            console.dir( xhr );
        }
    });
    return false;
});
