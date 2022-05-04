/*eslint object-shorthand: "off" */

function retrieveProgress(json ) {
    $.ajax({
        url: Urls.move_update(),
        data: {
            queryID: json.queryID
        },
        type: "POST",
        dataType: "json",
        success: function( json ) {
            $( "#move-status" ).html( json.message );
            $( "#subops" ).html( json.subops );
            if (json.status !== "move complete") setTimeout(function(){
                var data = {
                    queryID: json.queryID
                };
                retrieveProgress( data );
            }, 100);
        }
    });

}
function queryProgress(json ) {
    $.ajax({
        url: Urls.query_update(),
        data: {
            queryID: json.queryID,
            showDetailsLink: json.showDetailsLink
        },
        type: "POST",
        dataType: "json",
        success: function( json ) {
            $( "#qr-status" ).html( json.message );
            $( "#subops" ).html( json.subops );
            if (json.status === "not complete") setTimeout(function(){
                var data = {
                    queryID: json.queryID,
                    showDetailsLink: json.showDetailsLink
                };
                queryProgress( data );
            }, 500);
            if (json.status === "complete"){
                var data = {
                    queryID: json.queryID,
                };
                $.ajax({
                    url: Urls.move_update(),
                    data: data,
                    type: "POST",
                    dataType: "json",
                    success: function( json ) {
                        if (json.status === "not started") {
                            var moveHtml = '<div><button type="button" class="btn btn-default" id="move" data-id="'
                                + json.queryID
                                + '">Move</button></div>';
                            $( "#move-button").html( moveHtml );
                            $("#move").click(function(){
                                // console.log("In the move function");
                                var queryID = $(this).data("id");
                                // console.log(queryID);
                                $( "#move-button").html( "" );
                                $.ajax({
                                    url: Urls.start_retrieve(),
                                    data: {
                                        queryID: queryID
                                    },
                                    type: "POST",
                                    dataType: "json",
                                    success: function( json ) {
                                        // console.log("In the qr success function.");
                                        retrieveProgress( json );
                                    }
                                });
                            });
                        }
                        else {
                            retrieveProgress( {queryID: json.queryID });
                        }
                    }
                });
            }
        }
    });

}


$(document).ready(function(){
    // Submit post on submit
    var form = $("form#post-form");
    form.submit(function(event) {
        event.preventDefault();
        // console.log("ajax form submission function called successfully.");
        $("#move-status" ).html( "" );
        form = $(this);
        // console.log(form);
        var serializedForm = form.serialize();
        $.ajax({ type: "POST",
            url: $(this).attr("action"),
            data: serializedForm,
            dataType: "json",
            success: function( json ) {
                // This is only ever executed on the import page. When used on the details view it is None. (No link shown)
                json.showDetailsLink = "yes";
                queryProgress( json );
            },
            error: function( xhr, status, errorThrown ) {
                alert( "Sorry, there was a problem starting the job!" );
            }
        });
    });
});


