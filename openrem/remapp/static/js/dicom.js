$(document).ready(function(){
    $('#main-menu a').click(function (event) {
        // Prevents the default behavior which is
        // to load the page synchronously
        event.preventDefault();

        $('#main-ajax').load(this.href + ' #sharemenu *', function (data, status) {
            $bar = $('#notification-bar');

            if (status === 'success') {
                $bar.text('The page has been successfully loaded.');
            } else {
                $bar.text('An error occurred.');
            }

            $bar
                .slideDown('normal')
                .delay(2000)
                .slideUp('fast');
        });
    });

    $('#qr-go').click(function ( event ){
        event.preventDefault();

        $.ajax({
            url: "/openrem/admin/dicomstore/ajax_test2",
            data: {
                qr_id: 1
            },
            type: "POST",
            dataType: "json",
            success: function( json ) {
                query_progress( json );
            },
            error: function( xhr, status, errorThrown ) {
                alert( "Sorry, there was a problem starting the job!" );
                console.log( "Error: " + errorThrown );
                console.log( "Status: " + status );
                console.dir( xhr );
            },
            complete: function( xhr, status ) {
                alert( "The request is complete!" );
            }
        });
    });
});


function query_progress( json ) {
    $.ajax({
        url: "/openrem/admin/dicomstore/ajax_test3",
        data: {
            query_id: json.query_id
        },
        type: "POST",
        dataType: "json",
        success: function( json ) {
            alert( "I'm in test3!");
            $( '#qr-status' ).html( json.message);
            if (json.status != "complete") setTimeout(function(){
                var data = {
                    query_id: json.query_id
                }
                query_progress( data );
            }, 1000);
        },
        error: function( xhr, status, errorThrown ) {
            alert( "Sorry, there was a problem getting the status!" );
            console.log( "Error: " + errorThrown );
            console.log( "Status: " + status );
            console.dir( xhr );
        }

    });

}