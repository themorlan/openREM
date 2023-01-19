let query = [];

$(document).ready(function () {
    // Search for filterQuery parameter and load it if it exists
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);

    const filterQuery = urlParams.get('filterQuery')
    const test = JSON.parse(decodeURIComponent(filterQuery));
    if (test !== null) {
        query = test;
    }

    renderQuery();


    $('#saveFilter').on('click', function () {
        let new_filter = {};
        $.each($('#newExamFilter').serializeArray(), function (_, kv) {
            if (kv.value === "") {
                return;
            }
            new_filter[kv.name] = kv.value;
        })
        if (Object.keys(new_filter).length <= 0) {
            return;
        }

        if (Object.keys(query).length > 0) {
            query.push({
                "type": "operator",
                "value": "OR"
            })
        } 

        query.push({
            "type": "filter",
            "value": new_filter
        });
        renderQuery();
        $('#newFilterModal').modal('hide')
    });

    $('#SubmitQuery').on('click', function() {
        const token = $('#filterToken').val();
        fetch("", {
            method: "POST",
            credentials: "same-origin",
            headers: {
              "X-Requested-With": "XMLHttpRequest",
              "X-CSRFToken": token,
            },
            body: JSON.stringify({payload: query})
          })
          .then(response => {
            return response.text()
          })
          .then(data => {
            console.log(data)
            $("html").html(data)
          });
    });
});

function updateOperator(raw_el) {
    const el = $(raw_el);
    const pos = el.data('pos').toString();
    console.log(pos);
    let op = pos.split("-").reduce((acc, idx) => acc[idx], query);
    console.log(pos.split("-"))
    if (op["type"] !== "operator") {
        return;
    }
    console.log(el.val())
    op["value"] = el.val();
    renderQuery()
}

function updateFilter(pos) {
    console.log(pos)
}

function renderQuery(position='') {
    $('#myQuery').text(JSON.stringify(query));

    let content = "";

    query.forEach(function (el, idx) {
        if (el["type"] === "filter") {
            content += `
                <div class="row">
                    <div class="col-md-2">
                        <button onclick="updateFilter('${position}${idx}')">Edit filter</button>
                    </div>
                </div>
            `;
        } else if (el["type"] === "operator") {
            let value = el["value"];
            content += `
                <div class="row">
                    <div class="col-md-2">
                        <select class="form-control" onchange="updateOperator(this)" data-pos="${position}${idx}">
                            <option>${value}</option>
                            <option>OR</option>
                            <option>AND</option>
                        </select>
                    </div>
                </div>
            `;
        }
    });

    $('#advFilters').html(content);
    $('#filterQuery').val(encodeURIComponent(JSON.stringify(query)))

}