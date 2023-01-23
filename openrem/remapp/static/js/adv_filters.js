function createEntry(prev=null, next=null, parent=null) {
    return {
        prev: prev,
        next: next,
        parent: parent,
        type: "entry"
    };
}

function createFilter(fields={}, prev=null, next=null, parent=null) {
    let res = createEntry(prev, next, parent);
    res.fields = fields;
    res.type = "filter";
    return res;
}

function createOperator(operator="OR", prev=null, next=null, parent=null) {
    let res = createEntry(prev, next, parent);
    res.operator = operator;
    res.type = "operator";
    return res;
}
function createGroup(first=null, prev=null, next=null, parent=null) {
    let res = createEntry(prev, next, parent);
    res.first = first;
    res.type = "group";
    return res;
}

function makeid(length) {
    var result           = '';
    var characters       = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    var charactersLength = characters.length;
    for ( var i = 0; i < length; i++ ) {
        result += characters.charAt(Math.floor(Math.random() * charactersLength));
    }
    return result;
}

function getNewId() {
    let newId = makeid(16);
    while (newId in pattern) {
        newId = makeid(16);
    }
    return newId;
}

const rootGroup = createGroup();
let currentGroup = null;
let previousId = null;
let nextId = null;
let currentButton = null;
let currentFilter = null;
let doUpdate = false;

let pattern = {
    "root": rootGroup
};

$(document).ready(function () {
    // Search for filterQuery parameter and load it if it exists
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);

    const filterQuery = urlParams.get('filterQuery')
    const newPattern = JSON.parse(decodeURIComponent(filterQuery));
    if (newPattern !== null) {
        pattern = newPattern;
    }

    renderQuery();

    $('#newFilterModal').on('show.bs.modal', function (event) {
        var button = $(event.relatedTarget);
        currentGroup = button.data('group');
        previousId = button.data('previous');
        nextId = button.data('next');
        currentButton = button;
    });      

    $('#saveFilter').on('click', _ => {
        if (doUpdate === true) {
            updateFilter();
        } else {
            addFilter();
        }
        $('#newFilterModal').modal('hide');
        renderQuery();
    });
});

function updateOperator(id) {
    let op = pattern[id];
    if (op === undefined || op["type"] !== "operator") {
        return;
    }
    op["operator"] = $('#' + id).val();
    renderQuery();
}

function addFilter() {
    let filterFields = {};
    $.each($('#newExamFilter').serializeArray(), function (_, kv) {
        if (kv.value === "") {
            return;
        }
        let lookupType = null;
        if ($('#' + kv.name + "_lookupType").length) {
            lookupType = $('#' + kv.name + "_lookupType").val();
        }
        filterFields[kv.name] = [kv.value, lookupType];
    })
    if (Object.keys(filterFields).length <= 0) {
        return;
    }

    let newFilterId = getNewId();
    let newOperatorId = getNewId();

    if (previousId === undefined || !(previousId in pattern)) {
        previousId = null;
    }

    if (nextId === undefined || !(nextId in pattern)) {
        nextId = null;
    }

    if (previousId === null && nextId === null) {
        pattern[newFilterId] = createFilter(filterFields, null, null, currentGroup);
        pattern[currentGroup].first = newFilterId;
        currentButton.hide();
        return;
    } else if (previousId === null) {
        pattern[newFilterId] = createFilter(filterFields, null, newOperatorId, currentGroup);
        pattern[currentGroup].first = newFilterId;
        pattern[nextId].prev = newOperatorId;
        pattern[newOperatorId] = createOperator("OR", newFilterId, nextId, currentGroup);
        currentButton.data('previous', newOperatorId);
        return;
    } else if (nextId === null) {
        pattern[newFilterId] = createFilter(filterFields, newOperatorId, null, currentGroup);
        pattern[newOperatorId] = createOperator("OR", previousId, newFilterId, currentGroup);
        pattern[previousId].next = newOperatorId;
        currentButton.data('next', newOperatorId);
        return;
    }

    if (pattern[previousId].type === "operator") {
        pattern[newFilterId] = createFilter(filterFields, previousId, newOperatorId, currentGroup);
        pattern[newOperatorId] = createOperator("OR", newFilterId, nextId, currentGroup);
        currentButton.data('next', newFilterId);
        pattern[previousId].next = newFilterId;
        pattern[nextId].prev = newOperatorId;
    } else {
        pattern[newOperatorId] = createOperator("OR", previousId, newFilterId, currentGroup);
        pattern[newFilterId] = createFilter(filterFields, newOperatorId, nextId, currentGroup);
        currentButton.data('next', newOperatorId);
        pattern[previousId].next = newOperatorId;
        pattern[nextId].prev = newFilterId;
    }
}

function updateFilter() {
    if (currentFilter === undefined || currentFilter === null) {
        return;
    }
    let filterFields = {};
    $.each($('#newExamFilter').serializeArray(), function (_, kv) {
        if (kv.value === "") {
            return;
        }
        let lookupType = null;
        if ($('#' + kv.name + "_lookupType").length) {
            lookupType = $('#' + kv.name + "_lookupType").val();
        }
        filterFields[kv.name] = [kv.value, lookupType];
    })
    if (Object.keys(filterFields).length <= 0) {
        return;
    }
    pattern[currentFilter].fields = filterFields;
}

function openFilter(id) {
    let filter = pattern[id];
    if (filter === undefined || filter === null) {
        return;
    }
    $('#newFilterModal').modal('show');
    $(':input', '#newExamFilter').val('');
    $.each($('#newExamFilter').serializeArray(), function (_, kv) {
        let val = filter["fields"][kv.name];
        if (val === undefined) {
            return;
        }
        var [value, lookupType] = val;
        if ($('#' + kv.name + "_lookupType").length) {
            $('#' + kv.name + "_lookupType").val(lookupType);
        }
        $('#id_'+kv.name).val(value);
    });
    doUpdate = true;
    currentFilter = id;
    previousId = filter.prev;
    nextId = filter.next;
    currentGroup = filter.parent;
}

function removeFilter(id) {
    let filter = pattern[id];
    if (filter === undefined || filter === null) {
        return;
    }
    let prevOp = pattern[filter.prev];
    let nextOp = pattern[filter.next];


    if (prevOp === undefined) {
        prevOp = null;
    }

    if (nextOp === undefined) {
        nextOp = null;
    }

    if (prevOp === null && nextOp === null) {
        pattern[filter.parent].first = null;
        delete pattern[id];
    } else if (prevOp === null) {
        pattern[nextOp.next].prev = null;
        pattern[filter.parent].first = nextOp.next;
        delete pattern[filter.next]
        delete pattern[id];
    } else if (nextOp === null) {
        pattern[prevOp.prev].next = null;
        delete pattern[filter.prev]
    } else {
        pattern[nextOp.next].prev = filter.prev;
        pattern[filter.prev].next = nextOp.next;
        delete pattern[filter.next];
    }

    $('#' + id + '_row').remove();
    delete pattern[id];
    renderQuery();
}

function renderQuery(group='root', level=0) {
    $('#myQuery').text(JSON.stringify(pattern));

    let content = "";
    let currentId = pattern[group].first;
    
    while (currentId !== null) {
        let current = pattern[currentId];
        if (current.type === "filter") {
            content += `
                <div id="${currentId}_row">
                    <div class="row">
                        <a class="btn btn-success" data-toggle="modal" data-target="#newFilterModal"
                        data-group="${group}" data-previous="${current.prev}" data-next="${currentId}">+</a>
                    </div>
                    <div class="row">
                        <div class="col-md-2">
                            <a class="btn btn-primary" onclick="openFilter('${currentId}')"
                            data-group="${group}" data-previous="${current["prev"]}"
                            data-next="${current["next"]}">Edit filter</a>
                        </div>
                        <div class="col-md-2">
                            <button onclick="removeFilter('${currentId}')">X</button>
                        </div>
                        <div class="col-md-2">
                            <p>${JSON.stringify(current.fields)}</p>
                        </div>
                    </div>
                    <div class="row">
                        <a class="btn btn-success" data-toggle="modal" data-target="#newFilterModal"
                        data-group="${group}" data-previous="${currentId}" data-next="${current.next}">+</a>
                    </div>
                </div>
            `;
        } else if (current.type === "operator") {
            content += `
                <div class="row" id="${currentId}_row">
                    <div class="col-md-2">
                        <select id="${currentId}" class="form-control" onchange="updateOperator('${currentId}')">
                            <option>${current.operator}</option>
                            <option>OR</option>
                            <option>AND</option>
                        </select>
                    </div>
                </div>
            `;
        } else if (current.type === "group") {
            // TODO: Add ability to have nested groups
        }
        currentId = pattern[currentId].next;
    }
    $('#advFilters').html(content);
    $('#filterQuery').val(encodeURIComponent(JSON.stringify(pattern)))
}
