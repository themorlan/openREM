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
        // prevent using an existing id
        newId = makeid(16);
    }
    return newId;
}

const rootGroup = createGroup();
let currentGroupId = null;
let previousId = null;
let nextId = null;
let currentFilterId = null;
let isNewEntry = false;
let discardChanges = true;

let pattern = {
    "root": rootGroup
};

$(document).ready(function () {
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);
    
    const filterQuery = urlParams.get('filterQuery')
    const newPattern = JSON.parse(decodeURIComponent(filterQuery));
    if (newPattern !== null) {
        pattern = newPattern;
    }

    renderPattern();

    $('#saveFilter').on('click', _ => {
        discardChanges = false;
        saveFilter(currentFilterId);
        $('#newFilterModal').modal('hide');
        renderPattern();
    });

    $('#newFilterModal').on('hide.bs.modal', function (e) {
        if (e.namespace !== "bs.modal") {
            return;
        }
        if (isNewEntry && discardChanges) {
            removeFilter(currentFilterId);
            currentFilterId = null;
        }
        isNewEntry = false;
        discardChanges = true;
    });
});

function updateOperator(id) {
    let op = pattern[id];
    if (op === undefined || op["type"] !== "operator") {
        return;
    }
    op["operator"] = $('#' + id).val();
    renderPattern();
}

function saveFilter(id) {
    let filterFields = {};
    $.each($('#newExamFilter').serializeArray(), function (_, kv) {
        if (kv.value === "") {
            return;
        }
        let lookupType = null;
        if ($('#' + kv.name + "_lookupType").length) {
            lookupType = $('#' + kv.name + "_lookupType").val();
        }
        let invert = false;
        if ($('#' + kv.name + "_notToggle").length) {
            invert = $('#' + kv.name + "_notToggle").hasClass("active");
        }
        filterFields[kv.name] = [kv.value, lookupType, invert];
    })
    if (Object.keys(filterFields).length <= 0) {
        return;
    }
    pattern[id].fields = filterFields;
}

function addEntry(caller, newEntry, newEntryId) {
    currentGroupId = caller.data("group");
    previousId = caller.data("previous");
    nextId = caller.data("next");

    newEntry.parent = currentGroupId;
    let newOperatorId = getNewId();

    if (previousId === undefined || !(previousId in pattern)) {
        previousId = null;
    }

    if (nextId === undefined || !(nextId in pattern)) {
        nextId = null;
    }

    if (previousId === null && nextId === null) {
        pattern[currentGroupId].first = newEntryId;
    } else if (previousId === null) {
        pattern[currentGroupId].first = newEntryId;
        newEntry.next = newOperatorId;
        pattern[nextId].prev = newOperatorId;
        pattern[newOperatorId] = createOperator("OR", newEntryId, nextId, currentGroupId);
    } else if (nextId === null) {
        newEntry.prev = newOperatorId;
        pattern[newOperatorId] = createOperator("OR", previousId, newEntryId, currentGroupId);
        pattern[previousId].next = newOperatorId;
    } else {
        if (pattern[previousId].type === "operator") {
            newEntry.prev = previousId;
            newEntry.next = newOperatorId;
            pattern[newOperatorId] = createOperator("OR", newEntryId, nextId, currentGroupId);
            pattern[previousId].next = newEntryId;
            pattern[nextId].prev = newOperatorId;
        } else {
            newEntry.prev = newOperatorId;
            newEntry.next = nextId;
            pattern[newOperatorId] = createOperator("OR", previousId, newEntryId, currentGroupId);
            pattern[previousId].next = newOperatorId;
            pattern[nextId].prev = newEntryId;
        }
    }
    pattern[newEntryId] = newEntry;
}

function addFilter(caller) {
    let newFilterId = getNewId();
    const newFilter = createFilter({}, null, null, null);

    currentFilterId = newFilterId;

    addEntry($(caller), newFilter, newFilterId);

    isNewEntry = true;

    openFilter(currentFilterId);
}

function openFilter(id) {
    let filter = pattern[id];
    if (filter === undefined || filter === null) {
        return;
    }
    $('#newFilterModal').modal('show');
    $(':input', '#newExamFilter').val('');
    $('*[id*=lookupType]').each(function() {
        $(this).val("iexact");
    });
    $('*[id*=notToggle]').each(function() {
        $(this).removeClass("active");
    });   
    $.each($('#newExamFilter').serializeArray(), function (_, kv) {
        let val = filter["fields"][kv.name];
        if (val === undefined) {
            return;
        }
        var [value, lookupType, invert] = val;
        if ($('#' + kv.name + "_lookupType").length) {
            $('#' + kv.name + "_lookupType").val(lookupType);
        }
        if ($('#' + kv.name + "_notToggle").length && invert) {
            $('#' + kv.name + "_notToggle").addClass("active");
        }
        $('#id_'+kv.name).val(value);
    });
    currentFilterId = id;
    previousId = filter.prev;
    nextId = filter.next;
    currentGroupId = filter.parent;
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
        if (filter.parent !== "root") {
            removeFilter(filter.parent)
        }
    } else if (prevOp === null) {
        pattern[nextOp.next].prev = null;
        pattern[filter.parent].first = nextOp.next;
        delete pattern[filter.next]
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
    renderPattern();
}

function addGroup(caller) {    
    let newGroupId = getNewId();
    let newGroup = createGroup(null, null, null, null);
    
    currentFilterId = getNewId();

    addEntry($(caller), newGroup, newGroupId);

    newGroup.first = currentFilterId;
    pattern[newGroupId] = newGroup;
    
    let newFilter = createFilter({}, null, null, newGroupId);
    pattern[currentFilterId] = newFilter;

    renderPattern(); 
    openFilter(currentFilterId);
}

function loadFromLibrary() {
    let libraryId = $('#filterLibraryId').val();
    if (libraryId === NaN) {
        return;
    }

    $.get("/openrem/filters/" + libraryId, function(data) {
        if (data.pattern !== undefined && data.pattern !== null) {
            pattern = data.pattern;
            renderPattern();
            $('#submitQuery').click();
        }
    });
}

function deleteFromLibrary() {
    let libraryId = $('#filterLibraryId').val();
    if (libraryId === NaN) {
        return;
    }

    $.get("/openrem/filters/delete/" + libraryId, function(data) {
        renderPattern();
        $('#submitQuery').click();
    });
}

function saveToLibrary() {
    // Perofrming POST request to store a new pattern
    let libraryName = $('#newFilterLibraryName').val();
    if (libraryName === undefined || libraryName === null) {
        return;
    }
    $.post("/openrem/filters/add/", { libraryName: libraryName, pattern: JSON.stringify(pattern), csrfmiddlewaretoken: $('#postToken').val() }, function(data) {
        renderPattern();
        $('#submitQuery').click();
    });
}

function renderGroup(group="root", level=0) {
    let content = "";
    let currentId = pattern[group].first;
    
    while (currentId !== null) {
        const current = pattern[currentId];
        if (current.type === "filter") {
            content += `
                <div id="${currentId}_row">
                    ${getButtonTemplate(group, level, current.prev, currentId)}
                    <div class="row">
                        <div class="col-md-1 col-md-offset-${level} text-center">
                            <a class="btn btn-danger" onclick="removeFilter('${currentId}')">
                                <span class="glyphicon glyphicon-remove" aria-hidden="true"></span>
                            </a>
                        </div>
                        <div class="col-md-2 text-center">
                            <a class="btn btn-primary" onclick="openFilter('${currentId}')"
                            data-group="${group}" data-previous="${current["prev"]}"
                            data-next="${current["next"]}">Edit filter</a>
                        </div>
                        <div class="col-md-*">
                            <p>${JSON.stringify(current.fields)}</p>
                        </div>
                    </div>
                    ${getButtonTemplate(group, level, currentId, current.next)}
                </div>
            `;
        } else if (current.type === "operator") {
            content += `
                <div class="row" id="${currentId}_row">
                    <div class="col-md-2 col-md-offset-${level + 1}">
                        <select id="${currentId}" class="form-control text-center" onchange="updateOperator('${currentId}')">
                            <option>${current.operator}</option>
                            <option>OR</option>
                            <option>AND</option>
                        </select>
                    </div>
                </div>
            `;
        } else if (current.type === "group") {
            content += `
                <div id="${currentId}_row">
                    ${getButtonTemplate(group, level, current.prev, currentId)}
                    ${renderGroup(currentId, level + 1)}
                    ${getButtonTemplate(group, level, currentId, current.next)}
                </div>
            `;
        }
        currentId = pattern[currentId].next;
    }
    return content;
}

function renderPattern() {    
    const content = renderGroup();
    if (content.length > 0) {
        $('#advFilters').html(content);
    } else {
        $('#advFilters').html(`
            <div class="row">
                <div class="col-md-2 text-center">
                    <a class="btn btn-primary" onclick="addFilter(this)"
                    data-group="root" data-previous="null" data-next="null">Add filter</a>
                </div>
            </div>
        `);
    }
    $('#filterQuery').val(encodeURIComponent(JSON.stringify(pattern)))
}

function getButtonTemplate(group, level=0, prevId, nextId) {
    return `
        <div class="row" style="margin-top: 1em; margin-bottom: 1em;">
            <div class="col-md-2  col-md-offset-${level + 1} text-center">
                <a class="btn btn-success btn-xs" onclick="addFilter(this)"
                data-group="${group}" data-previous="${prevId}" data-next="${nextId}">
                    <span class="glyphicon glyphicon-plus" aria-hidden="true"></span>
                </a>
                <a class="btn btn-success btn-xs" onclick="addGroup(this)"
                data-group="${group}" data-previous="${prevId}" data-next="${nextId}">
                    <span class="glyphicon glyphicon-menu-hamburger" aria-hidden="true"></span>
                </a>
            </div>
        </div>
    `;
}
