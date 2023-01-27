import copy

TEMPLATE = (
    {
    "root": {
        "prev": None,
        "next": None,
        "parent": None,
        "type": "group",
        "first": "1"
    },
    "1": {
        "prev": None,
        "next": None,
        "parent": "root",
        "type": "filter",
        "fields": {
        
        }
    }
    }
)

def get_simple_query(field, value):
    temp = copy.deepcopy(TEMPLATE)
    temp["1"]["fields"][field] = [value, None, False]
    return temp

def get_simple_multiple_query(fields: dict):
    temp = copy.deepcopy(TEMPLATE)
    for (field, value) in fields.items():
        temp["1"]["fields"][field] = [value, None, False]
    return temp