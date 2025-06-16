import json

def loads(s):
    return json.loads(s)

def dumps(obj):
    return json.dumps(obj).encode()
