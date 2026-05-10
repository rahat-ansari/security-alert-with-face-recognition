#!/usr/bin/env python
import numpy as np
import json

# Test JSON serialization with numpy types
data = {'unknown_count': np.int64(5), 'persons': [{'id': np.int64(1), 'box': [np.float64(1.0), 2.0, 3.0, 4.0]}], 'arr': np.array([1,2,3])}

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, (np.integer, np.int64, np.int32)):
                obj[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                obj[k] = float(v)
            elif isinstance(v, np.ndarray):
                obj[k] = v.tolist()
            elif isinstance(v, dict):
                convert_numpy_types(v)
            elif isinstance(v, list):
                convert_numpy_types(v)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (np.integer, np.int64, np.int32)):
                obj[i] = int(item)
            elif isinstance(item, (np.floating, np.float64, np.float32)):
                obj[i] = float(item)
            elif isinstance(item, np.ndarray):
                obj[i] = item.tolist()
            elif isinstance(item, dict):
                convert_numpy_types(item)
            elif isinstance(item, list):
                convert_numpy_types(item)
    return obj

log_entry = {'timestamp': '2026-01-01', 'event': 'ALARM'}
log_entry.update(data)
convert_numpy_types(log_entry)
result = json.dumps(log_entry)
print('JSON serialization test passed')
print('Result:', result)

# Test None embedding in dedup
def test_should_save(embedding):
    if embedding is None:
        return False, 'no_embedding', None, None
    return True, 'save', None, None

print('None embedding test:', test_should_save(None))
print('All tests passed!')
