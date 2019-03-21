import numpy as np


def AutoRegressive(elem_param_fn, data_shape, params_shape, elem_idxs, elem_codec):
    def append(message, data, all_params=None):
        if not all_params:
            all_params = elem_param_fn(data)
        for idx in reversed(elem_idxs):
            elem_params = all_params[idx]
            elem_append, _ = elem_codec(elem_params, idx)
            message = elem_append(message, data[idx].astype('uint64'))
        return message

    def pop(message):
        data = np.zeros(data_shape, dtype=np.uint64)
        all_params = np.zeros(params_shape, dtype=np.float32)
        for idx in elem_idxs:
            all_params = elem_param_fn(data, all_params, idx)
            elem_params = all_params[idx]
            _, elem_pop = elem_codec(elem_params, idx)
            message, elem = elem_pop(message)
            data[idx] = elem
        return message, data
    return append, pop
