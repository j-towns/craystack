import autograd.numpy as np
from autograd import make_vjp
from autograd.extend import vspace, VSpace

def view_update(data, view_fun):
    view_vjp, item = make_vjp(view_fun)(data)
    item_vs = vspace(item)
    def update(new_item):
        assert item_vs == vspace(new_item), \
            "Please ensure new_item shape and dtype match the data view."
        diff = view_vjp(item_vs.add(new_item,
                                    item_vs.scalar_mul(item, -np.uint64(1))))
        return vspace(data).add(data, diff)
    return item, update

def softmax(x, axis=-1):
    max_x = np.max(x, axis=axis, keepdims=True)
    return np.exp(x - max_x) / np.sum(np.exp(x - max_x), axis=axis, keepdims=True)
