import numpy as np
import craystack.vectorans as vrans
import craystack.util as util


def NonUniform(enc_statfun, dec_statfun, precision):
    """
    Codec for symbols which are not uniformly distributed. The statfuns specify
    the following mappings:

        enc_statfun: symbol |-> start, freq
        dec_statfun: cf |-> symbol

    The interval [0, 1) is modelled by the range of integers
    [0, 2 ** precision). The operation performed by enc_statfun is used for
    compressing data and is visualised below for a distribution over a set of
    symbols {a, b, c, d}.

    0                                                         2 ** precision
    |    a              !b!              c              d         |
    |----------|--------------------|---------|-------------------|
               |------ freq --------|
             start

    Calling enc_statfun(b) must return the pair (start, freq), where start is
    the start of the interval representing the symbol b and freq is its width.
    Start and freq must satisfy the following constraints:

        0 <  freq
        0 <= start        <  2 ** precision
        0 <  start + freq <= 2 ** precision

    The value of start is analagous to the cdf of the distribution, evaluated
    at b, while freq is analagous to the pmf evaluated at b.

    The function dec_statfun essentially inverts enc_statfun. It is
    necessary for decompressing data, to recover the original symbol.

    0                                                         2 ** precision
    |    a               b               c              d         |
    |----------|-----+--------------|---------|-------------------|
                     ↑
                     cf

    For a number cf in the range [0, 2 ** precision), dec_statfun must return
    the symbol whose range cf lies in, which in the picture above is b.
    """
    def append(message, symbol):
        start, freq = enc_statfun(symbol)
        return vrans.append(message, start, freq, precision)

    def pop(message):
        cf, pop_fun = vrans.pop(message, precision)
        symbol = dec_statfun(cf)
        start, freq = enc_statfun(symbol)
        assert np.all(start <= cf) and np.all(cf < start + freq)
        return pop_fun(start, freq), symbol
    return append, pop

def repeat(codec, n):
    """
    Repeat codec n times.

    Assumes that symbols is a Numpy array with symbols.shape[0] == n. Assume
    that the codec doesn't change the shape of the ANS stack head.
    """
    append_, pop_ = codec
    def append(message, symbols):
        assert np.shape(symbols)[0] == n
        for symbol in reversed(symbols):
            message = append_(message, symbol)
        return message

    def pop(message):
        symbols = []
        for i in range(n):
            message, symbol = pop_(message)
            symbols.append(symbol)
        return message, np.asarray(symbols)
    return append, pop

def substack(codec, view_fun):
    append_, pop_ = codec
    def append(message, data):
        head, tail = message
        subhead, update = util.view_update(head, view_fun)
        subhead, tail = append_((subhead, tail), data)
        return update(subhead), tail
    def pop(message):
        head, tail = message
        subhead, update = util.view_update(head, view_fun)
        (subhead, tail), data = pop_((subhead, tail))
        return (update(subhead), tail), data
    return append, pop

def parallel(codecs, view_funs):
    """
    Run a number of independent codecs on different substacks. This could be
    executed in parallel, but when running in the Python interpreter will run
    in series.

    Assumes data is a list, arranged the same way as codecs and view_funs.
    """
    codecs = [substack(codec, view_fun)
              for codec, view_fun in zip(codecs, view_funs)]
    def append(message, symbols):
        assert len(symbols) == len(codecs)
        for (append, _), symbol in reversed(list(zip(codecs, symbols))):
            message = append(message, symbol)
        return message
    def pop(message):
        symbols = []
        for _, pop in codecs:
            message, symbol = pop(message)
            symbols.append(symbol)
        assert len(symbols) == len(codecs)
        return message, symbols
    return append, pop

def shape(message):
    head, _ = message
    def _shape(head):
        if type(head) is tuple:
            return tuple(_shape(h) for h in head)
        else:
            return np.shape(head)
    return _shape(head)
