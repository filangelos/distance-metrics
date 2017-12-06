import sys


def progress(count, total, status='', length=60):
    """Command Line Progress Bar.

    Parameters
    ----------
    count: int
        Current progress
    total: int
        Total progress value
    status: str
        Message to be printed next to bar
    length: int
        Length of progress bar, number of characters

    Examples
    --------
    >>> progress(1, 25, status='Example progress bar!')
    >>> [=-------------------------] 4% # Example progress bar!
    """
    _filled_len = int(round(length * count / float(total)))
    _percents = round(100.0 * count / float(total), 1)
    _bar = '*' * _filled_len + ' ' * (length - _filled_len)
    sys.stderr.write('\r[%s] %s%s # %s' % (_bar, _percents, '%', status))
    sys.stderr.flush()
