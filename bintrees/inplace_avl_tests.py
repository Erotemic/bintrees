from bintrees.avltree import height, AVLTree


def assert_avl_invariants(tree):
    r"""
    Args:
        tree (AVLTree):

    CommandLine:
        python -m bintrees.avltree assert_avl_invariants

    Example:
        >>> # DISABLE_DOCTEST
        >>> from bintrees.avltree import *  # NOQA
        >>> import bintrees
        >>> keys = list(range(10))
        >>> print('CPython Version')
        >>> tree = bintrees.AVLTree(list(zip(keys, keys)))
        >>> ascii_tree(tree._root)
        >>> assert_avl_invariants(tree)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from bintrees.avltree import *  # NOQA
        >>> import bintrees
        >>> keys = list(range(10))
        >>> print('Cython Version')
        >>> tree = bintrees.FastAVLTree(list(zip(keys, keys)))
        >>> ascii_tree(tree._root)
        >>> assert_avl_invariants(tree)
    """
    for node in tree._traverse_nodes():
        h1 = height(node.left)
        h2 = height(node.right)
        balance_factor = h1 - h2
        if abs(balance_factor) > 1:
            print('ERROR')
            print('node.key = %r' % (node.key,))
            print('node.left = %r' % (node.left,))
            print('node.right = %r' % (node.right,))
            print('h1 = %r' % (h1,))
            print('h2 = %r' % (h2,))
            print('balance_factor = %r' % (balance_factor,))
            raise AssertionError('Failed balance invariant')

    inorder_keys = [node.key for node in tree._traverse_nodes()]

    if sorted(inorder_keys) != inorder_keys:
        print('inorder_keys = %r' % (inorder_keys,))
        raise AssertionError('Failed order invariant')

    if tree.count != len(inorder_keys):
        raise AssertionError('count is inaccurate')


def test_join_cases(mode='cpython'):
    """
    CommandLine:
        python -m bintrees.avltree test_join_cases 'cython'
        python -m bintrees.avltree test_join_cases 'python'
    """
    import numpy as np
    import utool as ut
    import vtool as vt
    import bintrees
    if mode.lower() == 'python':
        _AVLTree_cls = bintrees.AVLTree
    elif mode.lower() == 'cython':
        _AVLTree_cls = bintrees.FastAVLTree
    else:
        raise ValueError(mode)

    # Exhaustively test that merging trees works
    # lowhigh_cases = [
    #     [1, 3], [1, 10], [1, 9], [3, 9], [2, 3], [3, 32],
    # ]
    lowhigh_cases = []
    n = 4
    for x in range(0, 2 ** n):
        for y in range(0, 2 ** n):
            lowhigh_cases += [[x, y]]

    offset = max(map(max, lowhigh_cases)) * 2
    directions = [0, 1]

    label =  '%s avl_join test' % (mode,)

    test_cases = list(ut.product(directions, lowhigh_cases))
    for direction, lowhigh in ut.ProgIter(test_cases, label=label):
        keys1 = np.arange(lowhigh[direction])
        keys2 = np.arange(lowhigh[1 - direction]) + offset

        mx1 = vt.safe_max(keys1, fill=0)
        mn2 = vt.safe_min(keys2, fill=mx1 + 2)
        key = value = int(mx1 + mn2) // 2

        self  = _AVLTree_cls(list(zip(keys1, keys1)))
        other = _AVLTree_cls(list(zip(keys2, keys2)))
        new = self.join_inplace(other, key, value)
        assert_avl_invariants(new)

        assert len(new) == len(keys1) + len(keys2) + 1
        assert self is new
        assert other._root is None


def debug_splice():
    r"""
    CommandLine:
        python -m bintrees.avltree debug_splice --show
    """
    import utool as ut
    import bintrees
    import plottool as pt
    import numpy as np
    size, start, stop = 9, 3, 6
    keys = np.arange(size)
    keys = list(range(size))
    self1  = bintrees.AVLTree(list(zip(keys, keys)))
    self2  = bintrees.FastAVLTree(list(zip(keys, keys)))

    inner1, outer1 = self1.copy().splice_inplace(start, stop)
    inner2, outer2 = self2.copy().splice_inplace(start, stop)

    nxkeys = []
    # ['key', 'balance']

    def plot(self, slx, sly):
        pnum = (2, 4, (slx, sly))
        pt.show_nx(self.to_networkx(nxkeys), fnum=1, pnum=pnum)
    plot(self1,  slice(0, 1), slice(0, 2))
    plot(inner1, slice(1, 2), 0)
    plot(outer1, slice(1, 2), 1)

    plot(self2,  slice(0, 1), slice(2, 4))
    plot(inner2, slice(1, 2), 2)
    plot(outer2, slice(1, 2), 3)
    ut.show_if_requested()


def debug_split():
    r"""
    CommandLine:
        python -m bintrees.avltree debug_split --show
    """
    import utool as ut
    import bintrees
    import plottool as pt
    import numpy as np
    size, key =  9, 3
    keys = np.arange(size)
    keys = list(range(size))
    self1  = bintrees.AVLTree(list(zip(keys, keys)))
    self2  = bintrees.FastAVLTree(list(zip(keys, keys)))

    left1, right1, flag1, value1 = self1.copy().split_inplace(key)
    left2, right2, flag2, value2 = self2.copy().split_inplace(key)

    nxkeys = []
    # ['key', 'balance']

    def plot(self, slx, sly):
        pnum = (2, 4, (slx, sly))
        pt.show_nx(self.to_networkx(nxkeys), fnum=1, pnum=pnum)
    plot(self1,  slice(0, 1), slice(0, 2))
    plot(left1, slice(1, 2), 0)
    plot(right1, slice(1, 2), 1)

    plot(self2,  slice(0, 1), slice(2, 4))
    plot(left1, slice(1, 2), 2)
    plot(right1, slice(1, 2), 3)
    ut.show_if_requested()


def debug_split_last():
    r"""
    CommandLine:
        python -m bintrees.avltree debug_split_last --show
    """
    import utool as ut
    import bintrees
    import plottool as pt
    import numpy as np
    size = 9
    keys = np.arange(size)
    keys = list(range(size))
    self1  = bintrees.AVLTree(list(zip(keys, keys)))
    self2  = bintrees.FastAVLTree(list(zip(keys, keys)))

    rest1 = self1.copy()
    rest2 = self2.copy()

    key1, value1 = rest1.split_last_inplace()
    key2, value2 = rest2.split_last_inplace()
    print('key1 = %r' % (key1,))
    print('key2 = %r' % (key2,))

    nxkeys = []
    # ['key', 'balance']

    def plot(self, slx, sly):
        pnum = (2, 4, (slx, sly))
        pt.show_nx(self.to_networkx(nxkeys), fnum=1, pnum=pnum)
    plot(self1,  slice(0, 1), slice(0, 2))
    plot(rest1, slice(1, 2), slice(0, 2))

    plot(self2,  slice(0, 1), slice(2, 4))
    plot(rest2, slice(1, 2), slice(2, 4))
    ut.show_if_requested()


def test_splice_cases(mode):
    """
    CommandLine:
        python -m bintrees.avltree test_splice_cases 'python'
        python -m bintrees.avltree test_splice_cases 'cython'
    """
    import numpy as np
    import utool as ut
    import bintrees
    if mode.lower() == 'python':
        _AVLTree_cls = bintrees.AVLTree
    elif mode.lower() == 'cython':
        _AVLTree_cls = bintrees.FastAVLTree
    else:
        raise ValueError(mode)
    label =  '%s avl_splice test' % (mode,)

    def test_case_gen():
        yield 0, 0, 0
        for size in range(10):
            for start in range(0, size):
                for stop in range(start, size + 1):
                    yield size, start, stop

    test_cases = list(test_case_gen())

    for size, start, stop in ut.ProgIter(test_cases, bs=0, label=label):
        print('  size,start,stop = %r,%r,%r' % (size, start, stop,))
        size = 10
        start = 3
        stop = 7
        keys = np.arange(size)

        self  = _AVLTree_cls(list(zip(keys, keys)))
        self_copy = self.copy()
        inside, outside = self_copy.splice_inplace(start, stop)

        slow  = bintrees.AVLTree(list(zip(keys, keys)))
        slow.splice_inplace(start, stop)
        # self.split_inplace(3)

        inside.recount()
        outside.recount()

        assert outside is self_copy
        assert_avl_invariants(inside)
        assert_avl_invariants(outside)

        invals = np.array(list(inside.values()))
        # outvals = np.array(list(inside.values()))

        import utool
        with utool.embed_on_exception_context:
            assert len(invals) == 0 or np.all(invals >= start)
            assert len(invals) == 0 or np.all(invals < stop)

        assert len(inside) + len(outside) == len(self)
        assert set(self.difference(outside).keys()) == set(inside.keys())
        assert len(inside.intersection(outside)) == 0

        # assert_avl_invariants(new)
        # assert len(new) == len(keys1) + len(keys2) + 1
        # assert self is new
        # assert other._root is None
    pass


def test_cython_join_cases():
    """
    CommandLine:
        python -m bintrees.avltree test_cython_join_cases
    """
    # Test Cython Version
    direction = 1
    lowhigh = [10, 20]
    offset = 20 * 2
    import numpy as np
    import bintrees
    keys1 = np.arange(lowhigh[direction])
    keys2 = np.arange(lowhigh[1 - direction]) + offset
    key = value = int(keys1.max() + keys2.min()) // 2
    self = bintrees.FastAVLTree(list(zip(keys1, keys1)))
    other = bintrees.FastAVLTree(list(zip(keys2, keys2)))

    self.join_inplace(other, key, value)


def test_split_cases(mode='cython'):
    """
    CommandLine:
        python -m bintrees.avltree test_split_cases 'python'
        python -m bintrees.avltree test_split_cases 'cython'
    """
    # import utool as ut
    import numpy as np
    import bintrees
    if mode.lower() == 'python':
        _AVLTree_cls = bintrees.AVLTree
    elif mode.lower() == 'cython':
        _AVLTree_cls = bintrees.FastAVLTree
    else:
        raise ValueError(mode)

    keys = list(range(0, 100, 2))
    for key in range(-2, max(keys) + 2):
        self = _AVLTree_cls(list(zip(keys, keys)))
        tree1, tree2, b, v = self.split_inplace(key)
        try:
            assert b == (key in keys)
            keys1 = np.array(list(tree1.keys()))
            keys2 = np.array(list(tree2.keys()))
            assert np.all(keys1 < key)
            assert np.all(keys2 > key)
            if b:
                assert v == key
            print('----')
            print('key = %r' % (key,))
            print('keys1 = %r' % (keys1,))
            print('keys2 = %r' % (keys2,))
            print('b = %r' % (b,))
            print('v = %r' % (v,))
        except AssertionError:
            print('key = %r' % (key,))
            print('keys1 = %r' % (keys1,))
            print('keys2 = %r' % (keys2,))
            print('b = %r' % (b,))
            print('v = %r' % (v,))
            raise


def speed_test():
    """
    CommandLine:
        python -m bintrees.avltree speed_test

        >>> from bintrees.avltree import *  # NOQA
    """
    import numpy as np
    import utool as ut
    low = 1000
    high = 10000
    keys1 = np.arange(low)
    keys2 = np.arange(high) + 2 * low
    key = value = int(keys1.max() + keys2.min()) // 2
    # self  = AVLTree(list(zip(keys1, keys1)))
    # other = AVLTree(list(zip(keys2, keys2)))
    import bintrees

    # Test Join - Assume disjoint
    for n in [1, 10]:
        print('--------------------------')
        verbose = n > 0
        for timer in ut.Timerit(n, 'new CPython avl_join', verbose=verbose):
            self1  = bintrees.AVLTree(list(zip(keys1, keys1)))
            other1 = bintrees.AVLTree(list(zip(keys2, keys2)))
            with timer:
                new1 = self1.join_inplace(other1, key, value)

        for timer in ut.Timerit(n, 'new Cython avl_join', verbose=verbose):
            self3  = bintrees.FastAVLTree(list(zip(keys1, keys1)))
            other3 = bintrees.FastAVLTree(list(zip(keys2, keys2)))
            with timer:
                new3 = self3.join_inplace(other3, key, value)
        try:
            assert set(new3) == set(new1)
        except AssertionError:
            ascii_tree(new1)
            ascii_tree(new3)
            print(ut.repr4(ut.set_overlaps(set(new1), set(new3))))
            print(ut.repr4(ut.set_overlaps(set(new1), set(self3))))
            print(ut.repr4(ut.set_overlaps(set(new1), set(other3))))
            # print(set(new3) - set(new1))
            raise

        for timer in ut.Timerit(n, 'old CPython join (as update)', verbose=verbose):
            self2  = bintrees.AVLTree(list(zip(keys1, keys1)))
            other2 = bintrees.AVLTree(list(zip(keys2, keys2)))
            with timer:
                self2.update(other2)
                new2 = self2
                new2[key] = value
        assert set(new2) == set(new1)

        for timer in ut.Timerit(n, 'old Cython join (as update)', verbose=verbose):
            self2  = bintrees.FastAVLTree(list(zip(keys1, keys1)))
            other2 = bintrees.FastAVLTree(list(zip(keys2, keys2)))
            with timer:
                self2.update(other2)
                new2 = self2
                new2[key] = value
        assert set(new2) == set(new1)

    return

    # Test Union - General sets
    for n in [1, 10]:
        for timer in ut.Timerit(n, 'NEW avl_union with avl_join', verbose=n > 1):
            self1  = AVLTree(list(zip(keys1, keys1)))
            other1 = AVLTree(list(zip(keys2, keys2)))
            with timer:
                new1 = self1.union_inplace(other1)

        for timer in ut.Timerit(n, 'ORIG fast union', verbose=n > 1):
            self2  = bintrees.FastAVLTree(list(zip(keys1, keys1)))
            other2 = bintrees.FastAVLTree(list(zip(keys2, keys2)))
            with timer:
                new2 = self2.union(other2)

        for timer in ut.Timerit(n, 'ORIG slow union', verbose=n > 1):
            self2  = AVLTree(list(zip(keys1, keys1)))
            other2 = AVLTree(list(zip(keys2, keys2)))
            with timer:
                new2 = self2.union(other2)

        try:
            assert set(new2) == set(new1)
            set(new2) - set(new1)
        except AssertionError:
            print(ut.repr4(ut.set_overlaps(set(new1), set(new2))))


def ascii_tree(root):
    if hasattr(root, '_root'):
        root = root._root
    yielder = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node is None:
            yielder.append(None)
        else:
            yielder.append(node)
            queue.append(node.left)
            queue.append(node.right)
    sequence = ['#' if n is None else str(n.key) for n in yielder]
    code = ','.join(sequence)
    # code = code.rstrip('#')
    import drawtree
    drawtree.draw_level_order('{' + code + '}')
    print([(n.key, n.balance) for n in yielder if n is not None])
