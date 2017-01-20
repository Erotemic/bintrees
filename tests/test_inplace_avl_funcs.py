from bintrees.avltree import height, AVLTree


def assert_avl_invariants(tree, verbose=0):
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
        error = abs(balance_factor) > 1
        if error or verbose:
            print('ERROR')
            ascii_tree(tree)
            print('node.key = %r' % (node.key,))
            print('node.left = %r' % (node.left,))
            print('node.right = %r' % (node.right,))
            print('h1 = %r' % (h1,))
            print('h2 = %r' % (h2,))
            print('balance_factor = %r' % (balance_factor,))
            if error:
                raise AssertionError('Failed balance invariant')

    inorder_keys = [node.key for node in tree._traverse_nodes()]

    if sorted(inorder_keys) != inorder_keys:
        print('inorder_keys = %r' % (inorder_keys,))
        raise AssertionError('Failed order invariant')

    if tree.count != len(inorder_keys):
        if tree.count != -1:
            raise AssertionError('count=%r is inaccurate(%r)' % (tree.count, len(inorder_keys)))


def test_join_cases():
    """
    CommandLine:
        python ~/code/bintrees/tests/test_inplace_avl_funcs.py test_join_cases

    Example:
        >>> # ENABLE_DOCTEST
        >>> from tests.test_inplace_avl_funcs import *  # NOQA
        >>> test_join_cases()
    """
    import numpy as np
    import utool as ut
    import vtool as vt
    import bintrees

    # Exhaustively test that joining trees works
    lowhigh_cases = []
    n = 4
    for x in range(0, 2 ** n):
        for y in range(0, 2 ** n):
            lowhigh_cases += [[x, y]]

    offset = max(map(max, lowhigh_cases)) * 2
    directions = [0, 1]

    test_cases = list(ut.product(directions, lowhigh_cases))
    for direction, lowhigh in ut.ProgIter(test_cases, label='test avl_join'):

        res_list = []
        cls_list = [bintrees.FastAVLTree, bintrees.AVLTree]
        for cls in cls_list:
            keys1 = np.arange(lowhigh[direction])
            keys2 = np.arange(lowhigh[1 - direction]) + offset

            mx1 = vt.safe_max(keys1, fill=0)
            mn2 = vt.safe_min(keys2, fill=mx1 + 2)
            key = value = int(mx1 + mn2) // 2

            self  = cls(list(zip(keys1, keys1)))
            other = cls(list(zip(keys2, keys2)))
            new = self.join_inplace(other, key, value)

            new.recount()
            assert_avl_invariants(new)

            assert len(new) == len(keys1) + len(keys2) + 1
            assert self is new
            assert other._root is None
            res_list.append(list(new.keys()))

        res0 = res_list[0]
        for res in res_list:
            assert res == res0, 'cython and python differ'


def test_splice_cases():
    """
    CommandLine:
        python ~/code/bintrees/tests/test_inplace_avl_funcs.py test_splice_cases

    Example:
        >>> # ENABLE_DOCTEST
        >>> from tests.test_inplace_avl_funcs import *  # NOQA
        >>> test_splice_cases()
    """
    import utool as ut
    import bintrees

    def test_case_gen():
        yield 0, 0, 0
        for size in range(10):
            for start in range(0, size):
                for stop in range(start, size + 1):
                    yield size, start, stop

    test_cases = list(test_case_gen())

    for size, start, stop in ut.ProgIter(test_cases, bs=0, label='avl_splice'):
        keys = list(range(size))

        res_list = []
        cls_list = [bintrees.FastAVLTree, bintrees.AVLTree]
        for cls in cls_list:
            self  = cls(list(zip(keys, keys)))
            self_copy = self.copy()
            inside, outside = self_copy.splice_inplace(start, stop)

            try:
                inside.recount()
                outside.recount()

                assert outside is self_copy, 'outside is the original tree'
                assert inside is not self_copy, 'inside must be new'

                assert_avl_invariants(inside)
                assert_avl_invariants(outside)

                inkeys = sorted(inside.keys())
                outkeys = sorted(inside.keys())

                assert len(inkeys) == 0 or all(k >= start for k in inkeys)
                assert len(inkeys) == 0 or all(k < stop for k in inkeys)

                assert len(inside) + len(outside) == len(self)
                assert set(self.difference(outside).keys()) == set(inside.keys())
                assert len(inside.intersection(outside)) == 0
                res_list.append((sorted(inkeys), sorted(outkeys)))
            except AssertionError:
                print('ERROR')
                print('  size,start,stop = %r,%r,%r' % (size, start, stop,))
                print('cls = %r' % (cls,))
                raise

        res0 = res_list[0]
        for res in res_list:
            assert res == res0, 'cython and python differ'


def test_split_cases():
    """

    CommandLine:
        python ~/code/bintrees/tests/test_inplace_avl_funcs.py test_split_cases

    Example:
        >>> # ENABLE_DOCTEST
        >>> from tests.test_inplace_avl_funcs import *  # NOQA
        >>> test_split_cases()
    """
    # import utool as ut
    import bintrees
    import utool as ut
    # Exhaustive test of every possible way you can split binary trees of size
    def gen_test_cases():
        for size in [0, 1, 2, 3, 4, 100]:
            for key in range(0 - 2, size + 2):
                yield size, key

    test_cases = list(gen_test_cases())
    for size, key in ut.ProgIter(test_cases, label='test_split'):
        # Stride ensures that key will be both in and not in self
        keys = list(range(0, size, 2))
        res_list = []
        cls_list = [bintrees.FastAVLTree, bintrees.AVLTree]
        for cls in cls_list:
            self = cls(list(zip(keys, keys)))
            tree1, tree2, b, v = self.split_inplace(key)
            try:
                assert b == (key in keys)
                keys1 = sorted(tree1.keys())
                keys2 = sorted(tree2.keys())
                assert all(k < key for k in keys1)
                assert all(k > key for k in keys2)
                if b:
                    assert v == key
                res_list.append((keys1, keys2, b, v ))
            except AssertionError:
                print('key = %r' % (key,))
                print('keys1 = %r' % (keys1,))
                print('keys2 = %r' % (keys2,))
                print('b = %r' % (b,))
                print('v = %r' % (v,))
                raise
        res0 = res_list[0]
        for res in res_list:
            assert res == res0


def union_speed_test():
    """
    CommandLine:
        python ~/code/bintrees/tests/test_inplace_avl_funcs.py union_speed_test
    """
    import utool as ut
    import numpy as np
    import bintrees
    num = 10000
    # Generate two sets of random non-contiguous numbers
    numbers = np.arange(-num, num)
    rng = np.random.RandomState(0)
    rng.shuffle(numbers)
    keys1 = numbers[0:num // 4:2]
    keys2 = numbers[1:num // 4:2]

    cls_list = [bintrees.AVLTree, bintrees.FastAVLTree]

    errors = []

    # Test Union - General sets
    for n in [1, 10]:
        verbose = n > 1
        for cls in cls_list:
            name = cls.__name__

            # Time of origional union
            for timer in ut.Timerit(n, name + ' union', verbose=verbose):
                self2  = AVLTree(list(zip(keys1, keys1)))
                other2 = AVLTree(list(zip(keys2, keys2)))
                with timer:
                    combo1 = self2.union(other2)

            # Time of inplace union
            for timer in ut.Timerit(n, name + ' union_inplace', verbose=verbose):
                self1  = cls(list(zip(keys1, keys1)))
                other1 = cls(list(zip(keys2, keys2)))
                with timer:
                    combo2 = self1.union_inplace(other1)

            # Time of copy + inplace union
            for timer in ut.Timerit(n, name + ' union_inplace', verbose=verbose):
                self1  = cls(list(zip(keys1, keys1)))
                other1 = cls(list(zip(keys2, keys2)))
                with timer:
                    combo3 = self1.copy().union_inplace(other1.copy())

            try:
                combo_keys1 = list(combo1.keys())
                combo_keys2 = list(combo2.keys())
                assert combo_keys1 == combo_keys2
            except AssertionError as ex:
                errors.append(ex)
                print('Error for')
                print(' * cls = %r' % (cls,))
                print(ut.repr4(ut.set_overlaps(set(combo1), set(combo2))))
                print('len(combo_keys1) = %r' % (len(combo_keys1),))
                print('len(combo_keys2) = %r' % (len(combo_keys2),))
                import utool
                utool.embed()

            try:
                assert list(combo1.keys()) == list(combo3.keys())
            except AssertionError as ex:
                errors.append(ex)
                print('Error for')
                print(' * cls = %r' % (cls,))
                print(ut.repr4(ut.set_overlaps(set(combo1), set(combo3))))

        assert not errors, 'there were errors'
        # assert list(combo2.keys()) == list(combo3.keys())
        #     assert set(new2) == set(new1)
        #     set(new2) - set(new1)
        # except AssertionError:
        #     print(ut.repr4(ut.set_overlaps(set(new1), set(new2))))


def speed_test():
    """
    CommandLine:
        python ~/code/bintrees/tests/test_inplace_avl_funcs.py speed_test

    Example:
        >>> # DISABLE_DOCTEST
        >>> from tests.test_inplace_avl_funcs import *  # NOQA
        >>> result = speed_test()
        >>> print(result)
    """
    import bintrees
    import numpy as np
    import utool as ut
    low = 1000
    high = 10000
    keys1 = np.arange(low)
    keys2 = np.arange(high) + 2 * low
    key = value = int(keys1.max() + keys2.min()) // 2

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


def debug_splice():
    r"""
    CommandLine:
        python -m tests.test_inplace_avl_funcs debug_splice --show
    """
    import utool as ut
    import bintrees
    import plottool as pt
    import numpy as np
    # size, start, stop = 9, 3, 6
    size, start, stop = 7, 1, 2
    keys = np.arange(size)
    keys = list(range(size))
    self1  = bintrees.AVLTree(list(zip(keys, keys)))
    self2  = bintrees.FastAVLTree(list(zip(keys, keys)))

    inner1, outer1 = self1.copy().splice_inplace(start, stop)
    inner2, outer2 = self2.copy().splice_inplace(start, stop)

    try:
        assert_avl_invariants(inner1)
        assert_avl_invariants(outer1)
        assert_avl_invariants(inner2)
        assert_avl_invariants(outer2)
    except AssertionError:
        pass

    nxkeys = ['key', 'balance']

    def plot(self, slx, sly):
        pnum = (2, 4, (slx, sly))
        pt.show_nx(self.to_networkx(nxkeys), fnum=1, pnum=pnum)
    plot(self1,  slice(0, 1), slice(0, 2))
    plot(inner1, slice(1, 2), 0)
    plot(outer1, slice(1, 2), 1)
    pt.set_xlabel('Python Splice [%r:%r]' % (start, stop))

    plot(self2,  slice(0, 1), slice(2, 4))
    plot(inner2, slice(1, 2), 2)
    plot(outer2, slice(1, 2), 3)
    pt.set_xlabel('Cython Splice [%r:%r]' % (start, stop))
    ut.show_if_requested()


def debug_join2():
    """
    CommandLine:
        python -m tests.test_inplace_avl_funcs debug_join2 --show
    """
    import numpy as np
    import bintrees
    import utool as ut
    # Test Cython Version
    keys1 = np.arange(0, 1)
    keys2 = np.arange(2, 7)

    self1 = bintrees.AVLTree(list(zip(keys1, keys1)))
    other1 = bintrees.AVLTree(list(zip(keys2, keys2)))

    self2 = bintrees.FastAVLTree(list(zip(keys1, keys1)))
    other2 = bintrees.FastAVLTree(list(zip(keys2, keys2)))

    new1 = self1.copy().join2_inplace(other1.copy())
    new2 = self2.copy().join2_inplace(other2.copy())

    nxkeys = []
    # ['key', 'balance']
    import plottool as pt

    def plot(self, slx, sly):
        pnum = (2, 4, (slx, sly))
        pt.show_nx(self.to_networkx(nxkeys), fnum=1, pnum=pnum)

    plot(self1,  slice(0, 1), slice(0, 1))
    plot(other1, slice(0, 1), slice(1, 2))
    plot(new1, slice(1, 2), slice(0, 2))
    pt.set_xlabel('Python Join2')

    plot(self2,  slice(0, 1), slice(2, 3))
    plot(other2, slice(0, 1), slice(3, 4))
    plot(new2, slice(1, 2), slice(2, 4))
    pt.set_xlabel('Cython Join2')
    ut.show_if_requested()


def debug_split():
    r"""
    CommandLine:
        python -m tests.test_inplace_avl_funcs debug_split --show
    """
    import utool as ut
    import bintrees
    import plottool as pt
    # size, key =  9, 3
    size, key =  7, 2
    # keys = list(range(2, 7))
    keys = list(range(size))
    self1  = bintrees.AVLTree(list(zip(keys, keys)))
    self2  = bintrees.FastAVLTree(list(zip(keys, keys)))

    left1, right1, flag1, value1 = self1.copy().split_inplace(key)
    left2, right2, flag2, value2 = self2.copy().split_inplace(key)

    assert_avl_invariants(left1)
    assert_avl_invariants(left2)
    assert_avl_invariants(right1)
    assert_avl_invariants(right2)

    nxkeys = ['key', 'balance']

    def plot(self, slx, sly):
        pnum = (2, 4, (slx, sly))
        pt.show_nx(self.to_networkx(nxkeys), fnum=1, pnum=pnum)
    plot(self1,  slice(0, 1), slice(0, 2))
    plot(left1, slice(1, 2), 0)
    plot(right1, slice(1, 2), 1)
    pt.set_xlabel('Python Split on %r' % (key,))

    plot(self2,  slice(0, 1), slice(2, 4))
    plot(left2, slice(1, 2), 2)
    plot(right2, slice(1, 2), 3)
    pt.set_xlabel('Cython Split on %r' % (key,))
    ut.show_if_requested()


def debug_split_last():
    r"""
    CommandLine:
        python -m tests.test_inplace_avl_funcs debug_split_last --show
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
    pt.set_xlabel('Python Split Last')

    plot(self2,  slice(0, 1), slice(2, 4))
    plot(rest2, slice(1, 2), slice(2, 4))
    pt.set_xlabel('Cython Split Last')
    ut.show_if_requested()


def debug_join():
    """
    CommandLine:
        python -m tests.test_inplace_avl_funcs debug_join --show

    SeeAlso:
        python -m bintrees.avltree AVLTree.join_inplace --show
    """
    import bintrees
    import utool as ut
    import numpy as np
    direction = 1
    lowhigh = [10, 20]
    lowhigh = [10, 20]
    offset = 20 * 2
    keys1 = np.arange(lowhigh[direction])
    keys2 = np.arange(lowhigh[1 - direction]) + offset
    key = value = int(keys1.max() + keys2.min()) // 2
    # keys1 = [0]
    # keys2 = [3, 4, 5, 6]
    # key = value = 2
    print('key = %r' % (key,))

    self1 = bintrees.AVLTree(list(zip(keys1, keys1)))
    other1 = bintrees.AVLTree(list(zip(keys2, keys2)))
    self1_copy = self1.copy()
    other1_copy = other1.copy()

    # if False:
    #     for x in [other1, other1_copy]:
    #         node_t = other1._root.__class__
    #         def new_node(k, b):
    #             n = node_t()
    #             n.key = k
    #             n.value = k
    #             n.balance = b
    #             return n
    #         # hand craft special case
    #         n = new_node(5, 2)
    #         n.right = new_node(6, 0)
    #         n.left = new_node(4, 1)
    #         n.left.left = new_node(3, 0)
    #         x._root = n
    #         print(x)

    self2 = bintrees.FastAVLTree(list(zip(keys1, keys1)))
    other2 = bintrees.FastAVLTree(list(zip(keys2, keys2)))

    new1 = self1_copy.join_inplace(other1_copy, key, value)
    new2 = self2.copy().join_inplace(other2.copy(), key, value)

    nxkeys = []
    nxkeys = ['key', 'balance']

    import plottool as pt

    def plot(self, slx, sly):
        pnum = (2, 4, (slx, sly))
        pt.show_nx(self.to_networkx(nxkeys), fnum=1, pnum=pnum)

    plot(self1,  slice(0, 1), slice(0, 1))
    plot(other1, slice(0, 1), slice(1, 2))
    plot(new1, slice(1, 2), slice(0, 2))
    pt.set_xlabel('Python Join on %r' % (key,))

    plot(self2,  slice(0, 1), slice(2, 3))
    plot(other2, slice(0, 1), slice(3, 4))
    plot(new2, slice(1, 2), slice(2, 4))
    pt.set_xlabel('Cython Join on %r' % (key,))
    ut.show_if_requested()


def debug_union():
    """
    CommandLine:
        python -m tests.test_inplace_avl_funcs debug_union --show
    """
    import bintrees
    import utool as ut
    import numpy as np

    num = 50
    # Generate two sets of random non-contiguous numbers
    numbers = np.arange(-num, num)
    rng = np.random.RandomState(0)
    rng.shuffle(numbers)
    keys1 = numbers[0:num // 4:2]
    keys2 = numbers[1:num // 4:2]

    self1 = bintrees.AVLTree(list(zip(keys1, keys1)))
    other1 = bintrees.AVLTree(list(zip(keys2, keys2)))

    self2 = bintrees.FastAVLTree(list(zip(keys1, keys1)))
    other2 = bintrees.FastAVLTree(list(zip(keys2, keys2)))

    new1 = self1.copy().union_inplace(other1.copy())
    new2 = self2.copy().union_inplace(other2.copy())

    nxkeys = []
    # ['key', 'balance']
    import plottool as pt

    def plot(self, slx, sly):
        pnum = (2, 4, (slx, sly))
        pt.show_nx(self.to_networkx(nxkeys, edge_labels=True), fnum=1, pnum=pnum)

    plot(self1,  slice(0, 1), slice(0, 1))
    pt.set_title('Python Input1')
    plot(other1, slice(0, 1), slice(1, 2))
    pt.set_title('Python Input2')
    plot(new1, slice(1, 2), slice(0, 2))
    pt.set_xlabel('Python Union')

    plot(self2,  slice(0, 1), slice(2, 3))
    pt.set_title('Cython Input1')
    plot(other2, slice(0, 1), slice(3, 4))
    pt.set_title('Cython Input2')
    plot(new2, slice(1, 2), slice(2, 4))
    pt.set_xlabel('Cython Union')
    ut.show_if_requested()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m tests.test_inplace_avl_funcs
        python -m tests.test_inplace_avl_funcs --allexamples
        python ~/code/bintrees/tests/test_inplace_avl_funcs.py --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
