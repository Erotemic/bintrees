#!/usr/bin/env python
#coding:utf-8
# Author:  mozman (python version)
# Purpose: avl tree module (Julienne Walker's unbounded none recursive algorithm)
# source: http://eternallyconfuzzled.com/tuts/datastructures/jsw_tut_avl.aspx
# Created: 01.05.2010
# Copyright (c) 2010-2013 by Manfred Moitzi
# License: MIT License

# Conclusion of Julienne Walker

# AVL trees are about as close to optimal as balanced binary search trees can
# get without eating up resources. You can rest assured that the O(log N)
# performance of binary search trees is guaranteed with AVL trees, but the extra
# bookkeeping required to maintain an AVL tree can be prohibitive, especially
# if deletions are common. Insertion into an AVL tree only requires one single
# or double rotation, but deletion could perform up to O(log N) rotations, as
# in the example of a worst case AVL (ie. Fibonacci) tree. However, those cases
# are rare, and still very fast.

# AVL trees are best used when degenerate sequences are common, and there is
# little or no locality of reference in nodes. That basically means that
# searches are fairly random. If degenerate sequences are not common, but still
# possible, and searches are random then a less rigid balanced tree such as red
# black trees or Andersson trees are a better solution. If there is a significant
# amount of locality to searches, such as a small cluster of commonly searched
# items, a splay tree is theoretically better than all of the balanced trees
# because of its move-to-front design.

from __future__ import absolute_import, division, print_function

from .abctree import ABCTree
from array import array
# import utool as ut
# print, rrr, profile = ut.inject2(__name__)

# __all__ = ['AVLTree']

MAXSTACK = 32


class Node(object):
    """Internal object, represents a tree node."""
    # __slots__ = ['left', 'right', 'balance', 'key', 'value']

    def __init__(self, key=None, value=None):
        self.left = None
        self.right = None
        self.key = key
        self.value = value
        self.balance = 0

    @property
    def xdata(self):
        """ compatibility with the C node_t struct """
        return self.balance

    @xdata.setter
    def xdata(self, value):
        """ compatibility with the C node_t struct """
        self.balance = value

    def __getitem__(self, key):
        """N.__getitem__(key) <==> x[key], where key is 0 (left) or 1 (right)."""
        return self.left if key == 0 else self.right

    def __setitem__(self, key, value):
        """N.__setitem__(key, value) <==> x[key]=value, where key is 0 (left) or 1 (right)."""
        if key == 0:
            self.left = value
        else:
            self.right = value

    def free(self):
        """Remove all references."""
        self.left = None
        self.right = None
        self.key = None
        self.value = None


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
        # for node in traverse_avl_nodes(tree._root):
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

    inorder_keys = [node.key for node in traverse_avl_nodes(tree._root)]

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


def traverse_avl_nodes(root):
    stack = []
    node = root
    yielder = []
    while stack or node is not None:
        if node is not None:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            yielder += [node]
            # yield node
            node = node.right
    return yielder


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


def avl_splice(root, start_key, stop_key):
    """
    Extracts a ordered slice from `root` and returns the inside and outside
    parts.

    O(log(n))
    """
    # import utool as ut
    # ut.cprint('----- SPLICE(PY) -', 'yellow')
    # print('-----')
    # print('root.key = %r' % (root.key,))
    # print('start_key = %r' % (start_key,))
    # print('stop_key = %r' % (stop_key,))

    # Split tree into three parts
    left, midright, start_flag, start_val = avl_split(root, start_key)
    # print('left.key = %r' % (left.key,))
    # print('midright.key = %r' % (midright.key,))

    middle, right, stop_flag, stop_val = avl_split(midright, stop_key)

    # print('left.key = %r' % (left.key,))
    # print('middle.key = %r' % (middle.key,))
    # print('right.key = %r' % (right.key,))

    # Insert the start_key back into the middle part if it was removed
    if start_flag:
        t_inner = avl_insert(middle, start_key, start_val)
    else:
        t_inner = middle
    # Recombine the outer parts
    if stop_flag:
        t_outer = avl_join(left, right, stop_key, stop_val)
    else:
        t_outer =  avl_join2(left, right)
    # ut.cprint('-----', 'yellow')
    return t_inner, t_outer


def avl_split_last(root):
    """
    Removes the maximum element from the tree

    O(log(n)) = O(height(root))
    """
    if root is None:
        raise IndexError('Empty tree has no maximum element')
    left, right = root.left, root.right
    if right is None:
        return (left, root.key, root.value)
    else:
        new_right, max_k, max_v = avl_split_last(right)
        new_root = avl_join(left, new_right, root.key, root.value)
        return (new_root, max_k, max_v)


def avl_join2(t1, t2):
    """
    join two trees without any intermediate key

    O(log(n) + log(m)) = O(r(t1) + r(t2))

    For AVL-Trees the rank r(t1) = height(t1) - 1
    """
    if t1 is None:
        return t2
    else:
        new_left, max_k, max_v = avl_split_last(t1)
        return avl_join(new_left, t2, max_k, max_v)


def avl_insert(root, key, value):
    """
    Functional version of insert

    O(log(n))
    """
    left, right, flag, old_val = avl_split(root, key)
    return avl_join(left, right, key, value)


def avl_remove(root, key, value):
    """
    Functional version of remove

    O(log(n))
    """
    left, right, flag, old_val = avl_split(root, key)
    return avl_join2(left, right)


def avl_intersection(t1, t2):
    """
    O(log(n)log(m))
    """
    if t1 is None:
        return None
    elif t2 is None:
        return None
    else:
        l2, r2 = t2[0], t2[1]
        k2, v2 = t2.key, t2.value
        l1, r1, b, bv = avl_split(t1, k2)
        tl = avl_intersection(l1, l2)
        tr = avl_intersection(r1, r2)
        if b:
            return avl_join(tl, tr, k2, v2)
        else:
            return avl_join2(tl, tr)


def avl_difference(t1, t2):
    """
    O(m log((n/m) + 1))
    """
    if t1 is None:
        return None
    elif t2 is None:
        return t1
    else:
        l2, r2 = t2[0], t2[1]
        k2 = t2.key
        l1, r1, b, bv = avl_split(t1, k2)
        tl = avl_difference(l1, l2)
        tr = avl_difference(r1, r2)
        return avl_join2(tl, tr)


def avl_union(t1, t2):
    """
    O(m log((n/m) + 1))
    This is sublinear and good improvement over O(mlog(n))
    """
    if t1 is None:
        return t2
    elif t2 is None:
        return t1
    else:
        l2, r2 = t2.left, t2.right
        k2, v2 = t2.key, t2.value
        l1, r1, b, bv = avl_split(t1, k2)
        tl = avl_union(l1, l2)
        tr = avl_union(r1, r2)
        return avl_join(tl, tr, k2, v2)


def avl_split(root, key):
    """
    O(log(n))

    Args:
        root (Node): tree root
        key (object): sortable key

    Returns:
        puple: (tl, tr, b, v)
            tl contains all keys in the tree less than key
            tr contains all keys in the tree greater than key
            b is a flag indicating if key in root
            v is the value of the key if it existed
    """
    debug_split = 0
    if debug_split:
        print('----- SPLIT -')
        print('root = %r' % (None if root is None else root.key,))
    # TODO: keep track of the size of the sets being avl_split if possible
    if root is None:
        if debug_split:
            print('Split NULL')
        part1 = root
        part2 = root
        return (part1, part2, False, None)
    else:
        l, r = root.left, root.right
        t_key = root.key
        t_val = root.value
        if key == t_key:
            part1 = l
            part2 = r
            if debug_split:
                print('Split Case1')
                print('part1 = %r' % (None if part1 is None else part1.key,))
                print('part2 = %r' % (None if part2 is None else part2.key,))
            return (part1, part2, True, t_val)
        elif key < t_key:
            if debug_split:
                print('Split Case2')
            ll, lr, b, bv = avl_split(l, key)
            new_right = avl_join(lr, r, t_key, t_val)
            part1 = ll
            part2 = new_right
            if debug_split:
                print('part1 = %r' % (None if part1 is None else part1.key,))
                print('part2 = %r' % (None if part2 is None else part2.key,))
            return (part1, part2, b, bv)
        else:
            if debug_split:
                print('Split Case3')
            rl, rr, b, bv = avl_split(r, key)
            new_left = avl_join(l, rl, t_key, t_val)
            part1 = new_left
            part2 = rr
            if debug_split:
                print('part1 = %r' % (None if part1 is None else part1.key,))
                print('part2 = %r' % (None if part2 is None else part2.key,))
            return (part1, part2, b, bv)


def avl_join(t1, t2, key, value):
    """
    Joins two trees `t1` and `t1` with an intermediate key-value pair

    Running Time:
        O(abs(r(t1) - r(t2)))
        O(abs(height(t1) - height(t2)))
    """
    if t1 is None and t2 is None:
        return avl_new_top(None, None, key, value, 0)
    elif t1 is None:
        # FIXME keep track of count if possible
        return avl_insert_iterative(t2, key, value)
    elif t2 is None:
        return avl_insert_iterative(t1, key, value)

    h1 = height(t1)
    h2 = height(t2)
    if h1 > h2 + 1:
        top = avl_join_dir(t1, t2, key, value, 1)
    elif h2 > h1 + 1:
        top = avl_join_dir(t1, t2, key, value, 0)
    else:
        # Insert at the top of the tree
        top = avl_new_top(t1, t2, key, value, 0)
    return top


def avl_join_dir_recursive(t1, t2, key, value, direction):
    """
    Recursive version of join_left and join_right
    TODO: make this iterative using a stack
    """
    other_side = 1 - direction

    if direction == 0:
        large, small = t2, t1
    elif direction == 1:
        large, small = t1, t2

    # Follow the spine of the larger tree
    spine = large[direction]
    rest = large[other_side]
    k_, v_ = large.key, large.value

    hsmall = height(small)
    hspine = height(spine)
    hrest = height(rest)

    if hspine <= hsmall + 1:
        t_ = avl_new_top(small, spine, key, value, direction)
        if height(t_) <= hrest + 1:
            return avl_new_top(t_, rest, k_, v_, direction)
        else:
            # Double rotation, but with a new node
            t_rotate = avl_rotate_single(t_, direction)
            t_merge = avl_new_top(rest, t_rotate, k_, v_, 0)
            return avl_rotate_single(t_merge, other_side)
    else:
        # Traverse down the spine in the appropriate direction
        if direction == 0:
            t_ = avl_join_dir_recursive(small, spine, key, value, direction)
        elif direction == 1:
            t_ = avl_join_dir_recursive(spine, t2, key, value, direction)
        t__ = avl_new_top(t_, rest, k_, v_, direction)
        if height(t_) <= hrest + 1:
            return t__
        else:
            return avl_rotate_single(t__, other_side)


avl_join_dir = avl_join_dir_recursive


def height(node):
    return node.balance if node is not None else -1


def avl_rotate_single(root, direction):
    """
    Single rotation, either 0 (left) or 1 (right).

    Figure:
                a,0 (left)
                ---------->
          a                   b
           \                /   \
            b             a       c
             \
              c
    """
    other_side = 1 - direction
    save = root[other_side]
    root[other_side] = save[direction]
    save[direction] = root
    rlh = height(root.left)
    rrh = height(root.right)
    slh = height(save[other_side])
    root.balance = max(rlh, rrh) + 1
    save.balance = max(slh, root.balance) + 1
    return save


def avl_rotate_double(root, direction):
    """
    Double rotation, either 0 (left) or 1 (right).

    Figure:
                    c,1 (right)
                    ----------->
           a              a             c
          /      b,0     /     a,1    /   \
         b       --->   b      -->  b      a
          \            /
           c          c
    """
    other_side = 1 - direction
    root[other_side] = avl_rotate_single(root[other_side], other_side)
    return avl_rotate_single(root, direction)


def avl_new_top(t1, t2, key, value, direction=0):
    """
    if direction == 0:
        (t1, t2) is (left, right)
    if direction == 1:
        (t1, t2) is (right, left)
    """
    top = Node(key, value)
    top[direction] = t1
    top[1 - direction] = t2
    top.balance = max(height(t1), height(t2)) + 1
    return top


def avl_insert_iterative(root, key, value):
    node_stack = []  # node stack
    dir_stack = array('I')  # direction stack
    done = False
    top = 0
    node = root
    # search for an empty link, save path
    while True:
        if key == node.key:  # update existing item
            node.value = value
            return
        direction = 1 if key > node.key else 0
        dir_stack.append(direction)
        node_stack.append(node)
        if node[direction] is None:
            break
        node = node[direction]

    # Insert a new node at the bottom of the tree
    node[direction] = Node(key, value)

    # Walk back up the search path
    top = len(node_stack) - 1
    while (top >= 0) and not done:
        direction = dir_stack[top]
        other_side = 1 - direction
        top_node = node_stack[top]
        left_height = height(top_node[direction])
        right_height = height(top_node[other_side])

        # Terminate or rebalance as necessary
        if left_height - right_height == 0:
            done = True
        if left_height - right_height >= 2:
            a = top_node[direction][direction]
            b = top_node[direction][other_side]

            # Determine which rotation is required
            if height(a) >= height(b):
                node_stack[top] = avl_rotate_single(top_node, other_side)
            else:
                node_stack[top] = avl_rotate_double(top_node, other_side)

            # Fix parent
            if top != 0:
                node_stack[top - 1][dir_stack[top - 1]] = node_stack[top]
            else:
                root = node_stack[0]
            done = True

        # Update balance factors
        top_node = node_stack[top]
        left_height = height(top_node[direction])
        right_height = height(top_node[other_side])

        top_node.balance = max(left_height, right_height) + 1
        top -= 1
    return root


class AVLTree(ABCTree):
    """
    AVLTree implements a balanced binary tree with a dict-like interface.

    see: http://en.wikipedia.org/wiki/AVL_tree

    In computer science, an AVL tree is a self-balancing binary search tree, and
    it is the first such data structure to be invented. In an AVL tree, the
    heights of the two child subtrees of any node differ by at most one;
    therefore, it is also said to be height-balanced. Lookup, insertion, and
    deletion all take O(log n) time in both the average and worst cases, where n
    is the number of nodes in the tree prior to the operation. Insertions and
    deletions may require the tree to be rebalanced by one or more tree rotations.

    The AVL tree is named after its two inventors, G.M. Adelson-Velskii and E.M.
    Landis, who published it in their 1962 paper "An algorithm for the
    organization of information."

    AVLTree() -> new empty tree.
    AVLTree(mapping) -> new tree initialized from a mapping
    AVLTree(seq) -> new tree initialized from seq [(k1, v1), (k2, v2), ... (kn, vn)]

    see also abctree.ABCTree() class.
    """
    def _new_node(self, key, value):
        """Create a new tree node."""
        self._count += 1
        return Node(key, value)

    def insert(self, key, value):
        """T.insert(key, value) <==> T[key] = value, insert key, value into tree."""
        if self._root is None:
            self._root = self._new_node(key, value)
        else:
            node_stack = []  # node stack
            dir_stack = array('I')  # direction stack
            done = False
            top = 0
            node = self._root
            # search for an empty link, save path
            while True:
                if key == node.key:  # update existing item
                    node.value = value
                    return
                direction = 1 if key > node.key else 0
                dir_stack.append(direction)
                node_stack.append(node)
                if node[direction] is None:
                    break
                node = node[direction]

            # Insert a new node at the bottom of the tree
            node[direction] = self._new_node(key, value)

            # Walk back up the search path
            top = len(node_stack) - 1
            while (top >= 0) and not done:
                direction = dir_stack[top]
                other_side = 1 - direction
                top_node = node_stack[top]
                left_height = height(top_node[direction])
                right_height = height(top_node[other_side])

                # Terminate or rebalance as necessary
                if left_height - right_height == 0:
                    done = True
                if left_height - right_height >= 2:
                    a = top_node[direction][direction]
                    b = top_node[direction][other_side]

                    # Determine which rotation is required
                    if height(a) >= height(b):
                        node_stack[top] = avl_rotate_single(top_node, other_side)
                    else:
                        node_stack[top] = avl_rotate_double(top_node, other_side)

                    # Fix parent
                    if top != 0:
                        node_stack[top - 1][dir_stack[top - 1]] = node_stack[top]
                    else:
                        self._root = node_stack[0]
                    done = True

                # Update balance factors
                top_node = node_stack[top]
                left_height = height(top_node[direction])
                right_height = height(top_node[other_side])

                top_node.balance = max(left_height, right_height) + 1
                top -= 1

    def remove(self, key):
        """T.remove(key) <==> del T[key], remove item <key> from tree."""
        if self._root is None:
            raise KeyError(str(key))
        else:
            node_stack = [None] * MAXSTACK  # node stack
            dir_stack = array('I', [0] * MAXSTACK)  # direction stack
            top = 0
            node = self._root

            while True:
                # Terminate if not found
                if node is None:
                    raise KeyError(str(key))
                elif node.key == key:
                    break

                # Push direction and node onto stack
                direction = 1 if key > node.key else 0
                dir_stack[top] = direction

                node_stack[top] = node
                node = node[direction]
                top += 1

            # Remove the node
            if (node.left is None) or (node.right is None):
                # Which child is not null?
                direction = 1 if node.left is None else 0

                # Fix parent
                if top != 0:
                    node_stack[top - 1][dir_stack[top - 1]] = node[direction]
                else:
                    self._root = node[direction]
                node.free()
                self._count -= 1
            else:
                # Find the inorder successor
                heir = node.right

                # Save the path
                dir_stack[top] = 1
                node_stack[top] = node
                top += 1

                while heir.left is not None:
                    dir_stack[top] = 0
                    node_stack[top] = heir
                    top += 1
                    heir = heir.left

                # Swap data
                node.key = heir.key
                node.value = heir.value

                # Unlink successor and fix parent
                xdir = 1 if node_stack[top - 1].key == node.key else 0
                node_stack[top - 1][xdir] = heir.right
                heir.free()
                self._count -= 1

            # Walk back up the search path
            top -= 1
            while top >= 0:
                direction = dir_stack[top]
                other_side = 1 - direction
                top_node = node_stack[top]
                left_height = height(top_node[direction])
                right_height = height(top_node[other_side])
                b_max = max(left_height, right_height)

                # Update balance factors
                top_node.balance = b_max + 1

                # Terminate or rebalance as necessary
                if (left_height - right_height) == -1:
                    break
                if (left_height - right_height) <= -2:
                    a = top_node[other_side][direction]
                    b = top_node[other_side][other_side]
                    # Determine which rotation is required
                    if height(a) <= height(b):
                        node_stack[top] = avl_rotate_single(top_node,
                                                            direction)
                    else:
                        node_stack[top] = avl_rotate_double(top_node,
                                                            direction)
                    # Fix parent
                    if top != 0:
                        node_stack[top - 1][dir_stack[top - 1]] = node_stack[top]
                    else:
                        self._root = node_stack[0]
                top -= 1

    def join_inplace(self, other, key, value):
        """
        Returns all elements from t1 and t2 as well as (key, val)

        Running time:
            O(|height(t1) − height(t2)|)

        References:
            Yihan Sun. Just Join for Parallel Ordered Sets. SPAA 2016
            https://www.cs.cmu.edu/~guyb/papers/BFS16.pdf

            S. Adams. Implementing sets effciently in a functional
            language. Technical Report CSTR 92-10, University of
            Southampton, 1992
            http://groups.csail.mit.edu/mac/users/adams/BB/92-10.ps


        CommandLine:
            python -m bintrees.avltree AVLTree.join_inplace --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from bintrees.avltree import *  # NOQA
            >>> import utool as ut
            >>> import numpy as np
            >>> keys1 = np.arange(10)
            >>> keys2 = np.arange(20) + 100
            >>> self  = AVLTree(list(zip(keys1, keys1)))
            >>> other = AVLTree(list(zip(keys2, keys2)))
            >>> key = value = int(keys1.max() + keys2.min()) // 2
            >>> new = self.copy().join_inplace(other.copy(), key, value)
            >>> import plottool as pt
            >>> pt.qt4ensure()
            >>> g1 = self.to_networkx(['key', 'balance'])
            >>> g2 = other.to_networkx(['key', 'balance'])
            >>> g_new = new.to_networkx(['key', 'balance'])
            >>> pt.show_nx(g1, fnum=1, pnum=(2, 2, (0, 0)))
            >>> pt.show_nx(g2, fnum=1, pnum=(2, 2, (1, 0)))
            >>> pt.show_nx(g_new, fnum=1, pnum=(2, 2, (slice(0, 2), 1)))
            >>> ut.show_if_requested()
        """
        minceil_key = -float('inf') if self.is_empty() else self.max_key()
        maxfloor_key = float('inf') if other.is_empty() else other.min_key()
        if not (minceil_key < key and key < maxfloor_key):
            raise ValueError('invalid join_inplace args %r < %r < %r' % (
                minceil_key, key, maxfloor_key))
        t1 = self._root
        t2 = other._root
        new_count = self._count + other._count + 1
        top = avl_join(t1, t2, key, value)
        # Two trees are now joined inplace
        self._root = top
        self._count = new_count
        # self assimilates all values from other
        other._root = None
        other._count = 0
        return self

    def splice_inplace(self, start_key, stop_key):
        """
        Extracts a ordered slice from `root` and returns the inside and outside
        parts. A new inside tree is returned. This tree becomes the outside.

        Args:
            start_key (object): sortable low value inclusive
            stop_key (object): sortable high value exclusive

        Returns:
            inner, outer

        CommandLine:
            python -m bintrees.avltree splice_inplace --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from bintrees.avltree import *  # NOQA
            >>> import bintrees
            >>> import utool as ut
            >>> import numpy as np
            >>> keys = np.arange(10)
            >>> self = bintrees.FastAVLTree(list(zip(keys, keys)))
            >>> self_copy = bintrees.FastAVLTree(list(zip(keys, keys)))
            >>> # self = bintrees.AVLTree(list(zip(keys, keys)))
            >>> # self_copy = self.copy()
            >>> start_key, stop_key = 3, 7
            >>> inner, outer = self_copy.splice_inplace(start_key, stop_key)
            >>> import plottool as pt
            >>> pt.qt4ensure()
            >>> g1 = self.to_networkx(['key', 'balance'])
            >>> g_in = inner.to_networkx(['key', 'balance'])
            >>> g_out = outer.to_networkx(['key', 'balance'])
            >>> pt.show_nx(g1, fnum=1, pnum=(2, 2, (slice(0, 2), 0)))
            >>> pt.show_nx(g_in, fnum=1, pnum=(2, 2, (0, 1)))
            >>> pt.show_nx(g_out, fnum=1, pnum=(2, 2, (1, 1)))
            >>> ut.show_if_requested()
        """
        t_inner, t_outer = avl_splice(self._root, start_key, stop_key)

        inner = AVLTree()
        inner._root = t_inner
        inner._count = 0 if t_inner is None else -1  # FIXME

        outer = self
        outer._root = t_outer
        outer._count = 0 if t_outer is None else -1  # FIXME
        return inner, outer

    def split_last_inplace(self):
        t1, k, v = avl_split_last(self._root)
        self._root = t1
        self._count -= 1
        return (k, v)

    def split_inplace(self, key):
        """
        Inplace split of `self` into two trees. Returns flag indicating
        This tree is destroyed.

        Running time:
            O(log(n))

        Returns:
            tuple (tl, tr, flag, val)
        """
        t1, t2, b, bv = avl_split(self._root, key)
        # print('t1 = %r' % (t1,))
        # print('t2 = %r' % (t2,))
        tree1 = AVLTree()
        tree1._root = t1
        tree1._count = 0 if t1 is None else -1  # FIXME

        tree2 = AVLTree()
        tree2._root = t2
        tree2._count = 0 if t2 is None else -1  # FIXME

        # We split our nodes into two other trees. Destroy ourself
        self._root = None
        self._count = 0
        return tree1, tree2, b, bv

    def union_inplace(self, other):
        """
        Inplace union of two trees.

        Args:
            self (AVLTree): tree of size n - union will happen inplace
            other (AVLTree): tree of size m - this tree is destroyed

        Running time:
            O(m log((n / m) + 1))
        """
        t1 = self._root
        t2 = other._root
        top = avl_union(t1, t2)
        # Two trees are now unioned inplace
        self._root = top
        other._root = None
        self._count = self._count
        other._count = 0
        return self


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m bintrees.avltree
        python -m bintrees.avltree --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
