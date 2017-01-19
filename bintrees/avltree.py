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
        return self.balance

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


def test_join_cases():
    """
    CommandLine:
        python -m bintrees.avltree test_join_cases
    """
    import numpy as np
    import utool as ut
    import vtool as vt
    # lowhigh_cases = [
    #     [1, 3],
    #     [1, 10],
    #     [1, 9],
    #     [3, 9],
    #     [2, 3],
    #     [3, 32],
    # ]
    import bintrees
    # _AVLTree_cls = bintrees.AVLTree
    _AVLTree_cls = bintrees.FastAVLTree

    # Exhaustively test that merging trees works
    lowhigh_cases = []
    n = 4
    for x in range(0, 2 ** n):
        for y in range(0, 2 ** n):
            lowhigh_cases += [[x, y]]

    offset = max(map(max, lowhigh_cases)) * 2
    directions = [0, 1]

    test_cases = list(ut.product(directions, lowhigh_cases))
    for direction, lowhigh in ut.ProgIter(test_cases, label='test avl_join'):
        keys1 = np.arange(lowhigh[direction])
        keys2 = np.arange(lowhigh[1 - direction]) + offset

        mx1 = vt.safe_max(keys1, fill=0)
        mn2 = vt.safe_min(keys2, fill=mx1 + 2)
        key = value = int(mx1 + mn2) // 2

        self  = _AVLTree_cls(list(zip(keys1, keys1)))
        other = _AVLTree_cls(list(zip(keys2, keys2)))
        # new = avl_tree_join(self, other, key, value)
        new = self.join(other, key, value)
        assert_avl_invariants(new)

        assert len(new) == len(keys1) + len(keys2) + 1
        assert self is new
        assert other._root is None


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

    self.join(other, key, value)


def test_split_cases():
    """
    CommandLine:
        python -m bintrees.avltree test_split_cases
    """
    # import utool as ut
    import numpy as np

    keys = list(range(0, 100, 2))
    for key in range(-2, max(keys) + 2):
        self = AVLTree(list(zip(keys, keys)))
        tree1, tree2, b = avl_split_tree(self, key)
        try:
            assert b == (key in keys)
            keys1 = np.array(list(tree1.keys()))
            keys2 = np.array(list(tree2.keys()))
            assert np.all(keys1 < key)
            assert np.all(keys2 > key)
            print('----')
            print('key = %r' % (key,))
            print('keys1 = %r' % (keys1,))
            print('keys2 = %r' % (keys2,))
            print('b = %r' % (b,))
        except AssertionError:
            print('key = %r' % (key,))
            print('keys1 = %r' % (keys1,))
            print('keys2 = %r' % (keys2,))
            print('b = %r' % (b,))
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
                new1 = self1.join(other1, key, value)

        for timer in ut.Timerit(n, 'new Cython avl_join', verbose=verbose):
            self3  = bintrees.FastAVLTree(list(zip(keys1, keys1)))
            other3 = bintrees.FastAVLTree(list(zip(keys2, keys2)))
            with timer:
                new3 = self3.join(other3, key, value)
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
                new1 = avl_tree_union(self1, other1)

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


def to_networkx(self, labels=['key', 'value']):
    import networkx as nx
    graph = nx.Graph()
    graph.add_node(0)
    queue = [[self._root, 0]]
    index = 0
    while queue:
        node = queue[0][0]  # Select front of queue.
        node_index = queue[0][1]
        label = ','.join([str(getattr(node, k)) for k in labels])
        graph.node[node_index]['label'] = label
        if node.left is not None:
            graph.add_node(node_index)
            graph.add_edges_from([(node_index, index + 1)])
            queue.append([node.left, index + 1])
            index += 1
        if node.right is not None:
            graph.add_node(node_index)
            graph.add_edges_from([(node_index, index + 1)])
            queue.append([node.right, index + 1])
            index += 1
        queue.pop(0)
    return graph


def avl_tree_join(self, other, key, value):
    """
    Returns all elements from t1 and t2 as well as (key, val)
    assert that t1.max() < key < t2.min()

    Running time:
        O(|height(t1) − height(t2)|)

    References:
        Just Join for Parallel Ordered Sets
        https://dx.doi.org/10.1145%2F2935764.2935768
        https://i.cs.hku.hk/~provinci/training2016/notes2.pdf

    CommandLine:
        python -m bintrees.avltree avl_tree_join --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from bintrees.avltree import *  # NOQA
        >>> import utool as ut
        >>> import numpy as np
        >>> left = 1
        >>> lowhigh = [1, 3]
        >>> keys1 = np.arange(lowhigh[1 - left])
        >>> keys2 = np.arange(lowhigh[left]) + 100
        >>> self  = AVLTree(list(zip(keys1, keys1)))
        >>> other = AVLTree(list(zip(keys2, keys2)))
        >>> key = value = int(keys1.max() + keys2.min()) // 2
        >>> t1 = self._root
        >>> t2 = other._root
        >>> new = avl_tree_join(self.copy(), other.copy(), key, value)
        >>> import plottool as pt
        >>> pt.qt4ensure()
        >>> #graph = to_networkx(self, ['key', 'balance'])
        >>> #pt.show_nx(graph, fnum=1, pnum=None)
        >>> labels = ['key', 'balance']
        >>> _ = pt.show_nx(to_networkx(self, labels), fnum=1, pnum=(2, 2, (0, 0)))
        >>> _ = pt.show_nx(to_networkx(other, labels), fnum=1, pnum=(2, 2, (1, 0)))
        >>> _ = pt.show_nx(to_networkx(new, labels), fnum=1, pnum=(2, 2, (slice(0, 2), 1)))
        >>> ut.show_if_requested()
    """
    minceil_key = -float('inf') if self.is_empty() else self.max_key()
    maxfloor_key = float('inf') if other.is_empty() else other.min_key()
    if not (minceil_key < key and key < maxfloor_key):
        raise ValueError('invalid avl_tree_join args %r < %r < %r' % (
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


def avl_split_tree(self, key):
    t = self._root
    t1, b, t2 = avl_split(t, key)
    # print('t1 = %r' % (t1,))
    # print('t2 = %r' % (t2,))
    tree1 = AVLTree()
    tree1._root = t1
    tree1._count = 0 if t1 is None else -1  # FIXME

    tree2 = AVLTree()
    tree2._root = t2
    tree2._count = 0 if t2 is None else -1  # FIXME
    return tree1, tree2, b


def avl_tree_union(self, other):
    """
    Inplace avl_union
    """
    t1 = self._root
    t2 = other._root
    top = avl_union(t1, t2)
    # Two trees are now unioned inplace
    self._root = other._root = top
    self._count = other._count = self._count + other._count
    return self


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


def avl_union(t1, t2):
    """
    Should take O(m log(n/m + 1)). Instead of O(m log(n))
    This is sublinear (good)
    """
    if t1 is None:
        return t2
    elif t2 is None:
        return t1
    else:
        l2, r2 = t2.left, t2.right
        k2, v2 = t2.key, t2.value
        l1, b, r1 = avl_split(t1, k2)
        tl = avl_union(l1, l2)
        tr = avl_union(r1, r2)
        return avl_join(tl, tr, k2, v2)


def avl_split(t, key):
    """
    Args:
        t (Node): tree node
        key (object): sortable key

    Returns:
        tuple: (tl, b, tr)
            tl contains all keys in t that are less than key
            b is a flag indicating if key in t
            tr contains all keys in t that are greater than key
    """
    # TODO: keep track of the size of the sets being avl_split if possible
    if t is None:
        return (t, False, t)
    else:
        l, r = t.left, t.right
        t_key = t.key
        t_val = t.value
        if key == t_key:
            return (l, True, r)
        elif key < t_key:
            ll, b, lr = avl_split(l, key)
            return (ll, b, avl_join(lr, r, t_key, t_val))
        else:
            rl, b, rr = avl_split(r, key)
            return (avl_join(l, rl, t_key, t_val), b, rr)


def avl_insert(root, key, value):
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


def avl_join(t1, t2, key, value):
    """ Executes a avl_join on avl nodes """
    if t1 is None and t2 is None:
        return avl_new_top(None, None, key, value, 0)
    elif t1 is None:
        # FIXME keep track of count if possible
        return avl_insert(t2, key, value)
    elif t2 is None:
        return avl_insert(t1, key, value)

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

    def join(self, other, key, value):
        return avl_tree_join(self, other, key, value)


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
