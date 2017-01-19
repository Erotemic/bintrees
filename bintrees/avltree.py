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
import utool as ut
print, rrr, profile = ut.inject2(__name__)

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


def height(node):
    return node.balance if node is not None else -1


def rotate_single(root, direction):
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


def rotate_double(root, direction):
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
    root[other_side] = rotate_single(root[other_side], other_side)
    return rotate_single(root, direction)


def expose(t):
    return t.left, t.key, t.value, t.right


def is_leaf(t):
    return t.left is None and t.right is None


def union(t1, t2):
    """
    Should take O(m log(n/m + 1)). Instead of O(m log(n))
    This is sublinear (good)
    """
    if is_leaf(t1):
        return t2
    elif is_leaf(t2):
        return t1
    else:
        l2, k2, v2, r2 = expose(t2)
        l1, b, r1 = split(t1, k2, v2)
        tl = union(l1, l2)
        tr = union(r1, r2)
        return join(tl, k2, tr)


def split(t, k):
    if is_leaf(t):
        return t, False, t
    else:
        L, m, R = expose(t)


def split_last(t):
    l, k, v, r = expose(t)
    if is_leaf(r):
        (l, k)
    else:
        t_, k_ = split_last(r)
        (join(l, k, t_), k_)


def join2(t1, t2):
    if t1.left is None and t2.right is None:
        t2
    else:
        t1_, k = split_last(t1)
        join(t1_, k, t2)


def new_top(left, right, key, value, direction=0):
    top = Node(key, value)
    top[direction] = left
    top[1 - direction] = right
    top.balance = max(height(left), height(right)) + 1
    return top


def traverse_avl_nodes(root):
    stack = []
    level = ut.ddict(list)
    node = root
    yielder = []
    while stack or node is not None:
        if node is not None:
            stack.append(node)
            node = node.left
        else:
            node = stack.pop()
            level[len(stack)].append(node)
            yielder += [node]
            # yield node
            node = node.right
    return yielder


def ascii_tree(root):
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
    # return
    # return yielder


def join_right(t1, t2, key, value):
    print('Join Right')
    print('Input: t1')
    ascii_tree(t1)
    print('Input: t2')
    ascii_tree(t2)
    # t1 is large
    # t2 is small
    l, c = t1.left, t1.right
    k_, v_ = t1.key, t1.value

    if height(c) <= height(t2) + 1:
        t_ = new_top(c, t2, key, value)
        if height(t_) <= height(l) + 1:
            ut.cprint('Case1', 'red')
            retvar = new_top(l, t_, k_, v_)
        else:
            ut.cprint('Case2', 'red')
            t_rotate = rotate_single(t_, 1)
            tmp = new_top(l, t_rotate, k_, v_)
            out = rotate_single(tmp, 0)
            retvar = out
    else:
        t_ = join_right(c, t2, key, value)
        t__ = new_top(l, t_, k_, v_)
        if height(t_) <= height(l) + 1:
            ut.cprint('Case3', 'red')
            retvar  = t__
        else:
            ut.cprint('Case4', 'red')
            retvar = rotate_single(t__, 0)
    return retvar


def join_left(t1, t2, key, value):
    print('Join Left')
    print('Input: t1')
    ascii_tree(t1)
    print('Input: t2')
    ascii_tree(t2)
    # t1 is small
    # t2 is large
    c, l = t2.left, t2.right
    k_, v_ = t2.key, t2.value

    if height(c) <= height(t1) + 1:
        t_ = new_top(t1, c, key, value)
        if height(t_) <= height(l) + 1:
            ut.cprint('Case1', 'red')
            retvar = new_top(t_, l, k_, v_)
        else:
            ut.cprint('Case2', 'red')
            t_rotate = rotate_single(t_, 0)
            tmp = new_top(l, t_rotate, k_, v_)
            out = rotate_single(tmp, 1)
            retvar = out
    else:
        t_ = join_left(t1, c, key, value)
        ut.cprint('CaseR', 'red')
        t__ = new_top(t_, l, k_, v_)
        if height(t_) <= height(l) + 1:
            ut.cprint('Case3', 'red')
            retvar  = t__
        else:
            ut.cprint('Case4', 'red')
            retvar = rotate_single(t__, 1)
    return retvar


DEBUG = 2


def join_dir(t1, t2, key, value, direction):
    debug = DEBUG
    if debug > 1:
        print('Join')
        print('Input: t1')
        ascii_tree(t1)
        print('Input: t2')
        ascii_tree(t2)

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
        t_ = new_top(small, spine, key, value, direction)
        if height(t_) <= hrest + 1:
            if debug:
                ut.cprint('Case1', 'red')
            retvar = new_top(t_, rest, k_, v_, direction)
        else:
            if debug:
                ut.cprint('Case2', 'red')
            # Double rotation, but with a new node
            t_rotate = rotate_single(t_, direction)
            tmp = new_top(rest, t_rotate, k_, v_, 0)
            retvar = rotate_single(tmp, other_side)
    else:
        if debug:
            ut.cprint('CaseR', 'red')
        # Traverse down the spine in the appropriate direction
        if direction == 0:
            t_ = join_dir(small, spine, key, value, direction)
        elif direction == 1:
            t_ = join_dir(spine, t2, key, value, direction)
        t__ = new_top(t_, rest, k_, v_, direction)
        if height(t_) <= hrest + 1:
            if debug:
                ut.cprint('Case3', 'red')
            retvar = t__
        else:
            if debug:
                ut.cprint('Case4', 'red')
            print('Pre-4')
            ascii_tree(t__)
            retvar = rotate_single(t__, other_side)
    if debug > 1:
        print('++++')
        print('Retvar:')
        ascii_tree(retvar)
    return retvar


def assert_avl_invariants(tree):
    for node in traverse_avl_nodes(tree._root):
        h1 = height(node.left)
        h2 = height(node.right)
        balance_factor = h1 - h2
        if abs(balance_factor) > 1:
            print('ERROR')
            ascii_tree(tree._root)
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
    lowhigh_cases = [
        [1, 3],
        [1, 10],
        # [1, 9],
        # [3, 9],
        # [2, 3],
        # [3, 32],
    ]
    # n = 3
    # for x in [1, 2]:
    #     for y in range(2 ** n):
    #         lowhigh_cases += [[x, 2 ** n + y]]

    test_cases = ut.product([0, 1], lowhigh_cases)
    for direction, lowhigh in test_cases:
        keys1 = np.arange(lowhigh[direction])
        keys2 = np.arange(lowhigh[1 - direction]) + 100
        key = value = int(keys1.max() + keys2.min()) // 2
        self  = AVLTree(list(zip(keys1, keys1)))
        other = AVLTree(list(zip(keys2, keys2)))
        debug = DEBUG
        if debug:
            ut.cprint('==========', 'yellow')
            if debug > 1:
                print('direction = %r' % (direction,))
                print('lowhigh = %r' % (lowhigh,))
                print('key = %r' % (key,))
        new = join(self, other, key, value)
        assert_avl_invariants(new)


def join(self, other, key, value):
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
        python -m bintrees.avltree join

    Example:
        >>> # DISABLE_DOCTEST
        >>> from bintrees.avltree import *  # NOQA
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
        >>> new = join(self.copy(), other.copy(), key, value)
        >>> import plottool as pt
        >>> pt.qt4ensure()
        >>> #graph = to_networkx(self, ['key', 'balance'])
        >>> #pt.show_nx(graph, fnum=1, pnum=None)
        >>> labels = ['key', 'balance']
        >>> _ = pt.show_nx(to_networkx(self, labels), fnum=1, pnum=(2, 2, (0, 0)))
        >>> _ = pt.show_nx(to_networkx(other, labels), fnum=1, pnum=(2, 2, (1, 0)))
        >>> _ = pt.show_nx(to_networkx(new, labels), fnum=1, pnum=(2, 2, (slice(0, 2), 1)))
    """
    assert self.max_key() < key and key < other.min_key()
    t1 = self._root
    t2 = other._root
    h1 = height(t1)
    h2 = height(t2)
    debug = DEBUG
    if h1 > h2 + 1:
        if debug:
            ut.cprint('right_joincase', 'green')
        top = join_dir(t1, t2, key, value, 1)
    elif h2 > h1 + 1:
        if debug:
            ut.cprint('left_joincase', 'green')
        top = join_dir(t1, t2, key, value, 0)
    else:
        if debug:
            ut.cprint('mid_joincase', 'green')
        # Insert at the top of the tree
        top = new_top(t1, t2, key, value)
    # print('New Top')
    # ascii_tree(top)
    # Two trees are now joined inplace
    self._root = other._root = top
    self._count = other._count = self._count + other._count + 1
    return self


def to_networkx(self, labels=['key', 'value']):
    import networkx as nx
    # import igraph as igraphs
    graph = nx.Graph()
    graph.add_node(0)
    queue = [[self._root, 0]]
    index = 0
    while queue:
        node = queue[0][0]  # Select front of queue.
        node_index = queue[0][1]
        # graph.node[node_index]['key'] = node.key
        # graph.node[node_index]['value'] = node.value
        # graph.node[node_index]['balance'] = node.balance
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


def show_avl_tree(tree, fnum=None, pnum=None):
    """
    >>> show_avl_tree(tree, pnum=(2, 1, 2), fnum=1)
    >>> pt.show_nx(mst, pnum=(2, 1, 1), fnum=1)

    """
    import plottool as pt
    pt.qt4ensure()
    pt.show_nx(graph, fnum=fnum, pnum=pnum)




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
                        node_stack[top] = rotate_single(top_node, other_side)
                    else:
                        node_stack[top] = rotate_double(top_node, other_side)

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
                        node_stack[top] = rotate_single(top_node, direction)
                    else:
                        node_stack[top] = rotate_double(top_node, direction)
                    # Fix parent
                    if top != 0:
                        node_stack[top - 1][dir_stack[top - 1]] = node_stack[top]
                    else:
                        self._root = node_stack[0]
                top -= 1


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
