#!/usr/bin/env python
#coding:utf-8
# Author:  mozman
# Purpose: Binary trees implemented in Cython/C
# Created: 28.04.2010
# Copyright (c) 2010-2013 by Manfred Moitzi
# License: MIT License

from .abctree import _ABCTree
from ctrees cimport *

DEF MAXSTACK = 64

cdef class NodeStack:
    """Simple stack for tree nodes."""
    cdef node_t* stack[MAXSTACK]
    cdef int stackptr

    def __cinit__(self):
        self.stackptr = 0

    cdef push(self, node_t* node):
        if self.stackptr >= MAXSTACK:
            raise RuntimeError("Stack overflow in NodeStack.push().")
        self.stack[self.stackptr] = node
        self.stackptr += 1

    cdef node_t* pop(self):
        if self.stackptr <= 0:
            raise RuntimeError("Stack underflow in NodeStack.pop().")
        self.stackptr -= 1
        return self.stack[self.stackptr]

    cdef bint is_empty(self):
        return self.stackptr == 0

cdef class _BaseTree:
    cdef node_t *root  # private (hidden) for CPython
    cdef readonly int count  # public readonly access for CPython

    def __cinit__(self, items=None):
        self.root = NULL
        self.count = 0

    def __init__(self, items=None):
        if items is not None:
            self.update(items)

    def __dealloc__(self):
        ct_delete_tree(self.root)

    def __getstate__(self):
        return dict(self.items())

    def __setstate__(self, state):
        self.update(state)

    def clear(self):
        ct_delete_tree(self.root)
        self.count = 0
        self.root = NULL

    def get_value(self, key):
        cdef node_t *result = ct_find_node(self.root, key)
        if result == NULL:
            raise KeyError(key)
        else:
            return <object> result.value

    def max_item(self):
        """Get item with max key of tree, raises ValueError if tree is empty."""
        cdef node_t *node = ct_max_node(self.root)
        if node == NULL:
            raise ValueError("Tree is empty")
        return <object>node.key, <object>node.value

    def min_item(self):
        """Get item with min key of tree, raises ValueError if tree is empty."""
        cdef node_t *node = ct_min_node(self.root)
        if node == NULL:
            raise ValueError("Tree is empty")
        return <object>node.key, <object>node.value

    def succ_item(self, key):
        """Get successor (k,v) pair of key, raises KeyError if key is max key
        or key does not exist.
        """
        cdef node_t *node = ct_succ_node(self.root, key)
        if node == NULL: # given key is biggest in tree
            raise KeyError(str(key))
        return <object> node.key, <object> node.value

    def prev_item(self, key):
        """Get predecessor (k,v) pair of key, raises KeyError if key is min key
        or key does not exist.
        """
        cdef node_t *node = ct_prev_node(self.root, key)
        if node == NULL: # given key is smallest in tree
            raise KeyError(str(key))
        return <object> node.key, <object> node.value

    def floor_item(self, key):
        """Get (k,v) pair associated with the greatest key less than or equal to
        the given key, raises KeyError if there is no such key.
        """
        cdef node_t *node = ct_floor_node(self.root, key)
        if node == NULL:  # given key is smaller than min-key in tree
            raise KeyError(str(key))
        return <object> node.key, <object> node.value

    def ceiling_item(self, key):
        """Get (k,v) pair associated with the smallest key greater than or equal to
        the given key, raises KeyError if there is no such key.
        """
        cdef node_t *node = ct_ceiling_node(self.root, key)
        if node == NULL:  # given key is greater than max-key in tree
            raise KeyError(str(key))
        return <object> node.key, <object> node.value

    def iter_items(self, start_key=None, end_key=None, reverse=False):
        """Iterate over the (key, value) items in ascending order
        if reverse is True iterate in descending order.
        """
        if self.count == 0:
            return
        cdef int direction = 1 if reverse else 0
        cdef int other = 1 - direction
        cdef bint go_down = True
        cdef NodeStack stack = NodeStack()
        cdef node_t *node

        node = self.root
        while True:
            if node.link[direction] != NULL and go_down:
                stack.push(node)
                node = node.link[direction]
            else:
                if (start_key is None or ct_compare(start_key, <object>node.key) < 1) and \
                    (end_key is None or ct_compare(end_key, <object>node.key) > 0):
                    yield <object>node.key, <object>node.value
                if node.link[other] != NULL:
                    node = node.link[other]
                    go_down = True
                else:
                    if stack.is_empty():
                        return  # all done
                    node = stack.pop()
                    go_down = False

    def pop_item(self):
        """ T.pop_item() -> (k, v), remove and return some (key, value) pair as a
        2-tuple; but raise KeyError if T is empty.
        """
        if self.count == 0:
            raise KeyError("pop_item(): tree is empty")

        cdef node_t *node = ct_get_leaf_node(self.root)
        key = <object> node.key
        value = <object> node.value
        self.remove(key)
        return key, value
    popitem = pop_item  # for compatibility to dict()

    def foreach(self, func, int order=0):
        """Visit all tree nodes and process tree data by func(key, Value).

        parm func: function(key, value)
        param int order: inorder = 0, preorder = -1, postorder = +1
        """
        if self.count == 0:
            return
        cdef NodeStack stack = NodeStack()
        cdef NodeStack tempstack = NodeStack()
        cdef node_t *node = self.root

        if order == 0:
            while not stack.is_empty() or node:
                if node:
                    stack.push(node)
                    node = node.link[0]
                else:
                    node = stack.pop()
                    func(<object>node.key, <object>node.value)
                    node = node.link[1]
        elif order == -1:
            stack.push(node)
            while not stack.is_empty():
                node = stack.pop()
                func(<object>node.key, <object>node.value)
                if node.link[1]:
                    stack.push(node.link[1])
                if node.link[0]:
                    stack.push(node.link[0])
        elif order == +1:
            tempstack.push(node)
            while not tempstack.is_empty():
                node = tempstack.pop()
                stack.push(node)
                if node.link[0]:
                    tempstack.push(node.link[0])
                if node.link[1]:
                    tempstack.push(node.link[1])
            while not stack.is_empty():
                node = stack.pop()
                func(<object>node.key, <object>node.value)

    @property
    def _root(self):
        return _create_dummy_node(<node_t*> self.root)

    def recount(self):
        """ O(n) recomputation of count (in case a split destroys it) """
        self.count = sum(1 for _ in self._traverse_nodes())


class DummyNode(object):
    def __init__(self):
        self.key = None
        self.value = None
        self.xdata = None
        self.left = None
        self.right = None

    @property
    def balance(self):
        return self.xdata

    @balance.setter
    def balance(self, value):
        self.xdata = value




cdef _create_dummy_node(node_t *root):
    if root == NULL:
        return None
    else:
        node = DummyNode()
        node.key = <object>root.key
        node.value = <object>root.value
        node.xdata = <int>root.xdata
        node.left = _create_dummy_node(root.link[0])
        node.right = _create_dummy_node(root.link[1])
        return node


cdef class _BinaryTree(_BaseTree):
    def insert(self, key, value):
        cdef int result = ct_bintree_insert(&self.root, key, value)
        if result < 0:
            raise MemoryError('Can not allocate memory for node structure.')
        self.count += result

    def remove(self, key):
        cdef int result
        result = ct_bintree_remove(&self.root, key)
        if result == 0:
            raise KeyError(str(key))
        else:
            self.count -= 1


class FastBinaryTree(_BinaryTree, _ABCTree):
    pass


cdef class _AVLTree(_BaseTree):
    def insert(self, key, value):
        cdef int result = avl_insert(&self.root, key, value)
        if result < 0:
            raise MemoryError('Can not allocate memory for node structure.')
        else:
            self.count += result

    def remove(self, key):
        cdef int result = avl_remove(&self.root, key)
        if result == 0:
            raise KeyError(str(key))
        else:
            self.count -= 1

    def join_inplace(self, other, key, value):
        minceil_key = -float('inf') if self.is_empty() else self.max_key()
        maxfloor_key = float('inf') if other.is_empty() else other.min_key()
        if not (minceil_key < key and key < maxfloor_key):
            raise ValueError('invalid avl_join_inplace args %r < %r < %r' % (
                minceil_key, key, maxfloor_key))
        # Join self and other inplace
        avl_join_inplace(&self.root, &(<_AVLTree> other).root, key, value)
        # All elements are assimilated into self
        new_count = self.count + other.count + 1
        self.count = new_count
        # Other is now empty
        (<_AVLTree> other).count = 0
        return self

    def split_inplace(self, key):
        cdef int flag
        cdef node_t *t_right;
        value_ptr = avl_split_inplace(&self.root, key, &flag, &t_right)
        cdef node_t *t_left = <node_t*> self.root

        value = <object> value_ptr
        # self becomes the left part
        # _AVLTree *outer = self
        left = self
        self.count = -1 if t_left != NULL else 0  # FIXME

        # Create a new container for the inner part
        right = FastAVLTree()
        (<_AVLTree> right).root = t_right
        (<_AVLTree> right).count = -1 if t_right != NULL else 0  # FIXME
        return left, right, bool(flag), value

    def split_last_inplace(self):
        if self.is_empty():
            raise IndexError('Empty tree has no maximum element')
        item = avl_split_last_inplace(&self.root)
        self.count -= 1
        return <tuple>item

    def splice_inplace(self, start_key, stop_key):
        cdef node_t *t_inner
        cdef node_t *t_outer
        inner = FastAVLTree()
        outer = self
        # only splice if start < stop, otherwise do nothing
        if start_key < stop_key:
            avl_splice_inplace(&self.root, start_key, stop_key,
                               &t_inner, &t_outer)

            # print('*t_inner = %r' % (t_inner,))
            # print('*t_outer = %r' % (t_outer,))

            # Create a new container for the inner part
            (<_AVLTree> inner).root = t_inner
            (<_AVLTree> inner).count = -1 if t_inner != NULL else 0  # FIXME

            # self becomes the outer part
            # _AVLTree *outer = self
            (<_AVLTree> outer).root = t_outer
            (<_AVLTree> outer).count = -1 if t_outer != NULL else 0  # FIXME
        return inner, outer


class FastAVLTree(_AVLTree, _ABCTree):
    pass


cdef class _RBTree(_BaseTree):
    def insert(self, key, value):
        cdef int result = rb_insert(&self.root, key, value)
        if result < 0:
            raise MemoryError('Can not allocate memory for node structure.')
        else:
            self.count += result

    def remove(self, key):
        cdef int result = rb_remove(&self.root, key)
        if result == 0:
            raise KeyError(str(key))
        else:
            self.count -= 1


class FastRBTree(_RBTree, _ABCTree):
    pass
