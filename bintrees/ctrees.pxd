#!/usr/bin/env python
#coding:utf-8
# Author:  mozman
# Created: 08.05.2010
# Copyright (c) 2010-2013 by Manfred Moitzi
# License: MIT License

cdef extern from "ctrees.h":
    ctypedef struct PyObject:
        pass

    ctypedef struct node_t:
        node_t *link[2]
        PyObject *key
        PyObject *value
        int xdata;

    int ct_compare(object key1, object key2)
    void ct_delete_tree(node_t *root)
    node_t *ct_find_node(node_t *root, object key)
    node_t *ct_get_leaf_node(node_t *node)
    PyObject *ct_get_item(node_t *root, object key)
    node_t *ct_max_node(node_t *root)
    node_t *ct_min_node(node_t *root)
    node_t *ct_succ_node(node_t *root, object key)
    node_t *ct_prev_node(node_t *root, object key)
    node_t *ct_floor_node(node_t *root, object key)
    node_t *ct_ceiling_node(node_t *root, object key)
    int ct_index_of(node_t *root, object key)
    node_t *ct_node_at(node_t *root, int index)

    # binary-tree functions
    int ct_bintree_insert(node_t **root, object key, object value)
    int ct_bintree_remove(node_t **root, object key)
    # avl-tree functions
    int avl_insert(node_t **root, object key, object value)
    int avl_remove(node_t **root, object key)

    # avl-tree join-based inplace functions
    void avl_join_inplace(node_t **root1, node_t **root2, object key, object value)
    void avl_join2_inplace(node_t **t1_addr, node_t **t2_addr);
    void avl_union_inplace(node_t **t1_addr, node_t **t2_addr);
    void avl_splice_inplace(node_t **root1, object start, object stop, node_t **t_inner, node_t **t_outer)
    PyObject *avl_split_inplace(node_t **root1, object key, int* o_flag, node_t **t_right)
    PyObject *avl_split_last_inplace(node_t **root);

    # rb-tree functions
    int rb_insert(node_t **root, object key, object value)
    int rb_remove(node_t **root, object key)
