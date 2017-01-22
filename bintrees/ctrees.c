/*
 * ctrees.c
 *
 *  Author: mozman
 *  Copyright (c) 2010-2013 by Manfred Moitzi
 *  License: MIT-License
 */

#include "ctrees.h"
#include "assert.h"
#include <Python.h>

#define LEFT 0
#define RIGHT 1
#define KEY(node) (node->key)
#define VALUE(node) (node->value)
#define LEFT_NODE(node) (node->link[LEFT])
#define RIGHT_NODE(node) (node->link[RIGHT])
#define LINK(node, dir) (node->link[dir])
#define XDATA(node) (node->xdata)
#define RED(node) (node->xdata)
#define BALANCE(node) (node->xdata)

#define REPR(pyobj) (PyString_AsString(PyObject_Repr(pyobj)))

static node_t *
ct_new_node(PyObject *key, PyObject *value, int xdata)
{
	node_t *new_node = PyMem_Malloc(sizeof(node_t));
	if (new_node != NULL) {
		KEY(new_node) = key;
		Py_INCREF(key);
		VALUE(new_node) = value;
		Py_INCREF(value);
		LEFT_NODE(new_node) = NULL;
		RIGHT_NODE(new_node) = NULL;
		XDATA(new_node) = xdata;
	}
	return new_node;
}

static void
ct_delete_node(node_t *node)
{
	if (node != NULL) {
		Py_XDECREF(KEY(node));
		Py_XDECREF(VALUE(node));
		LEFT_NODE(node) = NULL;
		RIGHT_NODE(node) = NULL;
		PyMem_Free(node);
	}
}

extern void
ct_delete_tree(node_t *root)
{
	if (root == NULL)
		return;
	if (LEFT_NODE(root) != NULL) {
		ct_delete_tree(LEFT_NODE(root));
	}
	if (RIGHT_NODE(root) != NULL) {
		ct_delete_tree(RIGHT_NODE(root));
	}
	ct_delete_node(root);
}

static void
ct_swap_data(node_t *node1, node_t *node2)
{
	PyObject *tmp;
	tmp = KEY(node1);
	KEY(node1) = KEY(node2);
	KEY(node2) = tmp;
	tmp = VALUE(node1);
	VALUE(node1) = VALUE(node2);
	VALUE(node2) = tmp;
}

int
ct_compare(PyObject *key1, PyObject *key2)
{
	int res;

	res = PyObject_RichCompareBool(key1, key2, Py_LT);
	if (res > 0)
		return -1;
	else if (res < 0) {
		PyErr_SetString(PyExc_TypeError, "invalid type for key");
		return 0;
		}
	/* second compare:
	+1 if key1 > key2
	 0 if not -> equal
	-1 means error, if error, it should happen at the first compare
	*/
	return PyObject_RichCompareBool(key1, key2, Py_GT);
}

extern node_t *
ct_find_node(node_t *root, PyObject *key)
{
	int res;
	while (root != NULL) {
		res = ct_compare(key, KEY(root));
		if (res == 0) /* key found */
			return root;
		else {
			root = LINK(root, (res > 0));
		}
	}
	return NULL; /* key not found */
}

extern node_t*
ct_get_leaf_node(node_t *node)
{
    if (node == NULL)
        return NULL;
    for(;;) {
        if (LEFT_NODE(node) != NULL)
            node = LEFT_NODE(node);
        else if (RIGHT_NODE(node) != NULL)
            node = RIGHT_NODE(node);
        else return node;
    }
}

extern PyObject *
ct_get_item(node_t *root, PyObject *key)
{
	node_t *node;
	PyObject *tuple;

	node = ct_find_node(root, key);
	if (node != NULL) {
		tuple = PyTuple_New(2);
		PyTuple_SET_ITEM(tuple, 0, KEY(node));
		PyTuple_SET_ITEM(tuple, 1, VALUE(node));
		return tuple;
	}
	Py_RETURN_NONE;
}

extern node_t *
ct_max_node(node_t *root)
/* get node with largest key */
{
	if (root == NULL)
		return NULL;
	while (RIGHT_NODE(root) != NULL)
		root = RIGHT_NODE(root);
	return root;
}

extern node_t *
ct_min_node(node_t *root)
// get node with smallest key
{
	if (root == NULL)
		return NULL;
	while (LEFT_NODE(root) != NULL)
		root = LEFT_NODE(root);
	return root;
}

extern int
ct_bintree_remove(node_t **rootaddr, PyObject *key)
/* attention: rootaddr is the address of the root pointer */
{
	node_t *node, *parent, *replacement;
	int direction, cmp_res, down_dir;

	node = *rootaddr;

	if (node == NULL)
		return 0; /* root is NULL */
	parent = NULL;
	direction = 0;

	while (1) {
		cmp_res = ct_compare(key, KEY(node));
		if (cmp_res == 0) /* key found, remove node */
		{
			if ((LEFT_NODE(node) != NULL) && (RIGHT_NODE(node) != NULL)) {
				/* find replacement node: smallest key in right-subtree */
				parent = node;
				direction = RIGHT;
				replacement = RIGHT_NODE(node);
				while (LEFT_NODE(replacement) != NULL) {
					parent = replacement;
					direction = LEFT;
					replacement = LEFT_NODE(replacement);
				}
				LINK(parent, direction) = RIGHT_NODE(replacement);
				/* swap places */
				ct_swap_data(node, replacement);
				node = replacement; /* delete replacement node */
			}
			else {
				down_dir = (LEFT_NODE(node) == NULL) ? RIGHT : LEFT;
				if (parent == NULL) /* root */
				{
					*rootaddr = LINK(node, down_dir);
				}
				else {
					LINK(parent, direction) = LINK(node, down_dir);
				}
			}
			ct_delete_node(node);
			return 1; /* remove was success full */
		}
		else {
			direction = (cmp_res < 0) ? LEFT : RIGHT;
			parent = node;
			node = LINK(node, direction);
			if (node == NULL)
				return 0; /* error key not found */
		}
	}
}

extern int
ct_bintree_insert(node_t **rootaddr, PyObject *key, PyObject *value)
/* attention: rootaddr is the address of the root pointer */
{
	node_t *parent, *node;
	int direction, cval;
	node = *rootaddr;
	if (node == NULL) {
		node = ct_new_node(key, value, 0); /* new node is also the root */
		if (node == NULL)
			return -1; /* got no memory */
		*rootaddr = node;
	}
	else {
		direction = LEFT;
		parent = NULL;
		while (1) {
			if (node == NULL) {
				node = ct_new_node(key, value, 0);
				if (node == NULL)
					return -1; /* get no memory */
				LINK(parent, direction) = node;
				return 1;
			}
			cval = ct_compare(key, KEY(node));
			if (cval == 0) {
				/* key exists, replace value object */
				Py_XDECREF(VALUE(node)); /* release old value object */
				VALUE(node) = value; /* set new value object */
				Py_INCREF(value); /* take new value object */
				return 0;
			}
			else {
				parent = node;
				direction = (cval < 0) ? LEFT : RIGHT;
				node = LINK(node, direction);
			}
		}
	}
	return 1;
}

static int
is_red (node_t *node)
{
	return (node != NULL) && (RED(node) == 1);
}

#define rb_new_node(key, value) ct_new_node(key, value, 1)

static node_t *
rb_single(node_t *root, int dir)
{
	node_t *save = root->link[!dir];

	root->link[!dir] = save->link[dir];
	save->link[dir] = root;

	RED(root) = 1;
	RED(save) = 0;
	return save;
}

static node_t *
rb_double(node_t *root, int dir)
{
	root->link[!dir] = rb_single(root->link[!dir], !dir);
	return rb_single(root, dir);
}

#define rb_new_node(key, value) ct_new_node(key, value, 1)

extern int
rb_insert(node_t **rootaddr, PyObject *key, PyObject *value)
{
    int new_node = 0;
	node_t *root = *rootaddr;

	if (root == NULL) {
		/*
		 We have an empty tree; attach the
		 new node directly to the root
		 */
		root = rb_new_node(key, value);
		new_node = 1;
		if (root == NULL)
			return -1; // got no memory
	}
	else {
		node_t head; /* False tree root */
		node_t *g, *t; /* Grandparent & parent */
		node_t *p, *q; /* Iterator & parent */
		int dir = 0;
		int last = 0;

		/* Set up our helpers */
		t = &head;
		g = NULL;
		p = NULL;
		RIGHT_NODE(t) = root;
		LEFT_NODE(t) = NULL;
		q = RIGHT_NODE(t);

		/* Search down the tree for a place to insert */
		for (;;) {
			int cmp_res;
			if (q == NULL) {
				/* Insert a new node at the first null link */
				q = rb_new_node(key, value);
				new_node = 1;
				p->link[dir] = q;
				if (q == NULL)
					return -1; // get no memory
			}
			else if (is_red(q->link[0]) && is_red(q->link[1])) {
				/* Simple red violation: color flip */
				RED(q) = 1;
				RED(q->link[0]) = 0;
				RED(q->link[1]) = 0;
			}

			if (is_red(q) && is_red(p)) {
				/* Hard red violation: rotations necessary */
				int dir2 = (t->link[1] == g);

				if (q == p->link[last])
					t->link[dir2] = rb_single(g, !last);
				else
					t->link[dir2] = rb_double(g, !last);
			}

			/*  Stop working if we inserted a new node. */
			if (new_node)
				break;

			cmp_res = ct_compare(KEY(q), key);
			if (cmp_res == 0) {       /* if key exists            */
				Py_XDECREF(VALUE(q)); /* release old value object */
				VALUE(q) = value;     /* set new value object     */
				Py_INCREF(value);     /* take new value object    */
				break;
			}
			last = dir;
			dir = (cmp_res < 0);

			/* Move the helpers down */
			if (g != NULL)
				t = g;

			g = p;
			p = q;
			q = q->link[dir];
		}
		/* Update the root (it may be different) */
		root = head.link[1];
	}

	/* Make the root black for simplified logic */
	RED(root) = 0;
	(*rootaddr) = root;
	return new_node;
}

extern int
rb_remove(node_t **rootaddr, PyObject *key)
{
	node_t *root = *rootaddr;

	node_t head = { { NULL } }; /* False tree root */
	node_t *q, *p, *g; /* Helpers */
	node_t *f = NULL; /* Found item */
	int dir = 1;

	if (root == NULL)
		return 0;

	/* Set up our helpers */
	q = &head;
	g = p = NULL;
	RIGHT_NODE(q) = root;

	/*
	 Search and push a red node down
	 to fix red violations as we go
	 */
	while (q->link[dir] != NULL) {
		int last = dir;
		int cmp_res;

		/* Move the helpers down */
		g = p, p = q;
		q = q->link[dir];

		cmp_res =  ct_compare(KEY(q), key);

		dir = cmp_res < 0;

		/*
		 Save the node with matching data and keep
		 going; we'll do removal tasks at the end
		 */
		if (cmp_res == 0)
			f = q;

		/* Push the red node down with rotations and color flips */
		if (!is_red(q) && !is_red(q->link[dir])) {
			if (is_red(q->link[!dir]))
				p = p->link[last] = rb_single(q, dir);
			else if (!is_red(q->link[!dir])) {
				node_t *s = p->link[!last];

				if (s != NULL) {
					if (!is_red(s->link[!last]) &&
						!is_red(s->link[last])) {
						/* Color flip */
						RED(p) = 0;
						RED(s) = 1;
						RED(q) = 1;
					}
					else {
						int dir2 = g->link[1] == p;

						if (is_red(s->link[last]))
							g->link[dir2] = rb_double(p, last);
						else if (is_red(s->link[!last]))
							g->link[dir2] = rb_single(p, last);

						/* Ensure correct coloring */
						RED(q) = RED(g->link[dir2]) = 1;
						RED(g->link[dir2]->link[0]) = 0;
						RED(g->link[dir2]->link[1]) = 0;
					}
				}
			}
		}
	}

	/* Replace and remove the saved node */
	if (f != NULL) {
		ct_swap_data(f, q);
		p->link[p->link[1] == q] = q->link[q->link[0] == NULL];
		ct_delete_node(q);
	}

	/* Update the root (it may be different) */
	root = head.link[1];

	/* Make the root black for simplified logic */
	if (root != NULL)
		RED(root) = 0;
	*rootaddr = root;
	return (f != NULL);
}

#define avl_new_node(key, value) ct_new_node(key, value, 0)
#define height(p) ((p) == NULL ? -1 : (p)->xdata)
#define avl_max(a, b) ((a) > (b) ? (a) : (b))

static node_t *
avl_single(node_t *root, int dir)
{
    node_t *save = root->link[!dir];
    int rlh, rrh, slh;

	/* Rotate */
	root->link[!dir] = save->link[dir];
	save->link[dir] = root;

	/* Update balance factors */
	rlh = height(root->link[0]);
	rrh = height(root->link[1]);
	slh = height(save->link[!dir]);

	BALANCE(root) = avl_max(rlh, rrh) + 1;
	BALANCE(save) = avl_max(slh, BALANCE(root)) + 1;

	return save;
}

static node_t *
avl_double(node_t *root, int dir)
{
	root->link[!dir] = avl_single(root->link[!dir], !dir);
	return avl_single(root, dir);
}


static node_t *
avl_new_top(node_t *t1, node_t *t2, PyObject *key, PyObject *value, int dir)
{
    node_t *node = avl_new_node(key, value);
    node->link[dir] = t1;
    node->link[1 - dir] = t2;
    BALANCE(node) = avl_max(height(t1), height(t2)) + 1;
    return node;
}

static node_t *
avl_join_dir_recursive(node_t *t1, node_t *t2, 
                       PyObject *key, PyObject *value,
                       const int dir)
{
    node_t *large, *small, *spine, *rest;
    PyObject *k_, *v_;
    int hsmall, hspine, hrest;
    node_t *t_rotate, *t_merge, *t_, *t__;
    const int other_side = 1 - dir;

    if (dir == 0) {
        large = t2;
        small = t1;
    }
    else {
        large = t1;
        small = t2;
    }

    // Follow the spine of the larger tree
    spine = large->link[dir];
    rest = large->link[other_side];
    k_ = KEY(large);
    v_ = VALUE(large);

    hsmall = height(small);
    hspine = height(spine);
    hrest = height(rest);

    if (hspine <= hsmall + 1) {
        t_ = avl_new_top(small, spine, key, value, dir);
        if (height(t_) <= hrest + 1) {
            return avl_new_top(t_, rest, k_, v_, dir);
        }
        else {
            // Double rotation, but with a new node
            t_rotate = avl_single(t_, dir);
            t_merge = avl_new_top(rest, t_rotate, k_, v_, other_side);
            return avl_single(t_merge, other_side);
        }
    }
    else {
        // Traverse down the spine in the appropriate dir
        if (dir == 0) {
            t_ = avl_join_dir_recursive(small, spine, key, value, dir);
        }
        else {
            t_ = avl_join_dir_recursive(spine, t2, key, value, dir);
        }
        t__ = avl_new_top(t_, rest, k_, v_, dir);
        if (height(t_) <= hrest + 1){
            return t__;
        }
        else {
            return avl_single(t__, other_side);
        }
    }
}


#define avl_join_dir avl_join_dir_recursive


static node_t *
avl_join(node_t *t1, node_t *t2, PyObject *key, PyObject *value)
{
    /* 
     * TODO: Instead of taking in a key/value pair this should 
     * accept a Node object. That way we can reuse Node objects
     * and just manipulate pointers without every allocating 
     * memory in these function. Do this for all avl_inplace functions.
     * Only the extern callers should ever create new nodes.
     */
    int h1, h2;
    node_t *top;
    if (t1 == NULL && t2 == NULL) {
        /*printf("Case 1\n");*/
        top = avl_new_top(NULL, NULL, key, value, 0);
    }
    else if (t1 == NULL) {
        /*printf("Case 2\n");*/
        // FIXME keep track of count if possible
        /*top = avl_insert_hack(t2, key, value);*/
        node_t **topaddr = &t2;
        avl_insert(topaddr, key, value);
        top = *topaddr;
    }
    else if (t2 == NULL) {
        /*printf("Case 3\n");*/
        /*top = avl_insert_hack(t1, key, value);*/
        node_t **topaddr = &t1;
        avl_insert(topaddr, key, value);
        top = *topaddr;
    }
    else {
        h1 = height(t1);
        h2 = height(t2);
        if (h1 > h2 + 1) {
            /*printf("Case 5\n");*/
            top = avl_join_dir(t1, t2, key, value, 1);
        }
        else if (h2 > h1 + 1) {
            /*printf("Case 6\n");*/
            top = avl_join_dir(t1, t2, key, value, 0);
        }
        else {
            /*printf("Case 7\n");*/
            // Insert at the top of the tree
            top = avl_new_top(t1, t2, key, value, 0);
        }
    }
    return top;
}


void print_node(const char* prefix, node_t *node){
    if (node == NULL) {
        printf("%s = NULL\n", prefix);
    }
    else {
        printf("%s->key = %s\n", prefix, REPR(KEY(node)));
    }
}


#define DEBUG_SPLIT 0

static void 
avl_split(node_t *root, PyObject *key,
          node_t** o_part1, node_t** o_part2, 
          int *o_flag, PyObject **o_value) {
    // # TODO: keep track of the size of the sets being avl_split if possible
#if DEBUG_SPLIT
    printf("--- SPLIT(C) \n");
    print_node("root", root);    
    printf("key = %s\n", REPR(key));
#endif
    // node_t *split_node;

    if (root == NULL) {
#if DEBUG_SPLIT
        printf("Split NULL\n");
#endif
        (*o_part1) = root;
        (*o_part2) = root;
        (*o_flag) = 0;
        // FIXME: Py_None needs to be treated like any other object 
        // wrt to reference counts. Do we need to do anything else here?
        (*o_value) = Py_None;
        // Did not find a node to splie
        // split_node = NULL;
    }
    else {
        PyObject *t_key, *t_val;
        node_t *l, *r;
        l = LEFT_NODE(root);
        r = RIGHT_NODE(root);
        t_key = KEY(root);
        t_val = VALUE(root);
        if (PyObject_RichCompareBool(key, t_key, Py_EQ) == 1) {
#if DEBUG_SPLIT
            printf("Split Case Hit\n");
#endif
            (*o_part1) = l;
            (*o_part2) = r;
            (*o_flag) = 1;
            (*o_value) = t_val;

            // We are effectively deleting this node
            // Should its children be removed?
            // LEFT_NODE(root) = NULL;
            // RIGHT_NODE(root) = NULL;
            // split_node = root;
        }
        else if (PyObject_RichCompareBool(key, t_key, Py_LT) == 1) {
#if DEBUG_SPLIT
            printf("Split Case Recurse Down 1\n");
#endif
            node_t *ll, *lr, *new_right;
            // split_node = 
            avl_split(l, key, &ll, &lr, o_flag, o_value);
#if DEBUG_SPLIT
            printf("Split Case Recurse Up 1\n");
#endif
            new_right = avl_join(lr, r, t_key, t_val);
            (*o_part1) = ll;
            (*o_part2) = new_right;
        }
        else {
#if DEBUG_SPLIT
            printf("Split Case Recurse 2\n");
#endif
            node_t *rl, *rr, *new_left;
            // split_node = 
            avl_split(r, key, &rl, &rr, o_flag, o_value);
#if DEBUG_SPLIT
            printf("Split Case Recurse Up 2\n");
#endif
            new_left = avl_join(l, rl, t_key, t_val);
            (*o_part1) = new_left;
            (*o_part2) = rr;
        }
    }
#if DEBUG_SPLIT
            print_node("part1", *o_part1);
            print_node("part2", *o_part2);
#endif
    // I think split should return the node that it removes
    // and potentially delete it?
    // return split_node
}


static node_t *
avl_insert_recusrive(node_t *root, PyObject *key, PyObject *value) {
    // Functional version of insert O(log(n))
    node_t *left, *right;
    int start_flag;
    PyObject *old_val;  //= NULL;

    avl_split(root, key, &left, &right, &start_flag, &old_val);
    if (old_val != NULL) {
        Py_XDECREF(old_val); // release old value object
        Py_INCREF(value); // take new value object
    }
    /*left, right, flag, old_val = avl_split(root, key)*/
    return avl_join(left, right, key, value);
}



static node_t *
avl_split_last(node_t *root, PyObject **o_max_key, PyObject **o_max_value)
{
    /*
    Removes the maximum element from the tree

    O(log(n)) = O(height(root))
    */
    node_t *new_right, *new_root;

    node_t *left = root->link[0];
    node_t *right = root->link[1];
    if (right == NULL) {
        (*o_max_key) = KEY(root);
        (*o_max_value) = VALUE(root);
        return left;
    }
    else {
        new_right = avl_split_last(right, o_max_key, o_max_value);
        new_root = avl_join(left, new_right, KEY(root), VALUE(root));
        return new_root;
    }
}


static node_t *
avl_join2(node_t *t1, node_t *t2) {
    /*
    join two trees without any intermediate key

    O(log(n) + log(m)) = O(r(t1) + r(t2))

    For AVL-Trees the rank r(t1) = height(t1) - 1
    */
    PyObject *max_k, *max_v;
    node_t *new_left;
    if (t1 == NULL) {
        return t2;
    }
    else {
        new_left = avl_split_last(t1, &max_k, &max_v);
        return avl_join(new_left, t2, max_k, max_v);
    }
}


static void avl_splice(node_t *root, PyObject *start_key, PyObject *stop_key,
                       node_t** t_inner, node_t** t_outer) {
    // Extracts a ordered slice from `root` and returns the inside and outside
    // parts.
    // O(log(n)) 
    
    int start_flag, stop_flag;
    PyObject *start_val, *stop_val;
    node_t *left, *midright, *middle, *right;

    /*printf("------- SPLICE (C)\n");             */
    /*print_node("root", root);*/
    /*printf("(start_key) %s\n", REPR(start_key));*/
    /*printf("(stop_key) %s\n", REPR(stop_key));*/

    // Split tree into three parts
    avl_split(root, start_key, &left, &midright, &start_flag, &start_val);
    /*print_node("left", left);*/
    /*print_node("midright", midright);*/
    avl_split(midright, stop_key, &middle, &right, &stop_flag, &stop_val);

    /*print_node("left", left);*/
    /*print_node("middle", middle);*/
    /*print_node("right", right);*/

    // Insert the start_key back into the middle part if it was removed
    if (start_flag == 1) {
        (*t_inner) = avl_insert_recusrive(middle, start_key, start_val);
    }
    else {
        (*t_inner) = middle;
    }
    // Recombine the outer parts
    if (stop_flag == 1) {
        (*t_outer) = avl_join(left, right, stop_key, stop_val);
    }
    else {
        (*t_outer) = avl_join2(left, right);
    }
    /*printf("start_flag %d\n", start_flag);*/
    /*printf("stop_flag %d\n", stop_flag);*/

    /*printf("KEY(*t_outer) %s\n", PyString_AsString(PyObject_Repr(KEY((*t_outer)))));*/
    /*printf("KEY(*t_inner) %s\n", PyString_AsString(PyObject_Repr(KEY((*t_inner)))));*/

    /*printf("*t_outer %p\n", *t_outer);*/
    /*printf("*t_inner %p\n", *t_inner);*/
    /*printf("t_outer %p\n", t_outer);  */
    /*printf("t_inner %p\n", t_inner);  */
}

#define DEBUG_UNION 0



static node_t *
 avl_union(node_t *t1, node_t *t2){
    /*
     * O(m log((n/m) + 1))                                   
     * This is sublinear and good improvement over O(mlog(n))
     */
#if DEBUG_UNION
     printf("------- UNION (C)\n");             
     print_node("t1", t1);
     print_node("t2", t2);
#endif

    if (t1 == NULL) {
        return t2;
    }
    else if (t2 == NULL){
        return t1;
    }
    else {
        node_t *left2 = LEFT_NODE(t2);
        node_t *right2 = RIGHT_NODE(t2);
        PyObject *key2 = KEY(t2);
        PyObject *val2 = VALUE(t2);

        node_t *left1, *right1;
        int flag;
        PyObject *val1;
        avl_split(t1, key2, &left1, &right1, &flag, &val1);
        // if flag: dereference val? (Should things be dereferenced in split?)
        // Split should return the node it removes I think

        node_t *left_combo = avl_union(left1, left2);
        node_t *right_combo = avl_union(right1, right2);
        return avl_join(left_combo, right_combo, key2, val2);
    }
 }




// --- Extern Funcs ---


extern PyObject *
avl_split_inplace(node_t **rootaddr, PyObject *key, int* o_flag, node_t **t_right)
{
    node_t *root = (*rootaddr);
    node_t *t_left;
    PyObject *value;

    avl_split(root, key, &t_left, t_right, o_flag, &value);
    
    // The root becomes the left tree
    (*rootaddr) = t_left;

    // Return the right tree
    return value;
}


extern PyObject *
avl_split_last_inplace(node_t **rootaddr)
{
	PyObject *tuple;
    PyObject *max_key, *max_value;
    node_t *root = (*rootaddr);

    (*rootaddr) = avl_split_last(root, &max_key, &max_value);

    tuple = PyTuple_New(2);
    PyTuple_SET_ITEM(tuple, 0, max_key);
    PyTuple_SET_ITEM(tuple, 1, max_value);
    return tuple;
}


extern void
avl_splice_inplace(node_t **rootaddr, PyObject *start_key, PyObject *stop_key,
                  node_t **t_inner, node_t **t_outer)
{
    node_t *root = (*rootaddr);
    avl_splice(root, start_key, stop_key, t_inner, t_outer);

    // The root becomes the outer tree
    (*rootaddr) = (*t_outer);
    // Return the inner tree
    /*return t_inner;*/
}


extern void
avl_join_inplace(node_t **t1_addr, node_t **t2_addr, PyObject *key, PyObject *value)
{
    node_t *top;
    top = avl_join(*t1_addr, *t2_addr, key, value);
    // Reassign root value item
	(*t1_addr) = top;
    // The nodes in t2 have been assimilated into t1. 
    // t2 should no longer contain any values
	(*t2_addr) = NULL;
}


extern void
avl_join2_inplace(node_t **t1_addr, node_t **t2_addr)
{
    node_t *top;
    top = avl_join2(*t1_addr, *t2_addr);
    // Reassign root value item
	(*t1_addr) = top;
    // The nodes in t2 have been assimilated into t1. 
    // t2 should no longer contain any values
	(*t2_addr) = NULL;
}


extern void
avl_union_inplace(node_t **t1_addr, node_t **t2_addr)
{
    node_t *top;
    top = avl_union(*t1_addr, *t2_addr);
    // Reassign root value item
	(*t1_addr) = top;
    // The nodes in t2 have been assimilated into t1. 
    // t2 should no longer contain any values
	(*t2_addr) = NULL;
}



extern int
avl_insert(node_t **rootaddr, PyObject *key, PyObject *value)
{
	node_t *root = *rootaddr;

	if (root == NULL) {
		root = avl_new_node(key, value);
		if (root == NULL)
			return -1; // got no memory
	}
	else {
		node_t *it, *up[32];
		int upd[32], top = 0;
		int done = 0;
		int cmp_res;

		it = root;
		/* Search for an empty link, save the path */
		for (;;) {
			/* Push direction and node onto stack */
			cmp_res = ct_compare(KEY(it), key);
			if (cmp_res == 0) {
                // update existing item
				Py_XDECREF(VALUE(it)); // release old value object
				VALUE(it) = value; // set new value object
				Py_INCREF(value); // take new value object
				return 0;
			}
			// upd[top] = it->data < data;
			upd[top] = (cmp_res < 0);
			up[top++] = it;

			if (it->link[upd[top - 1]] == NULL)
				break;
			it = it->link[upd[top - 1]];
		}

		/* Insert a new node at the bottom of the tree */
		it->link[upd[top - 1]] = avl_new_node(key, value);
		if (it->link[upd[top - 1]] == NULL)
			return -1; // got no memory

		/* Walk back up the search path */
		while (--top >= 0 && !done) {
			// int dir = (cmp_res < 0);
			int lh, rh, max;

			cmp_res = ct_compare(KEY(up[top]), key);

			lh = height(up[top]->link[upd[top]]);
			rh = height(up[top]->link[!upd[top]]);

			/* Terminate or rebalance as necessary */
			if (lh - rh == 0)
				done = 1;
			if (lh - rh >= 2) {
				node_t *a = up[top]->link[upd[top]]->link[upd[top]];
				node_t *b = up[top]->link[upd[top]]->link[!upd[top]];

                // Determine which rotation is required
				if (height( a ) >= height( b ))
					up[top] = avl_single(up[top], !upd[top]);
				else
					up[top] = avl_double(up[top], !upd[top]);

				/* Fix parent */
				if (top != 0)
					up[top - 1]->link[upd[top - 1]] = up[top];
				else
					root = up[0];
				done = 1;
			}
			/* Update balance factors */
			lh = height(up[top]->link[upd[top]]);
			rh = height(up[top]->link[!upd[top]]);
			max = avl_max(lh, rh);
			BALANCE(up[top]) = max + 1;
		}
	}
	(*rootaddr) = root;
	return 1;
}

extern int
avl_remove(node_t **rootaddr, PyObject *key)
{
	node_t *root = *rootaddr;
	int cmp_res;

	if (root != NULL) {
		node_t *it, *up[32];
		int upd[32], top = 0;

		it = root;
		for (;;) {
			/* Terminate if not found */
			if (it == NULL)
				return 0;
			cmp_res = ct_compare(KEY(it), key);
			if (cmp_res == 0)
				break;

			/* Push direction and node onto stack */
			upd[top] = (cmp_res < 0);
			up[top++] = it;
			it = it->link[upd[top - 1]];
		}

		/* Remove the node */
		if (it->link[0] == NULL ||
			it->link[1] == NULL) {
			/* Which child is not null? */
			int dir = it->link[0] == NULL;

			/* Fix parent */
			if (top != 0)
				up[top - 1]->link[upd[top - 1]] = it->link[dir];
			else
				root = it->link[dir];

			ct_delete_node(it);
		}
		else {
			/* Find the inorder successor */
			node_t *heir = it->link[1];

			/* Save the path */
			upd[top] = 1;
			up[top++] = it;

			while ( heir->link[0] != NULL ) {
				upd[top] = 0;
				up[top++] = heir;
				heir = heir->link[0];
			}
			/* Swap data */
			ct_swap_data(it, heir);
			/* Unlink successor and fix parent */
			up[top - 1]->link[up[top - 1] == it] = heir->link[1];
			ct_delete_node(heir);
		}

		/* Walk back up the search path */
		while (--top >= 0) {
			int lh = height(up[top]->link[upd[top]]);
			int rh = height(up[top]->link[!upd[top]]);
			int max = avl_max(lh, rh);

			/* Update balance factors */
			BALANCE(up[top]) = max + 1;

			/* Terminate or rebalance as necessary */
			if (lh - rh == -1)
				break;
			if (lh - rh <= -2) {
				node_t *a = up[top]->link[!upd[top]]->link[upd[top]];
				node_t *b = up[top]->link[!upd[top]]->link[!upd[top]];

				if (height(a) <= height(b))
					up[top] = avl_single(up[top], upd[top]);
				else
					up[top] = avl_double(up[top], upd[top]);

				/* Fix parent */
				if (top != 0)
					up[top - 1]->link[upd[top - 1]] = up[top];
				else
					root = up[0];
			}
		}
	}
	(*rootaddr) = root;
	return 1;
}

extern node_t *
ct_succ_node(node_t *root, PyObject *key)
{
	node_t *succ = NULL;
	node_t *node = root;
	int cval;

	while (node != NULL) {
		cval = ct_compare(key, KEY(node));
		if (cval == 0)
			break;
		else if (cval < 0) {
			if ((succ == NULL) ||
				(ct_compare(KEY(node), KEY(succ)) < 0))
				succ = node;
			node = LEFT_NODE(node);
		} else
			node = RIGHT_NODE(node);
	}
	if (node == NULL)
		return NULL;
	/* found node of key */
	if (RIGHT_NODE(node) != NULL) {
		/* find smallest node of right subtree */
		node = RIGHT_NODE(node);
		while (LEFT_NODE(node) != NULL)
			node = LEFT_NODE(node);
		if (succ == NULL)
			succ = node;
		else if (ct_compare(KEY(node), KEY(succ)) < 0)
			succ = node;
	}
	return succ;
}

extern node_t *
ct_prev_node(node_t *root, PyObject *key)
{
	node_t *prev = NULL;
	node_t *node = root;
	int cval;

	while (node != NULL) {
		cval = ct_compare(key, KEY(node));
		if (cval == 0)
			break;
		else if (cval < 0)
			node = LEFT_NODE(node);
		else {
			if ((prev == NULL) || (ct_compare(KEY(node), KEY(prev)) > 0))
				prev = node;
			node = RIGHT_NODE(node);
		}
	}
	if (node == NULL) /* stay at dead end (None) */
		return NULL;
	/* found node of key */
	if (LEFT_NODE(node) != NULL) {
		/* find biggest node of left subtree */
		node = LEFT_NODE(node);
		while (RIGHT_NODE(node) != NULL)
			node = RIGHT_NODE(node);
		if (prev == NULL)
			prev = node;
		else if (ct_compare(KEY(node), KEY(prev)) > 0)
			prev = node;
	}
	return prev;
}

extern node_t *
ct_floor_node(node_t *root, PyObject *key)
{
	node_t *prev = NULL;
	node_t *node = root;
	int cval;

	while (node != NULL) {
		cval = ct_compare(key, KEY(node));
		if (cval == 0)
			return node;
		else if (cval < 0)
			node = LEFT_NODE(node);
		else {
			if ((prev == NULL) || (ct_compare(KEY(node), KEY(prev)) > 0))
				prev = node;
			node = RIGHT_NODE(node);
		}
	}
	return prev;
}

extern node_t *
ct_ceiling_node(node_t *root, PyObject *key)
{
	node_t *succ = NULL;
	node_t *node = root;
	int cval;

	while (node != NULL) {
		cval = ct_compare(key, KEY(node));
		if (cval == 0)
			return node;
		else if (cval < 0) {
			if ((succ == NULL) ||
				(ct_compare(KEY(node), KEY(succ)) < 0))
				succ = node;
			node = LEFT_NODE(node);
		} else
			node = RIGHT_NODE(node);
	}
	return succ;
}
