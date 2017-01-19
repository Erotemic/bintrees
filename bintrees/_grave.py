


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


# DEBUG = 1
