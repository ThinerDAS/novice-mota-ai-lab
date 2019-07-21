#!/usr/bin/python -i

# one more optimization - if one action has no immediate good, it must have topological good, so only search immediately or never
# one more optimization - always prioritize the free meals

#from IPython import embed
with open('ai_info.data', 'r') as f:
    main_point_adj_d, conn_comp, main_point_desc, enemy_list, std_form, init_state, init_avail_set = eval(
        f.read())

m2cc = {i: set() for i in main_point_adj_d}
conn_comp_list = []
for i, j in conn_comp.items():
    idx = len(conn_comp_list)
    conn_comp_list.append(j)
    for k in i:
        m2cc[k].add(idx)

keyed_std_form = {i[0]: i for i in std_form}

sorted_atom_key = sorted(keyed_std_form)

# multiple tile may use one index
tile_to_idx = {}
for i, j in enumerate(sorted_atom_key):
    for k in keyed_std_form[j]:
        tile_to_idx[k] = i
init_state['total_dam'] = 0
init_state['curse'] = -1


def add_delta(d1, d2):
    d = dict(d1)
    for i in d2:
        d.setdefault(i, 0)
        d[i] += d2[i]
    return d


def ladd_delta(d1, d2):
    d = d1
    for i in d2:
        d.setdefault(i, 0)
        d[i] += d2[i]
    return d


class Mtutil(object):
    supported_spec = [0, 1, 2, 3, 4, 5, 6, 28]
    equip_effect = [
        (2, 0),
        (4, 0),
        (8, 0),
        (16, 0),
        (32, 0),
        (64, 0),
        (128, 0),
        (256, 0),
        (512, 0),
        (999, 99),
    ]

    def enemy_damage(self, hero, enemy):
        bhp = hero['hp']
        bat = hero['at']
        bdf = hero['df']
        bmf = hero['mf']
        mhp = enemy['hp']
        mat = enemy['at']
        mdf = enemy['df']
        mspec = enemy['spec']
        if hero['equip_level']:
            da, dd = self.equip_effect[hero['equip_level']-1]
            bat += da
            bdf += dd
        if isinstance(mspec, (int, long)):
            mspec = [mspec]
        for i in mspec:
            if i not in self.supported_spec:
                raise Exception('Unsupported special: {:d}'.format(i))
        mpoint = enemy['point']
        if 28 in mspec:
            # double 20
            if 'double20' in self.flags:
                bat *= 2
        if bat <= mdf:
            # cannot defeat
            return max(999999, bhp*4+4)
        bone = bat-mdf
        mone = max(0, mat-bdf)
        if 2 in mspec:
            # magic
            mone = mat
        if 3 in mspec:
            # solid
            bone = min(1, bone)
        count = max(mhp-1, 0)//bone
        if 1 in mspec:
            # speed
            count += 1
        if 4 in mspec:
            # double
            count *= 2
        if 5 in mspec:
            # triple
            count *= 3
        if 6 in mspec:
            # xn
            count *= enemy['n']
        return max(0, count*mone-bmf)


mtutil = Mtutil()


def dup_state(state):
    return dict(state)


def apply_event(state, event, coord):
    ty, obj = event
    if ty == 'enemy':
        dam = mtutil.enemy_damage(state, enemy_list[obj])
        if state['hp'] <= dam:
            return False
        state['hp'] -= dam
        assert dam >= heu1[coord], 'heuristic function larger than actual'
        state['total_dam'] += dam-heu1[coord]

        return True
    elif ty == 'door':
        if state[obj] >= 1:
            state[obj] -= 1
            return True
        else:
            return False
    elif ty == 'delta':
        ladd_delta(state, obj)
        return True
    else:
        print 'unhandled:', ty
        return False


def do_sequence(state, taken, l, curse=-1):
    # only implemented topological curse, keying curse & hp curse may be implemented later - if we do not need hp, we avoid taking it
    new_state = dup_state(state)
    if new_state['curse'] != -1:
        # make out what it equals and see whether i sats
        last = sorted_atom_key[new_state['curse']]
        last_l = keyed_std_form[last]
        for i in last_l:
            if sorted_atom_key[curse] in main_point_adj_d[i]:
                break
        else:
            # print 'curse take effect'
            return None, None
    new_state['curse'] = curse
    new_take = set()
    takes = False
    for i in l:
        if not apply_event(new_state, main_point_desc[i], i):
            return None, None
        if main_point_desc[i][0] == 'delta':
            takes = True
        for j in m2cc[i]:
            if j not in taken and j not in new_take:
                new_take.add(j)
                takes = True
                ladd_delta(new_state, conn_comp_list[j])
    if takes:
        new_state['curse'] = -1
    else:
        # print 'cursed'
        pass
    return new_state, new_take


def state_key(state):
    return state['hp']


def get_feed_back(feed_back, taken_set):
    return sum(feed_back[i] for i in taken_set)


def get_prior_target(been_set, state, avail, taken):
    s = set()
    for i, j in enumerate(sorted_atom_key):
        if i in been_set:
            continue
        if j not in avail:
            continue
        delta = dup_state(state)

        new_take = set()
        l = keyed_std_form[j]
        for k in l:
            ty, obj = main_point_desc[k]
            if ty == 'enemy':
                dam = mtutil.enemy_damage(state, enemy_list[obj])
                expect = heu1[k]
                assert dam >= expect
                '''
                    # consider the case where hp is too low
                    if dam == expect:
                        # cannot be reduced more
                        pass
                    else:
                        break
                '''
                if dam == 0:
                    pass
                else:
                    break
            elif ty == 'door':
                if delta[obj] >= 1:
                    delta[obj] -= 1
                else:
                    break
            elif ty == 'delta':
                ladd_delta(delta, obj)
            else:
                break
            for j in m2cc[k]:
                if j not in taken and j not in new_take:
                    new_take.add(j)
                    ladd_delta(delta, conn_comp_list[j])

        else:
            if any(j < state[i] for i, j in delta.items()):
                pass
            else:
                s.add(i)
    return s


def propagate(search_state, choice_d, THRESHOLD=262144, feed_back=None, max_totaldam=None):
    print 'state space', len(search_state)
    imprecise = False
    if len(search_state) > THRESHOLD:
        print 'Trimming state'
        if feed_back:
            def key_f(
                x): return search_state[x][0]['total_dam']-get_feed_back(feed_back, x)
            state_list = sorted(
                search_state, key=key_f)
            print 'threshold dam:', key_f(state_list[THRESHOLD])
        else:
            def key_f(x): return search_state[x][0]['total_dam']
            state_list = sorted(
                search_state, key=key_f)
            print 'threshold dam:', key_f(state_list[THRESHOLD])
        search_state = {i: search_state[i] for i in state_list[:THRESHOLD]}
        imprecise = True
    d = {}
    for i, j in search_state.items():
        st, avail, taken, vroute = j
        prior_target = get_prior_target(i, st, avail, taken)
        prior_been = False
        for t in range(2):
            if t == 1 and prior_been:
                break
            for h, k in enumerate(sorted_atom_key):
                if h in i:
                    continue
                if (t != 1) and (prior_target and h not in prior_target):
                    continue
                if k not in avail:
                    continue
                # prune - prerequisite
                if any(m not in i for m in prerequisite[h]):
                    continue
                # print vroute,'->',k
                l = keyed_std_form[k]
                new_state, new_take = do_sequence(st, taken, l, h)
                #new_state, new_take = do_sequence(st, taken, l)
                if new_state is None:
                    # print 'die on',k
                    continue
                if t == 0:
                    prior_been = True
                if max_totaldam and new_state['total_dam'] > max_totaldam:
                    # prune by overdamage
                    continue
                key = frozenset(list(i)+[h])
                if key in d:
                    last_best_state, new_avail, new_taken, _ = d[key]
                    if state_key(last_best_state) >= state_key(new_state):
                        continue
                else:
                    new_taken = taken.union(new_take)
                    new_avail = set(avail)
                    for m in l:
                        new_avail.update(main_point_adj_d[m])
                d[key] = (new_state, new_avail, new_taken, vroute+[h])
    return d, imprecise

# Astar algorithm
# first we need to determine the highest power for each mob

# algorithm weakness: cannot find the hidden truths - if both is unavailable, first can be harder to defeat


def max_state_without_nodes(s):
    fin_set = set()
    been_set = set(init_avail_set)
    q = list(init_avail_set)
    bound_delta = {}
    while q:
        v = q.pop()
        if v in s:
            continue
        fin_set.add(v)
        ty, obj = main_point_desc[v]
        if ty == 'delta':
            ladd_delta(bound_delta, obj)
        adj = main_point_adj_d[v]
        for j in adj:
            if j not in been_set:
                been_set.add(j)
                q.append(j)
    free_delta = reduce(
        add_delta, [j for i, j in conn_comp.items() if i.intersection(fin_set)], {})
    return add_delta(add_delta(init_state, free_delta), bound_delta), fin_set


def damage_of_one_without_another(c1, c2=None):
    # c2 is any, c1 must be enemy
    # c2 must not block c1, assumed
    # all assumption!
    hero = max_state_without_nodes(set([c1, c2]))[0]
    #hero['hp'] = 999999
    enemy = enemy_list[main_point_desc[c1][1]]
    return mtutil.enemy_damage(hero, enemy)


def get_block_damage(c):
    fin_set = max_state_without_nodes([c])[1]
    return {i: damage_of_one_without_another(i, c) for i in fin_set if main_point_desc[i][0] == 'enemy'}


def get_unblock_damage():
    return {i: damage_of_one_without_another(i)for i, j in main_point_desc.items() if j[0] == 'enemy'}

# then we determine the damage

# LEVEL 1
# v = {i: mtutil.enemy_damage(max_state_without_node(
#    i), enemy_list[j[1]]) if j[0] == 'enemy' else 0 for i, j in main_point_desc.items()}


heu1 = get_unblock_damage()

# LEVEL 2

heu2 = {i: get_block_damage(i) for i in main_point_adj_d}


def damage_of_one_with_prerequisite(c1):
    # c2 is any, c1 must be enemy
    # c2 must not block c1, assumed
    # all assumption!
    without_set = set([j for i in inv_prerequisite[tile_to_idx[c1]]
                       for j in keyed_std_form[sorted_atom_key[i]]])
    l = keyed_std_form[sorted_atom_key[tile_to_idx[c1]]]
    for i in l[l.index(c1):]:
        without_set.add(i)
    hero = max_state_without_nodes(without_set)[0]
    print 'mob', c1, 'best abil', hero
    #hero['hp'] = 999999
    enemy = enemy_list[main_point_desc[c1][1]]
    return mtutil.enemy_damage(hero, enemy)


def update_max_damage():
    return {i: damage_of_one_with_prerequisite(i)for i, j in main_point_desc.items() if j[0] == 'enemy'}


def get_max_damage(mon_c, state_key):
    v = heu1[mon_c]
    for i in heu2:
        if i not in state_key and mon_c in heu2[i]:
            v = max(v, heu2[i][mon_c])
    return v

# embed()


prerequisite = [set() for i in range(len(keyed_std_form))]


def transpose(sl):
    l = [set() for i in range(len(sl))]
    for i, j in enumerate(sl):
        for k in j:
            l[k].add(i)
    return l


def force_order(THRESHOLD=8192, max_totaldam=None):
    search_state = {frozenset(): (init_state, init_avail_set, set(), [])}
    for i in range(len(keyed_std_form)):
        print 'order hunting: Iteration', i+1, '/', len(keyed_std_form)
        new_search_state, imprecise = propagate(
            search_state, keyed_std_form, THRESHOLD, max_totaldam=max_totaldam)
        if imprecise:
            del new_search_state
            break
        else:
            search_state = new_search_state
            del new_search_state
    else:
        print 'This may make search too easy ... max is found'
        return
    return commit_order(search_state)


def generate_possible_pair(l):
    ls = set(l)
    s = set()
    for i, j in enumerate(l):
        for k in l[:i]:
            s.add((k, j))
    for i in l:
        for j in range(len(keyed_std_form)):
            if j not in ls:
                s.add((i, j))
    return s


def commit_order(search_state):
    '''
    For each two tile A,B
    a route may
    a) contain none
    b) contain both, A-B
    c) contain both, B-A
    d) contain A
    e) contain B
    b,d advocates that A<B
    c,e advocates that A>B
    a advocates that both may be true

    if a condition is not advocated, the contrary must be the truth and we can trim in later
    '''
    n = len(keyed_std_form)
    vec = [[0]*i for i in range(n)]
    s = set((x, y) for x in range(n) for y in range(n) if x != y)
    for v in search_state.values():
        s.intersection_update(generate_possible_pair(v[-1]))
    print 'we know {:d} constraints now'.format(len(s))
    #from IPython import embed;embed()
    for i, j in s:
        prerequisite[j].add(i)


print 'pretrim'
for i in [8192]:  # , 32768, 131072]:
    force_order(THRESHOLD=i)
    inv_prerequisite = transpose(prerequisite)
    heu1 = update_max_damage()


search_state = {frozenset(): (init_state, init_avail_set, set(), [])}


print 'Weak search'

for i in range(len(keyed_std_form)):
    print 'Iteration', i+1, '/', len(keyed_std_form)
    search_state, imprecise = propagate(search_state, keyed_std_form, 8192)
    # print 'state', search_state


assert len(search_state) == 1
fin_state, _, _, vroute = search_state.values()[0]
print 'hp', fin_state['hp']
print 'total_dam', fin_state['total_dam']
print 'route', vroute

max_totaldam = fin_state['total_dam']

for th in [8192, 32768, 65536]:  # we avoid making the force_order to finish the search directly
    print '='*60, 'refine lower bound of damage, ITER', th
    # update totaldam
    st = dup_state(init_state)
    taken = set()

    for i in vroute:
        k = sorted_atom_key[i]
        l = keyed_std_form[k]
        # print i, l
        new_state, new_take = do_sequence(st, taken, l, i)
        assert new_state
        st = new_state
        taken.update(new_take)
    print 'total_dam', st['total_dam']
    # make trim
    force_order(THRESHOLD=th, max_totaldam=st['total_dam'])
    inv_prerequisite = transpose(prerequisite)
    heu1 = update_max_damage()

print 'Strong search with feedback, prepare feedback'

st = dup_state(init_state)
taken = set()
feed_back = {}
dam_seq = []

for i in vroute:
    k = sorted_atom_key[i]
    l = keyed_std_form[k]
    print i, l
    new_state, new_take = do_sequence(st, taken, l, i)
    assert new_state
    feed_back[i] = new_state['total_dam']-st['total_dam']
    dam_seq.append(new_state['total_dam'])
    st = new_state
    taken.update(new_take)

print 'dam seq', dam_seq

print 'Strong search ...'

search_state = {frozenset(): (init_state, init_avail_set, set(), [])}

for i in range(len(keyed_std_form)):
    print 'Iteration', i+1, '/', len(keyed_std_form)
    search_state, imprecise = propagate(
        search_state, keyed_std_form, 262144, feed_back=feed_back, max_totaldam=max_totaldam)
    # print 'state', search_state


assert len(search_state) == 1
fin_state, _, _, vroute = search_state.values()[0]
print 'hp', fin_state['hp']
print 'route', vroute
with open('route_info.data', 'w') as f:
    f.write(str([sorted_atom_key[i] for i in vroute]))
# '''
