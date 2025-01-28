import networkx as nx
# NOTE: use original net


def create_graph(net):
    G = nx.MultiDiGraph()
    for t in net.transitions:
        G.add_node(t.name, label=t.label, type="t")
    for p in net.places:
        G.add_node(p.name, label=p.name, type="p")
    for a in net.arcs:
        G.add_edge(a.source.name, a.target.name)
    return G


def long_path(G, s, t_list):
    act_list = []
    for t in t_list:
        act_list.append(t.label)

    longest = {}
    for t in t_list:
        if t.label in longest.keys():
            continue
        path = nx.all_simple_paths(G, source=s, target=t.name)

        for n in path:
            count = 0
            for node in n:
                lab = G.nodes[node]['label']
                if G.nodes[node]['type'] == 't':
                    if not lab or 'tau' in lab or 'Inv' in lab:
                        continue
                    count += 1
                else:
                    continue
                if lab not in longest.keys() and lab in act_list:
                    longest[lab] = count
                elif lab in longest.keys() and longest[lab] < count:
                    longest[lab] = count
                else:
                    continue
    return longest


def p_length(out, net, im):
    G = create_graph(net)
    s = im.popitem()[0].name  # place's name of initial marking

    longest = long_path(G, s, net.transitions)
    longest_path = max(longest.values())

    length_dict = {}
    for elem in out:
        if 'Inv' in elem or 'tau' in elem:
            length_dict[elem] = 0
            continue
        position = longest[elem]
        length_dict[elem] = round(position / longest_path, 4)

    return length_dict


def optionality(out, id_c):
    # or_list = identify all or blocks, contains firsts branch activity
    # act_list = the lists contain all the activity for one or block
    opt_dict = {}

    for elem in out:
        if elem == "ASUBMITTED" or elem == "APARTLYSUBMITTED":
            print("a")
        choice = out[elem][id_c]
        if choice == 1:
            opt_dict[elem] = (1)
        else:
            opt_dict[elem] = round(1 / choice, 4)

    return opt_dict

def open_close(node):
    p = node.parent
    i = 0
    for n in p.children:
        if n == node:
            break
        i += 1
    if i == 1:
        #mit alten pnml files testen
        print("c")
    o = p.children[i-1]._get_label()
    c = p.children[i+1]._get_label()
    return o, c


def search_parallelism(node, p_list):
    if len(node._get_children()) == 0:
        for paral in p_list:
            if paral[0] > 0 and node._get_label():  # aggiungo nei parallelismi "aperti"
                c = p_list.index(paral)
                p_list[c].append(node._get_label())
        return p_list

    op = node._get_operator()
    if str(op) == "+":
        o, c = open_close(node)
        p_list.append([len(node.children), o, c])
        i = len(p_list) - 1
        for child in node.children:
            if p_list[i][0] > 0:
                search_parallelism(child, p_list)
                p_list[i][0] -= 1
    else:
        for child in node.children:
            search_parallelism(child, p_list)
    return p_list


def para_model(net, sp_list, o_c_list):
    G = create_graph(net)
    paral_list = []
    for i in range(0, len(sp_list)):
        paral_list.append([o_c_list[i][0], o_c_list[i][1]])

    for t in net.transitions:
        for i in range(0, len(sp_list)):
            if t.label == paral_list[i][0]:
                paral_list[i].pop(0)
                paral_list[i].insert(0,t.name)
                break
            elif t.label == paral_list[i][1]:
                paral_list[i].pop(1)
                paral_list[i].insert(1, t)
                break
            elif t.label in sp_list[i]:
                paral_list[i].append(t)
                break

    paral_activity = {}
    for p in paral_list:
        start = p.pop(0)
        act_dist = long_path(G, start, p)
        longest = max(act_dist.values())

        for activity in p:
            a = activity.label
            if 'Inv' in a or 'tau' in a:
                continue
            position = act_dist[a]
            paral_activity[a] = round(position / longest, 4)

    return paral_activity


def parallelism(tree, net, out, id_par):
    p_list = search_parallelism(tree, [[0]])
    p_list.pop(0)
    sp_list = []
    p_activity = set()

    open_close_list = []
    for x in p_list:
        x.pop(0)
        t_open = x.pop(0)
        t_close = x.pop(0)
        open_close_list.append((t_open, t_close))
        y = set()
        for e in x:
            if 'tau' in e or 'Inv' in e:
                continue
            else:
                y.add(e)
                p_activity.add(e)
        sp_list.append(y)

    superset = set()
    for i in range(0, len(sp_list)):
        for j in range(0, len(sp_list)):
            if sp_list[j].issuperset(sp_list[i]) and sp_list[j] != sp_list[i]:
                superset.add(i)

    superset = tuple(sorted(superset, reverse=True))

    for x in superset:
        open_close_list.remove(open_close_list[x])
        sp_list.remove(sp_list[x])

    p_model = para_model(net, sp_list, open_close_list)

    for elem in p_model:
        p_model[elem] = [p_model[elem], round(1 / out[elem][id_par], 4)]

    return p_model