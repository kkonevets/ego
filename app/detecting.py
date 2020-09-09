import matplotlib
matplotlib.use('Agg')

import networkx as nx
from networkx.algorithms import community
import ccmg_cassandra
from cassandra import query as cquery
import json
import itertools
from functools import partial
from typing import Optional
import logging
import igraph
from hlc import HLC
import graph_tool.all as gt
import cdlib.algorithms as cdlib
import numpy as np
import karateclub
import os
import tempfile
import multiprocessing
import subprocess


def sort_pair(a, b):
    return (a, b) if a < b else (b, a)


class Selector:
    def __init__(self, platform):
        self.cassandra = ccmg_cassandra.cassandra(request_timeout=60)
        query1 = "SELECT friends FROM connections \
                WHERE snid = ? and platform = '{}' allow filtering"

        query1 = query1.format(platform)
        self.query1 = self.cassandra.session().prepare(query1)

        query2 = "SELECT snid, friends FROM connections \
                WHERE snid in ? and platform = '{}' allow filtering"

        query2 = query2.format(platform)
        self.query2 = self.cassandra.session().prepare(query2)

        query3 = "SELECT snid, name, profile_text_name FROM people \
                WHERE snid in ? and platform = '{}' allow filtering"

        query3 = query3.format(platform)
        self.query3 = self.cassandra.session().prepare(query3)

    def ego_graph(self, snid: int):
        row = self.cassandra.session().execute(self.query1, [str(snid)],
                                               timeout=60).one()
        if not row or not row.friends:
            return None, None
        friends = row.friends.split()
        nodes = {int(_id) for _id in friends}
        edge_nodes = {}  # only nodes present in edges
        edges = set()

        rows = self.cassandra.session().execute(
            self.query2, [cquery.ValueSequence(friends)], timeout=60).all()
        for row in rows:
            if not row.friends:
                continue
            i = None
            for _id in map(int, row.friends.split()):
                if _id not in nodes:
                    continue
                if i is None:
                    i = edge_nodes.setdefault(int(row.snid), len(edge_nodes))
                j = edge_nodes.setdefault(_id, len(edge_nodes))
                edges.add(sort_pair(i, j))

        new_friends = map(str, edge_nodes)
        rows = self.cassandra.session().execute(
            self.query3, [cquery.ValueSequence(new_friends)],
            timeout=60).all()

        new_nodes = [None] * len(edge_nodes)
        for cur_snid, i in edge_nodes.items():
            new_nodes[i] = {'snid': cur_snid, 'name': 'not found'}

        for row in rows:
            cur_snid = int(row.snid)
            i = edge_nodes[cur_snid]
            name = row.profile_text_name if row.profile_text_name else row.name
            new_nodes[i]['name'] = name

        return new_nodes, edges


def test_edges():
    return [(0, 1), (0, 4), (4, 2), (2, 3), (3, 5), (1, 5), (1, 3), (0, 2),
            (2, 6), (3, 7), (6, 9), (7, 10), (8, 9), (9, 10), (10, 11),
            (11, 13), (12, 13), (8, 12), (9, 12), (10, 13)]


class BigClam:
    def __init__(self, snap_path):
        self.executable = os.path.join(snap_path, 'examples/bigclam/bigclam')
        assert os.path.isfile(self.executable)
        self.temp_dir = os.path.join(tempfile.mkdtemp(), '')

    def run(self, edges, c=-1, mc=2, xc=11, nc=None):
        fedges = os.path.join(self.temp_dir, 'edges.txt')
        with open(fedges, 'w') as f:
            for i, j in edges:
                f.write("%i\t%i\n" % (i, j))

        if nc is None:
            nc = xc - mc - 1
        nt = max(1, int(multiprocessing.cpu_count() / 2))
        # The program tries nc numbers from mc to xc
        cmd = '%s -o:%s -i:%s -c:%i -mc:%i -xc:%i -nc:%i -nt:%i' % (
            self.executable, self.temp_dir, fedges, c, mc, xc, nc, nt)
        result = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE)

        with open(os.path.join(self.temp_dir, 'cmtyvv.txt')) as f:
            s = f.read().rstrip()
            coms = [[int(c) for c in com.split('\t') if c]
                    for com in s.split('\n')]

        return coms

    def __del__(self):
        if hasattr(self, 'temp_dir'):
            import shutil
            shutil.rmtree(self.temp_dir)


def find_communities(nnodes, edges, alg, params=None):
    def membership2cs(membership):
        cs = {}
        for i, m in enumerate(membership):
            cs.setdefault(m, []).append(i)
        return cs.values()

    def connected_subgraphs(G: nx.Graph):
        for comp in nx.connected_components(G):
            sub = nx.induced_subgraph(G, comp)
            sub = nx.convert_node_labels_to_integers(sub,
                                                     label_attribute='old')
            yield sub

    def apply_subgraphs(algorithm, **params):
        cs = []
        for sub in connected_subgraphs(G):
            if len(sub.nodes) <= 3:
                coms = [sub.nodes]  # let it be a cluster
            else:
                coms = algorithm(sub, **params)
                if hasattr(coms, 'communities'):
                    coms = coms.communities

            for com in coms:
                cs.append([sub.nodes[i]['old'] for i in set(com)])
        return cs

    def karate_apply(algorithm, graph, **params):
        model = algorithm(**params)
        model.fit(graph)
        return membership2cs(model.get_memberships().values())

    if alg == 'big_clam':
        c = -1 if params['c'] == 'auto' else int(params['c'])
        cs = BigClam('../../snap').run(edges, c=c, xc=int(params['xc']))
    elif alg in ('gmm', 'kclique', 'lprop', 'lprop_async', 'fluid',
                 'girvan_newman', 'angel', 'congo', 'danmf', 'egonet_splitter',
                 'lfm', 'multicom', 'nmnf', 'nnsed', 'node_perception', 'slpa',
                 'GEMSEC', 'EdMot', 'demon'):
        G = nx.Graph()
        G.add_edges_from(edges)

        if alg == 'gmm':
            cs = community.greedy_modularity_communities(G)
        elif alg == 'kclique':
            params = {k: float(v) for k, v in params.items()}
            cs = community.k_clique_communities(G, **params)
        elif alg == 'lprop':
            cs = community.label_propagation_communities(G)
        elif alg == 'lprop_async':
            cs = community.asyn_lpa_communities(G, seed=0)
        elif alg == 'fluid':
            params = {k: int(v) for k, v in params.items()}
            params['seed'] = 0
            cs = apply_subgraphs(community.asyn_fluidc, **params)
        elif alg == 'girvan_newman':
            comp = community.girvan_newman(G)
            for cs in itertools.islice(comp, int(params['k'])):
                pass
        elif alg == 'angel':
            params = {k: float(v) for k, v in params.items()}
            cs = cdlib.angel(G, **params).communities
        elif alg == 'congo':  # too slow
            ncoms = int(params['number_communities'])
            cs = []
            for sub in connected_subgraphs(G):
                if len(sub.nodes) <= max(3, ncoms):
                    cs.append(sub.nodes)  # let it be a cluster
                else:
                    coms = cdlib.congo(sub,
                                       number_communities=ncoms,
                                       height=int(params['height']))
                    for com in coms.communities:
                        cs.append([sub.nodes[i]['old'] for i in set(com)])
        elif alg == 'danmf':  # no overlapping
            cs = apply_subgraphs(cdlib.danmf)
        elif alg == 'egonet_splitter':
            params['resolution'] = float(params['resolution'])
            cs = apply_subgraphs(cdlib.egonet_splitter, **params)
        elif alg == 'lfm':
            coms = cdlib.lfm(G, float(params['alpha']))
            cs = coms.communities
        elif alg == 'multicom':
            cs = cdlib.multicom(G, seed_node=0).communities
        elif alg == 'nmnf':
            params = {k: int(v) for k, v in params.items()}
            cs = apply_subgraphs(cdlib.nmnf, **params)
        elif alg == 'nnsed':
            cs = apply_subgraphs(cdlib.nnsed)
        elif alg == 'node_perception':  # not usable
            params = {k: float(v) for k, v in params.items()}
            cs = cdlib.node_perception(G, **params).communities
        elif alg == 'slpa':
            params["t"] = int(params["t"])
            params["r"] = float(params["r"])
            cs = cdlib.slpa(G, **params).communities
        elif alg == 'demon':
            params = {k: float(v) for k, v in params.items()}
            cs = cdlib.demon(G, **params).communities
        elif alg == 'GEMSEC':
            # gamma = float(params.pop('gamma'))
            params = {k: int(v) for k, v in params.items()}
            # params['gamma'] = gamma
            params['seed'] = 0
            _wrap = partial(karate_apply, karateclub.GEMSEC)
            cs = apply_subgraphs(_wrap, **params)
        elif alg == 'EdMot':
            params = {k: int(v) for k, v in params.items()}
            _wrap = partial(karate_apply, karateclub.EdMot)
            cs = apply_subgraphs(_wrap, **params)

    elif alg in ('infomap', 'community_leading_eigenvector', 'leig',
                 'multilevel', 'optmod', 'edge_betweenness', 'spinglass',
                 'walktrap', 'leiden', 'hlc'):
        G = igraph.Graph()
        G.add_vertices(nnodes)
        G.add_edges(edges)

        if alg == 'infomap':
            vcl = G.community_infomap(trials=int(params['trials']))
            cs = membership2cs(vcl.membership)
        elif alg == 'leig':
            clusters = None if params['clusters'] == 'auto' else int(
                params['clusters'])
            vcl = G.community_leading_eigenvector(clusters=clusters)
            cs = membership2cs(vcl.membership)
        elif alg == 'multilevel':
            vcl = G.community_multilevel()
            cs = membership2cs(vcl.membership)
        elif alg == 'optmod':  # too long
            membership, modularity = G.community_optimal_modularity()
            cs = membership2cs(vcl.membership)
        elif alg == 'edge_betweenness':
            clusters = None if params['clusters'] == 'auto' else int(
                params['clusters'])
            dendrogram = G.community_edge_betweenness(clusters, directed=False)
            try:
                clusters = dendrogram.as_clustering()
            except:
                return []
            cs = membership2cs(clusters.membership)
        elif alg == 'spinglass':  # only for connected graph
            vcl = G.community_spinglass(parupdate=True,
                                        update_rule=params['update_rule'],
                                        start_temp=float(params['start_temp']),
                                        stop_temp=float(params['stop_temp']))
            cs = membership2cs(vcl.membership)
        elif alg == 'walktrap':
            dendrogram = G.community_walktrap(steps=int(params['steps']))
            try:
                clusters = dendrogram.as_clustering()
            except:
                return []
            cs = membership2cs(clusters.membership)
        elif alg == 'leiden':
            vcl = G.community_leiden(
                objective_function=params['objective_function'],
                resolution_parameter=float(params['resolution_parameter']),
                n_iterations=int(params['n_iterations']))
            cs = membership2cs(vcl.membership)
        elif alg == 'hlc':
            algorithm = HLC(G, min_size=int(params['min_size']))
            cs = algorithm.run(None)

    elif alg in ("sbm", "sbm_nested"):
        np.random.seed(42)
        gt.seed_rng(42)

        G = gt.Graph(directed=False)
        G.add_edge_list(edges)

        deg_corr = bool(params['deg_corr'])
        B_min = None if params['B_min'] == 'auto' else int(params['B_min'])
        B_max = None if params['B_max'] == 'auto' else int(params['B_max'])

        if alg == "sbm":
            state = gt.minimize_blockmodel_dl(G,
                                              deg_corr=deg_corr,
                                              B_min=B_min,
                                              B_max=B_max)

            membership = state.get_blocks()
            cs = membership2cs(membership)
        if alg == "sbm_nested":
            state = gt.minimize_nested_blockmodel_dl(G,
                                                     deg_corr=deg_corr,
                                                     B_min=B_min,
                                                     B_max=B_max)
            levels = state.get_bs()
            level_max = int(params['level'])

            membership = {}
            for nid in range(nnodes):
                cid = nid
                level_i = len(levels)
                for level in levels:
                    cid = level[cid]
                    if level_i == level_max:
                        membership.setdefault(cid, []).append(nid)
                        break
                    level_i -= 1

            cs = membership.values()

    else:
        return None

    return list(cs)


def canvas_data(nodes, edges, cs):
    ngroups = len(cs)

    for node in nodes:
        node['group'] = []
    for gid, com in enumerate(cs):
        for i in com:
            nodes[i]['group'].append(gid)

    counts = {}
    colors = []
    for i, j in edges:
        g_i = nodes[i].get('group', [])
        g_j = nodes[j].get('group', [])
        c_i = counts.setdefault(i, {})
        c_j = counts.setdefault(j, {})

        c = ngroups  # clusters are different
        for a, b in itertools.product(g_i, g_j):
            if a == b:
                c = a  # clusters are equal
                c_i[a] = c_i.get(a, 0) + 1
                c_j[a] = c_j.get(a, 0) + 1

        if len(g_i) > 1 and len(g_j) > 1:
            c = ngroups + 1  # many to many relation

        colors.append(c)

    def max_count(i):
        c = counts[i]
        if c:
            return max(c.items(), key=lambda e: e[::-1])
        else:
            return (-1, -1)

    keys = [max_count(i) for i in range(len(nodes))]
    sorted_index = sorted(range(len(keys)), key=lambda i: keys[i])
    reindex = [None] * len(nodes)
    for i, j in enumerate(sorted_index):
        reindex[j] = i

    ncolors = max(colors) + 1
    data = {
        'ncolors': ncolors,
        'naux_colors': ncolors - ngroups,
        'nodes': [nodes[i]['snid'] for i in sorted_index],
        'groups': [nodes[i]['group'] for i in sorted_index],
        'info': [nodes[i]['name'] for i in sorted_index],
        'links':
        [(reindex[s], reindex[t], c) for (s, t), c in zip(edges, colors)]
    }

    return data
