import math
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.spatial

from sklearn.datasets import make_circles, make_moons, make_blobs

from pyspark import RDD, SparkConf, SparkContext


# Snap stanford

def get_clustering_data():
    """
    Retrieves all toy datasets from sklearn
    :return: circles, moons, blobs datasets.
    """
    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=.5,
                                 noise=.05)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05)
    blobs = make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples,
                        cluster_std=[1.0, 2.5, 0.5],
                        random_state=random_state)

    plt.figure(figsize=(9 * 2 + 3, 13))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                        hspace=.01)

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {})]

    return datasets


def create_distance_matrix(dataset):
    """
    Creates the distance matrix for a dataset with only vertices. Also adds the edges to a dict.
    :param dataset: dataset without edges
    :return: distance matrix, a dict of all edges and the total number of edges
    """
    vertices = []
    size = 0
    for line in dataset[0]:
        vertices.append([line[0], line[1]])
    d_matrix = scipy.spatial.distance_matrix(vertices, vertices, threshold=1000000)
    dict = {}

    # Run with less edges
    for i in range(len(d_matrix)):
        dict2 = {}
        for j in range(i, len(d_matrix)):
            if i != j:
                size += 1
                if dataset[1][i] == dataset[1][j]:
                    same_cluster = 1
                else:
                    same_cluster = 0

                distance = d_matrix[i][j]
                dict2[j] = (distance, same_cluster)

        dict[i] = dict2
    return d_matrix, dict, size, vertices


def partion_edges(E_list, k):
    E_list_partion = []
    random.shuffle(E_list)

    for i in range(len(E_list)):
        if i < k:
            E_list_partion.append({E_list[i]})
        else:
            E_list_partion[i % k].add(E_list[i])
    return E_list_partion


def get_key(item):
    """
    returns the sorting criteria for the edges. All edges are sorted from small to large values
    :param item: one item
    :return: returns the weight of the edge
    """
    return item[2]


def find_mst(E_list):
    """
    finds the mst of graph G = (U union V, E)
    :param U: vertices U
    :param V: vertices V
    :param E: edges of the graph
    :return: the mst and edges not in the mst of the graph
    """
    vertices = {vertex for pair in E_list for vertex in pair[:2]}

    E_list = sorted(E_list, key=get_key)
    print(E_list)
    connected_component = set()
    mst = []
    remove_edges = set()

    while len(mst) < len(vertices) - 1 and len(connected_component) < len(vertices):
        if len(E_list) == 0:
            break
        change = False
        i = 0
        while i < len(E_list):
            if len(connected_component) == 0:
                connected_component.add(E_list[i][0])
                connected_component.add(E_list[i][1])
                mst.append(E_list[i])
                change = True
                E_list.remove(E_list[i])
                break
            else:
                if E_list[i][0] in connected_component:
                    if E_list[i][1] in connected_component:
                        remove_edges.add(E_list[i])
                        E_list.remove(E_list[i])
                    else:
                        connected_component.add(E_list[i][1])
                        mst.append(E_list[i])
                        E_list.remove(E_list[i])
                        change = True
                        break
                elif E_list[i][1] in connected_component:
                    if E_list[i][0] in connected_component:
                        remove_edges.add(E_list[i])
                        E_list.remove(E_list[i])
                    else:
                        connected_component.add(E_list[i][0])
                        mst.append(E_list[i])
                        E_list.remove(E_list[i])
                        change = True
                        break
                else:
                    i += 1
        if not change:
            if len(E_list) != 0:
                connected_component.add(E_list[0][0])
                connected_component.add(E_list[0][1])
                mst.append(E_list[0])
                E_list.remove(E_list[0])
    for edge in E_list:
        remove_edges.add(edge)
    if len(mst) != len(vertices) - 1 or len(connected_component) != len(vertices):
        print("Warning: partition cannot have a full MST! Missing edges to create full MST.")
        # print("Error: MST found cannot be correct \n Length mst: ", len(mst), "\n Total connected vertices: ",
        #       len(connected_component), "\n Number of vertices: ", len(vertices))
    return mst, remove_edges


def reduce_edges(vertices, E, c, epsilon):
    """
    Uses PySpark to distribute the computation of the MSTs,
    Randomly partition the vertices twice in k subsets (U = {u_1, u_2, .., u_k}, V = {v_1, v_2, .., v_k})
    For every intersection between U_i and V_j, create the subgraph and find the MST in this graph
    Remove all edges from E that are not part of the MST in the subgraph
    :param vertices: vertices in the graph
    :param E: edges of the graph
    :param c: constant
    :param epsilon:
    :return:The reduced number of edges
    """
    conf = SparkConf().setAppName('MST_Algorithm')
    sc = SparkContext.getOrCreate(conf=conf)

    n = len(vertices)
    k = math.ceil(n ** ((c - epsilon) / 2))

    E_list = [(i, j, d) for i, neighbors in E.items() for j, d in neighbors.items()]

    E_list_partion = partion_edges(E_list, k)

    rddE_list = sc.parallelize(E_list_partion).map(lambda x: find_mst(x))
    both = rddE_list.collect()

    mst = []
    removed_edges = set()
    for i in range(len(both)):
        mst.append(both[i][0])
        for edge in both[i][1]:
            removed_edges.add(edge)

    sc.stop()
    return mst, removed_edges


def remove_edges(E, removed_edges):
    """
    Removes the edges, which are removed when generating msts
    :param E: current edges
    :param removed_edges: edges to be removed
    :param msts: edges in the msts
    :return: return the updated edge dict
    """
    for edge in removed_edges:
        if edge[1] in E[edge[0]]:
            del E[edge[0]][edge[1]]
    return E


def create_mst(V, E, epsilon, size,
               vertex_coordinates, plot_itermediate=False):
    """
    Creates the mst of the graph G = (V, E).
    As long as the number of edges is greater than n ^(1 + epsilon), the number of edges is reduced
    Then the edges that needs to be removed are removed from E and the size is updated.
    :param V: Vertices
    :param E: edges
    :param epsilon:
    :param m: number of machines
    :param size: number of edges
    :return: returns the reduced graph with at most np.power(n, 1 + epsilon) edges
    """
    n = len(V)
    c = math.log(size / n, n)

    while size > np.power(n, 1 + epsilon):
        print("C: ", c)
        mst, removed_edges = reduce_edges(V, E, c, epsilon)
        if plot_itermediate:
            plot_mst(vertex_coordinates, mst, True, False)
        E = remove_edges(E, removed_edges)
        print("Total edges removed in this iteration", len(removed_edges))
        size = size - len(removed_edges)
        print("New total of edges: ", size)
        c = (c - epsilon) / 2

    # Now the number of edges is reduced and can be moved to a single machine
    # V = set(range(n))

    """items = E.items()  # returns [(x, {y : 1})]
    edges = []
    for item in items:
        items2 = item[1].items()
        for item2 in items2:
            edges.append((item[0], item2[0], item2[1]))"""

    E_list = [(i, j, d) for i, neighbors in E.items() for j, d in neighbors.items()]
    mst, removed_edges = find_mst(E_list)

    return mst


def plot_mst(vertices, mst, intermediate=False, plot_cluster=False):
    x = []
    y = []
    c = []
    area = []
    colors = ["g", "b", "r", "c", "m", "y", "k", "darkorange", "dodgerblue", "deeppink", "khaki", "purple",
              "springgreen", "tomato", "slategray"]
    for i in range(len(vertices)):
        x.append(float(vertices[i][0]))
        y.append(float(vertices[i][1]))
        area.append(0.1)
        c.append("black")
    plt.scatter(x, y, c=c, s=area)
    if intermediate:
        cnt = 0
        for m in mst:
            for i in range(len(m)):
                linex = [float(x[int(m[i][0])])]
                liney = [float(y[int(m[i][0])])]
                linex.append(float(x[int(m[i][1])]))
                liney.append(float(y[int(m[i][1])]))
                plt.plot(linex, liney, colors[cnt])
            cnt = (cnt + 1) % len(colors)
    else:
        # TODO
        if plot_cluster:
            edges = sorted(mst, key=get_key, reverse=True)
            total_length = 0
            for edge in edges:
                total_length += edge[2]
            average_length = total_length / len(edges)
            print(average_length)
            print(edges[0], edges[1], edges[2], edges[3])
            print("Not yet implemented")
        else:
            for i in range(len(mst)):
                linex = [float(x[int(mst[i][0])])]
                liney = [float(y[int(mst[i][0])])]
                linex.append(float(x[int(mst[i][1])]))
                liney.append(float(y[int(mst[i][1])]))
                plt.plot(linex, liney)
    plt.show()
    return


def filter_edges_random(E, pc, between_cluster):
    e_filtered = dict(E)
    edges_to_remove = list()

    for edge_list_key in E:
        edge_list_per_vertex = E.get(edge_list_key)
        for edge in edge_list_per_vertex:
            edge_data = edge_list_per_vertex.get(edge)
            random_number_gen = random.randrange(0, 100)
            if between_cluster and random_number_gen < pc and edge_data[1] == 1:
                edges_to_remove.append((edge_list_key, edge, edge_data))
                print("Only removing edges in between clusters")
            elif (not between_cluster) and random_number_gen < pc:
                print("Removing all edges randomly")
                edges_to_remove.append((edge_list_key, edge, edge_data))

    for edge in edges_to_remove:
        e_filtered.get(edge[0]).pop(edge[1])

    return e_filtered


def main():
    """
    For every dataset, it creates the mst and plots the clustering
    """
    parser = ArgumentParser()
    parser.add_argument('--test', help="Used for smaller dataset and testing", action="store_true")
    parser.add_argument('--epsilon', help="epsilon [default=1/8]", type=float, default=1 / 8)
    parser.add_argument('--machines', help="Number of machines [default=1]", type=int, default=1)
    # parser.add_argument('--machines', help="Number of machines [default=1]", type=int, default=1)
    args = parser.parse_args()

    print("Start generating MST")
    if args.test:
        print("Test argument given")

    start_time = datetime.now()
    print("Starting time:", start_time)

    datasets = get_clustering_data()
    for dataset in datasets:
        timestamp = datetime.now()
        print("Start creating Distance Matrix...")
        dm, E, size, vertex_coordinates = create_distance_matrix(dataset[0])

        # print(E)
        print("Start sampling of edges")
        filter_edges_random(E, 10, False)
        """TODO: E processen aan de hand van wat we hebben besproken
        Ofwel, edges eruit flikkeren met een bepaald percentage,
        en dan een nieuwe E teruggeven die niet meer de 0 of 1
        bevat (geeft aan of ze in zelfde cluster zitten)

        De oude data structure ziet eruit als volgt, en dit is 
        wat we ook weer willen hebben om in de rest van de
        functies te stoppen: (elke edge komt maar één keer voor)
        E =
        {0: {1: 0.8, 2: 1.4, ...},
         1: {2: 0.7, 3: ...},
         ...}
        
        Dit is de nieuwe data structure, of wel ipv de afstand,
        een tuple met zowel de afstand als een indicator of ze
        tot dezelfde cluster behoren
        E =
        {0: {1: (0.8, 1), 2: (1.4, 0), ...},
         1: {2: (0.7, 1), 3: ...},
         ...}"""

        V = list(range(len(dm)))
        print("Size dataset: ", len(dm))
        print("Created distance matrix in: ", datetime.now() - timestamp)
        print("Start creating MST...")
        timestamp = datetime.now()
        mst = create_mst(V, E, epsilon=args.epsilon, size=size,
                         vertex_coordinates=vertex_coordinates)
        print("Found MST in: ", datetime.now() - timestamp)
        print("Start creating plot of MST...")
        timestamp = datetime.now()
        plot_mst(dataset[0][0], mst, False, False)
        print("Created plot of MST in: ", datetime.now() - timestamp)

    print("Done...")


if __name__ == '__main__':
    # Initial call to main function
    main()
