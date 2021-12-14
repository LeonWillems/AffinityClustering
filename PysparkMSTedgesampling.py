import math
from argparse import ArgumentParser
from datetime import datetime
import csv

import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.spatial

from sklearn.datasets import make_circles, make_moons, make_blobs

from pyspark import RDD, SparkConf, SparkContext



def get_clustering_data():
    n_samples = [150, 500, 1500]
    noise_config = [0.0, 1.0, 'lines']
    n_noise_points = 20
    noise_clusters = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    datasets = []

    for n_sample in n_samples:
        for noise in noise_config:
            settings = ['blobs', n_sample, noise]

            if noise != 'lines':
                cluster_stds = [std * (1 + noise) for std in [1.0, 2.0]]
                blob_data = make_blobs(n_samples=n_sample,
                                       cluster_std=cluster_stds,
                                       centers=2)

                settings = ['blobs', n_sample, noise]
                datasets.append([blob_data, settings])

            if noise == 'lines':
                blob_data = make_blobs(n_samples=n_sample,
                                       cluster_std=[1.0, 2.0],
                                       centers=2)

                y_mean, x_min, x_max = blob_data[0][:, 1].mean(), blob_data[0][:, 0].min(), blob_data[0][:, 0].max()

                noise_points = [[x_min + k * (x_max - x_min) / n_noise_points, y_mean] for k in
                                range(n_noise_points)]

                blob_data = (np.append(blob_data[0], noise_points, axis=0),
                             np.append(blob_data[1], noise_clusters, axis=0))

                datasets.append([blob_data, settings])

    for n_sample in n_samples:
        for noise in noise_config:
            circle_settings = ['circles', n_sample, noise]
            moon_settings = ['moons', n_sample, noise]

            if noise != 'lines':
                circle_data = make_circles(n_samples=n_sample,
                                           noise=noise)

                datasets.append([circle_data, circle_settings])

                moon_data = make_moons(n_samples=n_sample,
                                       noise=noise)

                datasets.append([moon_data, moon_settings])

            if noise == 'lines':
                circle_data = make_circles(n_samples=n_sample,
                                           noise=0.0)
                moon_data = make_moons(n_samples=n_sample,
                                       noise=0.0)

                circle_y_mean, circle_x_min, circle_x_max = \
                    circle_data[0][:, 1].mean(), circle_data[0][:, 0].min(), circle_data[0][:, 0].max()

                moon_y_mean, moon_x_min, moon_x_max = \
                    moon_data[0][:, 1].mean(), moon_data[0][:, 0].min(), moon_data[0][:, 0].max()

                circle_noise_points = [
                    [circle_x_min + k * (circle_x_max - circle_x_min) / n_noise_points, circle_y_mean]
                    for k in range(n_noise_points)]
                moon_noise_points = [[moon_x_min + k * (moon_x_max - moon_x_min) / n_noise_points, moon_y_mean]
                                     for k in range(n_noise_points)]

                circle_data = (np.append(circle_data[0], circle_noise_points, axis=0),
                               np.append(circle_data[1], noise_clusters, axis=0))
                moon_data = (np.append(moon_data[0], moon_noise_points, axis=0),
                             np.append(moon_data[1], noise_clusters, axis=0))

                datasets.append([circle_data, circle_settings])
                datasets.append([moon_data, moon_settings])

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
    print('len d_matrix:', len(d_matrix))
    dict = {}

    actual_clustering = []
    for k in range(len(set(dataset[1]))):
        actual_clustering.append([])

    # Run with less edges
    for i in range(len(d_matrix)):
        actual_clustering[dataset[1][i]].append(i)
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
    return d_matrix, dict, size, vertices, actual_clustering


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
    #print(E_list)
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


def plot_mst(vertices, mst, plot_cluster=True, num_clusters=2):
    """
    Plots the mst found
    :param vertices: vertex coordinates
    :param mst: minimal spanning tree to be plotted
    :param intermediate: bool to indicate if intermediate results should be plotted
    :param plot_cluster: bool to indicate if the clusters should have the same color
    :param num_clusters: number of clusters
    :return: nothing
    """
    x = []
    y = []
    z = []
    c = []
    area = []
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'darkorange', 'dodgerblue', 'deeppink', 'khaki', 'purple',
              'springgreen', 'tomato', 'slategray']

    for i in range(len(vertices)):
        x.append(float(vertices[i][0]))
        y.append(float(vertices[i][1]))
        if len(vertices[i]) == 3:
            z.append(vertices[i][2])
        area.append(0.1)
        c.append('k')


    if plot_cluster:
        edges = sorted(mst, key=get_key, reverse=True)
        removed_edges = []
        clusters = []
        for i in range(num_clusters - 1):
            edge = edges.pop(0)

            removed_edges.append(edge)
            clusters.append([edge[0]])
            clusters.append([edge[1]])
            linex = [float(x[edge[0]]), float(x[edge[1]])]
            liney = [float(y[edge[0]]), float(y[edge[1]])]
            """if len(z) > 0:
                linez = [float(z[edge[0]]), float(z[edge[1]])]
                #ax.plot(linex, liney, linez, "k")
            else:
                #plt.plot(linex, liney, "k")"""

        dict_edges = dict()
        for edge in edges:
            if edge[0] in dict_edges:
                dict_edges[edge[0]].append(edge[1])
            else:
                dict_edges[edge[0]] = [edge[1]]

        i = 0
        while i < len(clusters):
            pop = False
            for j in range(i):
                if clusters[i][0] in clusters[j]:
                    clusters.pop(i)
                    pop = True
                    break
            if pop:
                continue

            todo = []
            for j in range(clusters[i][0]):
                if j in dict_edges:
                    if clusters[i][0] in dict_edges[j]:
                        clusters[i].append(j)
                        todo.append(j)
            if clusters[i][0] in dict_edges:
                for j in range(len(dict_edges[clusters[i][0]])):
                    todo.append(dict_edges[clusters[i][0]][j])
                    clusters[i].append(dict_edges[clusters[i][0]][j])

            while len(todo) > 0:
                first = todo.pop()
                for k in range(first):
                    if k in dict_edges:
                        if first in dict_edges[k] and k not in clusters[i]:
                            clusters[i].append(k)
                            todo.append(k)
                if first in dict_edges:
                    for k in range(len(dict_edges[first])):
                        if dict_edges[first][k] not in clusters[i]:
                            clusters[i].append(dict_edges[first][k])
                            todo.append(dict_edges[first][k])
            i += 1
        for i in range(len(clusters)):
            clusters[i] = sorted(clusters[i])

        x_cluster = []
        y_cluster = []
        z_cluster = []
        c_cluster = []
        area_cluster = []

        for i in range(len(clusters)):
            for vertex in clusters[i]:
                x_cluster.append(float(vertices[vertex][0]))
                y_cluster.append(float(vertices[vertex][1]))
                if len(z) > 0:
                    z_cluster.append(float(vertices[vertex][2]))
                area_cluster.append(0.2)
                c_cluster.append(colors[i])
        if len(z_cluster) > 0:
            ax.scatter3D(x_cluster, y_cluster, z_cluster, c=c_cluster, s=area_cluster)
        else:
            plt.scatter(x_cluster, y_cluster, c=c_cluster, s=area_cluster)

        linez = []
        for i in range(len(mst)):
            if mst[i] in removed_edges:
                continue
            linex = [float(x[int(mst[i][0])]), float(x[int(mst[i][1])])]
            liney = [float(y[int(mst[i][0])]), float(y[int(mst[i][1])])]
            if len(z) > 0:
                linez = [float(z[int(mst[i][0])]), float(z[int(mst[i][1])])]
            """for j in range(len(clusters)):
                if mst[i][0] in clusters[j]:
                    if len(z) > 0:
                        #ax.plot3D(linex, liney, linez, c=colors[j])
                    else:
                        #plt.plot(linex, liney, c=colors[j])"""
    #plt.show()

    return clusters


def filter_edges_random(E, pc, between_cluster, size):
    e_filtered = dict(E)
    edges_to_remove = list()

    for edge_list_key in E:
        edge_list_per_vertex = E.get(edge_list_key)
        for edge in edge_list_per_vertex:
            edge_data = edge_list_per_vertex.get(edge)
            random_number_gen = random.randrange(0, 100)
            if between_cluster and random_number_gen < pc and edge_data[1] == 1:
                edges_to_remove.append((edge_list_key, edge, edge_data))
                #print("Only removing edges in between clusters")
            elif (not between_cluster) and random_number_gen < pc:
                #print("Removing all edges randomly")
                edges_to_remove.append((edge_list_key, edge, edge_data))

    for edge in edges_to_remove:
        e_filtered.get(edge[0]).pop(edge[1])
        size -= 1

    return e_filtered, size


def main(percentage, between_cluster, i):
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
    print('n_datasets:', len(datasets))

    dataset_counter = 1
    for dataset in datasets:
        print(f'==================== Currently at configuration {(i-1)*27 + dataset_counter} of 270 ====================')
        timestamp = datetime.now()
        print("Start creating Distance Matrix...")
        dm, E, size, vertex_coordinates, actual_clustering = create_distance_matrix(dataset[0])

        print("Start sampling of edges")
        E_filtered, size = filter_edges_random(E, percentage, between_cluster, size)

        E = {node: {neighbor: tup[0] for neighbor, tup in neighbors.items()}
             for node, neighbors in E_filtered.items() if len(neighbors)>0}

        total_vertices = len(dm)
        V = list(range(total_vertices))
        print("Size dataset: ", len(dm))
        print("Created distance matrix in: ", datetime.now() - timestamp)
        print("Start creating MST...")
        timestamp = datetime.now()
        mst = create_mst(V, E, epsilon=args.epsilon, size=size,
                         vertex_coordinates=vertex_coordinates)

        MST_found_in = datetime.now() - timestamp
        print("Found MST in: ", MST_found_in.total_seconds())
        print("Start creating plot of MST...")
        timestamp = datetime.now()

        predicted_clusters = plot_mst(dataset[0][0], mst, True)
        print('actual clusters:', len(actual_clustering))
        print('predic clusters:', len(predicted_clusters))

        print("Created plot of MST in: ", datetime.now() - timestamp)

        substraction = 0
        if total_vertices in [170, 520, 1520]:
            substraction = 20

        corrects = 0
        for cluster in range(2):
            for vertex in predicted_clusters[cluster][:int(total_vertices/2)-int(substraction/2)]:
                if vertex in actual_clustering[cluster]:
                    corrects += 1

        if corrects < (total_vertices-substraction)/2:
            corrects = total_vertices-substraction-corrects

        perc_correct = corrects/(total_vertices-substraction)
        print(f'{corrects} corrects out of {total_vertices-substraction} total, {perc_correct}')
        print()

        with open('experiment_results.csv', 'a', newline='\n') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            data_to_write = dataset[1] + [percentage, int(between_cluster),
                                          perc_correct, MST_found_in.total_seconds()]
            csv_writer.writerow(data_to_write)

        dataset_counter += 1



    print("Done...")


if __name__ == '__main__':
    # Initial call to main function
    main(0, False, 1)
