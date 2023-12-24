import time

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import Signal, QObject

from MyGraph import *
import copy
class Solution:
    pass

class Kruskal(Solution,QObject):
    def __init__(self, rawGraph):
        super().__init__()
        # self.ax=ax
        self.rawGraph = rawGraph
        self.newGraph = MyGraph()
        # plt.figure(1)
        # self.rawGraph.draw()
        self.nodesList =list(self.rawGraph.nxGraph.nodes)

        for i in self.nodesList:
            self.newGraph.nxGraph.add_node(i)


        self.newGraph.update_graph_properties()
        self.newGraph.pos = self.rawGraph.pos
        # plt.figure(2)
        # self.newGraph.draw()

        tempDict= sorted(nx.get_edge_attributes(self.rawGraph.nxGraph, 'weight').items(),
                 key=lambda item: item[1])
        self.SortedEdgesList = [item[0] for item in tempDict]
        self.SortedWeightsList = [item[1] for item in tempDict]
        self.TotalWeight = 0
        self.SingleCount = 0

    def addEdge(self,currentEdge,currentWeight,ax):
        self.newGraph.nxGraph.add_edge(u_of_edge=currentEdge[0],v_of_edge=currentEdge[1],weight=currentWeight)
        index = list(self.newGraph.nxGraph.edges).index(currentEdge)
        self.newGraph.EdgeColorList.insert(index,self.newGraph.EdgeColorSelected)
        self.newGraph.EdgeWidthList.append(self.newGraph.EdgeWidthDefault)
        self.newGraph.draw(ax)
        time.sleep(self.newGraph.DrawTimeWait)
        return index

    def removeEdge(self,currentEdge,index,currentEdgeWeight,ax):
        # self.rawGraph.EdgeColorList[index] = self.rawGraph.EdgeColorWrong
        self.newGraph.nxGraph.remove_edge(u=currentEdge[0],v=currentEdge[1])
        self.TotalWeight -= currentEdgeWeight
        del self.newGraph.EdgeColorList[index]
        del self.newGraph.EdgeWidthList[index]
        self.newGraph.draw(ax)
        time.sleep(self.newGraph.DrawTimeWait)

    def step(self,currentWeight,ax):
        for e in list(self.newGraph.nxGraph.edges):
            self.newGraph.NodeColorList[e[0]] = self.newGraph.NodeColorDone
            self.newGraph.NodeColorList[e[1]] = self.newGraph.NodeColorDone
        self.TotalWeight += currentWeight
        self.newGraph.EdgeColorList = [self.newGraph.EdgeColorDone]*len(self.newGraph.EdgeColorList)
        self.newGraph.draw(ax)
        time.sleep(self.newGraph.DrawTimeWait)

    def runAlgorithm(self):
        self.rawGraph.NodeColorList = self.newGraph.NodeColorList
        # time.sleep(self.newGraph.DrawTimeWait)
        for currentEdge in self.SortedEdgesList:
            if nx.is_connected(self.newGraph.nxGraph) is True:
                break
            if (len(self.newGraph.nxGraph.edges) == self.rawGraph.NodesNum-1):
                break
            currentEdgeWeight = self.SortedWeightsList[self.SortedEdgesList.index(currentEdge)]
            index = self.addEdge(currentEdge, currentEdgeWeight)
            if len(nx.cycle_basis(self.newGraph.nxGraph)) > 0:

                self.removeEdge(currentEdge,index,currentEdgeWeight)
                continue
            self.step(currentEdgeWeight)
        # plt.show()

    def SingleStepRun(self,ax1,ax2):
        self.rawGraph.draw(ax1)

        if nx.is_connected(self.newGraph.nxGraph) is True:
            self.rawGraph.draw(ax1)
            self.newGraph.draw(ax2)
            return False
        if (len(self.newGraph.nxGraph.edges) == self.rawGraph.NodesNum-1):
            return False
        currentEdge = self.SortedEdgesList[self.SingleCount]
        currentEdgeWeight = self.SortedWeightsList[self.SingleCount]
        index = self.addEdge(currentEdge, currentEdgeWeight,ax=ax2)
        # time.sleep(self.newGraph.DrawTimeWait)
        if len(nx.cycle_basis(self.newGraph.nxGraph)) > 0:
            index_2 = list(self.rawGraph.nxGraph.edges).index(currentEdge)
            self.rawGraph.EdgeColorList[index_2] = self.rawGraph.EdgeColorWrong
            self.rawGraph.draw(ax1)
            self.removeEdge(currentEdge,index,currentEdgeWeight,ax=ax2)
        self.step(currentEdgeWeight,ax=ax2)
        self.SingleCount+=1
        # plt.show()
        return True

    def resetNewGraph(self):
        self.__init__(self.rawGraph)
class Prim(Solution):
    def __init__(self, rawGraph):
        super().__init__()
        self.rawGraph = rawGraph
        self.newGraph = MyGraph()
        # plt.figure(1)
        # self.rawGraph.draw()
        self.nodesList = list(self.rawGraph.nxGraph.nodes)

        for i in self.nodesList:
            self.newGraph.nxGraph.add_node(i)

        self.newGraph.update_graph_properties()
        self.newGraph.pos = self.rawGraph.pos
        # plt.figure(2)
        # self.newGraph.draw()
        self.weightsDict = dict(
            sorted(nx.get_edge_attributes(self.rawGraph.nxGraph, 'weight').items(), key=lambda item: item[1]))
        self.TotalWeight = 0
        self.select_nodes = set()

    def changeEdgeColor(self,edge):
        pass

    def runAlgorithm(self):
        start_node = list(self.newGraph.nxGraph.nodes())[0]
        # 存储已经选择的顶点
        self.select_nodes.add(start_node)
        self.newGraph.NodeColorList[start_node] = self.newGraph.NodeColorDone
        self.rawGraph.NodeColorList = self.newGraph.NodeColorList
        time.sleep(self.newGraph.DrawTimeWait)
        while nx.is_connected(self.newGraph.nxGraph) is False:
            # 获取所有可达的边
            reachableEdge = list(self.rawGraph.nxGraph.edges(self.select_nodes))
            for i in range(len(reachableEdge)):
                if reachableEdge[i][0]>reachableEdge[i][1]:
                    reachableEdge[i]=reachableEdge[i][1],reachableEdge[i][0]

            # 将所有可达的边在原图上变色
            reachableEdgeIndices = [list(self.rawGraph.nxGraph.edges).index(element) for element in reachableEdge]
            for i in reachableEdgeIndices:
                self.rawGraph.EdgeColorList[i] = self.rawGraph.EdgeColorSelected
            self.rawGraph.draw()
            time.sleep(self.newGraph.DrawTimeWait)

            # 获取权值最小边
            currentEdge,currentWeight = (0,0), 101
            for key, value in self.weightsDict.items():
                if value < currentWeight :
                    if ((key[0] in self.select_nodes) and (key[1] not in self.select_nodes)) or ((key[0] not in self.select_nodes) and (key[1] in self.select_nodes)):
                        currentEdge = key
                        currentWeight = value

            # 向已选集中加入顶点，并且向newGraph加入边，变色并且渲染
            self.select_nodes.add(currentEdge[0])
            self.select_nodes.add(currentEdge[1])
            self.newGraph.nxGraph.add_edge(currentEdge[0],currentEdge[1],weight = currentWeight)
            self.newGraph.EdgeColorList.append(self.newGraph.EdgeColorDone)
            self.newGraph.EdgeWidthList.append(self.newGraph.EdgeWidthDefault)
            self.TotalWeight +=currentWeight
            for i in self.select_nodes:
                self.newGraph.NodeColorList[i] = self.newGraph.NodeColorDone
            self.newGraph.draw()
            time.sleep(self.newGraph.DrawTimeWait)
        plt.show()

    def SingleStepRun(self,ax_1,ax_2):
        if nx.is_connected(self.newGraph.nxGraph):
            self.rawGraph.draw(ax_1)
            self.newGraph.draw(ax_2)
            return True
        self.rawGraph.draw(ax_1)
        plt.pause(self.newGraph.DrawTimeWait)
        start_node = list(self.newGraph.nxGraph.nodes())[0]
        self.select_nodes.add(start_node)
        # 获取所有可达的边
        reachableEdge = list(self.rawGraph.nxGraph.edges(self.select_nodes))
        for i in range(len(reachableEdge)):
            if reachableEdge[i][0] > reachableEdge[i][1]:
                reachableEdge[i] = reachableEdge[i][1], reachableEdge[i][0]

        # 将所有可达的边在原图上变色
        reachableEdgeIndices = [list(self.rawGraph.nxGraph.edges).index(element) for element in reachableEdge]
        for i in reachableEdgeIndices:
            self.rawGraph.EdgeColorList[i] = self.rawGraph.EdgeColorSelected
        self.rawGraph.draw(ax_1)
        plt.pause(self.newGraph.DrawTimeWait)

        # 获取权值最小边
        currentEdge, currentWeight = (0, 0), 101
        for key, value in self.weightsDict.items():
            if value < currentWeight:
                if ((key[0] in self.select_nodes) and (key[1] not in self.select_nodes)) or (
                        (key[0] not in self.select_nodes) and (key[1] in self.select_nodes)):
                    currentEdge = key
                    currentWeight = value

        # 向已选集中加入顶点，并且向newGraph加入边，变色并且渲染
        self.select_nodes.add(currentEdge[0])
        self.select_nodes.add(currentEdge[1])
        self.newGraph.nxGraph.add_edge(currentEdge[0], currentEdge[1], weight=currentWeight)
        self.newGraph.EdgeColorList.append(self.newGraph.EdgeColorDone)
        self.newGraph.EdgeWidthList.append(self.newGraph.EdgeWidthDefault)
        self.TotalWeight += currentWeight
        for i in self.select_nodes:
            self.newGraph.NodeColorList[i] = self.newGraph.NodeColorDone
        self.newGraph.draw(ax_2)
        plt.pause(self.newGraph.DrawTimeWait)

    def resetNewGraph(self):
        self.__init__(self.rawGraph)


class CycleBreaking(Solution):
    def __init__(self, rawGraph):
        self.rawGraph = rawGraph
        self.rawGraph.NodeColorList = [self.rawGraph.NodeColorDone] * len(self.rawGraph.nxGraph.nodes)
        self.newGraph = copy.deepcopy(rawGraph)
        # self.newGraph.draw()
        self.TotalWeight = sum(nx.get_edge_attributes(self.rawGraph.nxGraph, 'weight').values())

    def removeEdge(self, currentEdge, weight,ax):
        print(currentEdge)
        index = list(self.newGraph.nxGraph.edges).index(currentEdge)
        self.newGraph.EdgeColorList[index] = self.newGraph.EdgeColorWrong
        self.newGraph.draw(ax)
        # time.sleep(self.newGraph.DrawTimeWait)

        self.newGraph.nxGraph.remove_edge(u=currentEdge[0], v=currentEdge[1])
        print(self.newGraph.nxGraph.edges)


        del self.newGraph.EdgeColorList[index]
        del self.newGraph.EdgeWidthList[index]
        self.TotalWeight -= weight
        self.newGraph.draw(ax)
        # time.sleep(self.newGraph.DrawTimeWait)

    def SingleStepRun(self, ax):
        self.rawGraph.NodeColorList = self.newGraph.NodeColorList
        # time.sleep(self.newGraph.DrawTimeWait)

        if len(nx.cycle_basis(self.newGraph.nxGraph)) <=0:
            self.newGraph.draw(ax)
            return True
        cycle_edges = nx.find_cycle(self.newGraph.nxGraph)
        print(cycle_edges)
        max_weight_edge = max(cycle_edges, key=lambda e: self.newGraph.nxGraph.get_edge_data(e[0], e[1])['weight'])
        if (max_weight_edge[0] > max_weight_edge[1]):
            max_weight_edge = max_weight_edge[1], max_weight_edge[0]
        max_weight = self.newGraph.nxGraph.get_edge_data(max_weight_edge[0], max_weight_edge[1])['weight']
        print(max_weight_edge)
        self.removeEdge(max_weight_edge, max_weight,ax)
        return False
        # plt.show()
    def resetNewGraph(self):
        self.__init__(self.rawGraph)

class Dijkstra(Solution):
    def __init__(self, rawGraph):
        super().__init__()
        self.rawGraph = rawGraph
        self.newGraph = MyGraph()
        # plt.figure(1)
        # self.rawGraph.draw()
        self.nodesList = list(self.rawGraph.nxGraph.nodes)

        for i in self.nodesList:
            self.newGraph.nxGraph.add_node(i)

        self.newGraph.update_graph_properties()
        self.newGraph.pos = self.rawGraph.pos
        # plt.figure(2)
        # self.newGraph.draw()
        self.weightsDict = dict(
            sorted(nx.get_edge_attributes(self.rawGraph.nxGraph, 'weight').items(), key=lambda item: item[1]))
        self.TotalWeight = 0
        self.parentList = [None] * len(self.nodesList)
        self.disList = [np.infty] * len(self.nodesList)

        self.currentNode = None
        self.selected_nodes = set()

    def SetStartNode(self,start_node):
        self.currentNode = start_node
        self.disList[start_node] = 0
        self.selected_nodes.add(self.currentNode)

    def SingleStepRun(self,ax1,ax2):
        if nx.is_connected(self.newGraph.nxGraph):
            self.rawGraph.draw(ax1)
            self.newGraph.draw(ax2)
            return True

        tempEdge = (-1, -1)
        tempWeight = np.infty
        # 获取所有可达的边
        reachableEdge = list(self.rawGraph.nxGraph.edges(self.selected_nodes))
        for i in range(len(reachableEdge)):
            if reachableEdge[i][0] > reachableEdge[i][1]:
                reachableEdge[i] = reachableEdge[i][1], reachableEdge[i][0]
        # 将所有可达的边在原图上变色
        reachableEdgeIndices = [list(self.rawGraph.nxGraph.edges).index(element) for element in reachableEdge]
        for i in reachableEdgeIndices:
            self.rawGraph.EdgeColorList[i] = self.rawGraph.EdgeColorSelected
        self.rawGraph.draw(ax1)
        time.sleep(self.newGraph.DrawTimeWait)
        for key, value in self.weightsDict.items():
            if (key[0] in self.selected_nodes) and (key[1] not in self.selected_nodes):
                if value + self.disList[key[0]] < self.disList[key[1]]:
                    self.disList[key[1]] = value + self.disList[key[0]]
                    self.parentList[key[1]] = key[0]
            elif (key[1] in self.selected_nodes) and (key[0] not in self.selected_nodes):
                if value + self.disList[key[1]] < self.disList[key[0]]:
                    self.disList[key[0]] = value + self.disList[key[1]]
                    self.parentList[key[0]] = key[1]

        for e in range(len(self.disList)):
            if (e not in self.selected_nodes) and self.disList[e] < tempWeight:
                tempEdge = (min(e, self.parentList[e]), max(e, self.parentList[e]))
                print(tempEdge)
                tempWeight = self.disList[e]

        if tempEdge[0] in self.selected_nodes:
            self.currentNode = tempEdge[1]
        else:
            self.currentNode = tempEdge[0]

        print(tempEdge[0])
        if tempEdge[0] != -1:

            # 向已选集中加入顶点，并且向newGraph加入边，变色并且渲染
            self.selected_nodes.add(tempEdge[0])
            self.selected_nodes.add(tempEdge[1])
            self.newGraph.nxGraph.add_edge(tempEdge[0], tempEdge[1], weight=tempWeight)
            self.newGraph.EdgeColorList.append(self.newGraph.EdgeColorDone)
            self.newGraph.EdgeWidthList.append(self.newGraph.EdgeWidthDefault)
            self.TotalWeight += tempWeight
            for i in self.selected_nodes:
                self.newGraph.NodeColorList[i] = self.newGraph.NodeColorDone
            self.newGraph.draw(ax2)
            time.sleep(self.newGraph.DrawTimeWait)
        else:
            self.newGraph.draw(ax2)
            time.sleep(self.newGraph.DrawTimeWait)
        return False
    def drawlast(self,ax1,ax2):
        self.rawGraph.draw(ax1)
        self.newGraph.draw(ax2)
    def resetNewGraph(self):
        self.__init__(self.rawGraph)
class Floyd(Solution):
    def __init__(self, rawGraph):
        super().__init__()
        self.rawGraph = rawGraph
        # self.rawGraph.draw()
        time.sleep(self.rawGraph.DrawTimeWait)
        self.nodesList = list(self.rawGraph.nxGraph.nodes)
        self.TotalWeights = None

    def runAlgorithm(self):
        # 获取节点数
        num_nodes = self.rawGraph.NodesNum

        # 初始化距离矩阵
        self.distance_matrix = np.copy(nx.to_numpy_matrix(self.rawGraph.nxGraph))
        print(self.distance_matrix)
        self.distance_matrix[self.distance_matrix == 0] = np.infty
        print(self.distance_matrix)
        np.fill_diagonal(self.distance_matrix, 0)
        print(self.distance_matrix)


        # 迭代更新距离矩阵
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    self.distance_matrix[i][j] = min(self.distance_matrix[i][j], self.distance_matrix[i][k] + self.distance_matrix[k][j])

        return self.distance_matrix

    def draw_result(self,ax):
        self.newGraph = DiGraph(self.distance_matrix)
        self.newGraph.update_graph_properties()
        self.newGraph.pos = self.rawGraph.pos
        self.newGraph.draw(ax)

class FloydWarshall(Solution):
    def __init__(self, rawGraph):
        super().__init__()
        self.rawGraph = rawGraph
        # self.rawGraph.draw()
        time.sleep(self.rawGraph.DrawTimeWait)
        self.nodesList = list(self.rawGraph.nxGraph.nodes)
        self.TotalWeights = None
        self.roads = []
    def runAlgorithm(self,ax):
        # 获取节点数
        num_nodes = self.rawGraph.NodesNum

        # 初始化距离矩阵和路径矩阵
        self.distance_matrix = np.copy(nx.to_numpy_matrix(self.rawGraph.nxGraph))
        print(self.distance_matrix)
        self.distance_matrix[self.distance_matrix==0] = np.infty
        print(self.distance_matrix)
        np.fill_diagonal(self.distance_matrix,0)
        print(self.distance_matrix)
        path_matrix = np.full((num_nodes, num_nodes), fill_value=np.nan)

        # 设置对角线上的元素为0
        np.fill_diagonal(self.distance_matrix, 0)

        # 初始化路径矩阵，对角线上的元素为节点本身
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and self.distance_matrix[i, j] != np.inf:
                    path_matrix[i, j] = i

        # Floyd-Warshall 算法核心部分
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if self.distance_matrix[i, k] + self.distance_matrix[k, j] < self.distance_matrix[i, j]:
                        self.distance_matrix[i, j] = self.distance_matrix[i, k] + self.distance_matrix[k, j]
                        path_matrix[i, j] = path_matrix[k, j]

        # 输出最短路径矩阵
        print("\n最短路径矩阵:")
        print(self.distance_matrix)
        self.draw_min_road(ax)

        # 输出最短路径
        print("\n最短路径:")
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and not np.isnan(path_matrix[i, j]):
                    path = [j]
                    while path[-1] != i:
                        path.append(int(path_matrix[i, path[-1]]))
                    path.reverse()
                    print(f"最短路径 {i} 到 {j}: {path}, 距离: {self.distance_matrix[i, j]}")
                    road = '最短路径 ' + str(i) + ' 到 ' + str(j) + ': ' + str(path) + ', 距离: ' + str(
                        self.distance_matrix[i, j])
                    self.roads.append(road)
    def draw_min_road(self,ax):
        self.newGraph = DiGraph(self.distance_matrix)
        self.newGraph.update_graph_properties()
        self.newGraph.pos = self.rawGraph.pos
        self.newGraph.draw(ax)
    def get_road_data(self):
        print(self.roads)
        return self.roads
class Hungarian(Solution):
    def __init__(self, rawGraph,X_num,Y_num):
        self.rawGraph = rawGraph
        self.newGraph = MyGraph()
        self.xNum = X_num
        self.yNum = Y_num
        self.nodesList = list(self.rawGraph.nxGraph.nodes)
        for i in self.nodesList:
            self.newGraph.nxGraph.add_node(i)
        self.newGraph.update_graph_properties()
        self.rawGraph.pos = nx.bipartite_layout(self.rawGraph.nxGraph,list(self.rawGraph.nxGraph.nodes)[:X_num])
        self.newGraph.pos = self.rawGraph.pos
        # time.sleep(self.rawGraph.DrawTimeWait)
        self.matchedEdges = set()
        self.x_saturated = set()
        self.y_saturated = set()
        self.node_index = 0

    def SingleStepRun(self,ax):
        reachableEdge = list(self.rawGraph.nxGraph.edges(self.node_index))
        print(f"reachableEdge {reachableEdge}")
        for e in reachableEdge:
            print(f"Judge Edge {e}")
            # 此处不用对e做规整，因为按照x索引，x的标号必然小于y
            if (e[0] not in self.x_saturated) and (e[1] in self.y_saturated) and (e not in self.matchedEdges):
                print(f"Judge First {e}")
                path = self.find_augment(e[1])
                if path is not None:
                    path.insert(0, e)
                    for _e in path:
                        if _e in self.matchedEdges:
                            self.remove_match(_e)

                        else:
                            self.add_match(_e)


                continue
            elif(e[0] not in self.x_saturated) and (e[1] not in self.y_saturated) and (e not in self.matchedEdges):
                print(f"Judge Second {e}")
                self.add_match(e)

                continue
        if self.node_index == self.xNum-1:
            self.newGraph.draw(ax)
            return False
        if self.node_index < self.xNum - 1:
            self.node_index += 1
        # plt.clf()
        print(self.newGraph.nxGraph.edges)
        self.newGraph.draw(ax)
        return True

    def remove_match(self,e):
        print(f"reomve {e}")
        # 先在匹配中删去该边
        self.matchedEdges.remove(e)
        # 再到新图中删去该边
        self.newGraph.nxGraph.remove_edge(u=e[0], v=e[1])
        self.newGraph.EdgeColorList.pop()
        self.newGraph.EdgeWidthList.pop()
    def add_match(self,e):
        print(f"add {e}")
        self.x_saturated.add(e[0])
        self.y_saturated.add(e[1])
        self.matchedEdges.add(e)
        # 变色、渲染
        self.newGraph.nxGraph.add_edge(u_of_edge=e[0], v_of_edge=e[1], weight=1)
        self.newGraph.EdgeColorList.append(self.newGraph.EdgeColorDone)
        self.newGraph.EdgeWidthList.append(self.newGraph.EdgeWidthDefault)
        self.newGraph.NodeColorList[e[0]] = self.newGraph.NodeColorDone
        self.newGraph.NodeColorList[e[1]] = self.newGraph.NodeColorDone
    def find_augment(self,start_node):
        for i in range(self.xNum,self.xNum+self.yNum):
            nodesLists = list(nx.all_simple_paths(self.rawGraph.nxGraph,source=start_node,target=i))
            print(f"={nodesLists}")

            for nodesList in nodesLists:

                print(f"=={nodesList}")
                path = []
                if len(nodesList)!=0 and len(nodesList)%2 != 0:
                    for i in range(len(nodesList)-1):
                        path.append(((min(nodesList[i], nodesList[i + 1]),max(nodesList[i], nodesList[i + 1]))))
                    evenEdge = path[0::2]
                    oddEdge = path[1::2]
                    if all(ve in self.matchedEdges for ve in evenEdge) and all(ve not in self.matchedEdges for ve in oddEdge) and (path[-1][-1] not in self.y_saturated):

                        return path

                else:
                    continue

class KM:
    def __init__(self, rawGraph):
        super().__init__()
        self.rawGraph = rawGraph
        self.newGraph = MyGraph()
        # self.rawGraph.draw()
        self.nodesList = list(self.rawGraph.nxGraph.nodes)

        # 初始化顶点颜色
        for i in self.nodesList:
            self.newGraph.nxGraph.add_node(i)

        self.newGraph.update_graph_properties()
        self.newGraph.pos = self.rawGraph.pos
        # self.newGraph.draw()

        self.TotalWeights = 0
        # self.newGraph.update_graph_properties()
        # self.newGraph.draw()
        self.Weights = 0

        self.left_nodes, self.right_nodes = nx.bipartite.sets(self.rawGraph.nxGraph)
        # self.left_nodes, self.right_nodes = list(nx.bipartite.sets(self.newGraph.nxGraph))
        self.left_nodes = list(self.left_nodes)
        self.right_nodes = list(self.right_nodes)
        # print('二部图中划分X中结点')
        # print(self.left_nodes)
        # print('二部图中划分Y中结点')
        # print(self.right_nodes)


    def runAlgorithm(self,ax):
        # KM算法实现
        # 创建一个平凡的可行节点标号 初始化标签 标签 lx 被初始化为每个节点的最大权重，而 ly 全部初始化为0
        lx = [max([edge[2].get('weight', 0) for edge in self.rawGraph.nxGraph.edges(node, data=True)])
              for node in self.left_nodes] #对每个节点，找到其相邻边中权重的最大值，形成 lx 列表，即节点标签的初始化
        # print('X中平凡的可行节点标号：')
        # print(lx)
        ly = [0] * len(self.right_nodes) #对每个节点，找到其相邻边中权重的最大值，形成 lx 列表，即节点标签的初始化
        # print('Y中平凡的可行节点标号：')
        # print(ly)
        # match = [-1] * len(self.newGraph.nxGraph.nodes) #用于记录匹配情况
        match = [-1] * len(self.right_nodes) #用于记录右侧节点Y中的匹配情况

        # 循环对每个节点执行KM算法的核心步骤，直到无法找到更多的增广路径
        # 通过 dfs 方法查找增广路径并更新标签 lx、ly，直至找不到增广路径
        for i in range(len(self.left_nodes)):
            slack = [float('inf')] * len(self.right_nodes) #初始化 slack 为长度为边数的全无穷大列表。
            visited_x = [False] * len(self.left_nodes) #用于记录在每轮增广路径查找中，哪些节点被访问过
            visited_y = [False] * len(self.right_nodes) #用于记录在每轮增广路径查找中，哪些边被访问过。
            while True:             #进入增广路径查找的循环，直到找不到增广路径 每轮增广路径查找开始前，重新初始化 visited_x visited_y
                visited_x = [False] * len(self.left_nodes)  # 用于记录在每轮增广路径查找中，哪些节点被访问过
                visited_y = [False] * len(self.right_nodes)  # 用于记录在每轮增广路径查找中，哪些节点被访问过
                if self.dfs(i, visited_x, visited_y, lx, ly, slack, match):#调用深度优先搜索函数 dfs 查找增广路径，如果找到了，跳出循环
                    break
                # 如果没找到，更新可行节点标号
                delta = min(slack[j] for j in range(len(self.right_nodes)) if not visited_y[j])
                #计算未访问过的边中，slack 值最小的边的 delta  slack=l(x)+l(y)-W(x,y) x属于X y属于Y-T
                for j in range(len(self.right_nodes)):
                    if visited_x[j]: #如果被访问过 那就是S中的结点 X中其他节点标号不变
                        lx[j] -= delta
                for j in range(len(self.right_nodes)):
                    if visited_y[j]: #如果被访问过 那就是T中的结点 Y中其他节点标号不变
                        ly[j] += delta
                    else:
                        slack[j] -= delta
        match_result = [(self.left_nodes[match[i]],self.right_nodes[i]) for i in range(len(match))]
        # print(match)
        # print('最后的匹配结果')
        # print(match_result)
        return match_result
    # dfs KM算法中深度优先搜索的核心方法 其实就是匈牙利算法，用于寻找增广路径
    # 在 KMAlgorithm 类的 dfs 方法中
    def dfs(self, x, visited_x, visited_y, lx, ly, slack, match):
        visited_x[x] = True
        # print(x)
        for y in range(len(self.right_nodes)):
            if visited_y[y]:
                continue
            # 计算当前节点和边的标签差值
            gap = lx[x] + ly[y] - self.rawGraph.nxGraph.get_edge_data(x, list(self.right_nodes)[y])['weight']

            if gap == 0:
                # 如果标签差值为0，说明找到了一条增广路径
                visited_y[y] = True  # 把找到的点y加入T

                if match[y] == -1 or self.dfs(match[y], visited_x, visited_y, lx, ly, slack, match):
                    match[y] = x
                    return True
            else:
                slack[y] = min(slack[y], gap)

        return False

    # 找到最优匹配后画最后的匹配图
    def draw_match_result(self,match_result,ax):
        for each in match_result:
            x,y = each
            weight = 0
            for edge in self.rawGraph.nxGraph.edges(x, data=True):
                if edge[1] == y:
                    # print(edge[2]['weight'])
                    weight = edge[2]['weight']
            self.newGraph.nxGraph.add_edge(x, y, weight=weight)
            index = list(self.newGraph.nxGraph.edges).index((x,y))
            self.newGraph.EdgeColorList.insert(index, self.newGraph.EdgeColorSelected)
            self.newGraph.EdgeWidthList.append(self.newGraph.EdgeWidthDefault)
        self.newGraph.draw(ax)