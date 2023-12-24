import networkx as nx
import matplotlib
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
matplotlib.use('Qt5Agg')
class MyGraph:
    EdgeColorDefault = 'black'
    EdgeColorSelected = 'pink'
    EdgeColorDone = 'green'
    EdgeColorWrong = 'lightcoral'

    NodeSizeDefault = 1000
    NodeColorDefault = 'lightgrey'
    NodeColorSelected = 'thistle'
    NodeColorDone = 'skyblue'
    DrawTimeWait = 1

    def __init__(self):
        self.nxGraph = nx.Graph()
        self.EdgesNum = None
        self.EdgeWidthDefault = 4.0
        self.NodesNum = None
        self.pos = None

    # 在子类中重写
    def getGraphObj(self,GraphMatrix):
        pass

    def draw(self,ax):
        # plt.clf()
        nx.draw_networkx(self.nxGraph, self.pos, with_labels=True, font_weight='bold', node_size=self.NodeSizeDefault,
                         node_color=self.NodeColorList, edge_color=self.EdgeColorList, width=self.EdgeWidthList, arrows=None,ax=ax)
        nx.draw_networkx_edge_labels(self.nxGraph, self.pos,
                                     edge_labels={(i, j): d['weight'] for i, j, d in self.nxGraph.edges(data=True)},ax=ax)
        plt.show(block=False)
        # plt.pause(1)
    def update_graph_properties(self):
        # 获取节点数和边数
        self.EdgesNum = self.nxGraph.number_of_edges()
        self.EdgeWidthList = [self.EdgeWidthDefault] * self.EdgesNum
        self.EdgeColorList = [self.EdgeColorDefault] * self.EdgesNum
        self.NodesNum = self.nxGraph.number_of_nodes()
        self.NodeColorList = [self.NodeColorDefault] * self.NodesNum
        self.pos = nx.spring_layout(self.nxGraph,self.NodesNum)

    def resetGraph(self, GraphMatrix):
        pass
class MatrixGenerator:
    # 无向图，无权
    @staticmethod
    def UGMatrix_without_weights_complex(nodes):
        matrix = np.array([[random.choice([0, 1]) for _ in range(nodes)] for _ in range(nodes)])
        while nx.is_connected(nx.from_numpy_matrix(matrix)) is False:
            matrix = MatrixGenerator.UGMatrix_without_weights_complex(nodes)
        return matrix

    # 无向简单图，无权
    @staticmethod
    def UGMatrix_without_weights_simple(nodes):
        matrix = np.random.randint(2, size=(nodes, nodes))
        matrix = (matrix + matrix.T) % 2
        for i in range(nodes):
            matrix[i][i] = 0
        while nx.is_connected(nx.from_numpy_matrix(matrix)) is False:
            matrix = MatrixGenerator.UGMatrix_without_weights_simple(nodes)
        return matrix

    # 无向简单图，带权
    @staticmethod
    def UGMatrix_with_weights_simple(nodes):
        matrix = np.random.randint(low=1,high=100, size=(nodes, nodes))
        # matrix = (matrix + matrix.T)
        for i in range(nodes):
            matrix[i][i] = 0
        mask = np.random.randint(2, size=(nodes, nodes))
        matrix = mask*matrix
        while nx.is_connected(nx.from_numpy_matrix(matrix)) is False:
            matrix = MatrixGenerator.UGMatrix_with_weights_simple(nodes)
        return matrix

    # 有向图，无权
    @staticmethod
    def DGM_without_weights_complex(nodes):
        matrix = np.array([[random.choice([0, 1]) for _ in range(nodes)] for _ in range(nodes)])
        while nx.is_connected(nx.from_numpy_matrix(matrix)) is False:
            matrix = MatrixGenerator.DGM_without_weights_complex(nodes)
        return matrix

    # 有向简单图，无权
    @staticmethod
    def DGM_without_weights_simple(nodes):
        matrix = np.array([[random.choice([0, 1]) for _ in range(nodes)] for _ in range(nodes)])
        for i in range(nodes):
            matrix[i][i] = 0
        while nx.is_connected(nx.from_numpy_matrix(matrix)) is False:
            matrix = MatrixGenerator.DGM_without_weights_simple(nodes)
        return matrix

    # 有向简单图，带权
    @staticmethod
    def DGM_with_weights_simple(nodes):
        matrix = np.random.randint(low=1, high=100, size=(nodes, nodes))
        for i in range(nodes):
            matrix[i][i] = 0
        while nx.is_connected(nx.from_numpy_matrix(matrix)) is False:
            matrix = MatrixGenerator.DGM_with_weights_simple(nodes)
        return matrix

    # 二部图
    @staticmethod
    def bipartite_graph_adjacency_matrix(X_nodes, Y_nodes):
        # 创建零矩阵
        matrix = np.zeros((X_nodes + Y_nodes, X_nodes + Y_nodes), dtype=int)
        # 连接 X 和 Y 的节点，保持对称性 生成二部图
        for i in range(X_nodes):
            for j in range(X_nodes, X_nodes + Y_nodes):
                # 随机设置连接或不连接，可以根据需求修改
                connection = np.random.randint(low=0,high=2)
                # 设置对称位置
                matrix[i, j] = connection
                matrix[j, i] = connection
        mask = np.random.randint(2, size=(X_nodes+Y_nodes, X_nodes+Y_nodes))
        matrix = mask * matrix

        while nx.is_connected(nx.from_numpy_matrix(matrix)) is False:
            matrix = MatrixGenerator.bipartite_graph_adjacency_matrix(X_nodes,Y_nodes)
        return matrix

    @staticmethod
    def weighted_bipartite_graph_adjacency_matrix(nodes):
        # 创建零矩阵
        zero_matrix = np.zeros((nodes, nodes), dtype=int)
        matrix = np.random.randint(low=1, high=100, size=(nodes, nodes))
        temp1 = np.append(zero_matrix, matrix, axis=1)
        temp2 = np.append(matrix.T, zero_matrix, axis=1)
        matrix = np.append(temp1,temp2,axis = 0)
        while nx.is_connected(nx.from_numpy_matrix(matrix)) is False:
            matrix = MatrixGenerator.weighted_bipartite_graph_adjacency_matrix(nodes)
        return matrix

    @staticmethod
    def convert_to_numpy_matrix_UDG(input_matrix):
        # 将字符串矩阵转为矩阵

        matrix = None

        for i in range(len(matrix)):
            matrix[i][i] = 0

        while (nx.is_connected(nx.from_numpy_matrix(matrix)) is False) or (matrix.shape[0]!=matrix.shape[1]):
            matrix = MatrixGenerator.UGMatrix_with_weights_simple(len(matrix))
        return matrix

    @staticmethod
    def convert_to_numpy_matrix_DG(input_matrix):
        # 将字符串矩阵转为矩阵

        matrix = None

        for i in range(len(matrix)):
            matrix[i][i] = 0

        while (nx.is_connected(nx.from_numpy_matrix(matrix)) is False) or (matrix.shape[0]!=matrix.shape[1]):
            matrix = MatrixGenerator.DGM_with_weights_simple(len(matrix))
        return matrix

    @staticmethod
    def HungrianExample(matrix):
        rows, cols = np.shape(matrix)
        zero_matrix = np.zeros((rows, cols), dtype=int)
        temp1 = np.append(zero_matrix, matrix, axis=1)
        temp2 = np.append(matrix.T, zero_matrix, axis=1)
        return np.append(temp1,temp2,axis = 0)
# class MatrixGenerator:
#     # 无向图，无权
#     @staticmethod
#     def UGMatrix_without_weights_complex(nodes):
#         adjacency_matrix = np.array([[random.choice([0, 1]) for _ in range(nodes)] for _ in range(nodes)])
#         return adjacency_matrix
#
#     # 无向简单图，无权
#     @staticmethod
#     def UGMatrix_without_weights_simple(nodes):
#         matrix = np.random.randint(2, size=(nodes, nodes))
#         matrix = (matrix + matrix.T) % 2
#         for i in range(nodes):
#             matrix[i][i] = 0
#         return matrix
#
#     # 无向简单图，带权
#     @staticmethod
#     def UGMatrix_with_weights_simple(nodes):
#         matrix = np.random.randint(low=1,high=100, size=(nodes, nodes))
#         # matrix = (matrix + matrix.T)
#         for i in range(nodes):
#             matrix[i][i] = 0
#         mask = np.random.randint(2, size=(nodes, nodes))
#         matrix = mask*matrix
#         return matrix
#
#     # 有向图，无权
#     @staticmethod
#     def DGM_without_weights_complex(nodes):
#         matrix = np.array([[random.choice([0, 1]) for _ in range(nodes)] for _ in range(nodes)])
#         return matrix
#
#     # 有向简单图，无权
#     @staticmethod
#     def DGM_without_weights_simple(nodes):
#         matrix = np.array([[random.choice([0, 1]) for _ in range(nodes)] for _ in range(nodes)])
#         for i in range(nodes):
#             matrix[i][i] = 0
#         return matrix
#
#     # 有向简单图，带权
#     @staticmethod
#     def DGM_with_weights_simple(nodes):
#         matrix = np.random.randint(low=1, high=100, size=(nodes, nodes))
#         for i in range(nodes):
#             matrix[i][i] = 0
#         return matrix
#
#     # 二部图
#     @staticmethod
#     def bipartite_graph_adjacency_matrix(X_nodes, Y_nodes):
#         # 创建零矩阵
#         matrix = np.zeros((X_nodes + Y_nodes, X_nodes + Y_nodes), dtype=int)
#         # 连接 X 和 Y 的节点，保持对称性 生成二部图
#         for i in range(X_nodes):
#             for j in range(X_nodes, X_nodes + Y_nodes):
#                 # 随机设置连接或不连接，可以根据需求修改
#                 connection = np.random.randint(low=0,high=2)
#                 # 设置对称位置
#                 matrix[i, j] = connection
#                 matrix[j, i] = connection
#         return matrix
#
#     @staticmethod
#     def weighted_bipartite_graph_adjacency_matrix(nodes):
#         # 创建零矩阵
#         zero_matrix = np.zeros((nodes, nodes), dtype=int)
#         matrix = np.random.randint(low=1, high=100, size=(nodes, nodes))
#         temp1 = np.append(zero_matrix, matrix, axis=1)
#         temp2 = np.append(matrix.T, zero_matrix, axis=1)
#         return np.append(temp1,temp2,axis = 0)
#
    @staticmethod
    def convert_to_numpy_matrix(input_matrix):
        # 将字符串矩阵转换为二维列表
        matrix_list = [list(map(int, row.split())) for row in input_matrix.splitlines()]
        # 将二维列表转换为NumPy矩阵
        numpy_matrix = np.array(matrix_list)
        return numpy_matrix


class UDiGraph(MyGraph):
    def __init__(self,GraphMatrix):
        super().__init__()
        self.nxGraph = self.getGraphObj(GraphMatrix)
        self.update_graph_properties()

    def getGraphObj(self,GraphMatrix):
        return nx.from_numpy_matrix(GraphMatrix)

    def resetGraph(self, GraphMatrix):
        pass
class DiGraph(MyGraph):

    def __init__(self,GraphMatrix):
        super().__init__()
        self.GraphMatrix = GraphMatrix
        self.nxGraph = self.getGraphObj(GraphMatrix)
        self.update_graph_properties()

    def getGraphObj(self,GraphMatrix):
        return nx.DiGraph(GraphMatrix)

    def resetGraph(self,GraphMatrix):
        self.GraphMatrix = GraphMatrix
        self.nxGraph = self.getGraphObj(GraphMatrix)
        self.update_graph_properties()
        return self.nxGraph