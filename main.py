
import sys

import matplotlib.pyplot as plt
import networkx as nx
import os
import platform

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QHeaderView

import MyGraph
from MySolution import Kruskal, Prim, CycleBreaking, Dijkstra, Floyd, FloydWarshall, Hungarian, KM
# IMPORT / GUI AND MODULES AND WIDGETS
# ///////////////////////////////////////////////////////////////
from modules import *
from widgets import *
os.environ["QT_FONT_DPI"] = "96" # FIX Problem for High DPI and Scale above 100%

# SET AS GLOBAL WIDGETS
# ///////////////////////////////////////////////////////////////
widgets = None

class MainWindow(QMainWindow):
    def __init__(self):
        plt.ion()
        QMainWindow.__init__(self)

        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title = "图论算法展示"
        description = "图论算法展示"
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # TOGGLE MENU
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))


        UIFunctions.uiDefinitions(self)


        #左边菜单按钮点击事件
        widgets.btn_p.clicked.connect(self.buttonClick)
        widgets.btn_k.clicked.connect(self.buttonClick)
        widgets.btn_po.clicked.connect(self.buttonClick)
        # widgets.btn_save.clicked.connect(self.buttonClick)
        widgets.btn_d.clicked.connect(self.buttonClick)
        widgets.btn_floyd.clicked.connect(self.buttonClick)
        widgets.btn_fw.clicked.connect(self.buttonClick)
        widgets.btn_x.clicked.connect(self.buttonClick)
        widgets.btn_km.clicked.connect(self.buttonClick)
        widgets.btn_gk.clicked.connect(self.buttonClick)
        widgets.btn_ek.clicked.connect(self.buttonClick)
        widgets.btn_nk.clicked.connect(self.buttonClick)
        widgets.btn_gp.clicked.connect(self.buttonClick)
        widgets.btn_ep.clicked.connect(self.buttonClick)
        widgets.btn_np.clicked.connect(self.buttonClick)
        widgets.btn_gpo.clicked.connect(self.buttonClick)
        widgets.btn_epo.clicked.connect(self.buttonClick)
        widgets.btn_npo.clicked.connect(self.buttonClick)
        widgets.btn_gd.clicked.connect(self.buttonClick)
        widgets.btn_ed.clicked.connect(self.buttonClick)
        widgets.btn_nd.clicked.connect(self.buttonClick)
        widgets.btn_gf.clicked.connect(self.buttonClick)
        widgets.btn_ef.clicked.connect(self.buttonClick)
        widgets.btn_gfw.clicked.connect(self.buttonClick)
        widgets.btn_efw.clicked.connect(self.buttonClick)
        widgets.btn_gx.clicked.connect(self.buttonClick)
        widgets.btn_ex.clicked.connect(self.buttonClick)
        widgets.btn_nx.clicked.connect(self.buttonClick)
        widgets.btn_gkm.clicked.connect(self.buttonClick)
        widgets.btn_ekm.clicked.connect(self.buttonClick)
        self.widget = widgets



        # 左边栏
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)
        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)


        self.show()

        #是否使用亮色主题
        useCustomTheme = False
        themeFile = "themes\py_dracula_light.qss"

        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)

        # SET HOME PAGE AND SELECT MENU
        # ///////////////////////////////////////////////////////////////
        widgets.stackedWidget.setCurrentWidget(widgets.k_page)
        widgets.btn_k.setStyleSheet(UIFunctions.selectMenu(widgets.btn_k.styleSheet()))


    # BUTTONS Setting
    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # Prim
        if btnName == "btn_p":
            widgets.stackedWidget.setCurrentWidget(widgets.p_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        if btnName == "btn_p":
            widgets.stackedWidget_2.setCurrentWidget(widgets.page_p2)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # Floyd
        if btnName == "btn_floyd":
            widgets.stackedWidget.setCurrentWidget(widgets.floyd_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        if btnName == "btn_floyd":
            widgets.stackedWidget_2.setCurrentWidget(widgets.page_floyd2)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # Kruskal
        if btnName == "btn_k":
            widgets.stackedWidget.setCurrentWidget(widgets.k_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        if btnName == "btn_k":
                widgets.stackedWidget_2.setCurrentWidget(widgets.page_k2)
                UIFunctions.resetStyle(self, btnName)
                btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        #破圈法
        if btnName == "btn_po":
            widgets.stackedWidget.setCurrentWidget(widgets.po_page) # SET PAGE
            UIFunctions.resetStyle(self, btnName) # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet())) # SELECT MENU
        if btnName == "btn_po":
            widgets.stackedWidget_2.setCurrentWidget(widgets.page_po2)  # SET PAGE
            UIFunctions.resetStyle(self, btnName)  # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))  # SELECT MENU

        # Dijsktra
        if btnName == "btn_d":
            widgets.stackedWidget.setCurrentWidget(widgets.d_page)  # SET PAGE
            UIFunctions.resetStyle(self, btnName)  # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))  # SELECT MENU
        if btnName == "btn_d":
            widgets.stackedWidget_2.setCurrentWidget(widgets.page_d2)  # SET PAGE
            UIFunctions.resetStyle(self, btnName)  # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))  # SELECT MENU

        # Floyd Warshall
        if btnName == "btn_fw":
            widgets.stackedWidget.setCurrentWidget(widgets.fw_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        if btnName == "btn_fw":
            widgets.stackedWidget_2.setCurrentWidget(widgets.page_fw2)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # 匈牙利算法
        if btnName == "btn_x":
            widgets.stackedWidget.setCurrentWidget(widgets.x_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        if btnName == "btn_x":
            widgets.stackedWidget_2.setCurrentWidget(widgets.page_x2)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # KM
        if btnName == "btn_km":
            widgets.stackedWidget.setCurrentWidget(widgets.km_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        if btnName == "btn_km":
            widgets.stackedWidget_2.setCurrentWidget(widgets.page_km2)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))


        # 按钮绑定Kruskal
        if btnName == "btn_gk":
            value = self.widget.lineEdit_k.text()
            plain_text_content = self.widget.plainTextEdit_k.toPlainText()
            if value:
                graph_matrix = MyGraph.MatrixGenerator.UGMatrix_with_weights_simple(int(value))
            else:
                graph_matrix = MyGraph.MatrixGenerator.convert_to_numpy_matrix(plain_text_content)
            self.ud_graph = MyGraph.UDiGraph(GraphMatrix=graph_matrix)
            self.k_algorithm = Kruskal(rawGraph=self.ud_graph)
            self.ud_graph.draw(ax=self.widget.canvas.figure.add_subplot(111))
            self.widget.canvas.draw()

        if btnName == "btn_ek":
            self.widget.canvas1.draw()
            self.k_algorithm = Kruskal(rawGraph=self.ud_graph)
            self.timer = QTimer(self.widget.canvas1)
            self.timer.timeout.connect(self.draw_kruskal_graph)
            self.timer.start(1000)  # 1000 milliseconds = 1 second
        if btnName == "btn_nk":
            self.widget.canvas1.draw()
            if self.k_algorithm.SingleStepRun(ax1=self.widget.canvas.figure.add_subplot(111),
                                                  ax2=self.widget.canvas1.figure.add_subplot(111)):
                matrix_str = np.array2string(nx.adjacency_matrix(self.k_algorithm.newGraph.nxGraph).toarray())
                self.widget.plainTextEdit_k2.setPlainText(matrix_str)
            self.widget.canvas.draw()
            self.widget.canvas1.draw()

        # 按钮绑定Prim
        if btnName == "btn_gp":
            value = self.widget.lineEdit_p.text()
            plain_text_content = self.widget.plainTextEdit_p.toPlainText()
            if value:
                graph_matrix = MyGraph.MatrixGenerator.UGMatrix_with_weights_simple(int(value))
            else:
                graph_matrix = MyGraph.MatrixGenerator.convert_to_numpy_matrix(plain_text_content)
            self.ud_graph = MyGraph.UDiGraph(GraphMatrix=graph_matrix)
            self.p_algorithm = Prim(rawGraph=self.ud_graph)
            self.ud_graph.draw(ax=self.widget.canvas_pg.figure.add_subplot(111))
            self.widget.canvas_pg.draw()
        if btnName == "btn_ep":
            self.widget.canvas_pe.draw()
            self.p_algorithm = Prim(rawGraph=self.ud_graph)
            self.timer_p = QTimer(self.widget.canvas_pe)
            self.timer_p.timeout.connect(self.draw_prim_graph)
            self.timer_p.start(1000)
        if btnName == "btn_np":
            self.widget.canvas_pe.draw()
            if self.p_algorithm.SingleStepRun(ax_1=self.widget.canvas_pg.figure.add_subplot(111),
                                              ax_2=self.widget.canvas_pe.figure.add_subplot(111)):
                matrix_str = np.array2string(nx.adjacency_matrix(self.p_algorithm.newGraph.nxGraph).toarray())
                self.widget.plainTextEdit_p2.setPlainText(matrix_str)
            self.widget.canvas_pg.draw()
            self.widget.canvas_pe.draw()


        # 按钮绑定破圈法
        if btnName == "btn_gpo":
            value = self.widget.lineEdit_po.text()
            plain_text_content = self.widget.plainTextEdit_po.toPlainText()
            if value:
                graph_matrix = MyGraph.MatrixGenerator.UGMatrix_with_weights_simple(int(value))
            else:
                graph_matrix = MyGraph.MatrixGenerator.convert_to_numpy_matrix(plain_text_content)
            self.ud_graph = MyGraph.UDiGraph(GraphMatrix=graph_matrix)
            self.c_algorithm = CycleBreaking(rawGraph=self.ud_graph)
            # print('-----------------------')
            self.ud_graph.draw(ax=self.widget.canvas_pog.figure.add_subplot(111))
            self.widget.canvas_pog.draw()
        if btnName == "btn_epo":
            self.widget.canvas_poe.draw()
            self.c_algorithm = CycleBreaking(rawGraph=self.ud_graph)

            self.timer_c = QTimer(self.widget.canvas_poe)
            self.timer_c.timeout.connect(self.draw_cycle_graph)
            self.timer_c.start(1000)
        if btnName == "btn_npo":
            self.widget.canvas_poe.draw()
            if self.c_algorithm.SingleStepRun(ax=self.widget.canvas_poe.figure.add_subplot(111)):
                matrix_str = np.array2string(nx.adjacency_matrix(self.c_algorithm.newGraph.nxGraph).toarray())
                self.widget.plainTextEdit_po2.setPlainText(matrix_str)
            self.widget.canvas_poe.draw()


        # 绑定Dijsktra
        if btnName == "btn_gd":
            value = self.widget.lineEdit_d.text()
            plain_text_content = self.widget.plainTextEdit_d.toPlainText()
            if value:
                graph_matrix = MyGraph.MatrixGenerator.UGMatrix_with_weights_simple(int(value))
            else:
                graph_matrix = MyGraph.MatrixGenerator.convert_to_numpy_matrix(plain_text_content)
            self.ud_graph = MyGraph.UDiGraph(GraphMatrix=graph_matrix)
            self.d_algorithm = Dijkstra(rawGraph=self.ud_graph)
            # print('-----------------------')
            self.ud_graph.draw(ax=self.widget.canvas_d.figure.add_subplot(111))
            self.widget.canvas_d.draw()
        if btnName == "btn_ed":
            self.widget.canvas_d2.draw()
            self.d_algorithm = Dijkstra(rawGraph=self.ud_graph)

            self.timer_d = QTimer(self.widget.canvas_d2)
            self.timer_d.timeout.connect(self.draw_dijsktra_graph)
            self.timer_d.start(1000)
        if btnName == "btn_nd":
            self.widget.canvas_d.draw()
            self.d_algorithm.SetStartNode(0)
            if self.d_algorithm.SingleStepRun(ax1=self.widget.canvas_d.figure.add_subplot(111),
                                              ax2=self.widget.canvas_d2.figure.add_subplot(111)):
                matrix_str = np.array2string(nx.adjacency_matrix(self.d_algorithm.newGraph.nxGraph).toarray())
                self.widget.plainTextEdit_d2.setPlainText(matrix_str)
            self.widget.canvas_d.draw()
            self.widget.canvas_d2.draw()


        # 绑定Floyd
        if btnName == "btn_gf":
            value = self.widget.lineEdit_f.text()
            plain_text_content = self.widget.plainTextEdit_f.toPlainText()
            if value:
                graph_matrix = MyGraph.MatrixGenerator.UGMatrix_with_weights_simple(int(value))
            else:
                graph_matrix = MyGraph.MatrixGenerator.convert_to_numpy_matrix(plain_text_content)
            self.ud_graph = MyGraph.UDiGraph(GraphMatrix=graph_matrix)
            # print('-----------------------')
            self.ud_graph.draw(ax=self.widget.canvas_f.figure.add_subplot(111))
            self.widget.canvas_f.draw()
        if btnName == "btn_ef":
            self.widget.canvas_f2.draw()
            self.f_algorithm = Floyd(rawGraph=self.ud_graph)
            self.f_distance = self.f_algorithm.runAlgorithm()
            self.f_algorithm.draw_result(ax=self.widget.canvas_f2.figure.add_subplot(111))
            matrix_str = np.array2string(nx.adjacency_matrix(self.f_algorithm.newGraph.nxGraph).toarray())
            self.widget.plainTextEdit_f2.setPlainText(matrix_str)
            self.widget.canvas_f2.draw()



        # 绑定Floyd Warshall
        if btnName == "btn_gfw":
            value = self.widget.lineEdit_fw.text()
            plain_text_content = self.widget.plainTextEdit_fw.toPlainText()
            if value:
                graph_matrix = MyGraph.MatrixGenerator.UGMatrix_with_weights_simple(int(value))
            else:
                graph_matrix = MyGraph.MatrixGenerator.convert_to_numpy_matrix(plain_text_content)
            self.ud_graph = MyGraph.UDiGraph(GraphMatrix=graph_matrix)
            # print('-----------------------')
            self.ud_graph.draw(ax=self.widget.canvas_fw.figure.add_subplot(111))
            self.widget.canvas_fw.draw()
        if btnName == "btn_efw":
            self.widget.canvas_fw2.draw()
            self.fw_algorithm = FloydWarshall(rawGraph=self.ud_graph)
            self.fw_distance = self.fw_algorithm.runAlgorithm(ax=self.widget.canvas_fw2.figure.add_subplot(111))
            self.fw_algorithm.draw_min_road(ax=self.widget.canvas_fw2.figure.add_subplot(111))
            self.widget.canvas_fw2.draw()
            self.text_list = self.fw_algorithm.get_road_data()
            # 将列表中的元素连接成一个字符串
            text_from_function = "\n".join(self.text_list)
            # 设置QTextBrowser的文本
            # 设置文本颜色为黑色
            text_format = QTextCharFormat()
            text_format.setForeground(QColor(0, 0, 0))  # RGB值为黑色
            # 设置 QTextBrowser 的默认文本格式
            self.widget.textBrowser.setCurrentCharFormat(text_format)

            self.widget.textBrowser.setPlainText(text_from_function)
            matrix_str = np.array2string(nx.adjacency_matrix(self.fw_algorithm.newGraph.nxGraph).toarray())
            self.widget.plainTextEdit_fw2.setPlainText(matrix_str)



        # 绑定匈牙利
        if btnName == "btn_gx":
            self.value = self.widget.lineEdit_x.text()
            self.value1 = self.widget.lineEdit_y.text()
            plain_text_content = self.widget.plainTextEdit_x.toPlainText()
            if self.value and self.value1:
                graph_matrix = MyGraph.MatrixGenerator.bipartite_graph_adjacency_matrix(int(self.value),int(self.value))
                self.row,self.clos = int(self.value),int(self.value)

            else:
                graph_matrix = MyGraph.MatrixGenerator.convert_to_numpy_matrix(plain_text_content)
                self.row,self.clos = np.shape(graph_matrix)
                graph_matrix = MyGraph.MatrixGenerator.HungrianExample(graph_matrix)
            self.ud_graph = MyGraph.UDiGraph(GraphMatrix=graph_matrix)
            self.x_algorithm = Hungarian(rawGraph=self.ud_graph,X_num=self.row,Y_num=self.clos)
            self.ud_graph.draw(ax=self.widget.canvas_x.figure.add_subplot(111))
            self.widget.canvas_x.draw()

        if btnName == "btn_ex":
            self.widget.canvas_x2.draw()
            self.x_algorithm = Hungarian(rawGraph=self.ud_graph, X_num=self.row, Y_num=self.clos)
            self.timer_x = QTimer(self.widget.canvas_d2)
            self.timer_x.timeout.connect(self.draw_hungarian_graph)
            self.timer_x.start(1000)

        if btnName == "btn_nx":
            self.widget.canvas_x.draw()
            if not self.x_algorithm.SingleStepRun(ax=self.widget.canvas_x2.figure.add_subplot(111)):
                matrix_str = np.array2string(nx.adjacency_matrix(self.x_algorithm.newGraph.nxGraph).toarray())
                self.widget.plainTextEdit_x2.setPlainText(matrix_str)
            self.widget.canvas_x.draw()
            self.widget.canvas_x2.draw()

        # 绑定KM
        if btnName == "btn_gkm":
            value = self.widget.lineEdit_km.text()
            plain_text_content = self.widget.plainTextEdit_km.toPlainText()
            if value:
                graph_matrix = MyGraph.MatrixGenerator.weighted_bipartite_graph_adjacency_matrix(int(value))
                print(graph_matrix)
            else:
                graph_matrix = MyGraph.MatrixGenerator.convert_to_numpy_matrix(plain_text_content)
                graph_matrix = MyGraph.MatrixGenerator.HungrianExample(graph_matrix)
                print(graph_matrix)
            self.ud_graph = MyGraph.UDiGraph(GraphMatrix=graph_matrix)
            # print('-----------------------')
            self.ud_graph.draw(ax=self.widget.canvas_km.figure.add_subplot(111))
            self.widget.canvas_km.draw()
        if btnName == "btn_ekm":
            self.widget.canvas_km2.draw()
            self.km_algorithm = KM(rawGraph=self.ud_graph)
            self.match_result = self.km_algorithm.runAlgorithm(
                ax=self.widget.canvas_km2.figure.add_subplot(111))


            self.km_algorithm.draw_match_result(self.match_result,
                                            ax=self.widget.canvas_km2.figure.add_subplot(111))
            matrix_str = np.array2string(nx.adjacency_matrix(self.km_algorithm.newGraph.nxGraph).toarray())
            self.widget.plainTextEdit_km2.setPlainText(matrix_str)
            self.widget.canvas_km2.draw()
    def draw_kruskal_graph(self):
        # 清除之前的图形
        # self.canvas.figure.clear()
        if not self.k_algorithm.SingleStepRun(ax1=self.widget.canvas.figure.add_subplot(111),ax2=self.widget.canvas1.figure.add_subplot(111)):
            matrix_str = np.array2string(nx.adjacency_matrix(self.k_algorithm.newGraph.nxGraph).toarray())
            self.widget.plainTextEdit_k2.setPlainText(matrix_str)
            self.timer.stop()
        print('------------------')
        self.widget.canvas.draw()
        self.widget.canvas1.draw()
    def draw_prim_graph(self):
        # 清除之前的图形
        # self.canvas.figure.clear()
        if  self.p_algorithm.SingleStepRun(ax_1=self.widget.canvas_pg.figure.add_subplot(111),ax_2=self.widget.canvas_pe.figure.add_subplot(111)):
            matrix_str = np.array2string(nx.adjacency_matrix(self.p_algorithm.newGraph.nxGraph).toarray())
            self.widget.plainTextEdit_p2.setPlainText(matrix_str)
            self.timer_p.stop()
        print('------------------')
        self.widget.canvas_pg.draw()
        self.widget.canvas_pe.draw()
    def draw_cycle_graph(self):
        # 清除之前的图形
        # self.canvas.figure.clear()
        if self.c_algorithm.SingleStepRun(ax=self.widget.canvas_poe.figure.add_subplot(111)):
            matrix_str = np.array2string(nx.adjacency_matrix(self.c_algorithm.newGraph.nxGraph).toarray())
            self.widget.plainTextEdit_po2.setPlainText(matrix_str)
            self.timer_c.stop()
        print('------------------')
        self.widget.canvas_poe.draw()
    def draw_dijsktra_graph(self):
        # 清除之前的图形
        # self.canvas.figure.clear()
        self.d_algorithm.SetStartNode(0)
        if  self.d_algorithm.SingleStepRun(ax1=self.widget.canvas_d.figure.add_subplot(111),
                                              ax2=self.widget.canvas_d2.figure.add_subplot(111)):
            matrix_str = np.array2string(nx.adjacency_matrix(self.d_algorithm.newGraph.nxGraph).toarray())
            self.widget.plainTextEdit_d2.setPlainText(matrix_str)
            self.timer_d.stop()
        print('------------------')
        self.widget.canvas_d.draw()
        self.widget.canvas_d2.draw()
    def draw_hungarian_graph(self):
        # 清除之前的图形
        # self.canvas.figure.clear()
        if  not self.x_algorithm.SingleStepRun(ax=self.widget.canvas_x2.figure.add_subplot(111)):
            matrix_str = np.array2string(nx.adjacency_matrix(self.x_algorithm.newGraph.nxGraph).toarray())
            self.widget.plainTextEdit_x2.setPlainText(matrix_str)
            self.timer_x.stop()
        print('------------------')
        self.widget.canvas_x.draw()
        self.widget.canvas_x2.draw()


    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    #定义鼠标发生时间
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        # if event.buttons() == Qt.RightButton:
        #     print('Mouse click: RIGHT CLICK')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
