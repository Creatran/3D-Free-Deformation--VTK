## #!/usr/local/bin/python3

from __future__ import print_function
import sys
import os

try:
	# PyQt4
	from PyQt4 import QtCore, QtGui
	from PyQt4.QtGui import *
	from PyQt4.QtCore import QSettings
	from PyQt4.uic import loadUiType
	print('Using PyQT4')
except ImportError:
	try:
		from PyQt5.QtCore import Qt,QSettings
		from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow,QFileDialog
		from PyQt5.QtGui import QStandardItemModel,QStandardItem,QImage,QTransform,QPixmap
		from PyQt5.uic import loadUiType
		print('Using PyQT5')
	except ImportError:
		sys.exit('No PyQt found in our system!')

# VTK
import vtk
from scenemanager import SceneManager

# We'll need to access home directory
from os.path import expanduser

# numpy imports
import numpy as np
import scipy.io as sio
import math
from copy import copy
import time
import util
from ui import Ui_MainWindow
"""
Part of the code refered https://github.com/Anthony-Xu/3D-FFD-in-VTK.
It helps we understand how to use VTK and build the grid for FFD.
We also added many new features like load file, present movement of the 3D object,
add a GUI and write the FFD code in a fast way.
"""
# load GUI
#Ui_MainWindow = loadUiType("mainwindow.ui")[0]

class MainWindow(QMainWindow, Ui_MainWindow):
	def __init__(self, parent=None):
		super(MainWindow, self).__init__(parent)
		self.setupUi(self)
		self.initVTK()


	def initVTK(self):
		self.show()		# We need to call QVTKWidget's show function before initializing the interactor
		self.SceneManager = SceneManager(self.vtkContext)

	def FileOpen(self):
		title = "Open File"
		flags = QFileDialog.ShowDirsOnly
		dbpath = QFileDialog.getExistingDirectory(self,
						title,
						expanduser("~"),
						flags)

	def FileExit(self):
		app.quit()

	def ShowAboutDialog(self):
		title = QString("About Qt VTK Skeleton")
		text = QString("Qt Vtk Skeleton \n")
		text.append("Minimal code to implement a basic\n")
		text.append("Qt application with a VTK widget \n")
		text.append("\n")
		aboutbox = QMessageBox(QMessageBox.Information,title,text,0,self,Qt.Sheet )
		aboutbox.show()
		#aboutbox.exec()

	def SetViewXY(self):
		# Ensure UI is sync (if view was selected from menu)
		self.radioButtonXY.setChecked(True)
		self.SceneManager.SetViewXY()

	def SetViewXZ(self):
		# Ensure UI is sync (if view was selected from menu)
		self.radioButtonXZ.setChecked(True)
		self.SceneManager.SetViewXZ()

	def SetViewYZ(self):
		# Ensure UI is sync (if view was selected from menu)
		self.radioButtonYZ.setChecked(True)
		self.SceneManager.SetViewYZ()

	def Snapshot(self):
		self.SceneManager.Snapshot()

	def ToggleVisualizeAxis(self, visible):
		# Ensure UI is sync
		self.actionVisualize_Axis.setChecked(visible)
		self.checkVisualizeAxis.setChecked(visible)
		self.SceneManager.ToggleVisualizeAxis(visible)

	def ToggleVisibility(self, visible):
		# Ensure UI is sync
		self.actionVisibility.setChecked(visible)
		self.checkVisibility.setChecked(visible)
		self.SceneManager.ToggleVisibility(visible)

	def changePath(self):
        # Find the path of obj
		open = QFileDialog()
		self.path = open.getOpenFileName()
		print(self.path)
		# Ui_MainWindow.lineEdit.setText(self.path[0])
		# self.path = open.getExistingDirectory()
		#Ui_MainWindow.lineEdit.setText(self.path[0])
	def changePath2(self):
        # Find the path of FFD
		open = QFileDialog()
		self.path2 = open.getOpenFileName()
		print(self.path2)

	def restart_total(self):
        # connect to the reset_all button
		print("Restart the whole stage")
		start = time.clock()
		count = 0
		global spherelist_position
		for i in range(totalsphere):
			spherelist[i].SetCenter(index2realworld(i))
		spherelist_position = [spherelist[i].GetCenter() for i in range(totalsphere)]
		for i in range(totalsphere):
			x, y, z = index2xyz(i)
			if (x + y + z) % 2 == 0:
				n = neighbor(i)
				for j in n:
					x1, y1, z1 = index2realworld(i)
					x2, y2, z2 = index2realworld(j)
					sourcelist[count].SetPoint1(x1, y1, z1)
					sourcelist[count].SetPoint2(x2, y2, z2)
					mapperlist[count].SetInputConnection(sourcelist[count].GetOutputPort())
					actorlist[count].SetMapper(mapperlist[count])
					mw.SceneManager.ren.AddActor(actorlist[count])
					count = count + 1
		face_data = actions.restart()
		face_mapper.SetInputData(face_data)
		face_actor = vtk.vtkActor()
		face_actor.SetMapper(face_mapper)
		face_actor.GetProperty().SetColor(0.5, 0.9, 0.95)
		mw.SceneManager.ren.AddActor(face_actor)
		print(time.clock() - start)

	def save_grid(self):
        # Connect to the save_FFD button
		file_name = self.path2[0]
		print("Save the grid")
		with open(file_name, "w") as file:
			file.write("#dimension#")
			file.write("\n")
			file.write("3")
			file.write("\n")
			file.write("#one to one#")
			file.write("\n")
			file.write("1")
			file.write("\n")
			file.write("#control grid size#")
			file.write("\n")
			file.write("3")
			file.write("\n")
			file.write("3")
			file.write("\n")
			file.write("3")
			file.write("\n")
			file.write("#control grid spacing#")
			file.write("\n")
			file.write("3")
			file.write("\n")
			file.write("3")
			file.write("\n")
			file.write("3")
			file.write("\n")
			file.write("# offsets of the control points#")
			file.write("\n")
			for i in range(totalsphere):
				pos = spherelist[i].GetCenter()
				pos_ori = index2realworld(i)
				delta = [-pos_ori[i] + pos[i] for i in range(3)]
				for item in delta:
					file.write(str(item))
					file.write(" ")
				file.write("\n")

		return 0

	def load_grid(self):
		"""
        读取FFF文件，并且进行调整
        :param file_name:
        :return:
        """
        # Connect to the load_FFD button
		print("Load the grid and update")
		file_name = self.path2[0]
		file = open(file_name, "r")
		lines = file.readlines()
		file.close()
		xl, yl, zl = lines[5:8]
		s1, s2, s3 = lines[7:10]
		begin = False
		lst_pos = []
		for line in lines:
			# print(line)
			if begin == True:
				# lst_pos.append([float(item) for item in line.split(" ")[0]])
				# print(line.split(" "))
				pos = []
				for item in line.split(' ')[0:3]:
					try:
						pos.append(float(item))
					except ValueError:
						print(item)
				lst_pos.append(pos)
			if 'offsets' in line:
				begin = True

		# print(lst_pos)
		# print(lst_pos)
		face_data = actions.restart()
		face_mapper.SetInputData(face_data)
		face_actor = vtk.vtkActor()
		face_actor.SetMapper(face_mapper)
		face_actor.GetProperty().SetColor(0.5, 0.9, 0.95)
		mw.SceneManager.ren.AddActor(face_actor)

		set_pos(lst_pos)
		return 0

	def load_obj(self):
        # Connect to the load_obj button
		#file_names=["obj/1.obj", "obj/2.obj", "obj/3.obj", "obj/4.obj", "obj/5.obj", "obj/6.obj", "obj/7.obj","obj/8.obj", "obj/10.obj"]
		lst_obj = []
		file_name=self.path[0]
		if ".txt" in file_name:
			print("txt")
			f=open(file_name)
			file_names=[]
			line=f.readline()
			while line:
				line=line.replace('\n','')
				file_names.append(line)
				line=f.readline()
			f.close()
			print(file_names)
			for filename in file_names:
				print(filename)
				reader = vtk.vtkOBJReader()
				# reader = vtk.vtkOBJImporter()
				reader.SetFileName(filename)
				# reader.SetTexturePath(filename)
				reader.Update()
				lst_obj.append(norm(reader.GetOutput()))
		else:
			print("obj")
			reader = vtk.vtkOBJReader()
			# reader = vtk.vtkOBJImporter()
			reader.SetFileName(file_name)
			# reader.SetTexturePath(filename)
			reader.Update()
			lst_obj.append(norm(reader.GetOutput()))
		global actions
		actions = movement(lst_obj)
		self.restart()
		return 0

	def show_last_action(self):
        # Connect to the last_one button
		print("Show the last action")
		# for i in range(actions.lenth):
		start = time.clock()
		face_data = FFD_trans(actions.last())
		face_mapper.SetInputData(face_data)
		face_actor = vtk.vtkActor()
		face_actor.SetMapper(face_mapper)
		face_actor.GetProperty().SetColor(0.5, 0.9, 0.95)
		mw.SceneManager.ren.AddActor(face_actor)
		while True:
			if time.clock() - start > 1 / fps:
				break
		print(time.clock() - start)

	def show_next_action(self):
        # Connect to the next_one button
		print("Show the next action")
		# for i in range(actions.lenth):
		start = time.clock()
		face_data = FFD_trans(actions.next())
		face_mapper.SetInputData(face_data)
		face_actor = vtk.vtkActor()
		face_actor.SetMapper(face_mapper)
		face_actor.GetProperty().SetColor(0.5, 0.9, 0.95)
		mw.SceneManager.ren.AddActor(face_actor)
		while True:
			if time.clock() - start > 1 / fps:
				break
		print(time.clock() - start)

	def restart(self):
        # Connect to the rest button
		print("Restart the action")
		start = time.clock()
		face_data = FFD_trans(actions.restart())
		face_mapper.SetInputData(face_data)
		face_actor = vtk.vtkActor()
		face_actor.SetMapper(face_mapper)
		face_actor.GetProperty().SetColor(0.5, 0.9, 0.95)
		mw.SceneManager.ren.AddActor(face_actor)
		print(time.clock() - start)


app = QApplication(sys.argv)
mw = MainWindow()

#  xl means how long is x. For example, if xl==2, then there are 3 control points in x-axe.
#  In fact, this is not the real world distance. I set real world distance to be 1.
"""There are how many edges in one line"""
xl = 5
yl = 5
zl = 5

# this is the time for delay the action
fps = 10.0


"""
Here are codes for indexing the control points
"""

def xyz2index(x, y, z):
    'For example, xyz2index(0,0,0)=0, xyz2index(0,0,1)=1'
    index = (xl + 1) * (yl + 1) * z + (xl + 1) * y + x
    return index


def index2xyz(i):
    'For example, index2xyz(1)=(0,0,1)'
    z = i // ((xl + 1) * (yl + 1))
    y = (i - z * (xl + 1) * (yl + 1)) // (xl + 1)
    x = i % (xl + 1)
    return x, y, z


def xyz2realworld(x, y, z):
    'Input x, y, z is int. Output xr, yr, zr is float. 0<=xr<=1.'
    xr = float(x) / xl
    yr = float(y) / yl
    zr = float(z) / zl
    return xr, yr, zr


def index2realworld(i):
    'Input an index. Output the position of that point in realworld. 0<=xr<=1.'
    if i >= (xl + 1) * (yl + 1) * (zl + 1):
        print('Error! Index not exists!')
        return 0
    x, y, z = index2xyz(i)
    xr, yr, zr = xyz2realworld(x, y, z)
    return xr, yr, zr


def neighbor(i):
    'i is index, return its neighbor points'
    x, y, z = index2xyz(i)
    n = []
    if x > 0:
        n.append(xyz2index(x - 1, y, z))
    if x < xl:
        n.append(xyz2index(x + 1, y, z))
    if y > 0:
        n.append(xyz2index(x, y - 1, z))
    if y < yl:
        n.append(xyz2index(x, y + 1, z))
    if z > 0:
        n.append(xyz2index(x, y, z - 1))
    if z < zl:
        n.append(xyz2index(x, y, z + 1))
    return n


def Do_FFD(obj, event):
    """
    Code for Do_FFD transform, though the code done many necessary assignment, the operation is quite fast.
    :param obj:
    :param event:
    :return:
    """
    print("Do FFD transformation")
    start = time.clock()
    count = 0
    global spherelist_position
    spherelist_position = [spherelist[i].GetCenter() for i in range(totalsphere)]
    # print(spherelist_position)
    for i in range(totalsphere):
        x, y, z = index2xyz(i)
        if (x + y + z) % 2 == 0:
            n = neighbor(i)
            for j in n:
                # x1, y1, z1 = spherelist[i].GetCenter()
                # x2, y2, z2 = spherelist[j].GetCenter()
                x1, y1, z1 = spherelist_position[i]
                x2, y2, z2 = spherelist_position[j]
                sourcelist[count].SetPoint1(x1, y1, z1)
                sourcelist[count].SetPoint2(x2, y2, z2)
                mapperlist[count].SetInputConnection(sourcelist[count].GetOutputPort())
                actorlist[count].SetMapper(mapperlist[count])
                mw.SceneManager.ren.AddActor(actorlist[count])
                count = count + 1
    face_data = FFD_trans(actions.now())
    face_mapper.SetInputData(face_data)
    face_actor = vtk.vtkActor()
    face_actor.GetProperty().SetColor(0.5, 0.9, 0.95)
    face_actor.SetMapper(face_mapper)
    mw.SceneManager.ren.AddActor(face_actor)
    print(time.clock() - start)


def add_speed(obj, event):
    # make fps bigger
    global fps
    fps += 1
    print("now max the fps is " + str(fps))


def slow_speed(obj, event):
    # make fps smaller
    global fps
    fps -= 1
    if fps <= 0:
        fps = 1
        print("the fps should be positive")
    print("now max the fps is " + str(fps))

def set_pos(lst_pos1):
    """
    为控制点进行赋值，给定一个序列
    :param lst_pos:
    :return:
    """
    # global spherelist
    for i in range(totalsphere):
        x, y, z = index2realworld(i)
        dx,dy,dz=lst_pos1[i]
        print(lst_pos1)
        print("dx")
        print([dx,dy,dz])
        print([dx + x, dy + y, dz + z])
        spherelist[i].SetCenter(lst_pos1[i][0] + x, lst_pos1[i][1] + y, lst_pos1[i][2] + z)  # 设置中心
        spherelist[i].SetRepresentationToSurface()  # 不懂
        spherelist[i].On()

    Do_FFD(1, 1)
    return 0

f = [1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0]


def B(i, l, s):
    return f[l] / f[i] / f[l - i] * pow(s, i) * pow(1 - s, l - i)


def norm(polygonPolyData):
    from copy import copy
    # 按照id顺序获取polydata的vertex坐标
    coord = []
    position = [0.0, 0.0, 0.0]
    for i in range(polygonPolyData.GetNumberOfPoints()):
        polygonPolyData.GetPoint(i, position)
        t = copy(position)
        coord.append(t)

    # 记录极大极小值，进行归一化
    max_lst = [-1000, -1000, -1000]
    min_lst = [1000, 1000, 1000]
    for i in range(polygonPolyData.GetNumberOfPoints()):
        position = coord[i]
        for i in range(len(min_lst)):
            if position[i] < min_lst[i]:
                min_lst[i] = position[i]
        for i in range(len(max_lst)):
            if position[i] > max_lst[i]:
                max_lst[i] = position[i]

    print(len(coord))
    # print max_lst
    scale = 1 / max([max_lst[i] - min_lst[i] for i in range(3)])
    bias = [0.5 - (max_lst[i] - min_lst[i]) * scale / 2 for i in range(3)]

    def change(point):
        return [(point[i] - min_lst[i]) * scale + bias[i] for i in range(3)]

    # 按照cell的id顺序获取构成cell的vertex的id
    cell_ids = []
    for i in range(polygonPolyData.GetNumberOfCells()):
        cell = polygonPolyData.GetCell(i)
        nPoints = cell.GetNumberOfPoints()
        vertex_ids = []
        for j in range(nPoints):
            vertex_ids.append(cell.GetPointId(j))
        cell_ids.append(vertex_ids)

    points = vtk.vtkPoints()
    for i in range(len(coord)):
        points.InsertNextPoint(change(coord[i]))
        # points.InsertNextPoint(coord[i])

    cells = vtk.vtkCellArray()

    for vertex_ids in cell_ids:
        polygon = vtk.vtkPolygon()
        cell_num = len(vertex_ids)
        polygon.GetPointIds().SetNumberOfIds(cell_num)
        id0 = 0
        for i in vertex_ids:
            polygon.GetPointIds().SetId(id0, i)
            id0 += 1
        cells.InsertNextCell(polygon)

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(cells)
    return polygonPolyData


def old2new(old):
    # the math code for FFD
    s, t, u = old
    s_new = 0
    t_new = 0
    u_new = 0
    for p in range(totalsphere):
        x, y, z = spherelist_position[p]
        i, j, k = index2xyz(p)
        b = B(i, xl, s) * B(j, yl, t) * B(k, zl, u)
        s_new += b * x
        t_new += b * y
        u_new += b * z
    return [s_new, t_new, u_new]


def FFD_trans(polygonPolyData):
    # 按照id顺序获取polydata的vertex坐标
    coord = []
    position = [0.0, 0.0, 0.0]
    for i in range(polygonPolyData.GetNumberOfPoints()):
        polygonPolyData.GetPoint(i, position)
        coord.append(copy(position))
    # coord=[polygonPolyData.GetPoint(i, [0.0, 0.0, 0.0]) for i in range(polygonPolyData.GetNumberOfPoints())]
    # 按照cell的id顺序获取构成cell的vertex的id
    cell_ids = []
    for i in range(polygonPolyData.GetNumberOfCells()):
        cell = polygonPolyData.GetCell(i)
        nPoints = cell.GetNumberOfPoints()
        vertex_ids = []
        for j in range(nPoints):
            vertex_ids.append(cell.GetPointId(j))
        cell_ids.append(vertex_ids)

    #new_coord = list(map(old2new, coord))
    # get Bernstein matrix B and deform matrix P
    # set the new_coord as BP
    P = np.zeros((totalsphere, 3))
    for i in range(totalsphere):
        x, y, z = spherelist[i].GetCenter()
        P[i] = [x, y, z]
    B = util.get_stu_deformation_matrix(coord, [xl, yl, zl])
    new_coord = np.dot(B, P)
    # 重构polydata
    points = vtk.vtkPoints()
    for i in range(len(coord)):
        points.InsertNextPoint(new_coord[i])
        # points.InsertNextPoint(coord[i])

    cells = vtk.vtkCellArray()

    for vertex_ids in cell_ids:
        polygon = vtk.vtkPolygon()
        cell_num = len(vertex_ids)
        polygon.GetPointIds().SetNumberOfIds(cell_num)
        id0 = 0
        for i in vertex_ids:
            polygon.GetPointIds().SetId(id0, i)
            id0 += 1
        cells.InsertNextCell(polygon)

    polygonPolyData = vtk.vtkPolyData()
    polygonPolyData.SetPoints(points)
    polygonPolyData.SetPolys(cells)
    return polygonPolyData

# draw sphere
totalsphere = (xl + 1) * (yl + 1) * (zl + 1)
spherelist = []
for i in range(totalsphere):
    sphereWidget = vtk.vtkSphereWidget()  # 产生一个点
    sphereWidget.SetInteractor(mw.SceneManager.iren)  # 用iren进行互动
    x, y, z = index2realworld(i)
    sphereWidget.SetCenter(x, y, z)  # 设置中心
    sphereWidget.SetRadius(0.01)  # 设置直径
    sphereWidget.SetRepresentationToSurface()  # 不懂
    sphereWidget.On()
    spherelist.append(sphereWidget)  # 球球的列表
spherelist_position = [spherelist[i].GetCenter() for i in range(totalsphere)]
# draw lines
sourcelist = []
mapperlist = []
actorlist = []
for i in range(3 * totalsphere):  # 首先创建一些vtk的线
    sourcelist.append(vtk.vtkLineSource())
    mapperlist.append(vtk.vtkPolyDataMapper())
    actorlist.append(vtk.vtkActor())

count = 0
for i in range(totalsphere):
    x, y, z = index2xyz(i)
    if (x + y + z) % 2 == 0:
        n = neighbor(i)
        for j in n:  # 对于每一个连着得线段
            x1, y1, z1 = spherelist[i].GetCenter()
            x2, y2, z2 = spherelist[j].GetCenter()
            sourcelist[count].SetPoint1(x1, y1, z1)
            sourcelist[count].SetPoint2(x2, y2, z2)
            mapperlist[count].SetInputConnection(sourcelist[count].GetOutputPort())
            actorlist[count].SetMapper(mapperlist[count])
            actorlist[count].GetProperty().SetColor(0.8, 0.3, 0.2)
            mw.SceneManager.ren.AddActor(actorlist[count])
            count = count + 1


"""-----------------------------------------------------"""
lst_obj = []
file_names=["Deer.obj"]
#file_names = ["obj/1.obj", "obj/2.obj",
#              "obj/3.obj",  "obj/4.obj","obj/5.obj","obj/6.obj","obj/7.obj","obj/8.obj","obj/10.obj",]
for filename in file_names:
    reader = vtk.vtkOBJReader()
    # reader = vtk.vtkOBJImporter()
    reader.SetFileName(filename)
    # reader.SetTexturePath(filename)
    reader.Update()
    lst_obj.append(norm(reader.GetOutput()))


class movement():
    """
    class for store the frames
    """
    def __init__(self, lst):
        self.lst = lst
        self.lenth = len(lst)
        self.position = 0

    def next(self):
        if self.position == self.lenth - 1:
            self.position = 0
            return self.lst[self.position]
        else:
            self.position += 1
            return self.lst[self.position]

    def now(self):
        return self.lst[self.position]

    def last(self):
        if self.position == 0:
            self.position = self.lenth - 1
            return self.lst[self.position]
        else:
            self.position -= 1
            return self.lst[self.position]

    def first(self):
        return self.lst[0]

    def restart(self):
        self.position = 0
        return self.lst[0]


actions = movement(lst_obj)

face_mapper = vtk.vtkPolyDataMapper()
face_data_ori = reader.GetOutput()
face_data_ori = norm(face_data_ori)

face_mapper.SetInputData(actions.first())
face_actor = vtk.vtkActor()
face_actor.GetProperty().SetColor(0.5, 0.9, 0.95)
face_actor.SetMapper(face_mapper)
mw.SceneManager.ren.AddActor(face_actor)
"""-----------------------------------------------------"""

# set interaction
for i in range(totalsphere):
    spherelist[i].AddObserver("InteractionEvent", Do_FFD)
#
# butt = []
# actor_lst = []
# total_butt = 8
# lst_butt = [vtk.vtkSphereWidget() for i in range(total_butt)]
# for i in range(total_butt):
#     sphereWidget = lst_butt[i]  # 产生一个点
#     sphereWidget.SetInteractor(mw.SceneManager.iren)  # 用iren进行互动
#     x, y, z = index2realworld(i)
#     sphereWidget.SetCenter(1.5, 0, (i - 2) / 3.0)  # 设置中心
#     sphereWidget.SetRadius(0.05 + 0.01 * i)  # 设置直径
#     sphereWidget.SetRepresentationToSurface()  # 不懂
#     sphereWidget.On()
#     butt.append(sphereWidget)
#
# butt[0].AddObserver("InteractionEvent", restart_total)
# butt[1].AddObserver("InteractionEvent", restart)
# butt[2].AddObserver("InteractionEvent", show_next_action)
# butt[3].AddObserver("InteractionEvent", show_last_action)
# butt[4].AddObserver("InteractionEvent", add_speed)
# butt[5].AddObserver("InteractionEvent", slow_speed)
# butt[6].AddObserver("InteractionEvent", save_grid)
# butt[7].AddObserver("InteractionEvent", load_grid)

mw.show()
#mw.SceneManager.iren.Initialize()
#mw.SceneManager.orient.SetInteractor(mw.SceneManager.iren )
mw.SceneManager.iren.Start()
sys.exit(app.exec_())
