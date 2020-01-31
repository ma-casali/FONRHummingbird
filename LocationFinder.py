from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from mpl_toolkits import mplot3d as plt3d
from sympy import *


rec_pos = np.array([(0.2,0.05,0,0,0,0),(0,0,0.2,0.05,0,0),(0,0,0,0,0.3,0.05)]) #((x_n),(y_n),(z_n)) n = 0,1,2,3,4,5
SM_1 = np.array([(rec_pos[0][0],rec_pos[0][1]), (rec_pos[1][0],rec_pos[1][1]),(rec_pos[2][0],rec_pos[2][1])])
SM_2 = np.array([(rec_pos[0][2],rec_pos[0][3]), (rec_pos[1][2],rec_pos[1][3]),(rec_pos[2][2],rec_pos[2][3])])
SM_3 = np.array([(rec_pos[0][4],rec_pos[0][5]), (rec_pos[1][4],rec_pos[1][5]),(rec_pos[2][4],rec_pos[2][5])])

print(SM_1[0][1],SM_2[1][1],SM_3[2][1])

SM = np.array([SM_1,SM_2,SM_3])

def Energymaker(SM):

	# test positions
	number = 1000
	r_x = np.linspace(-20,20,num = number)
	r_y = .2*np.sin(r_x)
	r_z = .2*np.cos(r_x)

	def d(SM,r_x,r_y,r_z):

		SM_1, SM_2, SM_3 = SM

		# initiate lists of distances
		d00,d01,d10,d11,d20,d21 = [[],[],[],[],[],[]]

		# append all values for distances across range of path (max ind = number - 1)
		t = 0
		for i in r_x:
			d00.append(np.sqrt((r_x-SM_1[0][0])**2+(r_y-SM_1[1][0])**2+(r_z-SM_1[2][0])**2))
			d01.append(np.sqrt((r_x-SM_1[0][1])**2+(r_y-SM_1[1][1])**2+(r_z-SM_1[2][1])**2))
			d10.append(np.sqrt((r_x-SM_2[0][0])**2+(r_y-SM_2[1][0])**2+(r_z-SM_2[2][0])**2))
			d11.append(np.sqrt((r_x-SM_2[0][1])**2+(r_y-SM_2[1][1])**2+(r_z-SM_2[2][1])**2))
			d20.append(np.sqrt((r_x-SM_3[0][0])**2+(r_y-SM_3[1][0])**2+(r_z-SM_3[2][0])**2))
			d21.append(np.sqrt((r_x-SM_3[0][1])**2+(r_y-SM_3[1][1])**2+(r_z-SM_3[2][1])**2))
			t += 1

		#transition to numpy array
		d00, d01 = np.array(d00[0]),np.array(d01[0])
		d10, d11 = np.array(d10[0]),np.array(d11[0])
		d20, d21 = np.array(d20[0]),np.array(d21[0])

		return np.array([d00,d01,d10,d11,d20,d21])

	# call d() function
	d = d(SM, r_x, r_y, r_z)

	# set intensity as proportional to 1/d**2. 
	# divide by each other, and all other terms for intensity fall out so this is the only important dependency
	I = d**-2

	I00,I01,I10,I11,I20,I21 = I

	return I, d

#---main function----------------------------------------

def position_estimate(I,SM,t):

	Isetx = np.array([I[0][t],I[1][t]])
	Isety = np.array([I[2][t],I[3][t]])
	Isetz = np.array([I[4][t],I[5][t]])

	def abfind(I,SM):

		if SM[2][1] != 0:
			R = .125
		else:
			R = .075

		k = min(I)/max(I)

		a = R**2 * (1-k)**2/(1+k)**2 #this value is actually a**2
		b = R**2 * (4*k)/(1+k)**2 #similarly this is b**2

		if I[0]<I[1]:
			neg = True
		else:
			neg = False

		return a,b,neg

	a1, b1, neg1 = abfind(Isetx,SM_1)
	a2, b2, neg2 = abfind(Isety,SM_2)
	a3, b3, neg3 = abfind(Isetz,SM_3)

	#Centers of Recorders
	c1 = .2 - (SM_1[0][0]-SM_1[0][1])/2
	c2 = .2 - (SM_2[1][0]-SM_2[1][1])/2
	c3 = .3 - (SM_3[2][0]-SM_3[2][1])/2

	def position(p,*data):
		x,y,z = p
		a1,b1,a2,b2,a3,b3,c1,c2,c3 = data
		return((x-c1)**2/a1 - y**2/b1 - z**2/b1 - 1,
			(y-c2)**2/a2 - x**2/b2 - z**2/b2 - 1,
			(z-c3)**2/a3 - x**2/b3 - y**2/b3 - 1)

	data = (a1,b1,a2,b2,a3,b3,c1,c2,c3)
	if neg1 == True:
		pos1 = SM_1[0][0]
	else:
		pos1 = SM_1[0][1]

	if neg2 == True:
		pos2 = SM_2[1][0]
	else:
		pos2 = SM_2[1][1]

	if neg3 == True:
		pos3 = SM_3[2][0]
	else:
		pos3 = SM_3[2][1]

	p0 = (pos1,pos2,pos3)

	x,y,z = fsolve(position,p0,args = data)

	x,y,z = [[x],[y],[z]]


	return x,y,z

#---executable code----------------

I, d = Energymaker(SM)

t = 0
posx, posy, posz = [[],[],[]]

for i in I[0]:

	x,y,z = position_estimate(I,SM,t)
	posx.append(x[0])
	posy.append(y[0])
	posz.append(z[0])
	t += 1

x = np.array(posx)
y = np.array(posy)
z = np.array(posz)
print(x,y,z)

#---test position------------------

number = 500
r_x = np.linspace(0,.3,num = number)
r_y = .2*np.sin(r_x)
r_z = .2*np.cos(r_x)


#---plotting-----------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot3D(SM_1[0],SM_1[1],SM_1[2], linestyle = 'none', marker = 'o')
ax.plot3D(SM_2[0],SM_2[1],SM_2[2], linestyle = 'none', marker = 'o')
ax.plot3D(SM_3[0],SM_3[1],SM_3[2], linestyle = 'none', marker = 'o')
ax.plot3D(x,y,z, color = 'r')
ax.plot3D(r_x,r_y,r_z)

plt.show()