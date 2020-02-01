#---import libraries---------------
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as plt3d


#---functions----------------------
def Energymaker(SM): #function to create theoretical intensities as a function of test distances from microphone.

	#test path 
	number = 1000
	r_x = np.linspace(-0.25,0.25,num = number)
	r_y = r_x
	r_z = r_x

	#distance from each microphone
	def d(SM,r_x,r_y,r_z):

		SM_1, SM_2, SM_3 = SM

		d00,d01,d10,d11,d20,d21 = [[],[],[],[],[],[]]

		t = 0
		for i in r_x:
			d00.append(np.sqrt((r_x-SM_1[0][0])**2+(r_y-SM_1[1][0])**2+(r_z-SM_1[2][0])**2))
			d01.append(np.sqrt((r_x-SM_1[0][1])**2+(r_y-SM_1[1][1])**2+(r_z-SM_1[2][1])**2))
			d10.append(np.sqrt((r_x-SM_2[0][0])**2+(r_y-SM_2[1][0])**2+(r_z-SM_2[2][0])**2))
			d11.append(np.sqrt((r_x-SM_2[0][1])**2+(r_y-SM_2[1][1])**2+(r_z-SM_2[2][1])**2))
			d20.append(np.sqrt((r_x-SM_3[0][0])**2+(r_y-SM_3[1][0])**2+(r_z-SM_3[2][0])**2))
			d21.append(np.sqrt((r_x-SM_3[0][1])**2+(r_y-SM_3[1][1])**2+(r_z-SM_3[2][1])**2))
			t += 1

		d00, d01 = np.array(d00[0]),np.array(d01[0])
		d10, d11 = np.array(d10[0]),np.array(d11[0])
		d20, d21 = np.array(d20[0]),np.array(d21[0])

		return np.array([d00,d01,d10,d11,d20,d21])

	d = d(SM, r_x, r_y, r_z)

	# for ratio of 'I's all other terms fall out so d**-2 term is only important dependency
	I = d**-2

	I00,I01,I10,I11,I20,I21 = I

	return I, d

#---main function---

def position_estimate(I,SM,t): #estimate positions from positions of microphones and Intensity data

	Isetx = np.array([I[0][t],I[1][t]])
	Isety = np.array([I[2][t],I[3][t]])
	Isetz = np.array([I[4][t],I[5][t]])

	def abfind(I,SM): #finds both a, b for hyperboloid (b = c for microphones)

		if SM[2][1] != 0:
			R = .125
		else:
			R = .075

		k = min(I)/max(I)

		a = R**2 * (1-k)**2/(1+k)**2 #this value is actually a**2
		b = R**2 * (4*k)/(1+k)**2 #similarly this is b**2
		
		# I1 > I2 means d1 < d2 means that source is closer to more positive mic
		# neg affects bias
		if I[0]>I[1]: 
			neg = True
		else:
			neg = False

		return a,b,neg

	a1, b1, neg1 = abfind(Isetx,SM_1)
	a2, b2, neg2 = abfind(Isety,SM_2)
	a3, b3, neg3 = abfind(Isetz,SM_3)

	# centers of microphone pairs (recorders)
	c1 = .05 + (SM_1[0][0]-SM_1[0][1])/2
	c2 = .05 + (SM_2[1][0]-SM_2[1][1])/2
	c3 = .035 - (SM_3[2][0]-SM_3[2][1])/2

	def position(p,*data): #function to find zeros for
		x,y,z = p
		a1,b1,a2,b2,a3,b3,c1,c2,c3 = data
		return((x-c1)**2/a1 - y**2/b1 - z**2/b1 - 1,
			(y-c2)**2/a2 - x**2/b2 - z**2/b2 - 1,
			(z-c3)**2/a3 - x**2/b3 - y**2/b3 - 1)

	# bias the starting values towards mic that picked up highest intensity
	# first indice is x,y or z, second is close to origin [0] or far from origin [1]
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

	data = (a1,b1,a2,b2,a3,b3,c1,c2,c3)
	p0 = (pos1,pos2,pos3)
	x,y,z = fsolve(position,p0,args = data)

	return x,y,z

#---defining terms-----------------

#position of recorders
rec_pos = np.array([(0.21,0.05,0,0,0,0),(0,0,0.21,0.05,0,0),(0,0,0,0,0.3,0.035)]) #((x_n),(y_n),(z_n)) n = 0,1,2,3,4,5

#each recorder (SM_n) position as (x,y,z)
SM_1 = np.array([(rec_pos[0][0],rec_pos[0][1]), (rec_pos[1][0],rec_pos[1][1]),(rec_pos[2][0],rec_pos[2][1])])
SM_2 = np.array([(rec_pos[0][2],rec_pos[0][3]), (rec_pos[1][2],rec_pos[1][3]),(rec_pos[2][2],rec_pos[2][3])])
SM_3 = np.array([(rec_pos[0][4],rec_pos[0][5]), (rec_pos[1][4],rec_pos[1][5]),(rec_pos[2][4],rec_pos[2][5])])

SM = np.array([SM_1,SM_2,SM_3])

#test position
number = 1000
r_x = np.linspace(-0.25,0.25,num = number)
r_y = r_x
r_z = r_x

#---executable code----------------

I, d = Energymaker(SM)

t = 0
posx, posy, posz = [[],[],[]]

for i in I[0]:  #loop to create found positons for each step of test path

	x,y,z = position_estimate(I,SM,t)
	posx.append(x)
	posy.append(y)
	posz.append(z)
	t += 1

x = np.array(posx)
y = np.array(posy)
z = np.array(posz)

#---plotting-----------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot3D(SM_1[0],SM_1[1],SM_1[2], linestyle = 'none', marker = 'o')# mark recorders
ax.plot3D(SM_2[0],SM_2[1],SM_2[2], linestyle = 'none', marker = 'o')
ax.plot3D(SM_3[0],SM_3[1],SM_3[2], linestyle = 'none', marker = 'o')

ax.plot3D(x,y,z, color = 'r') #, linestyle = 'none', marker = '.')
ax.plot3D([x[759],x[760]],[y[759],y[760]],[z[759],z[760]], color = 'r', linestyle = 'none', marker = 'o') #mark where change in closest mic to sound source happens

ax.plot3D(r_x,r_y,r_z, 'g')#test path

plt.show()