#---import libraries---------------
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as plt3d


#---functions----------------------
def Energymaker(SM,r): #function to create theoretical intensities as a function of test distances from microphone.

	#test path 
	r_x, r_y, r_z, number = r

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

	return I

#---main function---

def position_estimate(I,SM,t,number): #estimate positions from positions of microphones and Intensity data

	Isetx = np.array([I[0][t],I[1][t]])
	Isety = np.array([I[2][t],I[3][t]])
	Isetz = np.array([I[4][t],I[5][t]])

	def k(I):
		if max(I) == I[0]: #this process comes from solving k**2 * d1 **2 = d2 **2 for x and y
			mult = 1 # if k is on side of d1
		else:
			mult = -1 # if k is on side of d2
		return np.sqrt(min(I)/max(I)), mult

	def c(R,shift):
		return R + shift

	def m(R,k,c,mult):
		return c + mult*(R * (k**2 + 1)/(k**2 - 1))

	def l_sqd(R,k):
		return R **2 * (((k**2 + 1)/(k**2 - 1))**2 - 1)

	def position(p,*data): #function to find zeros for
		x,y,z = p
		m1,m2,m3,l1,l2,l3 = data

		return(
			(x+m1)**2+y**2+z**2-l1,
			(y+m2)**2+x**2+z**2-l2,
			(z+m3)**2+y**2+x**2-l3)

	R1, R2 = [.08,.08]
	R3 = .1325

	k1, mult1 = k(Isetx)
	k2, mult2 = k(Isety)
	k3, mult3 = k(Isetz)
	
	c1,c2,c3 = c(R1,0.05),c(R2,0.05),c(R3,0.035)
	m1,m2,m3 = m(R1,k1,c1,mult1), m(R2,k2,c2,mult2), m(R3,k3,c3,mult3)
	l1,l2,l3 = l_sqd(R1,k1), l_sqd(R2,k2), l_sqd(R3,k3)
	
	data = (m1,m2,m3,l1,l2,l3)
	bias = np.mean(np.array([np.sqrt(1/I[0][t]),np.sqrt(1/I[2][t]),np.sqrt(1/I[4][t])]))
	pos = bias*(bias/2)
	pos1, pos2, pos3 = (c1 + mult1*pos), (c2 + mult2*pos), (c3 + mult3*pos)
	p = (pos1,pos2,pos3)

	x,y,z = fsolve(position,p,args = data,factor = 10)

	return -1*x,-1*y,-1*z

#---defining terms-----------------

#position of recorders
rec_pos = np.array([(0.05,0.21,0,0,0,0),(0,0,0.05,0.21,0,0),(0,0,0,0,0.035,0.3)]) #((x_n),(y_n),(z_n)) n = 0,1,2,3,4,5

#each recorder (SM_n) position as (x,y,z)
SM_1 = np.array([(rec_pos[0][0],rec_pos[0][1]), (rec_pos[1][0],rec_pos[1][1]),(rec_pos[2][0],rec_pos[2][1])])
SM_2 = np.array([(rec_pos[0][2],rec_pos[0][3]), (rec_pos[1][2],rec_pos[1][3]),(rec_pos[2][2],rec_pos[2][3])])
SM_3 = np.array([(rec_pos[0][4],rec_pos[0][5]), (rec_pos[1][4],rec_pos[1][5]),(rec_pos[2][4],rec_pos[2][5])])

SM = np.array([SM_1,SM_2,SM_3])

#test position
number = 500
r_x = np.linspace(-10*np.pi,10*np.pi,num = number)
r_y = 100*np.cos(r_x)
r_z = 100*np.sin(r_x)
r = (r_x,r_y,r_z, number)

#---executable code----------------

I = Energymaker(SM,r)
t = 0
posx, posy, posz = [[],[],[]]

for i in I[0]:  #loop to create found positons for each step of test path

	x,y,z = position_estimate(I,SM,t,number)
	posx.append(x)
	posy.append(y)
	posz.append(z)
	t += 1

x = np.array(posx)
y = np.array(posy)
z = np.array(posz)

#---producing error readouts-------

x_err = 100*(r_x-x)/r_x
y_err = 100*(r_y-y)/r_y
z_err = 100*(r_z-z)/r_z

print('average x error: ',np.median(x_err),'%')
print('average y error: ',np.median(y_err),'%')
print('average z error: ',np.median(z_err),'%')

x_err = list(x_err)

for i in x_err:
	t = x_err.index(i)
	if i >= 5.00:
		print("Error greater than 5% for x at ", t, x[t],y[t],z[t])
	if y_err[t] >= 5.00:
		print("Error greater than 5% for y at ", t, x[t],y[t],z[t])
	if z_err[t] >= 5.00:
		print("Error greater than 5% for z at ", t, x[t],y[t],z[t])


#---plotting-----------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# ax.set_xlim3d(-2, 2)
# ax.set_ylim3d(-2, 2)
# ax.set_zlim3d(-2, 2)

ax.plot3D(SM_1[0],SM_1[1],SM_1[2], linestyle = 'none', marker = 'o') # mark recorders
ax.plot3D(SM_2[0],SM_2[1],SM_2[2], linestyle = 'none', marker = 'o')
ax.plot3D(SM_3[0],SM_3[1],SM_3[2], linestyle = 'none', marker = 'o')

ax.plot3D(x,y,z, color = 'r', linestyle = 'none',marker = '.')
ax.plot3D(r_x,r_y,r_z, 'g', linestyle = 'none', marker = '.')#test path

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot3D(x,y,z, color = 'r')

plt.show()
