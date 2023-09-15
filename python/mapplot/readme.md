This is a case study using python /numpy/ pandas to visualzation the data.

key code :
(`x=np.linspace(-3,3,100)
y=2*x+1
y1=x**2

#X Y limitation
plt.xlim((-1,2))
plt.ylim((-2,3))

ticks=np.linspace(-2,2,11)
print(ticks)

plt.xticks(ticks)
plt.yticks([-1,0,1,2,3],['level1','level2','level3','level4','level5'])

#Legend
L1,= plt.plot(x,y,c='red',linewidth=1,linestyle="--")
L2,= plt.plot(x,y1,c='blue',linewidth=5,linestyle="-")
plt.legend(handles=[L1,L2],labels=['test1','test2'],loc='best')

plt.show()`)



