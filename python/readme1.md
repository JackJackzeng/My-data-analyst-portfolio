This is a case study using python /NumPy/ Pandas/Matplot tools to train the **titanic data** learning using different mathematic algorithms. The data source can be downlaod from the public website, such as [Titanic dataset](https://www.kaggle.com/datasets/hesh97/titanicdataset-traincsv)

key code :
```
x=np.linspace(-3,3,100)
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

plt.show()
```

If you are interested reviewing and testing the project please use my [**Google CoLab notebook**](https://colab.research.google.com/drive/1FhjdinLX9dejz4spkXrhXDHuENXQM3Fq#updateTitle=true&folderId=1Q9EqShSEW9F3ULWA9Z6sSSbFlLBSQTmO)

[Click here to return to the main page](../README.md)
