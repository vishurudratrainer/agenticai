l1=[]
l1=list()
print(l1)
l1.append(22)
l1.append(33)
print(l1)
print(l1[0])
print(l1[1])
l1[0]=44
print(l1)
l1.remove(44)#deleting by value
print(l1)
del l1[0]
print(l1)
l2=[22,44]
for i in l2:
    print(i)
for i,v in enumerate(l2):
    print(i,v)
print(len(l2))
l2.clear()
print(len(l2))
l3=[10,20,30]
l3.reverse()
print(l3)

