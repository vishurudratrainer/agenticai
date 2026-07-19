
#Tuple is immutable
t1=tuple()
t1=()
print(t1)
t1=(10,20)
print(t1)
t1=(10,)
print(t1)
print(t1[0])
#t1[0]=33 will not work since tuple is immutable
t1=(10,20,30)
for i in t1:
    print(i)

t1=("Hello","Fello")
print(t1)

t1=(1,"Hello")
print(t1)