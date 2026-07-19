s1=set()
print(s1)
s1.add(22)
s1.add(22)
s1.add(22)
print(s1)
s2={10,20,30}
print(s2)
for i in s2:
    print(i)

s1={10,20,30}
s2={20,30}
s3={40,50}
print(s1.intersection(s2))
print(s1.issuperset(s2))
print(s2.issubset(s1))
print(s1.isdisjoint(s3))
print(s1-s2)