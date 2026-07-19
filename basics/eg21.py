#Within a class 
# each and every function will have self as first argument
# any properties/avriable we have created in constructor
# __init__
class Sample:
    def __init__(self,a):#constructor
        self.a=a
    def display(self):
        print(self.a)
    def add(self,a,b):
        return a+b

s1=Sample(20)#calls the constructor
s1.display()
print(s1.a)
print(s1.add(20,30))