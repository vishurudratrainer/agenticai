class Parent:
    def __init__(self,n1):
        self.n1=n1
    def display(self):
        print("n1",self.n1)
class Child1(Parent):#Inheritance
    def __init__(self,n1,n2):
        Parent.__init__(self,n1)#Calling base class constructor
        self.n2=n2

c1=Child1(22,33)
print(c1.n1)
c1.display()
print(c1.n2)