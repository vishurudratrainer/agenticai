class Parent1:
    def __init__(self,n1):
        self.n1=n1
    def display(self):
        print("n1",self.n1)

class Parent2:
    def __init__(self,n2):
        self.n2=n2
    def display(self):
        print("n2",self.n2)

class Child(Parent2,Parent1):#Sequence
    def __init__(self,n1,n2):
        Parent1.__init__(self,n1)
        Parent2.__init__(self,n2)

c=Child(22,33)
c.display()