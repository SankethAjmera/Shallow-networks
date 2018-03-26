#Implementing Stack class in Pyhton
#Stack is a LIFO type class
class stack:

    def __init__(self): #Initialising an instance of the class stack
        self.stack = []

    def __repr__(self):
        return '<stack({})>'.format(self.stack)

    def isempty(self):
        return self.stack == []

    def push(self,x):
        return self.stack.append(x)

    def pop(self):
        if self.isempty():
            return "underflow"
        else:
            return self.stack.pop()

S = stack()
S.push(100)
S.push(5)
S.pop()
print(S)
