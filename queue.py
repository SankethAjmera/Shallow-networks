#Implementing Queue class in Pyhton
#Stack is a FIFO type class

class queue(object):


    def __init__(self,n): #Initialising an instance of the class Queue
        self.queue = ['']*n  # n is the length of the queue
        self.n = n   #All entries except enqueue are empty ('')
        self.head = 0
        self.tail = 0

    def __repr__(self): #To represent the queue in readable format
        return '<stack({})>'.format(self.queue)

    def enqueue(self,x):
        self.queue[self.tail] = x
        if self.tail == self.n-1:
            self.tail = 0
        else:
            self.tail = self.tail+1


    def dequeue(self):
        x = self.queue[self.head]
        if self.head == self.n-1:
            self.head = 0
        else:
            self.head = self.head+ 1
        return x

#TEST CASES

Q = queue(10)
Q.enqueue(5)
Q.enqueue(6)
Q.enqueue(8)
print(Q.dequeue())
print(Q.dequeue())
print(Q.dequeue())
print(Q.dequeue())
print(Q)
