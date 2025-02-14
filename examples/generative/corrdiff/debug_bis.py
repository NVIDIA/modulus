
# class classA(object):
#     def __init__(self, x):
#         self.x = x

#     def __call__(self):
#         return self.x + 5

#     def doA(self, y):
#         return self.x + y


# class classB(classA):
#     def __new__(cls, x):
#         out = super().__new__(cls)
#         out._x = x
#         return out

#     def __init__(self, z):
#         super().__init__(z - 3)
#         self.z = z

#     def doB(self, m):
#         return m * self.x + 10 * m * self.z


# b = classB(10)
# b.doB(10.0)
# b.doA(0.1)

def fun(a, b, c, d=False, **kwargs):
    print(a, b, c, d, kwargs)
    return

fun(1, 2, 3, True, k=None)
