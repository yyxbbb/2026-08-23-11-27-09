import numpy as np
def cos(a,b):
    return a@b/(np.linalg.norm(a)*np.linalg.norm(b))
a=np.array([1,2])
b=np.array([3,4])
c=np.array([-2,1])
cheng=a@b
cheng1=np.linalg.norm(a)*np.linalg.norm(b)*cos(a,b)
print(cheng)
print(cheng1)
print(np.isclose(a@b,b@a))
print(cos(a,b))
print(cos(a,c))
