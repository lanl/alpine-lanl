## plot a gsussian distribution
import numpy as np
import matplotlib.pyplot as plt

mean = 2.55
std = np.sqrt(20.18) 
variance = np.square(std)
x = np.arange(mean-2,mean+2,.01)
f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))

plt.plot(x,f)
plt.ylabel('Probability Density',size=15)
plt.xlabel('Particle Density',size=15)
plt.title('Statistical representation of the target bubble feature',size=15)
# plt.show()

plt.savefig('bubble_feature.png')