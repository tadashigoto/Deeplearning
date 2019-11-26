import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
subplot = fig.add_subplot(1,1,1)
subplot.set_xlim(1,12)
subplot.scatter(range(1,13),[1,2,3,4,5,6,7,8,9,10,11,12])
linex = np.linspace(1,12,100)
liney = linex+3
subplot.plot(linex,liney)
plt.show()
