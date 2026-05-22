import matplotlib.pyplot as plt

x = [1,2,3]

plt.plot(x, [2,4,6], label="Temperature")
plt.plot(x, [1,3,5], label="Sales")
plt.legend()
plt.show()