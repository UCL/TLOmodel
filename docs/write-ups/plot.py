

import matplotlib.pyplot as plt

# Define x and y axis values
x_values = [0, 15686.54, 3660.09, 20929.22]
y_values = [0, 2.0227129, 0.0572584, 1.7867897]

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color='blue')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)  # Horizontal line at y=0
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)  # Vertical line at x=0
plt.xlabel('DALYs averted')
plt.ylabel('Difference in costs')
plt.title('Cost effectiveness plane')
plt.legend()
plt.grid(True)
plt.show()
