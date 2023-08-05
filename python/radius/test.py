import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size


fig = plt.figure(figsize=(6, 6))

# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(1.0), Size.Fixed(4.5)]
v = [Size.Fixed(0.7), Size.Fixed(2)]  # Set y-axis width to 0.8 inches

divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

# The width and height of the rectangle are ignored.
ax = fig.add_axes(divider.get_position(),
                  axes_locator=divider.new_locator(nx=1, ny=1))

ax.plot([1,2,3,4,5,6,7])


ax.yaxis.set_major_formatter(lambda y, pos: f"{y:>5.2f}")

plt.show()