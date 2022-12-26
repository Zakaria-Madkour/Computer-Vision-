import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from redis import Redis

cli = Redis('localhost')



def grab_frame():
    # include here the code that takes the pipeline images from the server
    return cli.get('Shared_Place')

# create five subplots
ax1 = plt.subplot(2, 3, 1)
ax2 = plt.subplot(2, 3, 2)
ax3 = plt.subplot(2, 3, 3)
ax4 = plt.subplot(2, 3, 4)
ax5 = plt.subplot(2, 3, 5)

image_list = grab_frame()

#create five image plots
im1 = ax1.imshow(image_list[0])
im2 = ax2.imshow(image_list[1])
im3 = ax3.imshow(image_list[2])
im4 = ax4.imshow(image_list[3])
im5 = ax5.imshow(image_list[4])


def update(i):
    image_list = grab_frame()
    im1 = ax1.imshow(image_list[0])
    im2 = ax2.imshow(image_list[1])
    im3 = ax3.imshow(image_list[2])
    im4 = ax4.imshow(image_list[3])
    im5 = ax5.imshow(image_list[4])


ani = FuncAnimation(plt.gcf(), update, interval=200)
plt.show()
