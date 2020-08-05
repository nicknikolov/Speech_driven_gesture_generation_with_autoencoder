import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import pandas as pd
from scipy.spatial.transform import Rotation as R
import argparse

def draw_plane():
    # https://stackoverflow.com/questions/3461869/plot-a-plane-based-on-a-normal-vector-and-a-point-in-matlab-or-matplotlib
    point  = np.array([1, 2, 3])
    normal = np.array([1, 1, 2])

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))

    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z)
    plt.show()

def animate_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.

    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)

    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    return scatters

def parse_openface(filepath):
    data = pd.read_csv(filepath, sep=', ', engine='python')

    # swap axis because matplotlib is WRONG
    # Ts = data[['pose_Tz', 'pose_Tx', 'pose_Ty']].to_numpy()
    # Ts[:, 2] = -Ts[:, 2]
    # Rs = data[['pose_Rz', 'pose_Rx', 'pose_Ry']].to_numpy()

    Ts = data[['pose_Tz', 'pose_Tx', 'pose_Ty']].to_numpy()
    Rs = data[['pose_Rz', 'pose_Rx', 'pose_Ry']].to_numpy()

    Ts_dir = Ts.copy()
    # Ts_dir[:, 0] += 50
    Ts_dir[:, 2] -= 50

    for i in range(Ts_dir.shape[0]):
        t = Ts[i, :]
        dir = Ts_dir[i, :] - t
        r = Rs[i, :]
        # r[2] = -r[2]

        rot = R.from_rotvec(r)
        new_dir = rot.apply(dir) + t
        Ts_dir[i, :] = new_dir

    # print(Ts_dir.shape)

    # 30 to 20fps
    Ts = np.delete(Ts, slice(None, None, 3), axis=0)
    Ts_dir = np.delete(Ts_dir, slice(None, None, 3), axis=0)

    data = np.expand_dims(Ts, axis=1)
    data = np.stack([Ts, Ts_dir], axis=1)
    print('openface data shape', data.shape)

    # print(np.max(data[:, 0, 2]))
    # print(np.shape(data)) # (101, 2, 3)
    # print(range(data[0].shape[0])) # 0..2 so for all points

    return data

def parse_decoded(filepath):
    data_raw = np.loadtxt(filepath)

    # for prediction with 6 coords
    data = np.zeros((data_raw.shape[0], 2, 3))
    data[:, 0, :] = data_raw[:,0:3] # translation
    data[:, 1, :] = data_raw[:,3:6] # look direction

    return data


def main(save=True):
    """
    Creates the 3D figure and animates it with the input data.

    Args:
        data (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
    """

    parser = argparse.ArgumentParser(
        description='Animate head translation and look direction')
    parser.add_argument('--input', '-i', required=True,
                        help='input filename')
    parser.add_argument('--output', '-o', required=True,
                        help='output filename')
    parser.add_argument('--openface', dest='openface', action='store_true')
    parser.add_argument('--no-openfacee', dest='openface', action='store_false')
    parser.set_defaults(openface=False)

    args = parser.parse_args()

    data = None
    if (args.openface):
        data = parse_openface(args.input)
    else:
        data = parse_decoded(args.input)


    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Initialize scatters
    # scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]
    scatters = [ ax.scatter(data[0][0,0:1], data[0][0,1:2], data[0][0,2:], marker='o'),
                 ax.scatter(data[0][1,0:1], data[0][1,1:2], data[0][1,2:], marker='d')]

    # Number of iterations
    iterations = len(data)

    xmin = np.min(data[:, 0, 0]) - 5
    xmax = np.max(data[:, 0, 0]) + 5

    ymin = np.min(data[:, 0, 1]) - 5
    ymax = np.max(data[:, 0, 1]) + 5

    zmin = np.min(data[:, 0, 2]) - 5
    zmax = np.max(data[:, 0, 2]) + 5

    # Setting the axes properties
    ax.set_xlim3d([xmin, xmax])
    ax.set_xlabel('X')

    ax.set_ylim3d([ymin, ymax])
    ax.set_ylabel('Y')

    ax.set_zlim3d([zmin, zmax])
    ax.set_zlabel('Z')

    # Provide starting angle for the view.
    ax.view_init(25, 20)

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                       interval=50, blit=False, repeat=False)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(args.output, writer=writer)

    # plt.show()


if __name__ == '__main__':
    main()
