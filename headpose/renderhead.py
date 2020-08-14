import pyrender
import trimesh
import numpy as np
import pandas as pd
import cv2
import os.path
from scipy.spatial.transform import Rotation as R
from subprocess import call
from scipy.io import wavfile
import tempfile

def parse_openface(filepath):
    data = pd.read_csv(filepath, sep=', ', engine='python')

    # move to the center, just as in the training data
    Ts = data[['pose_Tx', 'pose_Ty', 'pose_Tz']].to_numpy()
    mean_pose = Ts.mean(axis=(0))
    Ts = Ts - mean_pose
    Rs = data[['pose_Rx', 'pose_Ry', 'pose_Rz']].to_numpy()

    return Ts, Rs

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    SOURCE: https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def encode_rot(rotvec):
    # positive z in OpenCV coord system
    unitx = np.array([-1, 0, 0])
    unity = np.array([0, -1, 0])
    unitz = np.array([0, 0, 1])
    coord_sys = np.array([unitx, unity, unitz])
    r = R.from_rotvec(rotvec)
    coord_sys = r.apply(coord_sys)
    return coord_sys

def decode_rot(encoded):
    # positive z in OpenCV coord system
    unitx = np.array([-1, 0, 0])
    unity = np.array([0, -1, 0])
    unitz = np.array([0, 0, 1])
    decodedx = decode_axis(encoded[0], unitx)
    decodedy = decode_axis(encoded[1], unity)
    decodedz = decode_axis(encoded[2], unitz)

    # holy cow, this can't be the right way
    pitch = (decodedy[0] + decodedz[0]) / 2
    yaw = (decodedx[1] + decodedz[1]) / 2
    roll = (decodedx[2] + decodedy[2]) / 2

    decoded = np.array([pitch, yaw, roll])
    return decoded

def decode_axis(lookdir, unit):
    rm = rotation_matrix_from_vectors(unit, lookdir)
    r = R.from_matrix(rm)
    return r.as_rotvec()

def render_head(t, r):
    mesh = trimesh.load_mesh('headpose/head.obj')
    mesh.vertices = (r @ mesh.vertices.T).T
    mesh.vertices += t/1000

    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    intensity = 0.5

    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 0.5])
    scene.add(camera, pose=camera_pose)

    light = pyrender.SpotLight(color=np.ones(3), intensity=2.0,
                                   innerConeAngle=np.pi/16.0)
    scene.add(light, pose=camera_pose)
    # pyrender.Viewer(scene)

    W = 800
    H = 800

    try:
        r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        color, _ = r.render(scene)
    except Exception as e:
        print('pyrender: Failed rendering frame: ', e)
        color = np.zeros((H, W, 3), dtype='uint8')

    return color[..., ::-1]

def render_sequence_meshes(audio_fname, Ts, Rs, out_path):
    FPS = 20

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), FPS, (800, 800), True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (800, 800), True)

    for i in range(Ts.shape[0]):
        t = Ts[i]

        rotvec = Rs[i]

        encoded = encode_rot(rotvec)
        decoded = decode_rot(encoded)

        # move to openGL coord system
        t[1] = -t[1]
        t[2] = -t[2]
        decoded[1] = -decoded[1]
        decoded[2] = -decoded[2]

        # pass as rot matrix
        r = R.from_rotvec(decoded)
        img = render_head(t, r.as_matrix())
        writer.write(img)

    writer.release()

    video_fname = os.path.join(out_path, 'video.mp4')
    cmd = ('ffmpeg' + ' -y -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
        audio_fname, tmp_video_file.name, video_fname)).split()
    call(cmd)

    # cmd = 'ffmpeg' + ' -y -i new_data/video.mp4 -i obama/annotated/annotated32.mp4 -filter_complex [0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid] -map [vid] -c:v libx264 -crf 23 -preset veryfast new_data/merge.mp4'
    # call(cmd)

ex32 = 'obama_processed/train/labels/pose32.csv'
audio32 = 'obama_processed/train/inputs/audio32.wav'

Ts, Rs = parse_openface(ex32)
# 30 to 20fps
Ts = np.delete(Ts, slice(None, None, 3), axis=0)
Rs = np.delete(Rs, slice(None, None, 3), axis=0)

# t = Ts[0]
# rotvec = Rs[0]
# render_head(t, r)

render_sequence_meshes(audio32, Ts, Rs, 'new_data')


