import cv2
import tqdm
import torch
import numpy as np
import threading
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

model_path = "face_landmarker/face_landmarker.task"


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [
        face_blendshapes_category.category_name
        for face_blendshapes_category in face_blendshapes
    ]
    face_blendshapes_scores = [
        face_blendshapes_category.score
        for face_blendshapes_category in face_blendshapes
    ]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(
        face_blendshapes_ranks,
        face_blendshapes_scores,
        label=[str(x) for x in face_blendshapes_ranks],
    )
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(
            patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top"
        )

    ax.set_xlabel("Score")
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def face_landmarker(input, output, detector, id):
    videoin = cv2.VideoCapture(input)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")
    if int(major_ver) < 3:
        fps = videoin.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = videoin.get(cv2.CAP_PROP_FPS)

    nr = int(videoin.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = tqdm.tqdm(desc=f"ID: {id}", total=nr)
    face = []
    blendshapes = [0] * 53

    print("T", id)
    while True:
        _, frame = videoin.read()
        bar.update()
        if frame is None:
            break
        time = len(face) / fps * 1000
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(frame)
        if len(detection_result.face_blendshapes) > 0:
            blendshapes = [time] + [
                face_blendshapes_category.score
                for face_blendshapes_category in detection_result.face_blendshapes[
                    0
                ]
            ]
            # out = [[landmark.x, landmark.y, landmark.z] for landmark in detection_result.face_landmarks[0]]
        else:
            blendshapes[0] = time
        face.append(blendshapes)
    torch.save(torch.tensor(face).bfloat16(), f"{output}")


if __name__ == "__main__":
    import sys
    import os
    inf = sys.argv[1]
    outf = sys.argv[2]
    
    os.makedirs(outf, exist_ok=True)
    fls = set(os.listdir(inf))
    lfls = len(fls)
    fls_l = threading.Lock()

    def dland(i):
        # STEP 2: Create an FaceLandmarker object.
        base_options = python.BaseOptions(
            model_asset_path=model_path, delegate=mp.tasks.BaseOptions.Delegate.CPU if i > 4 else mp.tasks.BaseOptions.Delegate.GPU
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        while True:
            with fls_l:
                if len(fls) == 0:
                    break
                f = fls.pop()
            id = f"{lfls-len(fls)}/{lfls}"
            face_landmarker(inf+"/"+f, outf+"/"+f, detector, id)

    for i in range(20):
        threading.Thread(target=dland,kwargs={"i":i}).start()
# ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']
# ['中性', '眉毛左下', '眉毛右下', '眉内上', '眉毛外上左', '眉毛外上右', '脸颊蓬松', '脸颊左斜视', '脸颊斜视'右', '眼睛向左眨眼', '眼睛向右眨眼', '眼睛向左向下看', '眼睛向右向下看', '眼睛向左看', '眼睛向右看', '眼睛向左向外看', '眼睛向右看'，'眼睛向左看'，'眼睛向右看'，'眼睛向左斜视'，'眼睛向右斜视'，'眼睛向左张开'，'眼睛向右张开'，'下巴向前'， '下颌左', '下颌张开', '下颌右', '嘴闭合', '嘴左酒窝', '嘴酒窝右', '嘴皱眉左', '嘴皱眉右', '嘴漏斗', '嘴左”、“嘴下左下”、“嘴下下右”、“嘴按左”、“嘴按右”、“嘴撅起”、“嘴右”、“嘴卷下”、“嘴卷上” ', '嘴耸肩下', '嘴耸肩上', '嘴向左微笑', '嘴微笑向右', '嘴向左伸展', '嘴向右伸展', '嘴上左上', '嘴上右上', '鼻子向左冷笑', '鼻子向右冷笑']
