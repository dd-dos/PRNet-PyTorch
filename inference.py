import argparse
import logging
import time
import cv2
import torch
import numpy as np
from package.main import FacePatternModel

logging.getLogger().setLevel(logging.INFO)

def video_infer(args):
    '''
    3D landmarks video inference
    '''
    if args.video_path:
        cap = cv2.VideoCapture(f"{args.video_path}")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter('filename.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)
    else:
        cap = cv2.VideoCapture(0)

        if args.save:
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            size = (frame_width, frame_height)
            out = cv2.VideoWriter('prnet.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, size)
    
    model = FacePatternModel(args.model_path)
    face_detector = torch.jit.load(args.face_detector_path) 

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        if args.video_path is not None:
            frame = cv2.resize(frame, (max(frame_width, frame_height), max(frame_width, frame_height)))
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

        if args.flip:
            frame = cv2.flip(frame, 0)

        if args.width_press:
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            # import ipdb; ipdb.set_trace(context=10)
            pressed_frame = cv2.resize(frame, (int(frame_width/4), frame_height))
            canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            index = int(3*frame_width/8)
            # import ipdb; ipdb.set_trace(context=10)
            canvas[:, index:index+int(frame_width/4), :] = pressed_frame

            frame = canvas

        key = cv2.waitKey(1) & 0xFF

        time_0 = time.time()

        with torch.no_grad():
            bboxes, _ = face_detector.forward(torch.from_numpy(frame))

        if len(bboxes) != 0:
            frame = model.draw_landmarks(frame, bboxes)

        if args.video_path is not None:
            result.write(frame)

        if args.save:
            out.write(frame)

        logging.info("reference time: {}".format(time.time()-time_0))
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

        cv2.imshow("", frame)

        if key == ord("q"):
            break
    
    cap.release()
    if args.video_path is not None:
        result.release()
        print("The video was successfully saved")

    if args.save:
        out.release()
        print("The video was successfully saved")

    cv2.destroyAllWindows()


def cam_cap():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF

        rd_id = uuid4()
        if key == ord("c"):
            cv2.imwrite("input/{}.jpg".format(rd_id),frame)

        cv2.imshow("", frame)

        if key == ord("q"):
            break


if __name__=="__main__":
    P = argparse.ArgumentParser(description='landmark')
    P.add_argument('--face-detector-path', type=str, required=True, help='path to retinaface detector to use')
    P.add_argument('--model-path', type=str, required=True, help='path to landmarks model')
    P.add_argument('--video-path', type=str, help='path to video for inference')
    P.add_argument('--flip', action='store_true', help='flip input frame')
    P.add_argument('--save', action='store_true', help='save inference video')
    P.add_argument('--width-press', action='store_true', help='press image along the width')

    args = P.parse_args()
    video_infer(args)

