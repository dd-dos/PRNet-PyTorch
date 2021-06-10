import argparse
import logging
import time

import cv2
import torch
import torchvision

from models import resfcn256
from package.main import FacePatternModel

logging.getLogger().setLevel(logging.INFO)

def video_infer(args):
    '''
    3D landmarks video inference
    '''
    if args.video_path:
        cap = cv2.VideoCapture("sample_rotate.mp4")
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter('filename.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)
    else:
        cap = cv2.VideoCapture(0)
    
    model = FacePatternModel(args.model_path)
    face_detector = torch.jit.load(args.face_detector_path) 

    while True:
        ret, frame = cap.read()

        if args.video_path is not None:
            frame = cv2.resize(frame, (max(frame_width, frame_height), max(frame_width, frame_height)))
        
        if not ret:
            break
        
        # frame = cv2.flip(frame, 0)
        key = cv2.waitKey(1) & 0xFF

        time_0 = time.time()

        with torch.no_grad():
            bboxes, _ = face_detector.forward(torch.from_numpy(frame))

        if len(bboxes) != 0:
            frame = model.draw_landmarks(frame, bboxes)

        if args.video_path is not None:
            result.write(frame)

        logging.info("reference time: {}".format(time.time()-time_0))
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

        cv2.imshow("", frame)

        if key == ord("q"):
            break
    
    cap.release()
    if args.video_path is not None:
        result.release()
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
    args = P.parse_args()
    video_infer(args)

