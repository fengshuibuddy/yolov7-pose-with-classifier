import cv2
import csv
import time
import torch
from my_utils import parse_opt
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer


@torch.no_grad()
def run(device,
        pose_weights,
        source,
        output_path,
        csv_filename,
        classifier_weights=None  # Not needed in this script
        ):
    # list to store time
    time_list = []
    # list to store fps
    fps_list = []

    # select device
    device = select_device(device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(pose_weights, map_location=device)  # load FP32 model
    _ = model.eval()

    # video path
    video_path = source

    # pass video to videocapture object
    if video_path.isnumeric():
        cap = cv2.VideoCapture(int(video_path))
    else:
        cap = cv2.VideoCapture(video_path)
    # check if videocapture not opened
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')

    # get video frame width
    frame_width = int(cap.get(3))

    # get video frame height
    frame_height = int(cap.get(4))

    # code to write a video
    vid_write_image = letterbox(cap.read()[1], frame_width, stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"{video_path.split('/')[-1].split('.')[0]}"
    out = cv2.VideoWriter(f"{output_path}/{out_video_name}_keypoint.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (resize_width, resize_height))

    # count no of frames
    frame_count = 0
    # count total fps
    total_fps = 0

    with open(csv_filename, "a") as file:
        # loop until cap opened or video not complete
        while cap.isOpened:

            print("Frame {} Processing".format(frame_count))

            # get frame and success from video capture
            ret, frame = cap.read()
            # if success is true, means frame exist
            if ret:

                # store frame
                orig_image = frame

                # convert frame to RGB
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, frame_width, stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))

                # convert image data to device
                image = image.to(device)

                # convert image to float precision (cpu)
                image = image.float()

                # start time for fps calculation
                start_time = time.time()

                # get predictions
                with torch.no_grad():
                    output, _ = model(image)

                # Apply non-max suppression
                output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                                 kpt_label=True)
                output = output_to_keypoint(output)
                im0 = image[0].permute(1, 2, 0) * 255
                im0 = im0.cpu().numpy().astype(np.uint8)

                """
                Здесь уже идет основная мысль, что мы записываем эти Координаты в csv (пока что без даты)
                """
                print(output[0])
                print(output[0].shape)
                print(type(output[0]))
                my_writer = csv.writer(file, delimiter=',')
                my_writer.writerow(output[0])

                # reshape image format to (BGR)
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
                for idx in range(output.shape[0]):
                    plot_skeleton_kpts(im0, output[idx, 7:].T, 3)
                    xmin, ymin = (output[idx, 2] - output[idx, 4] / 2), (output[idx, 3] - output[idx, 5] / 2)
                    xmax, ymax = (output[idx, 2] + output[idx, 4] / 2), (output[idx, 3] + output[idx, 5] / 2)

                    # Plotting key points on Image
                    cv2.rectangle(im0, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(255, 0, 0),
                                  thickness=1, lineType=cv2.LINE_AA)

                # Calculation for FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1

                # append FPS in list
                fps_list.append(total_fps)

                # append time in list
                time_list.append(end_time - start_time)

                # add FPS on top of video
                cv2.putText(im0, f'FPS: {int(fps)}', (11, 100), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)

                cv2.imshow('image', im0)
                out.write(im0)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    cap.release()
    # cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


# main function
def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    options = parse_opt()

    print(options.device)
    print(options.pose_weights)

    strip_optimizer(options.device, options.pose_weights)
    main(options)
