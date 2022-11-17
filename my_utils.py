import argparse
import matplotlib.pyplot as plt


# function for plot fps and time comparison graph
def plot_fps_time_comparison(time_list, fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list, 'b', label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_weights', type=str,
                        default='./weights/yolov7-w6-pose.pt',
                        help='model path(s)')

    parser.add_argument('--classifier_weights', type=str,
                        # default='./weights/decision_tree_model.sav',
                        default='decision_tree_model.sav',

                        help='model path(s)')

    parser.add_argument('--csv_filename', type=str,
                        # default='./keypoints_csv/keypoints_data.csv',
                        default='keypoints_data.csv',
                        help='model path(s)')

    # parser.add_argument('--source', type=str, default='aidyn_waving.mp4', help='video/0 for webcam')
    parser.add_argument('--source', type=str,
                        default='./test_videos/aidyn-standing-waving-home.mp4',
                        # default='./test_videos/aidyn_waving.mp4',
                        # default='./test_videos/aidyn_standing.mp4',

                        help='video/0 for webcam')

    parser.add_argument('--output_path', type=str,
                        default='./output_videos',
                        help='video/0 for webcam')

    parser.add_argument('--device', type=str, default='0', help='cpu/0,1,2,3(gpu)')  # device arguments

    opt = parser.parse_args()
    return opt
