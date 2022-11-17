# yolov7-pose-estimation

### Features
- YOLOv7 Pose with Decision Tree Classifier
### Running
- Clone the repository.
```
git clone https://github.com/fengshuibuddy/yolov7-pose-with-classifier
```

- Goto the cloned folder.
```
cd yolov7-pose-with-classifier
```

- Create a virtual envirnoment (Recommended, If you dont want to disturb python packages)
```
### For Linux Users, I am using Anaconda
conda create --name yolov7-pose python=3.8
conda activate yolov7-pose
```

- Upgrade pip with mentioned command below.
```
pip install --upgrade pip
```

- Install requirements with mentioned command below.

```
pip install -r requirements.txt
pip install xtcocotools
```

- If you're utilizing GPU, you need to install pytorch cuda version. I am using the version CUDA >= 11.3
- Check the CUDA version.
``` 
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

- Download yolov7 pose estimation weights from [link](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt) and move them to the working directory {yolov7-pose-estimation}


- Run the code with mentioned command below.
```
python pose_estimate_with_classifier.py

#if you want to change source file
python pose_estimate_with_classifier.py --source "your custom video.mp4"

#For CPU
python pose_estimate_with_classifier.py --source "your custom video.mp4" --device cpu

#For GPU
python pose_estimate_with_classifier.py --source "your custom video.mp4" --device 0

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python pose_estimate_with_classifier.py --source "your IP Camera Stream URL" --device 0

#For WebCam
python pose_estimate_with_classifier.py --source 0

#For External Camera
python pose_estimate_with_classifier.py --source 1
```

- Output file will be created in the working directory with name <b>["./output_videos/your-file-name-without-extension"+"_keypoint.mp4"]</b>

#### RESULTS

<table>
  <tr>
    <td>Football Match Pose-Estimation</td>
     <td>Cricket Match Pose-Estimation</td>
     <td>FPS and Time Comparision</td>
     <td>Live Stream Pose-Estimation</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/62513924/185089411-3f9ae391-ec23-4ca2-aba0-abf3c9991050.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185228806-4ba62e7a-12ef-4965-a44a-6b5ba9a3bf28.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185324844-20ce3d48-f5f5-4a17-8b62-9b51ab02a716.png" width=640 height=180></td>
    <td><img src="https://user-images.githubusercontent.com/62513924/185587159-6643529c-7840-48d6-ae1d-2d7c27d417ab.png" width=640 height =180></td>
  </tr>
 </table>

#### References
- https://github.com/WongKinYiu/yolov7
- https://github.com/augmentedstartups/yolov7
- https://github.com/augmentedstartups
- https://learnopencv.com/yolov7-object-detection-paper-explanation-and-inference/
- https://github.com/ultralytics/yolov5

#### RizwanMunawar's Medium Articles
- https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- https://medium.com/augmented-startups/roadmap-for-computer-vision-engineer-45167b94518c
- https://medium.com/augmented-startups/yolor-or-yolov5-which-one-is-better-2f844d35e1a1
- https://medium.com/augmented-startups/train-yolor-on-custom-data-f129391bd3d6
- https://medium.com/augmented-startups/develop-an-analytics-dashboard-using-streamlit-e6282fa5e0f
