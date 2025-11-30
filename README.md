# TASE
## RL based Humanoid Locomotion 
### Authors: Sandeep, Sudhanva, and Srikar

This work showcases how a MuJoCo Humanoid robot can be trained to walk using RL algorithms well-suited for continuous control tasks. The project also provides a pre-trained model and a workflow that initializes the humanoid’s posture using body-keypoints extracted from an input image supplied by the user.

### Results
Demo GIF:
![Demo](tase/demo_4_.gif)

Input image:
![4](https://github.com/user-attachments/assets/9ee96f67-7bb7-4f2d-8a96-ac64b37585ea)
Pose estimation: Mediapipe
![4_annotated](https://github.com/user-attachments/assets/8bcacd60-c42a-4730-a186-b9029f208749)
Do check out the demo video of the locomotion has been pushed in the repository.

### Setup
```
cd tase
python3 -m venv venv
source venv/bin/activate
```
If Windows the venv activation is as follows:
```
.\venv\Scripts\activate
```
Requirements 
```
pip install -r requirements.txt
```
Run:
```
python3 rl_test.py
```
Add input image in same folder, and give its path as input
```
model1.png
```



