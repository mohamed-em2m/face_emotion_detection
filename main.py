import cv2
import numpy as np
from dlib import get_frontal_face_detector
import keras as k

# Load the pre-trained model
model = k.models.load_model('fernet (1).h5')

# Initialize the HOG detector and video writer
hog = get_frontal_face_detector()
force = cv2.VideoWriter_fourcc(*'mp4v')

# Define emotion labels and colors
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
colors = {0: (0,0,255), 1: (100,0,100), 2: (200,50,123), 3: (0,255,255), 4: (0,255,0), 5: (255,255,0), 6: (255,0,0)}

# Preprocess image function
def reshape(img):
    img = cv2.resize(img, (48, 48))
    return img.reshape(1, 48, 48, 1)

# Load video and set up writer
vid = cv2.VideoCapture("path/to/your/video.mp4")
write = cv2.VideoWriter('emotion_output.mp4', force, 25, (int(vid.get(3)), int(vid.get(4))))

frame_count = 0
while True:
    ret, frame = vid.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for block in hog(gray, 1):
        face_img = gray[block.top():block.bottom(), block.left():block.right()]
        if face_img.size > 0:
            img2 = reshape(face_img)
            pred = model.predict(img2)
            emotion = labels[np.argmax(pred)]
            cv2.rectangle(frame, (block.left(), block.top()), (block.right(), block.bottom()), colors[np.argmax(pred)], 3)
            cv2.putText(frame, emotion, (block.left(), block.top()-15), cv2.FONT_ITALIC, 1, colors[np.argmax(pred)], 3)
    write.write(frame)
    frame_count += 1
    print(f"Processed frame {frame_count}")

write.release()
vid.release()
cv2.destroyAllWindows()
