from faceDetection import face_detection
from trainModel import train_model

# Now we shall start to train_model, that would return a model
trained_model = train_model()   # with default parameters
print("[*] Model Trained.")

print("[*] Now Loading Model for Face Detection.")
face_detection(trained_model)