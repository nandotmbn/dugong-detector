import tensorflow as tf
import cv2 as cv
import numpy as np

class_names = ["Dugong Exist", "No Dugong"]
loaded_model_1 = tf.keras.models.load_model("dugong_detector_3")
cap = cv.VideoCapture('Sea.mp4')

if (cap.isOpened() == False):
    print("Error opening video stream or file")

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        cv_image = np.asarray(cv_image)
        cv_image = tf.convert_to_tensor(cv_image, dtype=tf.uint8)
        cv_image = tf.image.resize(cv_image, size=[224, 224])
        cv_image = cv_image/255.

        pred = loaded_model_1.predict(tf.expand_dims(cv_image, axis=0))
        if len(pred[0]) > 1:
            pred_class = class_names[pred.argmax()]
        else:
            pred_class = class_names[int(tf.round(pred)[0][0])]

        cv.putText(frame, "Prediksi: " + pred_class, (20, 40 - 2),
                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow("Detector", frame)
        print(f"Prediction: {pred_class}")

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
