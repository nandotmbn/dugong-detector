import tensorflow as tf
import cv2 as cv

image_filename = "dataset_dugongs_detector/test/dugong/1.PNG"
class_names = ["Dugong Exist", "No Dugong"]

def load_and_prep_image(filename, img_shape=224):
  img = tf.io.read_file(filename)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.resize(img, size = [img_shape, img_shape])
  img = img/255.
  print(img[112][112])
  return img

def pred_and_plot(model, filename, class_names):
  img = load_and_prep_image(filename)
  img_to_detect = cv.imread(filename)
  pred = model.predict(tf.expand_dims(img, axis=0))
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

  cv.putText(img_to_detect, "Prediksi: " + pred_class, (20, 40 - 2),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
  cv.imwrite("Detection.jpg", img_to_detect)
  print(f"Prediction: {pred_class}")

loaded_model_1 = tf.keras.models.load_model("dugong_detector_3")
pred_and_plot(loaded_model_1, image_filename, ["Dugong Exist", "No Dugong"])

