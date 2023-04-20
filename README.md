# Dugong Detector

Model dibuat dengan konsep multiclass classification model.
Menggunaan framework machine learning TensorFlow.

Kita akan membedakan 2 jenis penggambaran penampakan.

1. Tidak terdapat dugong (No Dugong / Gambar Laut)
2. Terdapat dugong (Dugong Exist)

Pengembangan dilakukan dengan beberapa langkah, yakni:

1. Memuat data
2. Preprocess data
3. Membuat arsitektur model (memulai dengan model sederhana)
4. Cocok kan model (Lampaui batas kecocokan untuk memastikan bahwa model bekerja)
5. Evaluasi model
6. Sesuaikan dengan mengganti hyperparameter yang berbeda untuk meningkatkan kualitas model
7. Ulangi hingga mencapai kepuasan

## Requirement

Kebutuhan yang dibutuhkan dibagi menjadi 2, yakni fase training dan inference.

### Training Requirement

#### Kebutuhan Training

> 1. Tensorflow
> 2. Pathlib
> 3. Numpy
> 4. OS
> 5. Matplotlib
> 6. Random

#### Kebutuhan Testing

> 1. Sklearn

### Inference

### Image Inference

> 1. Tensorflow
> 2. OpenCV

### Sequential Image (Video) Inference

> 1. Tensorflow
> 2. OpenCV
> 3. Numpy

## How to run

### Training Run

1. Menentukan direktori dataset

    ```py
    # Menentukan direktori dataset
    train_dir = "dataset_dugongs_detector/train"
    test_dir = "dataset_dugongs_detector/test"
    ```

2. Menentukan class

    ```py
    import pathlib
    import numpy as np

    data_dir = pathlib.Path(train_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    class_names
    ```

3. Setting seed, preprocess datagen dan load data

    ```py
    # set seed
    tf.random.set_seed(10)

    # Preprocess data (get add of the pixel values between 0 and 1)
    train_datagen = ImageDataGenerator(rescale=1/255.)
    valid_datagen = ImageDataGenerator(rescale=1/255.)

    # Load data
    train_data = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), class_mode="binary")
    test_data = train_datagen.flow_from_directory(test_dir, target_size=(224, 224), class_mode="binary")
    ```

4. Jalankan training

    ```py
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Activation

    model_1 = Sequential([ # Tiny VGG with Binary Classification
        Conv2D(10, 3, input_shape=(224, 224, 3)),
        Activation(activation="relu"),
        Conv2D(10, 3, activation="relu"),
        MaxPool2D(),
        Conv2D(10, 3, activation="relu"),
        Conv2D(10, 3, activation="relu"),
        MaxPool2D(),
        Flatten(),
        Dense(1, activation="sigmoid")
    ])

    model_1.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    history_1 = model_1.fit(train_data, epochs=100, steps_per_epoch=len(train_data), validation_data=test_data, validation_steps=len(test_data))
    ```

5. Evaluasi model

   ```py
   model_1.evaluate(test_data)
   ```

6. Tes model

    ```py
    def load_and_prep_image(filename, img_shape=224):
      """
      Reads an image from filename, turns it into a tensor
      and reshapes it to (img_shape, img_shape, colour_channel).
      """
      # Read in target file (an image)
      img = tf.io.read_file(filename)

      # Decode the read file into a tensor & ensure 3 colour channels 
      # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
      img = tf.image.decode_image(img, channels=3)

      # Resize the image (to the same size our model was trained on)
      img = tf.image.resize(img, size = [img_shape, img_shape])

      # Rescale the image (get all values between 0 and 1)
      img = img/255.
      return img

    def pred_and_plot(model, filename, class_names):
      """
      Imports an image located at filename, makes a prediction on it with
      a trained model and plots the image with the predicted class as the title.
      """
      # Import the target image and preprocess it
      img = load_and_prep_image(filename)

      # Make a prediction
      pred = model.predict(tf.expand_dims(img, axis=0))

      print(pred)

      # Get the predicted class
      if len(pred[0]) > 1: # check for multi-class
        pred_class = class_names[pred.argmax()] # if more than one output, take the max
      else:
        pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

      # Plot the image and predicted class
      plt.imshow(img)
      plt.title(f"Prediction: {pred_class}")
      plt.axis(False);
    
    pred_and_plot(model_1, "YOUR_IMAGE_YOU_WANT_TO_DETECT", class_names)
    ```

7. Training graphic loss and validation curves

    ```py
    # Plot the validation and training data separately
    def plot_loss_curves(history):
      """
      Returns separate loss curves for training and validation metrics.
      """ 
      loss = history.history['loss']
      val_loss = history.history['val_loss']

      accuracy = history.history['accuracy']
      val_accuracy = history.history['val_accuracy']

      epochs = range(len(history.history['loss']))

      # Plot loss
      plt.plot(epochs, loss, label='training_losses')
      plt.plot(epochs, val_loss, label='val_losses')
      plt.title('Loss')
      plt.xlabel('Epochs')
      plt.legend()

      # Plot accuracy
      plt.figure()
      plt.plot(epochs, accuracy, label='training_accuracy')
      plt.plot(epochs, val_accuracy, label='val_accuracy')
      plt.title('Accuracy')
      plt.xlabel('Epochs')
      plt.legend();
    
    plot_loss_curves(history_1)
    ```

8. Testing graphic loss and validation curves

    ```py
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

    def get_confusion_matrix(model, validation_generator):
        is_binary = len(validation_generator.class_indices) == 2
        all_predictions = np.array([])
        all_labels = np.array([])
        for i in range(len(validation_generator)):
            x_batch, y_batch = validation_generator[i]
            predictions = model.predict(x_batch)
            if (not is_binary):
                predictions = np.argmax(predictions, axis=1)
            else:
                predictions = (predictions > .5) * 1
            all_predictions = np.concatenate([all_predictions, predictions])

            if (not is_binary):
                labels = np.argmax(y_batch, axis = 1)
            else:
                labels = y_batch
            all_labels = np.concatenate([all_labels, labels])

        return tf.math.confusion_matrix(all_predictions, all_labels)

    Y_pred = model_1.predict_generator(test_data, 7)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(test_data.classes, y_pred)
    print(cm)
    print(classification_report(test_data.classes, y_pred))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.classes)
    disp.plot()
    ```

### Inference Run

#### Image Inference Run

```py
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
```

#### Video Inference Run

```py
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
```
