from keras.models import load_model
import cv2
import numpy as np

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)


class_names = open("labels.txt", "r").readlines()


np.set_printoptions(suppress=True)

image_paths = ['sunflower1.jpg','rose2.jpg','lotus3.jpg','jasmine4.jpg','tulip1.jpeg','lilly1.jpg']  


current_index = 0


display_width = 500
display_height = 500


def update_display(image_index):
 
    image = cv2.imread(image_paths[image_index])

    image = cv2.resize(image, (display_width, display_height))

    input_image = cv2.resize(image, (224, 224))

    image_input = np.asarray(input_image, dtype=np.float32).reshape(1, 224, 224, 3)

    image_input = (image_input / 127.5) - 1
    
    prediction = model.predict(image_input)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  
    confidence_score = prediction[0][index]

    label = f"{class_name}: {confidence_score:.2f}"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

 
    cv2.imshow("Image", image)


update_display(current_index)

while True:
    key = cv2.waitKey(0)

    if key == ord('q'):
        break

    elif key == ord('n'):
        current_index = (current_index + 1) % len(image_paths)
        update_display(current_index)

cv2.destroyAllWindows()