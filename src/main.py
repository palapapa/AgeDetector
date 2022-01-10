import face_recognition
import argparse
import cv2
import keras
import gtts
import googletrans
import numpy
import time
import playsound
import os
import threading

def playsound_async(path):
    playsound.playsound(path)

def main():
    argument_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argument_parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default="models/model.h5",
        help="The path to the h5 model.",
        metavar="path",
        dest="model_path"
    )
    argument_parser.add_argument(
        "-l",
        "--label",
        type=str,
        required=False,
        default="models/labels.txt",
        help="The path to the labels file.",
        metavar="path",
        dest="labels_path"
    )
    argument_parser.add_argument(
        "-c",
        "--camera",
        type=int,
        required=False,
        default=0,
        help="The number of the webcam to use.",
        metavar="id",
        dest="camera_id"
    )
    argument_parser.add_argument(
        "-s",
        "--scan-scaling",
        type=float,
        required=False,
        default=0.25,
        help="The scaling factor to use while scanning faces. A smaller number means faster detection but lower accuracy and vice versa.",
        metavar="scale-factor",
        dest="scaling_factor"
    )
    argument_parser.add_argument(
        "-d",
        "--scan-delay",
        type=int,
        required=False,
        default=0,
        help="The amount of time to delay between each scan.",
        metavar="ms",
        dest="scanning_delay"
    )
    argument_parser.add_argument(
        "--language",
        type=str,
        required=False,
        default="en",
        help="The language of the warning voice.",
        metavar="ISO-code",
        dest="language"
    )
    arguments = argument_parser.parse_args()
    video_capture = cv2.VideoCapture(arguments.camera_id)
    is_camera_on, _ = video_capture.read()
    if not is_camera_on:
        print("The camera failed to turn on. Please try another camera number.")
        return
    model = keras.models.load_model(arguments.model_path, compile=False)
    labels = []
    with open(arguments.labels_path) as label_file:
        for line in label_file.readlines():
            labels.append(line.strip())
    time_of_last_scan = 0
    faces = []
    minor_detected_speech = "Minor detected"
    minor_detected_speech_translated = googletrans.Translator().translate(minor_detected_speech, arguments.language, "en").text
    time_of_last_minor_detection = 0
    speech_location = f"voices/minor_detected_{arguments.language}.mp3"
    if not os.path.exists("voices"):
        os.makedirs("voices")
    gtts.gTTS(minor_detected_speech_translated, lang=arguments.language).save(speech_location)
    exposure = 128
    while True:
        _, frame = video_capture.read()
        processed_frame = cv2.flip(frame, 1)
        rgb_small_frame = cv2.resize(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), (0, 0), fx=arguments.scaling_factor, fy=arguments.scaling_factor)
        if (time.monotonic() - time_of_last_scan) * 1000 > arguments.scanning_delay:
            faces = face_recognition.api.face_locations(rgb_small_frame)
            time_of_last_scan = time.monotonic()
        for face in faces:
            x1 = int(face[3] * (1 / arguments.scaling_factor))
            y1 = int(face[0] * (1 / arguments.scaling_factor))
            x2 = int(face[1] * (1 / arguments.scaling_factor))
            y2 = int(face[2] * (1 / arguments.scaling_factor))
            processed_frame = cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cropped_face = frame[y1:y2, x1:x2]
            cropped_face = cv2.resize(cropped_face, (224, 224))
            data = numpy.ndarray((1, 224, 224, 3), numpy.float)
            data[0] = cropped_face / 127 - 1
            prediction = labels[numpy.argmax(model.predict(data))]
            if prediction == "minor" and time.monotonic() - time_of_last_minor_detection > 5:
                threading.Thread(target=playsound_async, args=(speech_location,), daemon=True).start()
                time_of_last_minor_detection = time.monotonic()
            processed_frame = cv2.rectangle(processed_frame, (x1, y1 - 25), (x2, y1), (0, 255, 0), cv2.FILLED)
            processed_frame = cv2.putText(processed_frame, prediction, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow("Age Detector(Esc to quit)", processed_frame)
        if cv2.waitKey(1) == 27: # esc key
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
