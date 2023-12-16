import argparse
import tkinter as tk
import time
import cv2
import imutils
import tflite_runtime.interpreter as tf
from imutils.video import VideoStream, FPS
from threading import Thread
from tkinter import scrolledtext

class VariableUpdater(Thread):
    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback
        self.running = True
        self.product_number = 1  # Initialize product number counter

    def run(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", type=str, help="path to input video file")
        ap.add_argument("-t", "--tracker", type=str, default="mil", help="OpenCV object tracker type")
        ap.add_argument("-m", "--model", type=str, default="cnn_model_s.tflite",
                        help="Name of tf lite model. It should be in the saved_models folder")
        args = vars(ap.parse_args())

        # Load the TFLite model and allocate tensors.
        interpreter = tf.Interpreter(model_path=f"saved_models/{args['model']}")
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        class_labels = ['Deficient', 'Normal']
        (major, minor) = cv2.__version__.split(".")[:2]

        if int(major) == 3 and int(minor) < 3:
            tracker = cv2.Tracker_create(args["tracker"].upper())
        else:
            OPENCV_OBJECT_TRACKERS = {
                "mil": cv2.TrackerMIL_create,
            }
            tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

        # Initialize the bounding box coordinates and label outside the loop
        initBB = None
        prev_label = None

        # Initialize lists to store the labels and scores
        labels_history = []
        scores_history = []

        # if a video path was not supplied, grab the reference to the webcam
        if not args.get("video", False):
            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()
            time.sleep(1.0)
        else:
            print("[INFO] starting test video...")
            vs = cv2.VideoCapture(args["video"])

        fps = None

        while self.running:
            frame = vs.read()
            frame = frame[1] if args.get("video", False) else frame
            time.sleep(0.5)

            if frame is None:
                print('End of video')
                break

            frame = imutils.resize(frame, width=500)
            (H, W) = frame.shape[:2]

            if initBB is not None:
                (success, box) = tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    track = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                fps.update()
                fps.stop()

                tracked = cv2.resize(track, (300, 300))
                img = tracked.astype('float32')

                interpreter.set_tensor(input_details[0]['index'], [img])
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                pred_label = class_labels[output_data.argmax()]
                pred_scores = output_data.max()

                if pred_label == 'Deficient':
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                info = [
                    ("Product Number", self.product_number),
                    ("Label", pred_label),
                    ("Probability", "{:.2f}".format(pred_scores)),
                ]

                # Update the GUI with the new information
                current_time = time.strftime("%H:%M:%S", time.localtime())
                self.update_callback(f"{self.product_number}\t\t{current_time}\t\t{pred_label}\t\t{pred_scores}")

                self.product_number += 1  # Increment product number counter

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
                tracker.init(frame, initBB)
                fps = FPS().start()

            elif key == ord("q"):
                break

        # After the loop ends, print the labels and scores history
        print("Labels History:")
        print(labels_history)
        print("Scores History:")
        print(scores_history)

        # if we are using a webcam, release the pointer
        if not args.get("video", False):
            vs.stop()
        else:
            vs.release()

        # close all windows
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Quality Track of a Piece")

        # Create a big title
        title_label = tk.Label(root, text="Quality Track of a Piece", font=('Garamond', 20, 'bold'))
        title_label.pack(pady=20)

        # Create a scrolled text widget for displaying time, label, and probability
        self.result_text = scrolledtext.ScrolledText(root, font=('Garamond', 18), foreground='black', wrap=tk.WORD)
        self.result_text.pack(expand=True, fill='both')

        # Initial column titles
        column_titles = ["Product Number", "Time", "Label", "Probability"]
        # Display initial column titles
        self.result_text.insert(tk.END, '\t\t'.join(column_titles) + '\n')

        # Add a line separator
        self.result_text.insert(tk.END, '-' * 60 + '\n')

        self.variable_updater = VariableUpdater(self.update_result)
        self.variable_updater.start()

        # Bind the window close event to stop the updater thread
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_result(self, value):
        # Append the new value to the scrolled text widget
        self.result_text.insert(tk.END, f"{value}\n")
        # Scroll to the bottom to always show the latest entry
        self.result_text.see(tk.END)

    def on_close(self):
        # Stop the updater thread before closing the window
        self.variable_updater.stop()
        self.variable_updater.join()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
