from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO = "vtest.avi"
CFG = "yolov4-tiny.cfg"
WEIGHTS = "yolov4-tiny.weights"
NAMES = "coco.names"
OUTPUT = "annotated.mp4"
CONF, NMS = 0.40, 0.40
INPUT = (416, 416)

for filename in [VIDEO, CFG, WEIGHTS, NAMES]:
    if not Path(filename).exists():
        raise FileNotFoundError(f"Missing required file: {filename}")

net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
names = open(NAMES, encoding="utf-8", errors="ignore").read().splitlines()
out_layers = net.getUnconnectedOutLayersNames()

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Unable to open {VIDEO}")

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter(
    OUTPUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H),
)

persons_per_frame = []
person_confidences = []
class_counts = {}
saved_frames = {}
frames_to_save = {0, 300, 600, 780}

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_number = len(persons_per_frame)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, INPUT, swapRB=True, crop=False)
    net.setInput(blob)
    boxes, scores, classes = [], [], []

    for out in net.forward(out_layers):
        for d in out:
            cls_scores = d[5:]
            c = int(np.argmax(cls_scores))
            conf = float(d[4] * cls_scores[c])
            if conf < CONF:
                continue
            cx, cy, w, h = d[0] * W, d[1] * H, d[2] * W, d[3] * H
            boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
            scores.append(conf)
            classes.append(c)

    keep = cv2.dnn.NMSBoxes(boxes, scores, CONF, NMS)
    keep = np.array(keep).flatten() if len(keep) else np.array([], dtype=int)
    persons = 0

    for i in keep:
        x, y, w, h = boxes[i]
        name = names[classes[i]]
        conf = scores[i]
        class_counts[name] = class_counts.get(name, 0) + 1
        if name == "person":
            persons += 1
            person_confidences.append(conf)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
        )

    persons_per_frame.append(persons)

    if frame_number in frames_to_save:
        saved_frames[frame_number] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    writer.write(frame)

cap.release()
writer.release()

print("Frames processed:", len(persons_per_frame))
print("Mean persons/frame:", np.mean(persons_per_frame))
print("Maximum persons/frame:", max(persons_per_frame))
print("Total detections:", sum(class_counts.values()))
print("\nDetections by class:")
for name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True):
    print(f"{name}: {count}")

# Figure 9.2: selected annotated frames
for frame_number in [0, 300, 600, 780]:
    if frame_number in saved_frames:
        plt.figure(figsize=(10, 6))
        plt.imshow(saved_frames[frame_number])
        plt.title(f"Annotated CCTV Frame {frame_number}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# Figure 9.3: persons per frame and confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(range(len(persons_per_frame)), persons_per_frame, linewidth=1)
axes[0].set_xlabel("Frame Number")
axes[0].set_ylabel("Number of Persons Detected")
axes[0].set_title("Persons Detected in Each Frame")
axes[0].grid(True)

axes[1].hist(person_confidences, bins=20, edgecolor="black")
axes[1].set_xlabel("Detection Confidence")
axes[1].set_ylabel("Number of Person Detections")
axes[1].set_title("Confidence of Person Detections")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("assignment9_detection_graphs.png", dpi=200)
plt.show()

print("Annotated video:", OUTPUT)
# vtest.avi has no ground-truth boxes, so accuracy/mAP is not reported.
