import cv2
import torch
import numpy as np
import config
from model import YOLOv3
from utils import non_max_suppression, cells_to_bboxes, plot_image

# Create model
model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

checkpoint = torch.load("my_checkpoint.pth.tar", map_location=config.DEVICE)
if "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval() # là chế độ inference, không tính gradient, giúp tăng tốc độ dự đoán

scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        out = model(img_tensor)
        bboxes = []
        for i in range(3):
            b = out[i].shape[0]
            bboxes += cells_to_bboxes(
                out[i], scaled_anchors[i], S=out[i].shape[2], is_preds=True
            )
        
        # Lọc qua NMS
        bboxes = non_max_suppression(
            bboxes[0],
            iou_threshold=config.NMS_IOU_THRESH,
            threshold=config.CONF_THRESHOLD,
            box_format="midpoint",
        )

    h, w, _ = frame.shape
    for box in bboxes:
        class_pred = int(box[0])
        prob = box[1]
        x, y, box_w, box_h = box[2:]

        # Chuyển đổi tọa độ từ chuẩn hóa sang pixel
        x1 = int((x - box_w / 2) * w)
        y1 = int((y - box_h / 2) * h)
        x2 = int((x + box_w / 2) * w)
        y2 = int((y + box_h / 2) * h)

        # Lấy nhãn lớp nếu có định nghĩa
        label = f"{config.PASCAL_CLASSES[class_pred] if hasattr(config, 'PASCAL_CLASSES') else class_pred}: {prob:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv3 Real-Time Detection", frame)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()