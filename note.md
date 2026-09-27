Với nền tảng là đã code một lần qua yolov3, thật sự vẫn chưa thấm lắm tại chưa dùng nhiều. Có các hướng đi hiện tại như sau:

    - Tiếp tục đào sâu vào detection qua việc cải biên lại mô hình yolo detection qua yolov3, mô hình đang còn rất nặng, lúc chạy trên máy dù là r7 5800h nhưng vẫn rất lag. Hiện tại cũng mới chỉ train thử trên tập dữ liệu PASCAL_VOC chứ chưa áp dụng các bộ dữ liệu thực tế hơn. Sẽ lên kaggle thử sức mô hình của mình với các bài toán có sẵn trên đấy
    - Hướng đi tiếp theo là học bài toán tracking, việc học tracking sẽ giải quyết thêm được nhiều bài toán thực tế hơn như xác định người lạ vào nhà, đếm vật thể, theo dõi vật thể, tính tốc độ tương đối một xe, thời gian một người trong một vùng nào đấy, các bài toán giao thông, bài toán theo dõi người nơi công cộng sư sân bay, nhà ga,....
    - Hướng cuối là sequence modeling, nó sẽ giải quyết các bài toán về xác định hành động của vật thể như phát hiện hành vi bất thường, Trajectory Prediction, Multi-modal sequence modeling. 

Có 3 mức độ can thiệp khi train bài toán classification: 

    - Can thiệp dữ liệu: nhận ra vấn đề của dữ liệu đang gặp phải để xử lý
        augmentation khác
        crop khác
        image size khác
        class balancing
        augmentation mạnh/yếu
    - Can thiệp các tham số cơ bản: tối ưu các tham số cơ bản của một mô hình có sẵn
        learning rate
        scheduler
        batch size
        optimizer
        weight decay
        number of epochs
        Các threshold
    - Can thiệp kiến trúc: nhận ra đặc thù trong dữ liệu và phần cứng để điều chỉnh kiến trúc phù hợp
        backbone A → backbone B
        freeze/unfreeze backbone
        thêm/bớt layer
        thay activation
        thay classifier head
        pretrained → train from scratch

Cách luyện tập trên kaggle:
    - 

Note:
    - Hiện tại chưa hiểu rõ về tác động của việc xử lý data như: (class imbalance, resolution, augumentation)
    - Chưa nói đến việc thay đổi kiến trúc mô hình, việc cải biên yolov3 cũng có thể giúp mình hiểu kiến trúc nào thì phù hợp với bài toán nào để train hơn
    - Mình đã biết yolov3 là chi, nhma ko biết cách dùng trong trường hợp như nào, thay đổi như nào để phù hợp trong từng bài toán
    - Ban đầu pick ngẫu nhiên các dataset nhỏ để tập thích ứng với một dataset mới, phân tích các dự đoán sai và rút ra mô hình đang mắc những nhược điểm nào, tìm những cách giải quyết những nhược điểm đó
        False Positive
        False Negative
        Wrong class
        Bad localization
        Small objects
        Occlusion
        Crowded scene
        Low-quality image
    - Sau khi thử các dataset nhỏ ngẫu nhiên, chuyển sang các dataset có các vấn đề đặc trưng, tìm cách giải quyết chuyên sâu các vấn đề của dataset ấy hơn:
        Small objects
        Class imbalance
        Occlusion
        Crowded objects


Ideas:
    - Count plants: https://www.youtube.com/watch?v=jTgc87HeAcw
    - Count people, vehicles: https://www.youtube.com/watch?v=Ro36g2PEkBo | https://www.youtube.com/watch?v=d1bky80NXeQ&pp=0gcJCS8MAYcqIYzv |
    - Detect vehicles speed
    - Read analog meter: https://www.youtube.com/watch?v=OC0Ar2YrytU&pp=0gcJCS8MAYcqIYzv
    - Smart farm: https://www.youtube.com/watch?v=2gi-UDz58Dw
        - Animals couting
        - Animal behaviors
        - Strange animal detection
    - Measure size of object: https://www.youtube.com/watch?v=xjH0e7kYJsU
    - People flow tracking: https://www.youtube.com/watch?v=COYUiWxthMc
    - AI for Manufacturing: https://www.youtube.com/watch?v=KCFd_DljJhs&t=2s

Do first:
    - Counting duck or something small: object nhỏ, lấn nhau
    - Fall detection: behavior detection but short
    - Animal behavior detection: trajectory of animal, behavior detection but longer
    - Stranger detection

    Nhận diện vật thể
    Nhận diện hành động
    Đếm vật thể
    Vật thể lạ
    Phân đoạn: đo khoảng cách, phát hiện lỗi
