from ultralytics import YOLO
import cv2
import os


def detect_on_image(image_path, model_path='yolov8n.pt', save_output=True):
    # Load Model YOLO
    model = YOLO(model_path)

 
    image = cv2.imread(image_path)

    results = model.predict(source=image, save=False, conf=0.5)  # Threshold 50%
    annotated_image = results[0].plot()


    cv2.imshow('Deteksi Gambar', annotated_image)


    if save_output:
        output_path = 'output_image.jpg'
        cv2.imwrite(output_path, annotated_image)
        print(f"Hasil deteksi disimpan di {output_path}")


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_on_video(video_path, model_path='yolov8n.pt', save_output=True):
   
    model = YOLO(model_path)

   
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error.")
        return

    # Output Video (Opsional)
    if save_output:
        output_path = 'output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Proses Deteksi
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi dengan YOLO
        results = model.predict(source=frame, save=False, conf=0.5)
        annotated_frame = results[0].plot()

        # Tampilkan Hasil di Jendela
        cv2.imshow('Deteksi Video', annotated_frame)

        # Simpan Hasil ke Video Baru
        if save_output:
            out.write(annotated_frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Bersihkan Resource
    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

    if save_output:
        print(f"Hasil deteksi disimpan di {output_path}")

# Fungsi Utama
def main():
    print("Pilih mode input:")
    print("1. Gambar")
    print("2. Video")
    choice = input("Masukkan pilihan (1/2): ")

    model_path = 'best.pt'  # Ganti dengan model terlatih jika diperlukan

    if choice == '1':
        image_path = input("Masukkan path gambar (contoh: path/to/image.jpg): ")
        if os.path.exists(image_path):
            detect_on_image(image_path, model_path)
        else:
            print("Error: File gambar tidak ditemukan.")
    elif choice == '2':
        video_path = input("Masukkan path video (contoh: path/to/video.mp4): ")
        if os.path.exists(video_path):
            detect_on_video(video_path, model_path)
        else:
            print("Error: File video tidak ditemukan.")
    else:
        print("Pilihan tidak valid. Silakan jalankan ulang program.")

if __name__ == "__main__":
    main()
