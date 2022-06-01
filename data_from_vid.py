import os
from mtcnn import MTCNN
import cv2
from PIL import Image
def data_from_vid(file_path):
  cap = cv2.VideoCapture(file_path)
  count = 0

  while True:
    success, frame = cap.read()
    if success:
        frame_id = int(round(cap.get(1)))
        cv2.imshow("frame", frame)
        k = cv2.waitKey(20)

        if frame_id % multiplier == 0:
                cv2.imwrite(f"data_from_vid/{count}.jpg", frame)
                print(f"Take a screenshot {count}")
                count += 1

        if k == ord(" "):
            cv2.imwrite(f"data_from_vid/{count}_1.jpg", frame)
            print(f"Take an extra screenshot {count}")
            count += 1
        elif k == ord("q"):
            print("Q pressed, closing the app")
            break

  cap.release()
  cv2.destroyAllWindows()

  
def cropping_photo(dir_path):
  os.chdir(dir_path)
  files=os.listdir()

  while len(files)>0:
    
    img = cv2.cvtColor(cv2.imread(files[0]), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    a=detector.detect_faces(img)
    if len(a)==0:
        print(files[0])
        files.pop(0)
    else:
        a=a[0]['box']
        im = Image.open(files[0])
        im_crop = im.crop((a[0],a[1], a[0]+a[2], a[1]+a[3]))
        im_crop.save(files[0], quality=95)
        files.pop(0)
