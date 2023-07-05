import cv2

class FaceExtraction():

    def __init__(self, dimension):
        self.dimension = dimension
        self.images_from_video_path = "data/images_from_video/"
        self.extracted_face_path = "data/extracted_face/"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')


    def extract_face_from_frames(self, fileName):
        image = cv2.imread(self.images_from_video_path + fileName)
        faces = self.face_cascade.detectMultiScale(image)
        for (x, y, w, h) in faces[:1]:
            find_face = image[y:y + h, x:x + w]
            find_face = cv2.resize(find_face, self.dimension)
            cv2.imwrite(self.extracted_face_path + fileName, find_face)
            print("face extracted from image, fileName ", fileName)
            
            






        
