from VehicleDetector import VehicleDetector
import cv2

image = cv2.imread("/home/joshu/LPR-proj/data/LPRdatasets/vehicleImages/testData/UK/frame_00075_jpg.rf.eb57e5662c71c7f455ff6095199f06ce.jpg")

Vdetector = VehicleDetector()
vehicles = Vdetector.detect(image)

print(len(vehicles))
print(vehicles)
