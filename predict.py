import numpy as np
import time
import cv2





def predict(INPUT_FILE):
    labels_file='/Users/priyamvada./Documents/animal_project/classes.names'
    CONFIG_FILE='/Users/priyamvada./Documents/animal_project/yolov3_custom_training1.cfg'
    WEIGHTS_FILE='/Users/priyamvada./Documents/animal_project/yolov3_custom_training1_last_100_working_12.48pm.weights'   

    CONFIDENCE_THRESHOLD=0.75

    classes = open(labels_file).read().strip().split("\n")
    print(classes,"1 :",classes[0])

    np.random.seed(4)

    try:
        net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    except:
        print('------ anh --------')



    #input_file = INPUT_FILE.read()
    image = cv2.imread(INPUT_FILE)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]



    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()


    print("[INFO] YOLO took {:.6f} seconds".format(end - start))



    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            
            if confidence > CONFIDENCE_THRESHOLD:
                # scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # derive top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID) 

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
        CONFIDENCE_THRESHOLD)
    print(idxs)

    COLORS = np.random.randint(0, 255, size=(len(classes), 3),
        dtype="uint8")
    
    to_return=[]
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
        

            color = [int(c) for c in COLORS[classIDs[i]]]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            print(" classIDs[i] :",classIDs[i]," classes[classIDs[i]] :",classes[classIDs[i]])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
            #a={'class:',classes[classIDs[i]],x,y,w,h}
            class_detected=classes[classIDs[i]]
            a=(class_detected,'(x,y),(w,h):',(x,y),(w,h) ,'\n')
            to_return.append(a)

    # show the output image
    #filename="detected.jpg"
    #cv2.imwrite(filename, image)
    #print('to_return:',to_return)
    resized_image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_AREA)
    return resized_image,to_return
    


INPUT_FILE='/Users/priyamvada./Documents/animal_project/Side/23512.jpg'

predict(INPUT_FILE)