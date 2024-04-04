import math
from unittest import skip
from cv2 import sqrt
from matplotlib import scale
import numpy as np
import argparse
import cv2
import random
import os
import time
import sys
import matplotlib.pyplot as plt
from   matplotlib.pyplot import cm, title, ylabel 
import statistics
from scipy.stats import norm
import pandas as pd
from fitter import Fitter

# check line 115 for own input path
# check line 346 for input
# command input: 

ref_length = 1.5 #Set Value
pixel_length = 1000 #Set Value
scale = round((pixel_length/ref_length) * 100/1000) #Scale 100µm

def parser(confidence, threshold, labels, config, weights, input_path, ips):
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', '--image_path', type=str, default = '')
    parser.add_argument('-video_path', '--video_path', type=str, default = '')
    parser.add_argument('-output', '--output_path', type=str, default = './detection')
    parser.add_argument('-weights', '--weights', type=str, default = find_directory(weights, os.getcwd()))
    parser.add_argument('-config', '--config', type=str, default = find_directory(config, os.getcwd()))
    parser.add_argument('-labels', '--labels', type=str, default = find_directory(labels, os.getcwd()))
    parser.add_argument('-confidence', '--confidence', type=float, default=confidence)
    parser.add_argument('-threshold', '--threshold', type=float, default=threshold)
    parser.add_argument('-no_gpu', '--no_gpu', default=False, action='store_true')
    parser.add_argument('-save', '--save', default=False, action='store_true')
    parser.add_argument('-show', '--show', default=False, action='store_true')
    parser.add_argument('-input_path','--input_path', type=str, required=True, help='Path to the folder containing the video(s).')
    parser.add_argument('-ips', '--ips', type=float, default=ips)
    return parser.parse_args()

def check_arguments_errors(args):
    check = True
    msg = ''
    assert 0 < args.threshold < 1, "Threshold should be a float between zero and one (non-inclusive)"
    assert 0 < args.confidence < 1, "Confidence should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config):
        msg = "Invalid config path {}".format(os.path.abspath(args.config))
        check = False
    if not os.path.exists(args.weights):
        msg = "Invalid weight path {}".format(os.path.abspath(args.weights))
        check = False
    if not os.path.exists(args.labels):
        msg = "Invalid .names file path {}".format(os.path.abspath(args.labels))
        check = False
    if args.image_path and not os.path.exists(args.image_path):
        msg = "Invalid image path {}".format(os.path.abspath(args.image_path))
        check = False    
    return check, msg

#-----------------------------------------------------------------------------------------------------
def extract_frames(video_path, output_folder, ips):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"[INFO] Processing {os.path.basename(video_path)}")
    
    # Open video using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frames per second of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    interval = math.ceil(fps/ips)
    frame_count = 1
    image_count = 1
    path_list =[]
    print(fps, "", ips, "", interval)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # If the current frame number is a multiple of the fps, save the frame as an image
        if frame_count % interval == 0:
            file_name = os.path.join(output_folder, f"{image_count:03}.jpg")
            cv2.imwrite(file_name, frame)
            path_list.append(file_name)
            image_count += 1

        frame_count += 1

    cap.release()

    with open (os.path.join(output_folder, "paths.txt"), "w") as f:
        f.write("\n".join(path_list))

    print(f"[INFO] Extracted {image_count - 1} images.")

def monitor_and_extract():
    processed_videos = set()
    confidence = 0.9 # defaults confidence
    threshold = 0.5 # default threshold
    labels = "obj.names" #classes_file
    config = "yolov4_1ob.cfg" #config file
    weights= "yolov4_1ob_best.weights"  #weights_file
    input_path = "D:\\YOLO detection\\OpenCV\\yolov4-openCv\\test_yolo" #Check own input path
    ips = 1
    args = parser(confidence, threshold, labels, config, weights, input_path, ips) # use the provided customization
    check, msg = check_arguments_errors(args)
    current_directory = os.getcwd()
    if not check:
        print('[INFO] Program terminated due to an error in arguments')
        print(msg)
        sys.exit()
    else: 
        print('[INFO] Arguments checked, All clear')
    
    try:
        while True:
            video_files = [f for f in os.listdir(args.input_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv','.MOV'))]

            if not any(video for video in video_files if video not in processed_videos):
                print("[INFO] Waiting...Checking for new video every 10sec. To stop script press ctrl+c.")

            for video_file in video_files:
                if video_file not in processed_videos:
                    video_path = os.path.join(args.input_path, video_file)
                    initial_mtime = os.path.getmtime(video_path)
                    time.sleep(1)
                    final_mtime = os.path.getmtime(video_path)

                    if not initial_mtime == final_mtime:
                        print(f"[INFO] {os.path.basename(video_path)} is still being transferred to the destination folder")
                    while True:
                        time.sleep(5)
                        final_mtime = os.path.getmtime(video_path)
                        if initial_mtime == final_mtime:
                            break

                    output_folder = os.path.join(input_path, os.path.splitext(video_file)[0])
                    extract_frames(video_path, output_folder, ips)
                    processed_videos.add(video_file)               
                    
                    main(output_folder, args)
                    os.chdir(current_directory)
                    
                                   
            time.sleep(10)  # sleep for 10 seconds before checking again

    except KeyboardInterrupt:
        print("Stopping script...")

#------------------------------------------------------------------------
def hough_circle(img, overlay, boxes, classes, num_obj, droplet_num):
    rad=[]
    Droplet_diameter=[]
    box_diameter = []
    box_height = []
    image_new = img.copy()
    
    for i in range(num_obj):
        class_name= int(classes[i])
       
        if class_name == 0: #allowed_classes(droplet)
            #seperate coorodinates from box
            xmin,ymin,w,h =boxes[i]
            xmax=w+xmin
            ymax=h+ymin
            
            # get the subimage that makes up the bounded region 
            box = image_new[int(ymin):int(ymax), int(xmin):int(xmax)]
            

            if box.size == 0:
                print("Invalid crop dimensions")
                continue
            try:
                gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
                clip_hist_percent = 1

                #Calculate grayscale histogram
                hist = cv2.calcHist([gray],[0],None,[256],[0,256])
                hist_size = len(hist)

                #Calculate cumulative distribution from the histogram
                accumulator = []
                accumulator.append(float(hist[0]))
                for index in range(1, hist_size):
                    accumulator.append(accumulator[index -1] + float(hist[index]))

                #Locate Points to Clip
                maximum = accumulator[-1]
                clip_hist_percent = clip_hist_percent*(maximum/100.0)
                clip_hist_percent = clip_hist_percent/2.0

                #Locate left cut
                minimum_gray = 0
                while accumulator[minimum_gray] < clip_hist_percent:
                    minimum_gray += 1

                #Locate right cut
                maximum_gray = hist_size -1
                while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
                    maximum_gray -= 1

                #Calculate alpha and beta values
                alpha = 255 / (maximum_gray - minimum_gray)
                beta = -minimum_gray * alpha

            
                con_bri = cv2.convertScaleAbs(gray, alpha = alpha, beta = beta)#contrast and brightness 
                img_blur = cv2.medianBlur(con_bri, 3)# must be odd number   
                kernel = np.array ([ [0, -1,  0], 
                                     [-1, 5, -1], 
                                     [0, -1,  0]])
                img_sharp = cv2.filter2D(img_blur, -1, kernel)
                sf = 4  #scaling factor for resizing the image
                img_resize = cv2.resize(img_sharp, None, fx=sf, fy=sf, interpolation = cv2.INTER_AREA)
                
                diag = math.sqrt(((img_resize.shape[0])**2)+((img_resize.shape[1])**2))   
                rr= int(img_resize.shape[0]/2)         
                circles_img  = cv2.HoughCircles(img_resize, cv2.HOUGH_GRADIENT, dp = 1, minDist = 1000, param1 = 5, param2 = 15, minRadius = rr-10, maxRadius = rr+10)             
                
            except cv2.error as e:
                print(f"Error converting image to grayscale: {e}")


            if circles_img is None:
                circles_img = [[[0,0,0]]]
            else:
                circles_img  = np.uint16(np.around(circles_img))
                for j in circles_img[0, :]:   
                    k0=((j[0]/sf)+(xmin))
                    k1=((j[1]/sf)+(ymin))
                    k2=(j[2]/sf) 
                    k3=(diag/(sf*2))
                    k4=(img_resize.shape[1]/(sf*2))
                    if k2 in [0]:
                        None
                    else:     
                        
                        r = random.randint(0,255)
                        g = random.randint(0,255)
                        b = random.randint(0,255)

                        cv2.circle(overlay,(int(k0),int(k1)),int(k2),(r,g,b), 2)
                        cv2.circle(img,(int(k0),int(k1)),int(k2),(r,g,b), 2) 
                        cv2.circle(img, (int(k0),int(k1)),int(k2),(r, g, b), -1)
                        cv2.circle(overlay,(int(k0),int(k1)),1 ,(0,0,0),2)
                        cv2.circle(img,(int(k0),int(k1)),1 ,(0,0,0),2)
                        
                    k2 = ((k2*ref_length)/pixel_length)*1000*2
                    k3 = ((k3*ref_length)/pixel_length)*1000*2
                    k4 = ((k4*ref_length)/pixel_length)*1000*2
                    
                    rad.append(k2) 
                    Droplet_diameter.append(k2)
                    box_diameter.append(k3)
                    box_height.append(k4)
                    
    return(rad, Droplet_diameter)     
    

def stat(data):
        q1= np.percentile(data, 25)
        q3= np.percentile(data, 75)
        median = np.median(data)
        mean = np.mean(data)
        print("Median      :", median)
        print("Mean        :", mean)
        iqr = q3 - q1
        print("IQR         :", iqr)
        standd=np.std(data)
        print("Std. Dev.   :",standd)
        bin_width3 = (2 * iqr) * (len(data) ** (-1 / 3))# (freedman)'s
        bin_width = int(3.5*(standd)) * (len(data) ** (-1 / 3))# (scott)'s bin
        bin_width4 = (max(data) - min(data)) / (1 + 3.3 * np.log10(len(data)))# (sturges) 
        bin_width2 = (max(data) - min(data)) / np.sqrt(len(data))
        minimum_val = min(data)
        maximum_val = max(data)
        print("Min. value  :",minimum_val)
        print("Max. value  :",maximum_val)
        print("-------------------------------------")
        return q1, q3,median,mean,iqr,standd,bin_width,minimum_val,maximum_val


def output(df, index, name, droplet_diameter, number_droplets, detection_time, number_outRange, median, mean, IQR, stand_deviation, x_25, x_75, min_diameter, max_diameter, stand_error, confidence_score, threshhold, ref_length, pixel_length):
    df.loc[index, 'image_name'] = name
    df.loc[index, 'droplet_diameter [micrometer]'] = droplet_diameter
    df.loc[index, 'number_droplets'] = number_droplets
    df.loc[index, 'detection_time [s]'] = detection_time
    df.loc[index, 'number_outRange'] = number_outRange
    df.loc[index, 'median [micrometer]'] = median
    df.loc[index, 'mean [micrometer]'] = mean
    df.loc[index, 'IQR [micrometer]'] = IQR
    df.loc[index, 'stand_deviation [micrometer]'] = stand_deviation
    df.loc[index, 'x_25 [micrometer]'] = x_25
    df.loc[index, 'x_75 [micrometer]'] = x_75
    df.loc[index, 'min_diameter [micrometer]'] = min_diameter
    df.loc[index, 'max_diameter [micrometer]'] = max_diameter
    df.loc[index, 'stand_error'] = stand_error
    df.loc[index, 'confidence_score'] = confidence_score
    df.loc[index, 'threshhold'] = threshhold
    df.loc[index, 'ref_length [mm]'] = ref_length
    df.loc[index, 'pixel_length'] = pixel_length


def fitters(dropletdist,titles="",plotfile_name='histo',binn=20):
    print("")
    f= Fitter(dropletdist,distributions=['lognorm',"norm",],bins = binn)      
    f.fit()
    f.summary(title=titles)
    plt.savefig(plotfile_name+'.png',format='png',dpi=900)

def draw_boxes(image, labels, classes, scores, boxes, Diameter):
    
    for (classid, score, box, Diameter) in zip(classes, scores, boxes, Diameter):
        x = box[0] #top left  they are not normalized
        y = box[1] #top left    
        w = box[2] 
        h = box[3]
        
        label = np.array(classid)
        confidence = score * 100
        color = [255,0,0]
        
        if label == [0]:
            cv2.rectangle(image, box,4)
            
        else: 
            cv2.rectangle(image, box,4)
               
    return image

def make_prediction(net, image, confidence, threshold):
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(3552, 3552), scale=1/255, swapRB=True) #Change to your input size
    droplet_num = 0
    outRange_num = 0 
    
    classes, scores, boxes = model.detect(image, confidence,threshold)
    
    num_obj = int(len(classes))
    for x in classes:
        if x == [0]:  
            droplet_num+=1
        elif x== [1]:
            droplet_num+=1    
        else:
            None
    return classes, scores, boxes, num_obj, droplet_num,outRange_num


def find_directory(file,current_dir):
    for root, dirs, files in os.walk(current_dir):
        if file in files:
            return os.path.join(current_dir, file)
        elif dirs == '':
            return None
        else:
            for new_dir in dirs: 
                result = find_directory(file, os.path.join(current_dir, new_dir))
                if result:
                    return result
            return None
#-----------------------------------------------------------------------------------------------
def everything(lines, net, confid, threshold1, Diameter_array, labels, index, Diameter_array_continous):
    image = cv2.imread(lines)
    overlay = image.copy()
    labels = labels
    classes, scores, boxes, num_obj, droplet_num,outRange_num = make_prediction(net, image, confid, threshold1) # number of objects and droplet numbers
    
    #apply hough circle
    Diameter,Droplet_Diameter = hough_circle(image, overlay, boxes, classes, num_obj, droplet_num) #path_h
    Diameter_array.append(Droplet_Diameter)
    print('[INFO] Image' , index + 1, 'loaded')
    print("Objects detected by Yolov4      :"+ str(num_obj) + "\n" + "Circles detected by HoughCircle :"+ str(len(Droplet_Diameter))) # print both of them
    print("")
    image = draw_boxes(image, labels, classes, scores, boxes, Diameter)
    text_on_image =("The number of droplets detected: "+ str(len(Droplet_Diameter)))
    cv2.putText(image, text_on_image,(100,100), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.75,color=(127,126,0),thickness=2) # add the classes to draw the hough circle
    cv2.putText(overlay, text_on_image,(100,100), cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.75,color=(127,126,0),thickness=2)
    height, width, _ = image.shape
    top_left = (width - scale - 50, height - 50)
    bottom_right = (width - 50, height - 40)
    image = cv2.rectangle(image, top_left, bottom_right, (255,255,255), -1)
     
    return(image, overlay, Diameter, Droplet_Diameter, Diameter_array, outRange_num)

    
#-----------------------------------------------------------------------------------------------
def main(output_folder, args):

    labels = open(args.labels).read().strip().split('\n')
    net = cv2.dnn.readNet(args.weights, args.config)
    print('[INFO] Yolo network loaded')
    if  not args.no_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  
        print('[INFO] Detecting using GPU')
    else: 
        print('[INFO] GPU is disabled')
    
    if args.save or args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        print('[INFO] Output directory created')  
    make_prediction(net, np.zeros((3552,3552,3), np.uint8), args.confidence, args.threshold) # lazy run
    print('[INFO] Initialization is complete')
    if output_folder:
        paths = []
        txt = os.path.join(output_folder, "paths.txt")
        if os.path.splitext(txt)[1] == '.txt':
            paths = open(txt).read().strip().split('\n')
            print('[INFO] Multiple images are loaded for detection')

        else:
            paths.append(txt)
            print('[INFO] One image is loaded for detection')
        print("")
        index = 0
        Diameter_array = []
        Diameter_array_continous=[]
        outrange_continous = 0 
        confid = args.confidence
        threshold1 = args.threshold
        
        #################### Dataframe to store results for each image ##########################
        dataframe = pd.DataFrame(columns=['image_name', 'droplet_diameter [micrometer]', 'number_droplets', 'detection_time [s]','number_outRange', 'median [micrometer]', 'mean [micrometer]', 'IQR [micrometer]', 'stand_deviation [micrometer]','x_25 [micrometer]', 'x_75 [micrometer]', 'min_diameter [micrometer]', 'max_diameter [micrometer]', 'stand_error','confidence_score','threshhold','ref_length [mm]', 'pixel_length'])
        st = time.time()
        for line in paths:
            if index < len(paths):  # every Image should be investigated
                try: 
                    image, overlay, Diameter, Droplet_Diameter, Diameter_array, outRange_num=everything (line, net, confid,threshold1,Diameter_array,labels,index,Diameter_array_continous) # <------- Hier
                except:  
                    print('No object detected in single image')
                    index += 1
                    continue
                Diameter_array_continous = Diameter_array_continous + Droplet_Diameter
                outrange_continous = outrange_continous + outRange_num # total number of detected outrange
                
                alpha = 0.4
                image = cv2.addWeighted(image, alpha, overlay, 1-alpha, 0)
                if args.show:
                    cv2.namedWindow("YOLO Object Detection", cv2.WINDOW_NORMAL)
                    cv2.imshow('YOLO Object Detection', image)
                    cv2.waitKey()
                if args.save or args.output_path:
                    result_folder = os.path.join(args.output_path, os.path.basename(output_folder))
                    if not os.path.exists(result_folder):
                        os.makedirs(result_folder)
                    cv2.imwrite(f'{result_folder}/{os.path.split(paths[index])[1]}', image)
                    outName= str(index)
                    try: q1, q3,median,mean,iqr,standd,bin_width,minimum_val,maximum_val = stat(Diameter_array_continous) # <---------- Hier
                    except:
                        index += 1
                        continue
                    output(dataframe, index, os.path.split(line)[1], Droplet_Diameter, len(Droplet_Diameter), (time.time())-st, outRange_num, median, mean, iqr, standd, q1, q3, minimum_val, maximum_val, standd, args.confidence, args.threshold, ref_length, pixel_length)                       
                    print('')
                cv2.destroyAllWindows()
                index += 1
             
        OverallTime= (time.time())-st   
        print("Number of images to reach 600 droplets: "+str(index))
        print("Overall time:",OverallTime)  #To get the time for over all speed       
        
        print("")       
        print("\033[4mOverall Parameters\033[0m")
        os.chdir(result_folder)
        try:
            q1, q3,median,mean,iqr,standd,bin_width,minimum_val,maximum_val = stat(Diameter_array_continous)
        except:
            print('No droplets in all images')
        

        csv_name = os.path.basename(result_folder)[:-4] #Saves csv file with same name than folders name
        dataframe.to_csv(f'{csv_name}.csv', sep=';')
        
        try:
            bin_count = int((max(Diameter_array_continous) - min(Diameter_array_continous)) / bin_width)
            
            fitters(Diameter_array_continous,titles="Distribution",binn=20)
            plt.title ("Boxplot")
            plt.ylabel("Droplet size [\u03bcm]")
            plt.boxplot(Diameter_array_continous)
            #plt.savefig('boxplot.png',format='png',dpi=900)
            
        except: 
            print('No graphs')

if __name__ == '__main__':
    
    monitor_and_extract()
        

