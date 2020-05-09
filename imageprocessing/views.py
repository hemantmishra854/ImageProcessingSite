from django.shortcuts import render
from django.http import HttpResponse
from .models import Image
import cv2
import numpy
import os


def index(request):
    return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')


def contact(request):
    return render(request, 'contact.html')


def face_detection(request):
    return render(request, 'face-detection.html')


def process_image(request):
    return render(request, 'process_image.html')


def upload(request):
    uploaded_file = request.FILES['document']
    img = Image(image=uploaded_file)
    img.save()
    users = Image.objects.all()
    img = users[len(users) - 1].image
    cur_img = cv2.imread(img.path, 1)
    gray_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    img_path = r"C:\Users\Hemant\pycharm_projects\ImageProcessingSite\static\images"
    img_arr = img.name.split('/')
    cv2.imwrite(os.path.join(img_path, 'gray_'+img_arr[1]), gray_img)
    blur_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    cv2.imwrite(os.path.join(img_path, 'blur_' + img_arr[1]), blur_img)
    canny_img = cv2.Canny(blur_img, 150, 200)
    cv2.imwrite(os.path.join(img_path, 'canny_' + img_arr[1]), canny_img)
    kernel = numpy.ones((6, 6), numpy.uint8)
    dilate_img = cv2.dilate(canny_img, kernel, iterations=1)
    cv2.imwrite(os.path.join(img_path, 'dilate_' + img_arr[1]), dilate_img)
    erode_img = cv2.erode(dilate_img, kernel, iterations=1)
    cv2.imwrite(os.path.join(img_path, 'erode_'+img_arr[1]), erode_img)
    return render(request, 'upload.html', {'image': img.url, 'image_name': img_arr[1]})


def detect_face(request):
    uploaded_file = request.FILES['document']
    img = Image(image=uploaded_file)
    img.save()
    users = Image.objects.all()
    img = users[len(users) - 1].image
    cur_img = cv2.imread(img.path, 1)
    cur_img = cv2.resize(cur_img, (500, 500))
    gray_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    img_path = r"C:\Users\Hemant\pycharm_projects\ImageProcessingSite\static\images"
    img_arr = img.name.split('/')
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    faces = face_cascade.detectMultiScale(gray_img, 1.08, 5)
    for (x, y, w, h) in faces:
        cur_img = cv2.rectangle(cur_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = cur_img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            roi_color = cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('Face', cur_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(img_path, 'face_' + img_arr[1]), cur_img)
    return render(request, 'detect_face.html', {'image': img.url, 'image_name': img_arr[1]})


