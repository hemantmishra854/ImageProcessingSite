from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import *
from .forms import *
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
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('detect_face')
    else:
        form = ImageForm()
    return render(request, 'face_detection.html', {'form': form})


def process_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('upload')
    else:
        form = ImageForm()
    return render(request, 'process_image.html', {'form': form})


def upload(request):
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

    cv2.imwrite(os.path.join(img_path, 'face_' + img_arr[1]), cur_img)
    return render(request, 'detect_face.html', {'image': img.url, 'image_name': img_arr[1]})


def feature_matching(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('match_feature')
    else:
        form = ImageForm()
    return render(request, 'feature_matching.html',  {'form': form})


def match_feature(request):
    angle = 0
    if request.method == 'POST':
        angle = int(request.POST['angle'])

    users = Image.objects.all()
    img = users[len(users) - 1].image
    cur_img = cv2.imread(img.path, 1)
    gray_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    h, w = gray_img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 0.6)
    img2 = cv2.warpAffine(gray_img, rotation_matrix, (w, h))

    # ORB Detector
    orb = cv2.ORB_create(nfeatures=50)
    kp1, des1 = orb.detectAndCompute(gray_img, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matching_result = cv2.drawMatches(gray_img, kp1, img2, kp2, matches[:50], None, flags=2)

    img_path = r"C:\Users\Hemant\pycharm_projects\ImageProcessingSite\static\images"
    img_arr = img.name.split('/')
    cv2.imwrite(os.path.join(img_path, 'rot_' + img_arr[1]), img2)
    cv2.imwrite(os.path.join(img_path, 'matched_' + img_arr[1]), matching_result)
    return render(request, 'match_feature.html', {'image': img.url, 'image_name': img_arr[1]})


def compare_images(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('compare_images2')
    else:
        form = ImageForm()
    return render(request, 'compare_images.html',  {'form': form})


def compare_images2(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('compare_src_dst')
    else:
        form = ImageForm()
    return render(request, 'compare_images2.html',  {'form': form})


def compare_src_dst(request):
    messages = []
    users = Image.objects.all()
    img1 = users[len(users) - 1].image
    img2 = users[len(users) - 2].image
    cur_img1 = cv2.imread(img1.path, 1)
    cur_img2 = cv2.imread(img2.path, 1)
    difference = cv2.subtract(cur_img1, cur_img2)
    if cur_img1.shape == cur_img2.shape:
        cur_img1 = cv2.resize(cur_img1, (400, 400))
        cur_img2 = cv2.resize(cur_img2, (400, 400))
        difference = cv2.subtract(cur_img1, cur_img2)
        b, g, r = cv2.split(difference)
        messages.append("Both the Images have same size and channels")
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            messages.append("Both the images are completely equal")
        else:
            messages.append("Both the images are not equal")
    else:
        messages.append("Both the Images have not different size and channels")

    img_path = r"C:\Users\Hemant\pycharm_projects\ImageProcessingSite\static\images"
    img_arr = img1.name.split('/')
    cv2.imwrite(os.path.join(img_path, 'compared_' + img_arr[1]), difference)
    return render(request, 'compare_src_dst.html', {'image1': img1.url, 'image2': img2.url, 'messages': messages,
                                                    'image_name': img_arr[1]})

