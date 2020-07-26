from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import *
import math
from .forms import *
import cv2
import numpy
import os


def index(request):
    return render(request, 'index.html')


def about(request):
    return render(request, 'about.html')


def privacy(request):
    return render(request, 'privacy.html')


def terms_and_conditions(request):
    return render(request, 'terms_and_conditions.html')


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
    cv2.imwrite(os.path.join(img_path, 'gray_' + img_arr[1]), gray_img)
    blur_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    cv2.imwrite(os.path.join(img_path, 'blur_' + img_arr[1]), blur_img)
    canny_img = cv2.Canny(blur_img, 150, 200)
    cv2.imwrite(os.path.join(img_path, 'canny_' + img_arr[1]), canny_img)
    kernel = numpy.ones((6, 6), numpy.uint8)
    dilate_img = cv2.dilate(canny_img, kernel, iterations=1)
    cv2.imwrite(os.path.join(img_path, 'dilate_' + img_arr[1]), dilate_img)
    erode_img = cv2.erode(dilate_img, kernel, iterations=1)
    cv2.imwrite(os.path.join(img_path, 'erode_' + img_arr[1]), erode_img)
    return render(request, 'upload.html', {'image': img.url, 'image_name': img_arr[1]})


def detect_face(request):
    users = Image.objects.all()
    img = users[len(users) - 1].image
    cur_img = cv2.imread(img.path, 1)
    # cur_img = cv2.resize(cur_img, (500, 500))
    gray_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    img_path = r"C:\Users\Hemant\pycharm_projects\ImageProcessingSite\static\images"
    img_arr = img.name.split('/')
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    faces = face_cascade.detectMultiScale(gray_img, 1.08, 5)
    for (x, y, w, h) in faces:
        cur_img = cv2.rectangle(cur_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_color = cur_img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            roi_color = cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(img_path, 'face_' + img_arr[1]), cur_img)
    return render(request, 'detect_face.html', {'image': img.url, 'image_name': img_arr[1]})


def feature_matching(request):
    if request.method == 'POST':
        print('before initiating the form')
        form = ImageForm(request.POST, request.FILES)
        print('after initiating the form')
        if form.is_valid():
            print('before saving the form')
            form.save()
            print('after saving the form')
            return redirect('match_feature')
            #return redirect('success')
    else:
        form = ImageForm()

    print('before redirecting the form')
    return render(request, 'feature_matching.html', {'form': form})


def match_feature(request):
    angle = 0
    if request.method == 'POST':
        angle = int(request.POST['angle'])

    users = Image.objects.all()
    img = users[len(users) - 1].image
    cur_img = cv2.imread(img.path, 1)
    # gray_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
    h, w = cur_img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 0.6)
    img2 = cv2.warpAffine(cur_img, rotation_matrix, (w, h))

    # start of swift detector
    e1 = cv2.getTickCount()

    # sift detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(cur_img, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    no_of_good_points = len(good)
    total_points = 0
    if len(kp1) <= len(kp2):
        total_points = len(kp1)
    else:
        total_points = len(kp2)

    result = dict()
    result['kp1_sift'] = len(kp1)
    result['kp2_sift'] = len(kp2)
    result['good_points_sift'] = no_of_good_points
    result['match_percentage_sift'] = math.ceil(no_of_good_points / total_points * 100)

    # cv2.drawMatchesKnn expects list of lists as matches.
    matching_result_sift = cv2.drawMatchesKnn(cur_img, kp1, img2, kp2, good, None, flags=2)

    # end of sift detector
    e2 = cv2.getTickCount()
    # time taken by sift detector
    sift_time = (e2 - e1)/cv2.getTickFrequency()
    result['sift_time'] = sift_time

    img_path = r"C:\Users\Hemant\pycharm_projects\ImageProcessingSite\static\images"
    img_arr = img.name.split('/')

    cv2.imwrite(os.path.join(img_path, 'sift_' + img_arr[1]), matching_result_sift)

    # start of orb detector
    e1 = cv2.getTickCount()

    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(cur_img, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    no_of_good_points = len(matches)
    total_points = 0
    if len(kp1) <= len(kp2):
        total_points = len(kp1)
    else:
        total_points = len(kp2)

    result['kp1_orb'] = len(kp1)
    result['kp2_orb'] = len(kp2)
    result['good_points_orb'] = no_of_good_points
    result['match_percentage_orb'] = math.ceil(no_of_good_points / total_points * 100)

    matching_result = cv2.drawMatches(cur_img, kp1, img2, kp2, matches[:50], None, flags=2)

    # end of orb detector
    e2 = cv2.getTickCount()
    # time taken by orb detector
    orb_time = (e2 - e1)/cv2.getTickFrequency()

    result['orb_time'] = orb_time

    cv2.imwrite(os.path.join(img_path, 'rot_' + img_arr[1]), img2)
    cv2.imwrite(os.path.join(img_path, 'orb_' + img_arr[1]), matching_result)
    return render(request, 'match_feature.html', {'image': img.url, 'image_name': img_arr[1], 'result': result})


def compare_images(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('compare_images2')
    else:
        form = ImageForm()
    return render(request, 'compare_images.html', {'form': form})


def compare_images2(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('compare_src_dst')
    else:
        form = ImageForm()
    return render(request, 'compare_images2.html', {'form': form})


def compare_src_dst(request):
    messages = []
    users = Image.objects.all()
    img1 = users[len(users) - 1].image
    img2 = users[len(users) - 2].image
    cur_img1 = cv2.imread(img1.path, 1)
    cur_img2 = cv2.imread(img2.path, 1)

    img_path = r"C:\Users\Hemant\pycharm_projects\ImageProcessingSite\static\images"
    img_arr = img1.name.split('/')

    if cur_img1.shape == cur_img2.shape:
        difference = cv2.subtract(cur_img1, cur_img2)
        b, g, r = cv2.split(difference)
        cv2.imwrite(os.path.join(img_path, 'difference_' + img_arr[1]), difference)
        messages.append("Both the Images have same size and channels")
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            messages.append("Both the images are completely equal")
        else:
            messages.append("Both the images are not equal")
    else:
        cur_img1_res = cv2.resize(cur_img1, (400, 400))
        cur_img2_res = cv2.resize(cur_img2, (400, 400))
        difference = cv2.subtract(cur_img1_res, cur_img2_res)
        b, g, r = cv2.split(difference)
        cv2.imwrite(os.path.join(img_path, 'difference_' + img_arr[1]), difference)
        messages.append("Both the Images have different size and channels")

    # start of sift detector
    e1 = cv2.getTickCount()

    # sift detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(cur_img1, None)
    kp2, des2 = sift.detectAndCompute(cur_img2, None)

    # FlannBasedMatcher with default params
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])

    no_of_good_points = len(good)
    total_points = 0
    if len(kp1) <= len(kp2):
        total_points = len(kp1)
    else:
        total_points = len(kp2)

    result = dict()
    result['kp1'] = len(kp1)
    result['kp2'] = len(kp2)
    result['good_points'] = no_of_good_points
    result['match_percentage'] = math.ceil(no_of_good_points / total_points * 100)

    # cv2.drawMatchesKnn expects list of lists as matches.
    matching_result_flann = cv2.drawMatchesKnn(cur_img1, kp1, cur_img2, kp2, good, None, flags=2)

    # end of sift detector
    e2 = cv2.getTickCount()
    # time taken by sift detector
    sift_time = (e2 - e1)/cv2.getTickFrequency()
    result['sift_time'] = sift_time

    cv2.imwrite(os.path.join(img_path, 'flann_' + img_arr[1]), matching_result_flann)

    return render(request, 'compare_src_dst.html', {'image1': img1.url, 'image2': img2.url, 'messages': messages,
                                                    'image_name': img_arr[1], 'result': result})


def image_filter(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('filter_images')
    else:
        form = ImageForm()
    return render(request, 'image_filter.html', {'form': form})


def filter_images(request):
    users = Image.objects.all()
    img = users[len(users) - 1].image
    cur_img = cv2.imread(img.path, 1)
    kernel = numpy.ones((5, 5), numpy.float32) / 25
    img_path = r"C:\Users\Hemant\pycharm_projects\ImageProcessingSite\static\images"
    img_arr = img.name.split('/')

    filter2d = cv2.filter2D(cur_img, -1, kernel)
    cv2.imwrite(os.path.join(img_path, '2d_' + img_arr[1]), filter2d)

    bilateral_filter = cv2.bilateralFilter(cur_img, 9, 75, 75)
    cv2.imwrite(os.path.join(img_path, 'bilateral_' + img_arr[1]), bilateral_filter)

    median_filter = cv2.medianBlur(cur_img, 5)
    cv2.imwrite(os.path.join(img_path, 'median_' + img_arr[1]), median_filter)

    return render(request, 'filter_images.html', {'image': img.url, 'image_name': img_arr[1]})


def success(request):
    return HttpResponse('Success....')

