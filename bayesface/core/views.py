from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from core.models import Document
from core.forms import DocumentForm
from bayesface.settings import BASE_DIR, MEDIA_URL
import os
import faceByBayes


def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        imagepath = os.path.join(BASE_DIR, '') + uploaded_file_url
        print imagepath
        user_index, check_time, scores = faceByBayes.test_clf(imagepath.replace('/','\\'))
        user_image_url = MEDIA_URL + 'user/s' + str(int(user_index)) + '/1.bmp'
        return render(request, 'core/result.html', {
            'user_image_url': user_image_url,
            'check_time':check_time,
            'scores':scores
        })
    return render(request, 'core/simple_upload.html')
