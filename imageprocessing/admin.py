from django.contrib import admin
from .models import *


class ImageAdmin(admin.ModelAdmin):
    list_display = ['image']


# Register your models here.
admin.site.register(Image, ImageAdmin)
