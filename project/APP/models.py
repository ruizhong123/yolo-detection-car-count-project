from django.db import models

# Create your models here.

## change structure for database

class CountData_for_Line(models.Model):
    
    """Model to store count data"""
    
    date = models.CharField(max_length=8, blank=True, default="")
    linecount1 = models.IntegerField(default=0)
    linecount2 = models.IntegerField(default=0)
 
class CountData_for_polygon(models.Model):
    
    """Model to store count datatime"""
    
    datetime = models.CharField(max_length=8, blank=True, default="")
    polygon1count = models.IntegerField(default=0)
    polygon2count = models.IntegerField(default=0)
    


    
    
    
