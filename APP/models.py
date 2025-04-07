from django.db import models

# Create your models here.




## change structure for database 
class countData(models.Model):
    
    timestep = models.DateTimeField(auto_now_add=True)
    date = models.CharField(max_length=8)
    datetime = models.CharField(max_length=8)
    linecount1 = models.IntegerField()
    linecount2 = models.IntegerField()
    polygon1count = models.IntegerField()
    polygon2count = models.IntegerField()  
    
