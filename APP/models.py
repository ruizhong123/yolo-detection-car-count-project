from django.db import models

# Create your models here.

## change structure for database 
class CountData(models.Model):
    
    
    timestep = models.DateTimeField(auto_now_add=True, db_index=True)  # Index for faster filtering
    
    timestep = models.DateTimeField(auto_now_add=True)
    date = models.CharField(max_length=8,null=True, blank=True)
    datetime = models.CharField(max_length=8,null=True, blank=True)
    linecount1 = models.IntegerField(default=0)
    linecount2 = models.IntegerField(default=0)
    polygon1count = models.IntegerField(default=0)
    polygon2count = models.IntegerField(default=0)
    
    
    class Meta:
        indexes = [models.Index(fields=['timestep'])]  # Ensure index
    
