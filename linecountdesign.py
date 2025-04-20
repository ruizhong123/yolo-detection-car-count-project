## design linecount 

class linecount : 
    
    def __init__(self,initial_point,end_point):
        
        self.initial_point = initial_point
        self.end_point = end_point
        self.count = 0
        
    # if i have 4 point on box ,why should i do 
    # trigger line is a vector 
    def trigger(self,box):
        
        # initial point : (x1,x2) 、 end_point(x2,y2)
        # box output  : [(x1,y1),(x1,y2),(x2,y1),(x2,y2)] , i want to extract the center point in box as   
        # box certiod : (box[0]+box[2])/2 , (box[1]+box[2])/2
        
        
        # trigger_line [(x.,y.),(x.,y.)]
        # i want to design above and below for a image 
        ## first thing  we do slop for caculate y with x 
        
        # b = y2 - y1 / x2 - x1
        
        box_certiod = self.box[0]
         
        trigger_line = [self.initial_point,self.end_point]
        
        slop = trigger_line[1] = trigger_line[0]
        
        if  box_certiod[1] > b*self.box_x :
            
            self.in_count +=  1
            
        
        
        
        
          
      