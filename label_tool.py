import tkinter as tk
import tkinter.font as tkFont
from PIL import Image, ImageTk
import os,json,re,math
import sys
##sys.path.append(r'F:\Anaconda8\Lib\site-packages')
##sys.path.append(r'F:\Anaconda8\pkgs')
import tensorflow as tf
import numpy as np

def NMS(B,S,iouthreshold):
  l=[[],[]]
  
  while S.shape[0]>0:
    max_indices=tf.math.argmax(S[:,0])
    # print(S.shape[0])
    # print(S[:10,0])
    l[0].append(B[max_indices])
    l[1].append(S[max_indices])
    iou=Iou2(B[max_indices],B)
    # print(iou.shape,iou[:10],int(max_indices),iou[max_indices])
    # print(tf.where(iou<iouthreshold).shape)
    B=tf.gather_nd(B,tf.where(iou<iouthreshold))
    S=tf.gather_nd(S,tf.where(iou<iouthreshold))
  return l

def Iou2(a,b):#x1,y1,x2,y2
  if len(a.shape)==1:
    max=tf.where(a[:2]>b[:,:2],a[:2],b[:,:2])
    min=tf.where(a[2:]<b[:,2:],a[2:],b[:,2:])
    m=min-max
    s=m[:,0]*m[:,1]
    s1=(a[2]-a[0])*(a[3]-a[1])
    s2=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    iou=s/(s1+s2-s)
    iou=tf.where((max[:,0]<min[:,0]) & (max[:,1]<min[:,1]),iou,0)
    return iou
  elif len(a.shape)==2:
    print('Iou2 error')
  else:
    print('Iou2 error')


class App:
    def __init__(self,master):
        self.pic=None
        self.tkpic=None
        self.master=master
        self.tkpic=None
        self.cv_image=None
        self.outline='red'
        self.choose_outline='green'
        self.resolution_x=-1#改变后的图片大小
        self.resolution_y=-1
        self.ordinary_x=-1#原图片大小
        self.ordinary_y=-1
        self.canvas_x=700#画布大小
        self.canvas_y=400
        self.scale=-1#原图片/后图片
        self.init_widgets()
#         self.line_num=-1#当前选定的线段编号
#         self.exit_line=False#当前是否已经画完了四条线
        self.folder_state_init()
        self.file_state_init()
        self.picture_class_font=tkFont.Font(size=30, weight=tkFont.BOLD)
        self.normal_color='black'
        self.template_color='yellow'
        self.now_rectangle_color='red'
        self.preview_color='green'
        self.pc={}#picture_class
        self.lc={}#label_class
        self.tc={}#template_class
    def folder_state_init(self):
        self.pic_template={}
        self.filename=None
        self.file_state=-1#输入的文件名是否存在/为文件/文件夹 -1代表文件不存在 0代表文件 1代表文件夹
        self.filepath=None
        self.pf1=[]#picture_file
        self.pf2=[]
        self.jf1=[]#json_file
        self.jf2=[]
##        self.pc={}#picture_class
##        self.lc={}#label_class
##        self.tc={}#template_class
        self.file_num=-1
        self.label_template={}
        if self.alistbox.size()>0:
            self.alistbox.delete(0,self.alistbox.size()-1)
##        if self.blistbox.size()>0:
##            self.blistbox.delete(0,self.blistbox.size()-1)
##        if self.clistbox.size()>0:
##            self.clistbox.delete(0,self.clistbox.size()-1)
##        if self.dlistbox.size()>0:
##            self.dlistbox.delete(0,self.dlistbox.size()-1)
            
    def file_state_init(self):
        self.picture_class=None
        self.class_text=None
        self.rectangle={}#存放矩形的坐标{handler:[ax,ay,bx,by,label_class,text对象]}
        self.classname=None
        self.drawing_rectangle_type=0 # 0代表无操作，1代表正在画矩形，2代表拖动矩形,3表示ctrl按下，4表示ctrl松开
        self.now_rectangle=None
        self.model_rectangle={}
        self.model_now_rectangle=None
        self.model_class_name=None
        self.ax=-10 #可作为画矩形时时左上点坐标，也可作为移动矩形时初始坐标
        self.ay=-10
        self.mv_prevx=-10
        self.mv_prevy=-10
        self.scale=1
        self.offset=[0,0]
        self.model_label_detail={}
        
    def init_widgets(self):
        self.cv=tk.Canvas(self.master,width=self.canvas_x, height=self.canvas_y)
        #self.cv.bind('a',self.delete_rectangle)
        #picturecanvas.create_image(0,0,anchor=NW,image=tkpicture)
        #self.cv.bind('<Button-1>',self.B1)
        self.cv.bind("<Enter>",self.a)
        
        self.cv.bind('<ButtonRelease-1>',self.BR1)
        self.cv.bind('<B1-Motion>',self.B1m)
        self.cv.bind('a',self.alt_l)
        #self.cv.bind('b',self.bigger)
        self.cv.bind('<Button-2>',self.double_scale)
        self.cv.bind('<Double-Button-2>',self.return_scale)
        #self.cv.bind('<KeyRelease-Alt_L>',self.keyrelease_alt_l)
        #picturecanvas.bind('<B1-Motion>',bclick)
        #self.cv.bind('<ButtonRelease-1>',cclick)
        #self.cv.bind('<BackSpace>',returnclick)
        self.cv.place(x=0,y=0)
        
        self.ebutton=tk.Button(self.master,text='save',bg='red',width=5,height=1,command=self.save)
        self.ebutton.place(x=720,y=50)
        
        
        self.alabel=tk.Label(self.master,text="filepath",width=10,height=1,bg='green')
        self.alabel.place(x=0,y=420)
        
        self.aentry=tk.Entry(self.master,text="0",bg='green')
        self.aentry.place(x=0,y=450)
        
        self.abutton=tk.Button(self.master,text="list the file",command=self.listfile,bg='green')
        self.abutton.place(x=0,y=500)
        
        self.alistbox=tk.Listbox(self.master,height=10,width=10)
        self.alistbox.place(x=200,y=420)
        self.alistbox.bind('<Double-Button-1>',self.aprintlist)
        
        self.blabel=tk.Label(self.master,text='labelname',bg='green')
        self.blabel.place(x=0,y=550)
        
        self.bentry=tk.Entry(self.master,text='1',bg='green')
        self.bentry.place(x=720,y=300)
        
        self.bbutton=tk.Button(self.master,text="type the label class",command=self.makelabelclass,bg='green')
        self.bbutton.place(x=720,y=350)

        self.gbutton=tk.Button(self.master,text='add label property',command=self.addlabel,bg='green')
        self.gbutton.place(x=720,y=380)

        
        self.blistbox=tk.Listbox(self.master,height=10,width=20)
        self.blistbox.place(x=720,y=100)
        self.blistbox.bind('<Double-Button-1>',self.bprintlist)
        
        self.centry=tk.Entry(self.master,text='2',bg='green')
        self.centry.place(x=920,y=300)
        
        self.cbutton=tk.Button(self.master,text="type the picture class",command=lambda currentlabel=None,class_type='pic':self.makelabelclass(currentlabel,class_type),bg='green')
        self.cbutton.place(x=920,y=350)
        
        self.clistbox=tk.Listbox(self.master,height=10,width=20)
        self.clistbox.place(x=920,y=100)
        self.clistbox.bind('<Double-Button-1>',self.cprintlist)
        
        self.dbutton=tk.Button(self.master,text="next picture",command=self.next_pic,bg='green')
        #self.dbutton=tk.Button(self.master,text="next picture",command=self.delete_all,bg='green')
        self.dbutton.place(x=200,y=650)
        
        
        self.blabel=tk.Label(self.master,text='template',bg='green')
        self.blabel.place(x=400,y=410)

        self.dlistbox=tk.Listbox(self.master,height=10,width=20)
        self.dlistbox.place(x=400,y=430)
                #listbox.bind('<Button-1>',lambda event,string=string[i],listbox=listbox:self.preview_template(event,string,listbox))
        self.dlistbox.bind('<Double-Button-1>',lambda event,string='template',listbox=self.dlistbox:self.apply_template(event,string,listbox))
        self.dlistbox.bind('<Button-3>',lambda event,string='template',listbox=self.dlistbox:self.delete_template(event,string,listbox))

        self.dentry=tk.Entry(self.master,text='5',bg='green')
        self.dentry.place(x=400,y=620)

        self.ebutton=tk.Button(self.master,text='save',command=lambda string='template',entry=self.dentry,listbox=self.dlistbox:self.save_template(string,entry,listbox))
        self.ebutton.place(x=400,y=650)

        self.ibutton=tk.Button(self.master,text='model_predict',command=self.model_predict)
        self.ibutton.place(x=900,y=550)

        self.jbutton=tk.Button(self.master,text='load_model',command=self.load_model)
        self.jbutton.place(x=900,y=400)
        
        self.fentry=tk.Entry(self.master,text='model_path',bg='green')
        self.fentry.place(x=900,y=450)

        self.flabel=tk.Label(self.master,text='not load model',bg='green')
        self.flabel.place(x=900,y=500)

        self.glabel=tk.Label(self.master,text='model_predict_label',bg='green')
        self.glabel.place(x=720,y=420)
        
    def load_model(self):
##        sys.path.append(r'C:\Users\Administrator\AppData\Roaming\Python\Python36\site-packages')
##        import tensorflow as tf
        filename=self.fentry.get()
        print('asdfw')
        if filename !=None:
            if os.path.exists(filename):
                if '.h5' in filename:
                    self.model=tf.keras.models.load_model(filename,
                                        custom_objects={'custom_loss': self.custom_loss,
                                                        'custom_loss2':self.custom_loss2})
                    self.flabel.config(text='succeed loading model')
        print('vfdbdrt')
        self.default_anchor_size=tf.constant([[0.05416219,0.09319306],
                                              [0.1060535 ,0.06356987],
                                              [0.14459075,0.3840488 ],
                                              [0.11091662,0.15959485],
                                              [0.05781969,0.03113795]])
        self.abn=self.default_anchor_size.shape[0]
        config_file=filename.replace('.h5','-config.json')
        with open(config_file,'rb') as f:
            data=json.load(f)
        self.width=data['width']
        self.height=data['height']
        self.hs=data['hs']
        self.ws=data['ws']
        self.label_list=data['label_list']
        self.class_list=data['class_list']
        self.label_num=len(self.label_list)
        self.class_num=len(self.class_list)
        print(self.width,self.height,self.hs,self.ws,self.label_list,self.class_list,self.label_num,self.class_num)
        
    def  post_process(self,output,scorethreshold,iouthreshold):#one picture
      print(1*self.hs*self.ws*self.abn*(1+4+self.label_num))
      output=tf.reshape(output,(1,self.hs,self.ws,self.abn,1+4+self.label_num))
      indices=tf.where(output[...,0]>scorethreshold)
      output=tf.gather_nd(output,indices)
      S=tf.concat([output[:,0:1],output[:,5:]],axis=1)
      anchor=tf.gather_nd(self.default_anchor_size,indices[:,3:4])
      width=tf.exp(output[:,4])*anchor[:,0]
      height=tf.exp(output[:,3])*anchor[:,1]
      central_x=(tf.sigmoid(output[:,2])+tf.cast(indices[:,2],dtype=tf.float32))/self.ws
      central_y=(tf.sigmoid(output[:,1])+tf.cast(indices[:,1],dtype=tf.float32))/self.hs
      B=tf.stack([central_x-width/2,central_y-height/2,central_x+width/2,central_y+height/2],axis=1)
      l=NMS(B,S,iouthreshold)#x1,y1,x2,y2
      return l
    
    def seeing_result(self,scorethreshold,iouthreshold):
      dataset_x=tf.reshape(tf.convert_to_tensor(np.array(self.pic.resize((self.width,self.height),Image.BILINEAR)),dtype=tf.float32)/255.0,(1,self.height,self.width,3))
      y1,y2=self.model(dataset_x)
      l=self.post_process(y1,scorethreshold,iouthreshold)
      l0=tf.stack(l[0],axis=0)
      class_sort=tf.math.top_k(y2,3)
      self.makelabelclass(currentlabel=self.class_list[class_sort.indices[0,0]],class_type='class')
      text=''
      for i in range(3):
          text=text+self.class_list[class_sort.indices[0,i]]+'-'+'%.3f' %class_sort.values[0,i]+'\n'
      self.cv.itemconfig(self.class_text,text=text)
      if l0.shape[0]>0:
        for i in range(len(l[1])):
          label_sort=tf.math.top_k(l[1][i][1:],5)
          a=[int(l0[i,0]*self.resolution_x),int(l0[i,1]*self.resolution_y),int(l0[i,2]*self.resolution_x),int(l0[i,3]*self.resolution_y)]
          self.now_rectangle=self.cv.create_rectangle((a[0]-self.offset[0])*self.scale,(a[1]-self.offset[1])*self.scale,
                                                      (a[2]-self.offset[0])*self.scale,(a[3]-self.offset[1])*self.scale,
                                                      outline=self.now_rectangle_color)
          self.rectangle[self.now_rectangle]=[a[0],a[1],a[2],a[3],None,None]
          self.ax=-10
          self.ay=-10
          self.drawing_rectangle_type=0
          self.makelabelclass(currentlabel=self.label_list[label_sort.indices[0]],class_type='label')
          text='%.1f' % l[1][i][0]+'\n'
          #+self.label_list[label_sort.indices[0]]+'-'+'%.1f' %label_sort.values[0]
          self.cv.tag_bind(self.now_rectangle,'<Button-1>',lambda event, t=self.now_rectangle: self.choose_item_handler(event,t))
          self.cv.tag_bind(self.now_rectangle,'<Button-3>',lambda event, t=self.now_rectangle: self.delete_rectangle(event,t))
          self.cv.tag_bind(self.now_rectangle,'<Double-Button-1>',self.show_model_predict_label)
          for j in range(0,4):
            text=text+'\n'+self.label_list[label_sort.indices[j]]+'  '+'%.1f' %label_sort.values[j]
            if j!=0 and label_sort.values[j]>0:
                self.addlabel(self.label_list[label_sort.indices[j]])
          self.model_label_detail[self.now_rectangle]=text
          #self.cv.itemconfig(self.rectangle[self.now_rectangle][5],text=text)
          #self.rectangle[self.now_rectangle]
            
##      if l0.shape[0]>0:
##        colors = tf.cast(tf.tile(tf.constant([[1,0,0]]),(l0.shape[0],1)),dtype=tf.float32)
##        c=tf.stack((l0[:,1],l0[:,0],l0[:,3],l0[:,2]),axis=1)
##        dataset_x=tf.image.draw_bounding_boxes(dataset_x,tf.reshape(c,(1,-1,4)),colors)
    def model_predict(self):
        for keys,values in self.rectangle.items():
            print(keys,values)
            self.cv.delete(keys)
            self.cv.delete(values[5])
        if self.class_text != None:
            print(self.class_text)
            self.cv.delete(self.class_text)
        self.file_state_init()
        self.seeing_result(0,0.2)
    def custom_loss(self,target_y,pred_y):
        return 0
    def custom_loss2(self,traget_y,pred_y):
        return 0
    
    def apply_template(self,event,string,listbox):
        filename=self.filepath+'\\'+string+'.json'
        print(filename)
        if self.drawing_rectangle_type==0 and os.path.exists(filename):
            a=listbox.get(listbox.curselection()[0])
            with open(filename,'r') as f:
                b=json.load(f)
            for i in b:
                if a==i[0]:
                    c=i[1]
                    break
            else:
                print('not found template')
                return 
            for j in c:
                fixed_num=[int(num*self.ratio) for num in j[1]]
                rectangle=self.cv.create_rectangle(fixed_num,outline=self.normal_color)
                self.rectangle[rectangle]=fixed_num+[None,None]
                self.cv.tag_bind(rectangle,'<Button-1>',lambda event, t=rectangle: self.choose_item_handler(event,t))
                self.cv.tag_bind(rectangle,'<Button-3>',lambda event, t=rectangle: self.delete_rectangle(event,t))
                if j[0]!=None:
                    self.change_label_class(rectangle,j[0])
    
    def delete_template(self,event,string,listbox):
        filename=self.filepath+'\\'+string+'.json'
        a=listbox.get(listbox.curselection()[0])
        if os.path.exists(filename):
            with open(filename,'r') as f:
                b=json.load(f)
        else:
            print("not found template file")
        i=0
        while i < len(b):
            if a in b[i]:
                c=i
                break
            i+=1
        else:
            print("not found template")
            return
        del b[c]
        listbox.delete(c)
        with open(filename,'w') as f:
            json.dump(b,f,ensure_ascii=False)
            
    def save_template(self,string,entry,listbox):
        if len(self.rectangle)>0:
            a=entry.get()
            if a=='':
                return 
            new_list=[]
            filename=self.filepath+'\\'+string+'.json'
            for value in self.rectangle.values():
                new_list.append([value[4],[num/self.ratio for num in value[0:4]]])
            if os.path.exists(filename):
                with open(filename,'r') as f:
                    data=json.load(f)
            else:
                data=[]
            for i in range(len(data)):
                if data[i][0]==a:
                    data[i][1]=new_list
                    break
            else:
                self.dlistbox.insert('end',a)
                data.append([a,new_list])
            with open(filename,'w') as f:
                json.dump(data,f,ensure_ascii=False)

    def save(self):
        new_list=[]
        if self.picture_class != None:
            new_list.append(["class_name",self.picture_class])
        for value in self.rectangle.values():
            if value[4] !=None:
                new_list.append([value[4],[num/self.ratio for num in value[0:4]]])
        with open(self.filename+'.json','w') as f:
            json.dump(new_list,f,ensure_ascii=False)
    
    def load(self):
        if os.path.exists(self.filename+'.json'):
            a=[]
            with open(self.filename+'.json','r') as f:
                a=json.load(f)
                for d in a:
                    if 'class_name' in d[0]:
                        self.change_picture_class(d[1])
                    else:
                        fixed_num=[num*self.ratio for num in d[1]]
                        rectangle=self.cv.create_rectangle([int(i) for i in fixed_num],outline='black')
                        self.rectangle[rectangle]=fixed_num+[None,None]
                        self.cv.tag_bind(rectangle,'<Button-1>',lambda event, t=rectangle: self.choose_item_handler(event,t))
                        self.cv.tag_bind(rectangle,'<Button-3>',lambda event, t=rectangle: self.delete_rectangle(event,t))
                        self.change_label_class(rectangle,d[0])
        
    def bprintlist(self,event):
        if self.file_state>=0:
            self.makelabelclass(currentlabel=self.blistbox.get(self.blistbox.curselection()[0]))
        
    def cprintlist(self,event):
        if self.file_state>=0:
            self.makelabelclass(currentlabel=self.clistbox.get(self.clistbox.curselection()[0]),class_type='pic')
        
    def readjson(self):
        for i in self.pf2:
            filename=self.filepath+'\\'+i+'.json'
            with open(filename,'r') as f:
                c=json.load(f)
                for list1 in c:
                    if 'class_name' in list1[0]:
                        e=list1[1]
                        if e not in list(self.pc.keys()):
                            self.pc[e]=filename.split('\\')[-1].split('.')
                            self.clistbox.insert('end',e)
                    else:
                        f=list1[0]
                        for e in f.split('-'):
                            if e not in self.lc:
                                self.lc[e]=filename.split('\\')[-1].split('.')
                                self.blistbox.insert('end',e)
        filename=self.filepath+'\\template.json'
        if os.path.exists(filename):
            with open(filename,'r') as f:
                c=json.load(f)
                for list1 in c:
                    if list1[0] not in list(self.tc.keys()):
                      self.tc[list1[0]]=filename.split('\\')[-1].split('.')
                      self.dlistbox.insert('end',list1[0])
    def show_model_predict_label(self,event):
      self.glabel.config(text=self.model_label_detail[self.now_rectangle])
                    
    def aprintlist(self,event):
        a=self.alistbox.size()
        if self.file_state==1 and a>1:
            b=self.alistbox.curselection()
            self.file_num=b[0]
            self.filename=self.filepath+'\\'+self.alistbox.get(self.file_num)
            self.openpicture(self.filename+'.jpg')
                
    def delete_text(self,event,t):
        if self.rectangle[t][5] !=None:
            self.cv.delete(self.rectangle[t][5])
            self.rectangle[t][4:6]=[None,None]
        
    def delete_rectangle(self,event,t):
        self.cv.delete(t)
        self.cv.delete(self.rectangle[t][5])
        self.drawing_rectangle_type=0
        del self.rectangle[t]
        if self.now_rectangle == t:
            self.now_rectangle=None
        self.ax=-10
        self.ay=-10
        self.mv_prevx=-10
        self.mv_prevy=-10
            
    def delete_picture_class(self,event):
        if self.class_text != None:
            self.cv.delete(self.class_text)
        self.class_text=None
        self.picture_class=None

        
    def next_pic(self):
        a=self.alistbox.size()
        if self.file_state==1 and a>1:
            if self.file_num<a-1:
                self.file_num=self.file_num+1
            else:
                self.file_num=0
            self.filename=self.filepath+'\\'+self.alistbox.get(self.file_num)
            self.openpicture(self.filename+'.jpg')
        print("next_pic:",self.filename)
            
            
    
    def listfile(self):
        self.delete_all()
        self.file_state_init()
        self.folder_state_init()
        self.filepath=self.aentry.get()
        if os.path.exists(self.filepath):
            if os.path.isdir(self.filepath):
                self.file_state_init()
                self.file_state=1
                listdir=os.listdir(self.filepath)
                for i in listdir:
                    str1=i.split('.')
                    if len(str1)>1:
                        if str1[1]=='jpg':
                            if str1[0]+'.json' in listdir:
                                self.pf2.append(str1[0])
                            else:
                                self.pf1.append(str1[0])
                for i in range(len(self.pf1)):
                    self.alistbox.insert('end',self.pf1[i])
                    self.alistbox.itemconfig(i,fg='red')
                for i in range(len(self.pf2)):
                    self.alistbox.insert('end',self.pf2[i])
                self.readjson()
                self.file_num=0
                self.filename=self.filepath+'\\'+self.alistbox.get(self.file_num)
                self.openpicture(self.filename+'.jpg')
            else:
                self.file_state=0
                self.file_num=-1
                self.filename=self.filepath
                self.filepath=None
                self.openpicture(self.filename)
        else:
            self.file_state=-1
            self.file_num=-1
            self.filename=None
            self.filepath=None
            self.alabel.config(text='this file does not exist')
        
    def openpicture(self,filename):
        self.alabel.config(text=self.filename.split('\\')[-1])
        self.delete_all()
        print("openpicture:",self.filename)
        self.file_state_init()
        filename=self.filename+'.jpg'
        self.pic = Image.open(filename)
        width,height=self.pic.size
        if width!=self.ordinary_x or height!=self.ordinary_y:
            self.ordinary_x=width
            self.ordinary_y=height
            self.ratio=min(self.canvas_x/self.ordinary_x,self.canvas_y/self.ordinary_y)
            self.resolution_x=int(self.ordinary_x*self.ratio)
            self.resolution_y=int(self.ordinary_y*self.ratio)
        self.pic=self.pic.resize((self.resolution_x,self.resolution_y),Image.BILINEAR)
        self.tkpic = ImageTk.PhotoImage(image=self.pic,master=self.cv)
        self.cv_image=self.cv.create_image(0,0,anchor=tk.NW,image=self.tkpic)
#         self.cv.tag_bind(a,'<Double-Button-3>',lambda event,b=a:self.delete_pic(event,b))
        self.load()
    
#     def delete_pic(self,event,b):
#         self.cv.delete(b)
    
    def delete_all(self):
        print(type(self.rectangle))
        if self.cv_image!=None:
            self.cv.delete(self.cv_image)
        for keys,values in self.rectangle.items():
            self.cv.delete(keys)
            self.cv.delete(values[5])
        if self.class_text != None:
            self.cv.delete(self.class_text)
            
    def makelabelclass(self,currentlabel=None,class_type='label'):
        jsonfile=self.filename.split('.')[0]+'.json'
        if class_type=='label':
            if self.now_rectangle != None:
                if currentlabel==None:
                    currentlabel=self.bentry.get()
                self.change_label_class(self.now_rectangle,currentlabel)    
                if currentlabel not in list(self.lc.keys()):
                    self.lc[currentlabel]=self.filename.split('\\')[-1].split('.')
                    self.blistbox.insert('end',currentlabel)
        else:
            if currentlabel==None:
                currentlabel=self.centry.get()
            self.change_picture_class(currentlabel)
            if currentlabel not in list(self.pc.keys()):
                self.pc[currentlabel]=self.filename.split('\\')[-1].split('.')
                self.clistbox.insert('end',currentlabel)
    def addlabel(self,label=None):
        if label==None:
            label=self.bentry.get()
        t=self.now_rectangle
        if label!=None:
            self.rectangle[t][4]=self.rectangle[t][4]+'-'+label
        else:
            self.rectangle[t][4]=label
        r=self.rectangle[t]
        if r[5] != None:
            self.cv.itemconfig(r[5],text=self.rectangle[t][4])
        else:
            self.rectangle[t][5]=self.cv.create_text([(r[0]-self.offset[0])*self.scale,(r[3]-self.offset[1])*self.scale],text=self.rectangle[t][4],anchor="nw")
            self.cv.tag_bind(self.rectangle[t][5],'<Button-3>',lambda event, t=t: self.delete_text(event,t))

    def B1m(self,event):
        if self.drawing_rectangle_type==0:
            if self.now_rectangle!=None:
                self.cv.itemconfig(self.now_rectangle,outline='black')
                self.now_rectangle=None
            self.drawing_rectangle_type=1
            self.ax=event.x
            self.ay=event.y
        elif self.drawing_rectangle_type==1:
            if self.now_rectangle != None:
                self.cv.delete(self.now_rectangle)
            self.now_rectangle=self.cv.create_rectangle(self.ax,self.ay,event.x,event.y,outline='black',dash=2)
        elif self.drawing_rectangle_type==2:
            if self.ax>0 and self.ay >0 and self.now_rectangle != None:
                self.cv.move(self.now_rectangle,event.x-self.ax,event.y-self.ay)
                if self.rectangle[self.now_rectangle][5] != None:
                    self.cv.move(self.rectangle[self.now_rectangle][5],event.x-self.ax,event.y-self.ay)
                self.ax=event.x
                self.ay=event.y
            
    def BR1(self,event):
        if self.drawing_rectangle_type==1:
            if abs(self.ax-event.x) >5 or abs(self.ay-event.y)>5:
                self.cv.delete(self.now_rectangle)
                self.now_rectangle=self.cv.create_rectangle(self.ax,self.ay,event.x,event.y,outline=self.now_rectangle_color)
                self.rectangle[self.now_rectangle]=[min(self.ax,event.x)/self.scale+self.offset[0],min(self.ay,event.y)/self.scale+self.offset[1],max(self.ax,event.x)/self.scale+self.offset[0],max(self.ay,event.y)/self.scale+self.offset[1],None,None]
                self.ax=-10
                self.ay=-10
                self.drawing_rectangle_type=0
                self.cv.tag_bind(self.now_rectangle,'<Button-1>',lambda event, t=self.now_rectangle: self.choose_item_handler(event,t))
                self.cv.tag_bind(self.now_rectangle,'<Button-3>',lambda event, t=self.now_rectangle: self.delete_rectangle(event,t))
            else:
                self.cv.delete(self.now_rectangle)
                self.now_rectangle=None
                self.ax=-10
                self.ay=-10
                self.drawing_rectangle_type=0
            
        elif self.drawing_rectangle_type==2:
            if self.ax>0 and self.ay >0 and self.now_rectangle != None:
                self.cv.move(self.now_rectangle,event.x-self.ax,event.y-self.ay)
                self.ax=event.x
                self.ay=event.y
                m=self.rectangle[self.now_rectangle][0:4]
                self.rectangle[self.now_rectangle][0:4]=[m[0]+(event.x-self.mv_prevx)/self.scale,
                                                         m[1]+(event.y-self.mv_prevy)/self.scale,
                                                         m[2]+(event.x-self.mv_prevx)/self.scale,
                                                         m[3]+(event.y-self.mv_prevy)/self.scale]
            self.ax=-10
            self.ay=-10
            self.mv_prevx=-10
            self.mv_prevy=-10
            self.drawing_rectangle_type=0
                                                     
    def change_picture_class(self,class_name):
        if self.filename != None:
            self.picture_class=class_name
            if self.class_text==None:
                self.class_text=self.cv.create_text(0,0,text=self.picture_class,anchor='nw',font=self.picture_class_font,fill='red')
                self.cv.tag_bind(self.class_text,'<Button-3>',self.delete_picture_class)
            else:
                self.cv.itemconfig(self.class_text,text=self.picture_class)
                
    def change_label_class(self,t,label):
        self.rectangle[t][4]=label
        r=self.rectangle[t]
        if r[5] != None:
            self.cv.itemconfig(r[5],text=label)
        else:
            self.rectangle[t][5]=self.cv.create_text([(r[0]-self.offset[0])*self.scale,
                                                      (r[3]-self.offset[1])*self.scale],text=label,anchor="nw",fill='red')
            self.cv.tag_bind(self.rectangle[t][5],'<Button-3>',lambda event, t=t: self.delete_text(event,t))
                
            
    def double_scale(self,event):
        scale=2
        
        self.scale*=scale
        self.offset[0]+=(scale-1)*event.x/self.scale
        self.offset[1]+=(scale-1)*event.y/self.scale
        image=self.pic.crop((int(self.offset[0]),int(self.offset[1]),min(int(self.offset[0]+self.canvas_x/self.scale),self.resolution_x),min(int(self.offset[1]+self.canvas_y/self.scale),self.resolution_y)))
        image=image.resize((int(image.size[0]*self.scale),int(image.size[1]*self.scale)),Image.BILINEAR)
        self.cv.delete(self.cv_image)
        self.cv.scale('all',event.x,event.y,scale,scale)
        self.tkpic = ImageTk.PhotoImage(image=image)
        self.cv_image=self.cv.create_image(0,0,anchor=tk.NW,image=self.tkpic)
        self.cv.lower(self.cv_image)
        
    def return_scale(self,event):
        if self.scale==1:
            return
        self.cv.delete(self.cv_image)
        self.cv.scale('all',self.scale/(self.scale-1)*self.offset[0],self.scale/(self.scale-1)*self.offset[1],1/self.scale,1/self.scale)
        self.offset=[0,0]
        self.scale=1
        self.tkpic = ImageTk.PhotoImage(image=self.pic)
        self.cv_image=self.cv.create_image(0,0,anchor=tk.NW,image=self.tkpic)
#         for keys,values in self.rectangle.items():
#             self.cv.delete(keys)
#             if values[5]!=None:
#                 self.cv.delete(values[5])
#         self.rectangle={}
#         self.load()
        self.cv.lower(self.cv_image)
                            
    def choose_item_handler(self,event,t):
        if self.drawing_rectangle_type==3:
            if t in self.template_rectangle:
                self.template_rectangle.remove(t)
                self.cv.itemconfig(t,outline=self.normal_color)
            else:
                self.template_rectangle.append(t)
                self.cv.itemconfig(t,outline=self.template_color)
#         elif self.drawing_rectangle_type==4:
#             for m in self.template_rectangle:
#                 self.cv.itemconfig(t,outline='black')
#             self.template_rectangle.clear()
#             self.drawing_rectangle=0
        elif self.drawing_rectangle_type==0:
            if self.now_rectangle != None:
                self.cv.itemconfig(self.now_rectangle,outline=self.normal_color)
            self.now_rectangle=t
            self.cv.itemconfig(self.now_rectangle,outline=self.now_rectangle_color)
            self.drawing_rectangle_type=2
            self.mv_prevx=event.x
            self.ax=event.x
            self.mv_prevy=event.y
            self.ay=event.y
    
    
    def alt_l(self,event):
        if self.drawing_rectangle_type == 0:
            self.drawing_rectangle_type=3
            if self.now_rectangle!=None:
                self.cv.itemconfig(self.now_rectangle,outline=self.normal_color)
                self.now_rectangle=None
        elif self.drawing_rectangle_type==3:
            self.drawing_rectangle_type=0
            for i in self.template_rectangle:
                self.cv.itemconfig(i,outline=self.normal_color)
                self.template_rectangle=[]
    
    def a(self,event):
        self.cv.focus_set()
    
    def keyrelease_alt_l(self):
        pass
#         if drawing_rectangle_type==3:
#             drawing_rectangle_type=4
    
root=tk.Tk()
root.title("滔爸爸的标注器")
root.geometry('1100x700')
app=App(root)
#filename=r"F:\jupyter_notebook_my_code\pictures\fgo\picture1.jpg"




root.mainloop()
