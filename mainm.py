import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
st.set_page_config(page_title="Face Mask Detection System",page_icon="https://th.bing.com/th/id/OIP.wBxPRinK5gNZrLME7WjLwAHaH0?w=160&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7")
st.title("  FACE MASK DETECTION SYSTEM")
choice=st.sidebar.selectbox("My Menu",("HOME","URL","CAMERA","Feedback"))
if(choice=="HOME"):
    st.image("https://miro.medium.com/max/1140/1*Xu2czX_NgcACvnM4TQruFg.jpeg")    
    st.write("It is a Computer Vision machine learning application which can detect whether a person is wearing mask or not")
elif(choice=="URL"):
    url=st.text_input("Enter the url")
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1  
        btn2=st.button("Stop Detection")
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5",compile=False)
        vid=cv2.VideoCapture(url)
        if btn2:
            vid.release()
            st.experimental_rerun()
        
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                faces=facemodel.detectMultiScale(frame,5)
                for(x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l]
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
                    face_img= np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_img=(face_img/127.5)-1
                    pred=maskmodel.predict(face_img)[0][0]
                    if(pred>0.9):
                        path="data/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+1])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)                        
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")
elif(choice=="CAMERA"):
    cam=st.selectbox("Choose 0 for primary camera and 1 for seconday camera",("None",0,1))
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        i=1  
        btn2=st.button("Stop Detection")
        facemodel=cv2.CascadeClassifier("face.xml")
        maskmodel=load_model("mask.h5",compile=False)
        vid=cv2.VideoCapture(cam)
        if btn2:
            vid.release()
            st.experimental_rerun()
        
        while(vid.isOpened()):
            flag,frame=vid.read()
            if(flag):
                faces=facemodel.detectMultiScale(frame)
                for(x,y,l,w) in faces:
                    face_img=frame[y:y+w,x:x+l]
                    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
                    face_img= np.asarray(face_img, dtype=np.float32).reshape(1, 224, 224, 3)
                    face_img=(face_img/127.5)-1
                    pred=maskmodel.predict(face_img)[0][0]
                    if(pred>0.9):
                        path="data/"+str(i)+".jpg"
                        cv2.imwrite(path,frame[y:y+w,x:x+1])
                        i=i+1
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)                        
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
                window.image(frame,channels="BGR")
elif(choice=="Feedback"):
    st.markdown('<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSfKoxF-E03CCFog8ycsyNiaAkBqRh9Is-HbnhGQbfIl7_aUDw/viewform?embedded=true" width="640" height="1214" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>',unsafe_allow_html=True)
       
