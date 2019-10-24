from queue import Queue 
from threading import Thread
import threading
import time
import queue
from queue import Empty
import cv2
import numpy as np
  
# Object that signals shutdown 
_sentinel = object()



class FixedDropout(keras.layers.Dropout):
   def _get_noise_shape(self, inputs):
      if self.noise_shape is None:
         return self.noise_shape

      symbolic_shape = keras.backend.shape(inputs)
      noise_shape = [symbolic_shape[axis] if shape is None else shape
                    for axis, shape in enumerate(self.noise_shape)]
      return tuple(noise_shape)

def consumer2(in_q, out_q):
    while True:
        try:

            items = []
            while True:
                try:
                    items.append(in_q.get_nowait())
                except Empty as e:
                    break


            #print("========q2========")
            #print(len(items))
            #items = queue_get_all(in_q)
            for d in items:
                if d is _sentinel:
                    print("===========SENTINEL C2=============")
                    out_q.put(_sentinel)
                    return True
                    
                # Produce some data 
                out_q.put(d)
            time.sleep(0.1)
        except queue.Empty:
            break
            
    #print("============================================")
    #print(in_q.qsize(), out_q.qsize())
  
    # Put the sentinel on the queue to indicate completion 
    #out_q.put(_sentinel) 
    return True


class PreProcessing(object):
   def __init__(self):
      d_h, d_w = 1280, 720
      self.split_h, self.split_w = 940, 960
      self.h, self.w = 128, 128

   def process(self, preprocess_queue, prediction_queue):
      while True:
         try:
            #frame = preprocess_queue.get_nowait()
            frame = preprocess_queue.get()
            if frame is None:
               #time.sleep(0.5)
               #continue
               #preprocess_queue.put(None)
               print("pre frame is empty")
               prediction_queue.put(None)
               break

            orig = cv2.resize(frame, (self.split_h, self.split_w))
            orig_blur = np.divide(orig, 255, dtype=np.float32)
            orig_blur = cv2.blur(orig_blur,(int(self.split_h/16), int(self.split_w/16)),0)
            orig_float = np.divide(orig, 255, dtype=np.float32)


            image = cv2.resize(orig, (self.h, self.w), interpolation = cv2.INTER_AREA)
            image = image[..., ::-1] # switch BGR to RGB
            image = np.divide(image, 255, dtype=np.float32)
            image = image[np.newaxis, ...]

            print("preprocessing frame")
            prediction_queue.put((image, orig_float))
            print("pre prediction_queue size ")
            print(prediction_queue.qsize())
 
         except queue.Empty:
            break
         else:
            prediction_queue.put((image, orig_float))

      print('Exiting process {k}'.format(k=multiprocessing.current_process().ident))

      return True



class MaskPrediction(object):
   def __init__(self, model_params):
      #load the model from graph & setup the weights
      with open(model_params["graph_path"],'r') as f:
          model_json = json.load(f)

      self.model = model_from_json(model_json, custom_objects = {"swish": tf.nn.swish, "FixedDropout": FixedDropout})
      self.model.load_weights(model_params["weight_path"])

   def predict(self, prediction_queue, postprocess_queue):
      while True:
         #try:
         #payload = prediction_queue.get_nowait()
         print("predict prediction_queue size ")
         print(prediction_queue.qsize())
         payload = prediction_queue.get()
         if payload is None:
            #time.sleep(1)
            #continue
            break


         image, orig_float = payload[0], payload[1]
         print("================================ predicting frame===========================")

         pr_mask = self.model.predict(image)
         postprocess_queue.put((image, pr_mask, orig_float))
         #except queue.Empty:
         #   print("Empty predict queue")
         #   break
         #else:
         #   postprocess_queue.put((image, pr_mask, orig_float))

      print('Exiting process {k}'.format(k=multiprocessing.current_process().ident))
      return True


class PostProcessing(object):
   def __init__(self):
      d_h, d_w = 1280, 720
      self.split_h, self.split_w = 940, 960
      self.h, self.w = 128, 128

   def process(self, postprocess_queue, preprocess_queue):
      while True:
         try:
            #val = postprocess_queue.get_nowait()
            val = postprocess_queue.get()
            if val is None:
               #time.sleep(1)
               #continue
               break
            print("======================== post process frame =============================")
            (image, pr_mask, orig_float) = val[0], val[1], val[2]
            mask = pr_mask[..., 0].squeeze()

            mask_dst = cv2.resize(mask, (self.split_h, self.split_w), interpolation = cv2.INTER_CUBIC)
            mask_dst = cv2.blur(mask_dst,(15, 15),0) 
            new_image = np.multiply(orig_float, mask_dst[:,:,None], dtype=np.float32)


            new_image[:, :, 0] = np.where(mask_dst > 0.5,  new_image[:, :, 0], orig_blur[:, :, 0])
            new_image[:, :, 1] = np.where(mask_dst > 0.5,  new_image[:, :, 1], orig_blur[:, :, 0])
            new_image[:, :, 2] = np.where(mask_dst > 0.5,  new_image[:, :, 2], orig_blur[:, :, 0])
            color_and_mask = np.concatenate((orig_float, new_image), axis=1)


            cv2.imshow("Frame", color_and_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               #cv2.destroyAllWindows()
               print('display windows')
               
               preprocess_queue.put(None)
               postprocess_queue.put(None)


         except queue.Empty:
            print("Empty post process queue")
            break

      print('Exiting process {k}'.format(k=multiprocessing.current_process().ident))

      return True


class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.frame_buffer = Queue()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            self.frame_buffer.put(self.frame)
 
    def read(self, out_q):
        # return the frame most recently read
        #return self.frame
        items = []
        while True:
            try:
                items.append(self.frame_buffer.get_nowait())
            except Empty as e:
                break


        print("========WEB========")
        print(len(items))
            #items = queue_get_all(in_q)
        for d in items:
            if d is _sentinel:
                print("===========SENTINEL INCOMING=============")
                out_q.put(_sentinel)
            else:
                out_q.put(d)
        return True
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


# A thread that produces data 
def video_stream(stream, out_q, total_frames=1000):
    k=0
    while k<total_frames:
        (grabbed, frame) = stream.read()
        out_q.put(frame)
        k+=1
        #print(k)
    stream.release()
    print("=====================VIDEO===========")
    out_q.put(_sentinel)
    return True


def queue_get_all(q):
    items = []
    while 1:
        try:
            items.append(q.get_nowait())
        except Empty as e:
            break
    return items

# A thread that produces data 
def consumer1(in_q, out_q):
    while True:
        try:

            items = []
            while True:
                try:
                    items.append(in_q.get_nowait())
                except Empty as e:
                    break

            #items = queue_get_all(in_q)
            #print("========q1========")
            #print(len(items))
            for d in items:
                if d is _sentinel:
                    print("===========SENTINEL C1=============")
                    out_q.put(_sentinel)
                    return True
                    
                # Produce some data 
                out_q.put(d)
            time.sleep(0.05)
        except queue.Empty:
            break
            
    #print("============================================")
    #print(in_q.qsize(), out_q.qsize())
  
    # Put the sentinel on the queue to indicate completion 
    #out_q.put(_sentinel) 
    return True
  
# A thread that consumes data 
def consumer2(in_q, out_q):
    while True:
        try:

            items = []
            while True:
                try:
                    items.append(in_q.get_nowait())
                except Empty as e:
                    break


            #print("========q2========")
            #print(len(items))
            #items = queue_get_all(in_q)
            for d in items:
                if d is _sentinel:
                    print("===========SENTINEL C2=============")
                    out_q.put(_sentinel)
                    return True
                    
                # Produce some data 
                out_q.put(d)
            time.sleep(0.1)
        except queue.Empty:
            break
            
    #print("============================================")
    #print(in_q.qsize(), out_q.qsize())
  
    # Put the sentinel on the queue to indicate completion 
    #out_q.put(_sentinel) 
    return True
            

# A thread that consumes data 
def consumer3(in_q):
    t = time.time()
    while True:
        try:
            items = []
            while True:
                try:
                    items.append(in_q.get_nowait())
                except Empty as e:
                    break

            #items = queue_get_all(in_q)
            #print("========q3========")
            #print(len(items))
            for d in items:
                if d is _sentinel:
                    print("===========SENTINEL C3=============")
                    print("C3 done")
                    print(time.time() - t)
                    return True
                else:
                    #print('Show frame')
                    cv2.imshow("Frame", d)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                       cv2.destroyAllWindows()
                       print('display windows')
                       return True
                    #print(d)

            time.sleep(0.1)
        except queue.Empty:
            break

  
    # Put the sentinel on the queue to indicate completion 
    #out_q.put(_sentinel)

    return True

           
def main():
    time0 = time.time()
    #q1 = Queue() 
    preprocess_queue = Queue()
    prediction_queue = Queue()
    postprocess_queue = Queue()

    
    t0 = Thread(target = video_stream, args =(stream,q2, )).start()
    #t1 = Thread(target = consumer1, args =(q1,q2, )) 
    t2 = Thread(target = consumer2, args =(q2,q3, )) 
    t3 = Thread(target = consumer3, args =(q3, )) 
    
    #t0.start()
    #t1.start() 
    t2.start() 
    t3.start()
    print("=====================VIDEO===========")

    '''
    
    total_frames = 100
    stream = cv2.VideoCapture(0)
    #stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #stream.set(cv2.CAP_PROP_FPS, 30)
    #stream.set(cv2.CAP_PROP_CONVERT_RGB, False)
    stream.set(cv2.CAP_PROP_BUFFERSIZE, 4)
    k=0
    while k<total_frames:
        #print(k)
        (grabbed, frame) = stream.read()
        #frame = stream.grab()
        q2.put(frame)
        #q2.put(k)
        k+=1
        #frame = cv2.resize(frame, (480,640))
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF
        #print(k)
    stream.release()
    print("=====================VIDEO===========")
    '''
    q2.put(_sentinel)
    print(time.time() - time0)
    return True


    '''
    s=0
    while s<30:
        q1.put(s)
        s+=1
    
    s=0
    vs = WebcamVideoStream(src=0).start()
    #cam = cv2.VideoCapture(0)

    while s<100:
        s+=1
        #flag, frame= cam.read()
        vs.read(q1)

        #q1.put(frame)
        #cv2.imshow("Frame", frame)
        #key = cv2.waitKey(1) & 0xFF
    '''

      
    #q1.put(_sentinel)


    #cam.release()
    
    
    #vs.stop()



    #==========================================================










    cv2.destroyAllWindows()
    #print(q1.qsize(), q2.qsize(), q3.qsize())
    # Wait for all produced items to be consumed 
    #q1.close()
    #q2.close()
    #q3.close()
    return True

if __name__=='__main__':
    #t = time.time()
    main()
    #print(time.time() - t)