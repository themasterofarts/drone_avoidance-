#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# import tensorflow as tf
import sys
# import tensorflow.com
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow.compat.v1 as tf
import os
import argparse
import time
import datetime
from utils import *
from pydnet import *
from stack import stackImages
from display_img import display
from PIL import Image

# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=1, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')

args = parser.parse_args()

def main(_):

  with tf.Graph().as_default():
    height = args.height
    width = args.width
    placeholders = {'im0':tf.compat.v1.placeholder(tf.float32,[None, None, None, 3], name='im0')}

    with tf.compat.v1.variable_scope("model") as scope:
      model = pydnet(placeholders)

    init = tf.group(tf.compat.v1.global_variables_initializer(),
                   tf.compat.v1.local_variables_initializer())

    loader = tf.train.Saver()
    saver = tf.train.Saver()
    cam = cv2.VideoCapture(0)

    with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, args.checkpoint_dir)
        while True:
          for i in range(4):
            cam.grab()
          ret_val, img = cam.read() 
          img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
          im=img.copy()
          img = np.expand_dims(img, 0)
          start = time.time()
          disp = sess.run(model.results[args.resolution-1], feed_dict={placeholders['im0']: img})
          end = time.time()

          disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')
          # print(type(disp_color.shape))
          toShow = (np.concatenate((img[0], disp_color), 0)*255.).astype(np.uint8)
          toShow = cv2.resize(toShow, (int(width/2), height))

          disp_color = np.uint8(disp_color * 255.)
          print(disp_color.shape)


          w, h, _ = toShow.shape
          print(toShow.shape)

          roi = disp_color[0:int(9 * disp_color.shape[0] / 10), int(disp_color.shape[1] / 10):int(9 * disp_color.shape[1] / 10)]


          h_minf, h_maxf, s_minf, s_maxf, v_minf, v_maxf= 0, 112, 29, 255, 202, 255

          h_minc, h_maxc, s_minc, s_maxc, v_minc, v_maxc = 89, 179, 82, 255, 65, 255

          imgHSV = cv2.cvtColor(disp_color, cv2.COLOR_BGR2HSV)

          free_min = np.array([h_minf, s_minf, v_minf])
          free_max = np.array([h_maxf, s_maxf, v_maxf])
          maskf = cv2.inRange(imgHSV, free_min, free_max)
          # free = cv2.bitwise_and(disp_color, disp_color, img, mask=mask)
          free = cv2.bitwise_and(disp_color, disp_color, mask=maskf)

          col_min = np.array([h_minc, s_minc, v_minc])
          col_max = np.array([h_maxc, s_maxc, v_maxc])
          maskc = cv2.inRange(imgHSV, col_min, col_max)
          col = cv2.bitwise_and(disp_color, disp_color, mask=maskc)
          w,  h,_ =disp_color.shape



          x = int(w / 3)
          y = 2 * int(w / 3)
          u = int(h / 3)
          v = 2 * int(h / 3)

          s1 = disp_color[0:h, 0:x]
          s2 = disp_color[u:v, x:y]
          s3 = disp_color[0:h, y:w]
          s4 = disp_color[v:h, x:y]
          s5 = disp_color[0:u, x:y]

          s1 = np.array(s1)
          s2 = np.array(s2)
          s3 = np.array(s3)
          s4 = np.array(s4)
          s5 = np.array(s5)

          c1 = np.sum(s1 == 255)
          c2 = np.sum(s2 == 255)
          c3 = np.sum(s3 == 255)
          c4 = np.sum(s4 == 255)
          c5 = np.sum(s5 == 255)

          a1 = h * x
          a2 = (v - u) * (y - x)
          a3 = h * (w - y)
          a4 = (y - x) * (h - y)
          a5 = (y - x) * u
          # print((c1, c2, c3, c4, c5))
          # print(w,h)
          display(disp_color, h, w)
          cv2.imshow('pydnet', toShow)
          cv2.imshow("roi",roi)
          cv2.imshow('py-net', disp_color)
          cv2.imshow('pydnetfree', free)
          cv2.imshow('pydnetcol', col)
          k = cv2.waitKey(1)         
          if k == 1048603 or k == 27: 
            break  # esc to quit
          if k == 1048688:
            cv2.waitKey(0) # 'p' to pause

          print("Time: " + str(end - start))
          del img
          del disp
          del toShow
          
        cam.release()        

if __name__ == '__main__':
    # tf.app.run()
    tf.compat.v1.app.run()
