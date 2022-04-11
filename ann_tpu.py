import argparse
import time
import numpy as np
from PIL import Image
import matplotlib.image
from numpy import loadtxt

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter



def set_input_tensor(interpreter, input):
  input_details = interpreter.get_input_details()[0]
  tensor_index = input_details['index']
  input_tensor = interpreter.tensor(tensor_index)()
  
  scale, zero_point = input_details['quantization']
  quantized_input = np.uint8(input / scale + zero_point)
  input_tensor[:, :] = quantized_input

def predict_weather(interpreter, input):
  set_input_tensor(interpreter, input)
  start = time.perf_counter()
  interpreter.invoke()
  inference = time.perf_counter()-start
  print('Inference Time:',inference*1000,'ms')
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  return output


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file.')
  parser.add_argument('-i', '--input', required=False,
                      help='Image to be classified.')
  parser.add_argument('-k', '--top_k', type=int, default=1,
                      help='Max number of classification results')
  parser.add_argument('-t', '--threshold', type=float, default=0.0,
                      help='Classification score threshold')
  parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
  args = parser.parse_args()
  

  start_total = time.perf_counter()   
    
  interpreter = make_interpreter(*args.model.split('@'))
  interpreter.allocate_tensors()
  
  dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
  X = dataset[:,0:8]
  y = dataset[:,8]

  #X = []
  #X.append(np.array([6., 148., 72., 35., 0., 33.5, 0.627, 50.]))
  #X.append(np.array([1., 85., 66., 29., 0., 26.6, 0.351, 31.]))                      
  #X.append(np.array([8., 183., 64., 0., 0., 23.3, 0.672, 32.]))
  #X.append(np.array([1., 89., 66., 23., 94., 28.1, 0.167, 21.]))
  #X.append(np.array([0., 137., 40., 35., 168., 43.1, 2.288, 33.]))
  #X.append(np.array([5., 116., 74., 0., 0., 25.6, 0.201, 30.]))
  #X.append(np.array([3., 78., 50., 32., 88., 31, 0.248, 26.]))
  #X.append(np.array([10., 115., 0., 0., 0., 35.3, 0.134, 29.]))
  #X.append(np.array([2., 197., 7., 45., 543., 30.5, 0.158, 53.]))

  #y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
  
  t = 0
  infer = []

  print('\n----INFERENCE TIME----')
  print('Note: The first inference on Edge TPU is slow because it includes',
        'loading the model into Edge TPU memory.\n')
  
  count = 0
  for i in range(args.count): 
    pred = (predict_weather(interpreter, X[i]) > 0.5).astype(int)
    #print(pred, y[i])
    if pred == int(y[i]):
        count +=1
    #print('%.3fms' % (inference_time * 1000))
    #infer.append(inference_time * 1000)
    #t += (inference_time * 1000)


  #summ = sum(infer)-infer[0] 
  #lat = (summ)/(args.count-1)
  #print('\nCount:', (args.count * 1), 'inferences') 
  #print('Duration:','%.3f ms' % (t))  
  #print('Latency:', '%.3f ms' % (t / args.count))  
  #print('Avg Latency:', '%.3f ms' % (lat)) 
  #print('Throughput:', '%.3f FPS' % (1000 * (args.count-1) / summ))

  print('\n-------RESULTS--------\n')
  
  print('Accuracy:',count/768)
 
  #im1 = np.reshape(out[0],(113,150))
  
  inference_time_total = time.perf_counter() - start_total



  
if __name__ == '__main__':
  main()
