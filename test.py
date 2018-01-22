from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip

class load_idx:
    def __init__(self, file_name=None, fstream=None, file_handler=open):
        self.file_name = file_name
        self.fstream = fstream
        self.file_handler = file_handler
        self.magic_number = 0
        self.header_dtype = np.dtype(np.uint32).newbyteorder('>')
        if not (self.file_name is not None) ^ (self.fstream is not None):
            raise ValueError('Define either File Name or File Stream')
        elif self.file_name is not None:
            self.fstream = self.file_handler(self.file_name, 'rb')
    def get_magic_number(self):
        self.magic_number = np.frombuffer(self.fstream.read(4), dtype=self.header_dtype)
        return self.magic_number
    def _extract_header(self):
        mask_dim = int('0x000000ff',16)
        mask_datatype = int('0x0000ff00',16)
        no_of_dimensions = np.bitwise_and(self.magic_number, np.array(mask_dim, dtype=np.uint32))
        datatype_index = np.right_shift(np.bitwise_and(self.magic_number, np.array(mask_datatype, dtype=np.uint32)),8)
        if datatype_index == int('0x08',16):
            dt = np.dtype(np.uint8)
        elif datatype_index == int('0x09',16):
            dt = np.dtype(np.int8)
        elif datatype_index == int('0x0B',16):
            dt = np.dtype(np.int16)
        elif datatype_index == int('0x0C',16):
            dt = np.dtype(np.int32)
        elif datatype_index == int('0x0D',16):
            dt = np.dtype(np.float32)
        elif datatype_index == int('0x0E',16):
            dt = np.dtype(np.float64)
        dimensions = np.empty(no_of_dimensions, dtype=np.uint32)
        for i in range(no_of_dimensions):
            read_val = np.frombuffer(self.fstream.read(4),dtype=self.header_dtype)
            dimensions[i] = read_val
        return dimensions, dt
    def load_file(self):
        if self.magic_number == 0:
            self.get_magic_number()
        [dimensions, dt] = self._extract_header()
        total_bytes_to_be_read = np.prod(dimensions, dtype=np.int32)*dt.itemsize
        data = np.frombuffer(self.fstream.read(total_bytes_to_be_read),dtype=dt)
        data = np.reshape(data,dimensions)
        if self.file_name is not None:
            self.fstream.close()
        return data

class load_mnist(load_idx):
    def __init__(self, file_name, file_type, file_handler=open, convert_to_float = False, display_sample = 0):
        load_idx.__init__(self, file_name = file_name, file_handler=file_handler)
        self.file_type = file_type
        self.convert_to_float = convert_to_float
        self.display_sample = display_sample
        self.mnist_magic_number={'data':2051, 'label':2049}
        if self.file_type == 'label':
            self.display_sample = 0
    def load(self):
        self.get_magic_number()
        if self.mnist_magic_number[self.file_type] == self.magic_number:
            self.data = self.load_file()
            if self.convert_to_float:
                self.data = self.data.astype(np.float32)
                self.data = np.multiply(self.data, 1.0/255.0)
            if self.display_sample != 0:
                self.display_samples(self.display_sample)
            return self.data
        else:
            print('Given file is not mnist : (%s,%s)'%(self.file_name, self.file_type))
    def display_samples(self, how_many=5):
        size = self.data.shape[0]
        perm = np.random.permutation(size)
        perm = perm[:how_many]
        images = self.data[perm,:,:]
        for i in range(how_many):
            fig = plt.figure()
            plt.imshow(images[i], cmap='Greys_r')
    def display_images(self, number):
        if number.shape.__len__() > 1:
            print('Number should be 1D array')
        else:
            for i in number:
                fig = plt.figure()
                plt.imshow(self.data[i], cmap='Greys_r')

training_set_file_name ='train_images_idx3_ubyte_customize.gz'
training_labels_file_name ='train_labels_idx1_ubyte_customize.gz'
testing_set_file_name ='test_images_idx3_ubyte_customize.gz'

train_images_obj=load_mnist(training_set_file_name, 'data', file_handler=gzip.GzipFile, display_sample=0)
train_labels_obj=load_mnist(training_labels_file_name, 'label', file_handler=gzip.GzipFile)
test_images_obj=load_mnist(testing_set_file_name, 'data', file_handler=gzip.GzipFile, display_sample=0)

train_images = train_images_obj.load()
train_labels = train_labels_obj.load()
test_images = test_images_obj.load()
train_images = train_images.reshape(train_images.shape[0],np.prod(train_images.shape[1:]))
test_images = test_images.reshape(test_images.shape[0], np.prod(test_images.shape[1:]))

print(train_images.shape)
print(test_images.shape)
print(train_labels.shape)
train_images2 = train_images[50000:60000, :]
train_images1 = train_images[0:50000, :]

np.random.seed(20)
w2 = np.random.randn(100,784)
w3 = np.random.randn(80,100)
w4 = np.random.randn(10,80)
b2 = np.random.randn(100)
b3 = np.random.randn(80)
b4 = np.random.randn(10)
err2 = np.zeros(100)
err3 = np.zeros(80)
err4 = np.zeros(10)
delw2 = np.zeros((100,784))
delw3 = np.zeros((80,100))
delw4 = np.zeros((10,80))
delb2 = np.zeros(100)
delb3 = np.zeros(80)
delb4 = np.zeros(10)
z2 = np.zeros(100)
z3 = np.zeros(80)
z4 = np.zeros(10)
a2 = np.zeros(100)
a3 = np.zeros(80)
a4 = np.zeros(10)
aracost = np.zeros(60000)



def activfun(Z):
 temp = Z*(-1)
 temp1 = np.exp(temp)
 temp2 = temp1 + 1
 temp3 = temp2**(-1)
 return temp3

def convert(Z):
 temp = np.zeros(10)
 temp[Z] = 1
 return temp


def initialization(train_image):
 global w2
 global w3
 global w4
 global b2
 global b3
 global b4
 global z2
 global z3
 global z4
 global a2
 global a3
 global a4
 z2 = np.dot(w2,((train_image.astype(float))/255)) + b2
 a2 = activfun(z2)
 z3 = np.dot(w3,a2) + b3
 a3 = activfun(z3)
 z4 = np.dot(w4,a3) + b4
 a4 = activfun(z4)
 return


def delbanao(err_n, a_n):
 temp = np.zeros((err_n.shape[0], a_n.shape[0]))
 for bita in range(err_n.shape[0]):
  temp[bita] = np.array(a_n)
 for idx1, bita1 in enumerate(err_n):
  temp[idx1] = temp[idx1]*bita1
 return temp





def findel(train_image,train_label):
 global a2
 global a3
 global a4
 global w4
 global w3
 global w2
 global err2
 global err3
 global err4
 global delw2
 global delw3
 global delw4
 global delb2
 global delb3
 global delb4
 global b2
 global b3
 global b4
 err4 = (a4 - convert(train_label))*a4*(1-a4)
 err3 = (np.dot(w4.T, err4))*a3*(1-a3)
 err2 = (np.dot(w3.T, err3))*a2*(1-a2)
 delb2 = np.array(err2)
 delb3 = np.array(err3)
 delb4 = np.array(err4)
 delw2 = delbanao(err4, a3)
 delw3 = delbanao(err3, a2)
 delw2 = delbanao(err2, (train_image.astype(float))/255)
 w2 = w2 - (0.5*delw2)
 w3 = w3 - (0.5*delw3)
 w4 = w4 - (0.5*delw4)
 b2 = b2 - (0.5*delb2)
 b3 = b3 - (0.5*delb3)
 b4 = b4 - (0.5*delb4)
 return 


def load_pickle(fileName):
 with open(fileName, 'r') as fid:
  data = pickle.load(fid)
 return data




w2 = load_pickle('w2_final.pkl')
w3 = load_pickle('w3_final.pkl')
w4 = load_pickle('w4_final.pkl')
b4 = load_pickle('b4_final.pkl')
b3 = load_pickle('b3_final.pkl')
b2 = load_pickle('b2_final.pkl')
z2 = np.zeros(100)
z3 = np.zeros(100)
z4 = np.zeros(10)
a2 = np.zeros(100)
a3 = np.zeros(100)
a4 = np.zeros(10)

def activfun(Z):
 temp = Z*(-1)
 temp1 = np.exp(temp)
 temp2 = temp1 + 1
 temp3 = temp2**(-1)
 return temp3


def initialization(train_image):
 global w2
 global w3
 global w4
 global b2
 global b3
 global b4
 global z2
 global z3
 global z4
 global a2
 global a3
 global a4
 z2 = np.dot(w2,((train_image.astype(float))/255)) + b2
 a2 = activfun(z2)
 z3 = np.dot(w3,a2) + b3
 a3 = activfun(z3)
 z4 = np.dot(w4,a3) + b4
 a4 = activfun(z4)
 return



trai2= np.zeros(10000)

for idx, bita in enumerate(test_images):
 initialization(bita)
 temp = np.argmax(a4)
 trai2[idx] = temp

counterr = 0
for idx, bita in enumerate(trai2):
 if (bita == train_labels[idx]):
  counterr += 1


np.savetxt("test_predicted_class_label_2014EE30542.csv", trai2, delimiter = ",")




















