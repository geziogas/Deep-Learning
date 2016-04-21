# in this file there are some flag-variables which are usefull,
# for testing our code:
# "imType(boolean)" and thresholding(uint8) are used for binarizing our images.
# By setting the imType as "True" and setting thresholding, we get arrays only with 1s and 0s.
# Comment/Uncomment line 166 in order to choose if images should be also saved as .BMP files and
# labels saved as .TXT file.

import os
import glob
import numpy, scipy.misc
from PIL import Image

# Choose for visualization
imType = "normal"
# imType = "bw"
thresholding = 240

class SingleGntImage(object):
    def __init__(self, f):
        self.f = f

    def read_gb_label(self):
        label_gb = self.f.read(2)

        # check garbage label
        if label_gb.encode('hex') is 'ff':
            return True, None
        else:
            label_uft8 = label_gb.decode('gb18030').encode('utf-8')
            return False, label_uft8

    def read_special_hex(self, length):
        num_hex_str = ""
        
        # switch the order of bits
        for i in range(length):
            hex_2b = self.f.read(1)
            num_hex_str = hex_2b + num_hex_str

        return int(num_hex_str.encode('hex'), 16)

    def read_single_image(self):
        
        # zero-one value
        max_value = 255.0
        margin = 4
        
        # try to read next single image
        try:
            self.next_length = self.read_special_hex(4)
        except ValueError:
            print "Notice: end of file"
            return None, None, None, None, True

        # read the chinese utf-8 label
        self.is_garbage, self.label = self.read_gb_label()

        # read image width and height and do assert
        self.width = self.read_special_hex(2)
        self.height = self.read_special_hex(2)
        
        # Debug print of initial character size
        # print self.width," ",self.height

        assert self.next_length == self.width * self.height + 10

        # read image matrix
        image_matrix_list = []
        for i in range(self.height):
            row = []
            for j in range(self.width):                
                if imType == "bw":
                    a = (self.read_special_hex(1)*(-1))+255
                    if a < 255.0-thresholding:
                        row.append(0.0)
                    elif a > 255.0-thresholding:
                        row.append(255.0)
                    else: row.append(a)
                elif imType == "normal":
                    a = (self.read_special_hex(1)*(-1))+255
                    row.append(a)
            image_matrix_list.append(row)

        #print image_matrix_list
        # convert to numpy ndarray with size of 40 * 40 and add margin of 4
        self.image_matrix_numpy = \
            scipy.misc.imresize(numpy.array(image_matrix_list), \
            size=(40, 40)) #/ max_value
        self.image_matrix_numpy = numpy.lib.pad(self.image_matrix_numpy, \
            margin, self.padwithones)
        return self.label, self.image_matrix_numpy, \
            self.width, self.height, False

    def padwithones(self, vector, pad_width, iaxis, kwargs):
        # vector[:pad_width[0]] = 255.0 #Was 1 instead of 1.0
        # vector[-pad_width[1]:] = 255.0
        vector[:pad_width[0]] = 0.0 #Was 1 instead of 1.0
        vector[-pad_width[1]:] = 0.0
        return vector

class ReadGntFile(object):
    def __init__(self):
        pass

    def find_file(self):
        #print 'hello'
        file_extend = ".gnt"
        self.file_list = []


        # get all gnt files in the dir
        try:
            dir_path = os.path.join(os.path.split(__file__)[0])
        except NameError:  # We are the main py2exe script, not a module
            import sys
            dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        # dir_path = os.path.join(os.path.split(__file__)[0], "..", "data")
        # dir_path = "/home/george/Documents/Uppsala Universitet/Thesis/DeepLearning Thesis/Data-Sets/ChineseData/ChineseData/isolated_data"
        print dir_path
        for file_name in sorted(glob.glob(os.path.join(dir_path, 'data-set/*.gnt'))):
            self.file_list.append(file_name)
            print file_name


        return self.file_list

    def show_image(self):
        test_switch = False # Switch to extract few images (10)
        end_of_file = False
        count_file = 0
        count_single = 0
        width_list = []
        height_list = []
        mat3d = []
        label_mat = []

        #open all gnt files
        #print "hello2"
        findex = open('full-labels.txt','w+')
        # print len(self.file_list)
        for file_name in self.file_list:
            count_file = count_file + 1
            end_of_file = False
            with open(file_name, 'rb') as f:
                while not end_of_file:
                    count_single = count_single + 1
                    this_single_image = SingleGntImage(f)

                    # get the pixel matrix of a single image
                    label, pixel_matrix, width, height, end_of_file = \
                        this_single_image.read_single_image()

                    
                    # print pixel_matrix
                    width_list.append(width)
                    height_list.append(height)
                    # Debug purpose print every 10000 characters reading.
                    if (count_single % 10000) == 0:
                        print count_single, label, \
                            width, height, numpy.shape(pixel_matrix)

                    # if count_single==1:
                        # print pixel_matrix[:,38:40]
                    if not end_of_file: 
                        mat3d.append(pixel_matrix)   
                        label_mat.append(label)                    
                        # self.save_image(findex,pixel_matrix, label, count_single)
                        
                    # Debugging switches
                    if test_switch == True:
                        if count_single >= 2:
                           end_of_file = True
                        # a=np.random.randint(255,size=(400,400))


            print ("End of file #%i") % (count_file)
        
        imagesTo3DMatrix = numpy.asarray(mat3d)
        labelsToMatrix = numpy.asarray(label_mat)
        # print "Size of data: %r\n" %str(imagesTo3DMatrix.shape)
        print "size of data: (%r, %r, %r)" % (imagesTo3DMatrix.shape[0],imagesTo3DMatrix.shape[1],imagesTo3DMatrix.shape[2])
        print "size of labels: (%r)" % (labelsToMatrix.shape[0])
        print "Saving image files...\n"
        print "Saving array of images and labels to file...\n"
        numpy.save('imagesToArray',imagesTo3DMatrix)
        numpy.save('labelsToArray',labelsToMatrix)
        # a=np.load('imagesToArray.npy')

        findex.close()

    def save_image(self, findex, matrix, label, count):
        im = Image.fromarray(matrix,mode='L') # 8bit grayscale
        # findex = open('index-labels.txt','w+')
        # findex.write("%i %s\n"%(count,label))
        findex.write("%s\n"%(label))
        # findex.close()
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        name = ("tmp/test-op-%i.bmp") % (count)
        im.save(name)

def display_char_image():
    gnt_file = ReadGntFile()
    file_list = gnt_file.find_file()
    gnt_file.show_image()

if __name__ == '__main__':
    display_char_image()
