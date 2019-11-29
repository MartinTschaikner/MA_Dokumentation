"""
This script converts fda respectively octbin files into img files for interpolation method
"""
import os
from os import listdir
from os.path import isfile, join
from random import randrange
from shutil import copyfile
import shutil

file_root = os.path.dirname(os.path.realpath(__file__))
tempDir = 'img_temp_' + str(randrange(1000))
os.mkdir(tempDir)
file_path_fda = file_root + r'\Topcon_octbin_compare'
save_path_img = file_root + r'\img'
save_temp = file_root + '/' + tempDir

commandline = r'C:\Users\Martin\Desktop\Masterarbeit\octmarker64\convert_oct_data.exe --octpath=' \
              + file_path_fda + ' --outputPath=' + save_temp + ' -f img'

print(commandline)

os.system(commandline)

temp = [f for f in listdir(save_temp) if isfile(join(save_temp, f))]

for i in range(len(temp)):
    temp_file = temp[i]
    source = save_temp + '/' + temp[i]
    new_file_name = 'Topcon-' + temp[i].split('Unknown')[-2] + 'Optic Disc' + temp[i].split('Unknown')[-1]
    destination = save_path_img + '/' + new_file_name
    copyfile(source, destination)

shutil.rmtree(save_temp)
