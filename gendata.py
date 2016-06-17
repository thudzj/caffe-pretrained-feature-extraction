import os
import random

out = open("data_list", "w")
dirs = os.listdir('101_ObjectCategories')  
label = 0
for d in dirs:  
    print d
    files = os.listdir('101_ObjectCategories' + os.sep + d)
    for f in files:
        out.write('101_ObjectCategories' + os.sep + d + os.sep + f + " " + str(label) + "\n")
    label += 1
    
out.close()

fin = open("data_list")
lines = fin.readlines()
fin.close()

random.shuffle(lines)
test_num = int(len(lines) * 0.2)
out = open("test.txt","w")
for i in range(test_num):
    out.write(lines[i])
out.close()

out = open("train.txt","w")
for i in range(test_num, len(lines)):
    out.write(lines[i])
out.close()
