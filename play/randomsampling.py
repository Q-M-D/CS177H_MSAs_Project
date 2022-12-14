# read in Covid.asm files and randomly sample it into SIZE a3m files
# randomly sample SIZE0 sequences from each file
import os
import random

TIMES = 50
SIZE = 100

INPUT_PATH = './covid.a3m'
OUTPUT_PATH = './data_order/'

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    f = open(INPUT_PATH)
    # make a 1*10 list
    # 10 is the number of files
    old_data = []
    data = [[] for i in range(TIMES)]
    line_count = 0
    data_line = False
    is_first_line = 2
    
    for line in f:
        old_data.append(line)
    
    for i in range(TIMES):
        data[i].append(old_data[0])
        data[i].append(old_data[1])
    
    # delete the first two lines for old_data
    del old_data[0]
    del old_data[0]
    
    # get a random number list without repetition
    # random_num = [[] for i in range(TIMES)]
    # for i in range(TIMES):
    #     random_num[i] = random.sample(range(0, int(len(old_data)/2)), SIZE)
    #     random_num[i].sort()
        # print(random_num[i])
        
    # get the data
    # for i in range(TIMES):
    #     for j in range(SIZE):
    #         data[i].append(old_data[random_num[i][j]*2])
    #         data[i].append(old_data[random_num[i][j]*2+1])
    
    index = 0
    for i in range(TIMES):
        for j in range(SIZE):
            # print(index)
            data[i].append(old_data[index])
            data[i].append(old_data[index+1])
            index += 2
    
    
    # write the data into files
    for i in range(TIMES):
        f = open(OUTPUT_PATH + str(i) + '.a3m', 'w')
        for j in range(len(data[i])):
            f.write(data[i][j])
        f.close()
    