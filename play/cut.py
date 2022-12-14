# random cut 0~230 lines in ./data/* files
# output in ./cut_data/*

import os
import random

INPUT_PATH = './data/'
OUTPUT_PATH = './cut_data/'

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    for filename in os.listdir(INPUT_PATH):
        if os.path.exists(OUTPUT_PATH + filename):
            print(filename + " already exist")
            continue
        if filename.endswith(".a3m"):
            print(filename)
            f = open(INPUT_PATH + filename)
            f2 = open(OUTPUT_PATH + filename, 'w')
            # get a random number in (0, 230)
            random_num = random.randint(0, 230)
            line_count = 0
            for line in f:
                if line_count < 2 * random_num:
                    f2.write(line)
                else:
                    break
                line_count += 1
            f.close()
            f2.close()
