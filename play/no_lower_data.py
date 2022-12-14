import os
import re
# let ./data/* files be the input files
# for lines in each file:
# if line is even line:
# remove and delete the all lower case letters

INPUT_PATH = './data/'
OUTPUT_PATH = './no_lower_data/'

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
            for line in f:
                if line.startswith(">"):
                    f2.write(line)
                else:
                    line = re.sub('[a-z]', '', line)
                    f2.write(line)
            
            f.close()
            f2.close()