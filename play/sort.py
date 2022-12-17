import os

INPUT_PATH = './pred_out'
OUTPUT_PATH = './pred_out_sorted'

if __name__ == "__main__":
    f = open(INPUT_PATH)
    data = []
    for line in f:
        # remove '\n'
        line = line.strip('\n').split(' ')
        data.append(line)
    # sort by the second column
    data.sort(key=lambda x: x[1])
    # write the data into files
    f = open(OUTPUT_PATH, 'w')
    for i in range(len(data)):
        f.write(data[i][0] + ' ' + data[i][1] + '\n')
    f.close()
    