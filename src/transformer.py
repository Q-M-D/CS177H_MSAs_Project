import torch
import esm
import re
import os
import numpy as np


def translate(address):
    # if file not exist
    if not os.path.exists(address):
        return []
    f = open(address)
    sequence = []
    nameX = ''
    for line in f:
        if line.startswith('>'):
            nameX = line.replace('>', '').split()[0]
        else:
            dataX = line.strip()
            # remove and delete lowercase letter
            dataX = re.sub('[a-z]', '', dataX)
            sequence.append((nameX, dataX))
    f.close()
    return sequence


def get_info(is_train):
    if is_train:
        f = open('./train_set.txt')
    else:
        f = open('./test_set.txt')
    data = {}
    for line in f:
        if line != '':
            tmp = line.split(' ')
            tmp[1] = float(tmp[1].replace('\n', ''))
            data[tmp[0]] = tmp[1]
    f.close()
    return data



def transform(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model.eval()
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])
    token_representations = results["representations"][12]
    return token_representations

def write_transform_output(item, is_train, output,version):
    if is_train:
        f = open("./train_transform/version" + str(version) + "/"+ item + ".txt", "w")
    else:
        f = open("./test_transform/version" + str(version) + "/"+ item + ".txt", "w")
        # print path
        # print("./test_transform/version" + str(version) + "/"+ item + ".txt")
    f.write(str(output))
    f.close()
    print(item + " succeed")
        

def mul_transform(data, is_train, version=1):
    for item in data:
        if is_train:
            tmp = translate("./train/" + item + ".a3m")
            if os.path.exists("./train_transform/version" + str(version) + "/"+ item + ".txt"):
                print(item + " already exist")
                continue
        else:
            tmp = translate("./test/" + item + ".a3m")
            if os.path.exists("./test_transform/version" + str(version) + "/"+ item + ".txt"):
                print(item + " exist")
                continue
        print(item + " begin:")
        print(np.shape(tmp))
        
        # switch version
        # version 0: use first 256 sequences
        if version == 0:
            tmp = tmp[:256]
            matrix = transform(tmp)
            output = matrix[0][0][0].tolist()
            print(matrix.shape)
            
            write_transform_output(item, is_train, output,version)
            
        # version 1: negelect big sequences
        elif version == 1:
            if np.shape(tmp)[0] < 256:
                matrix = transform(tmp)                
                output = matrix[0][0][0].tolist()
                print(matrix.shape)
                
                write_transform_output(item, is_train, output,version)
            else:
                print("too big")
                continue
        
        # version 2: output by calculating the average float number for each element
        elif version == 2:
            tmp = tmp[:256]
            matrix = transform(tmp)
            output = np.zeros(768)
            print(matrix.shape)
            for i in range(0, len(tmp[0][1])):
                for j in range(0, 768):
                    output[j] += matrix[0][0][i][j]
            for j in range(0, 768):
                output[j] /= len(tmp[0][1])
            output = output.tolist()
            write_transform_output(item, is_train, output, version)
            # os._exit(0)



# data , is_train , version

# version0 use first 256 sequences
# mul_transform(get_info(False), False, 0)
# mul_transform(get_info(True), True, 0)

# version1 remove big sequences
# mul_transform(get_info(1), 1, 1)
# mul_transform(get_info(0), 0, 1)

# version2 take first 256 sequences and calculate average
mul_transform(get_info(0), 0, 2)
mul_transform(get_info(1), 1, 2)


# def hamming_distance_calculate(a, b, length):
#     distance = 0
#     for i in range(length):
#         if a[i] != b[i]:
#             distance += 1
#     return distance

# # apply Diversity Maximizing to choose top 256 sequences from sequence: This is a greedy strategy which starts from the reference and adds the sequence with highest average hamming distance to current set of sequences
# def diversity_maximum(sequence):
#     # get the reference sequence
#     reference = sequence[0]
#     # get the length of reference sequence
#     length = len(reference)
#     # get the number of sequence
#     num = len(sequence)
#     # get the hamming distance matrix
#     hamming_distance = np.zeros((num, num))
#     for i in range(num):
#         for j in range(i, num):
#             hamming_distance[i][j] = hamming_distance[j][i] = hamming_distance_calculate(sequence[i], sequence[j], length)
#     # get the average hamming distance matrix
#     average_hamming_distance = np.zeros((num, num))
#     for i in range(num):
#         for j in range(i, num):
#             average_hamming_distance[i][j] = average_hamming_distance[j][i] = (hamming_distance[i][j] + hamming_distance[j][i]) / 2
#     # get the average hamming distance of reference sequence
#     average_hamming_distance_reference = average_hamming_distance[0]
#     # get the index of sequence which has the highest average hamming distance to reference sequence
#     index = np.argmax(average_hamming_distance_reference)
#     # get the sequence which has the highest average hamming distance to reference sequence
#     new_sequence = sequence[index]
#     # delete the sequence which has the highest average hamming distance to reference sequence
#     del sequence[index]
#     # add the sequence which has the highest average hamming distance to reference sequence
#     sequence.append(new_sequence)
#     return sequence
    
#     # # choose the first sequence
#     # mid_ouput = []
#     # mid_ouput.append(sequence[0])
#     # # choose the sequence with highest average hamming distance
#     # for i in range(1, len(sequence)):
#     #     max_hamming = 0
#     #     max_index = 0
#     #     for j in range(i):
#     #         hamming = 0
#     #         for k in range(len(sequence[i])):
#     #             if sequence[i][k] != sequence[j][k]:
#     #                 hamming += 1
#     #         if hamming > max_hamming:
#     #             max_hamming = hamming
#     #             max_index = j
#     #     mid_ouput.append(sequence[max_index])
#     # return mid_ouput