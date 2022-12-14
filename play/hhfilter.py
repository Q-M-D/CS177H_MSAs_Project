import os

# input_path = "./test"
# output_path = "./test_hhfilter"
# input_path = "./data_final/"
# output_path = "./data_final_filter"
input_path = "./data_final/"
output_path = "./data_final_filter"
diff = "256"
verbose = "1"


if __name__ == "__main__":
    for filename in os.listdir(input_path):
        # if file already exist, skip
        if os.path.exists(output_path + "/" + filename):
            print(filename + " already exist")
            continue
        if filename.endswith(".a3m"):
            print(filename)
            os.system("hhfilter -i " + input_path + "/" + filename + " -o " + output_path + "/" + filename + " -diff " + diff + " -v " + verbose)
