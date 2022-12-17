import os

# path表示路径
path="./data/"
# 返回path下所有文件构成的一个list列表
filelist=os.listdir(path)
# 遍历输出每一个文件的名字和类型
count = 0
for item in filelist:
    # 输出指定后缀类型的文件
    # if(item.endswith('.jpg')):
        print(item)
        count += 1
print(filelist)
print(type(filelist))
print(count)