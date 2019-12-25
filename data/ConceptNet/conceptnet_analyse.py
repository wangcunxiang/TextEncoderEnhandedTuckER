import csv

ftst  = open('dev1.txt', 'r')
ftrn  = open('train100k.txt', 'r')

tst_data = ()
trn_count = {}


for line in ftst.readlines():
    line = line.split('\t')[1:2]
    for item in line:
        tst_data = tst_data + (item, )

for line in ftrn.readlines():
    line = line.split('\t')[1:2]
    for item in line:
        if item in trn_count:
            trn_count[item] = trn_count[item] + 1
        else:
            print(item)
            trn_count[item] = 1

empty_num = 0
overall_num = 0

for item in tst_data:
    if item in trn_count:
        overall_num += trn_count[item]
    else:
        empty_num += 1

print(empty_num, len(tst_data), overall_num)