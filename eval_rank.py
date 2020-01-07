import csv

data = []
with open('shuffle_all.csv', 'r', encoding='utf-8') as fr1:
    csv_reader1 = csv.reader(fr1)
    for i in csv_reader1:
        data.append(i)

dic = {}
for i in range(len(data)//10):
    for j in range(10):
        dic[data[i*10][0] + data[i*10][1] + data[i*10+j][2]] = float(data[i*10+j][5])

gamas = [6, 7, 8, 9, 10]
categories = ['hits10', 'hits3', 'hits1', 'mr', 'mrr', 'comet', 'shuffle']
# categories = ['hits10', 'hits3', 'hits1', 'mr', 'mrr', 'comet']
for gama in gamas:
    print('gamas = ' + str(gama))
    for cg in categories:
        s = 0
        fr3 = open('rank_{}.csv'.format(cg), 'w', encoding='utf-8')
        csv_writer = csv.writer(fr3)
        with open('results_{}.txt'.format(cg), 'r', encoding='utf-8') as fr2:
            lines = fr2.readlines()
            for i in range(len(lines)//10):
                if lines[i * 10].split('\t')[0][:-1] + lines[i * 10].split('\t')[1] + lines[i * 10].split('\t')[2][:-1] in dic:
                    tmp = 0
                    t_all = 0
                    for j in range(10):
                        tmp += dic[lines[i*10+j].split('\t')[0][:-1] + lines[i*10+j].split('\t')[1] + lines[i*10+j].split('\t')[2][:-1]]
                        t_all += dic[lines[i*10+j].split('\t')[0][:-1] + lines[i*10+j].split('\t')[1] + lines[i*10+j].split('\t')[2][:-1]] * (j + 1)
                    if tmp <= gama:
                        s += t_all
                        for j in range(10):
                            csv_writer.writerow([lines[i*10+j].split('\t')[0][:-1], lines[i*10+j].split('\t')[1], lines[i*10+j].split('\t')[2][:-1],dic[lines[i*10+j].split('\t')[0][:-1] + lines[i*10+j].split('\t')[1] + lines[i*10+j].split('\t')[2][:-1]]])
        print(cg + ' = ' + str(s))

    f = open('cal_comet.txt', 'r', encoding="utf8")
    all = 0
    list_tmp = []
    for num, line in enumerate(f.readlines()):
        k = float(line)
        list_tmp.append(k)
        if num%10 == 9:
            tmp = 0
            t_all = 0
            list_tmp.sort(reverse=True)
            for id, i in enumerate(list_tmp):
                tmp += i
                t_all += i * (id%10 +1)
            if tmp <=gama:
                all += t_all
            list_tmp.clear()
    print('best: ' + str(all))