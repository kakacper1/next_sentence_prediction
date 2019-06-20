path1 = "./test_50_50_true_swapped.tsv"
path2 = "./test_50_50_true_swapped.tsv"

target_coding = "utf-8"


f= open(path1, 'rb') as f:
content= f.read()
f.close()


f= open('new'+ path1, 'w', encoding=target_coding)
f.write(content)
f.close()

print("done_1")


path1 = "./test_50_50_true_swapped.tsv"
path2 = "./test_50_50_true_random.tsv"


f= open(path2, 'rb') as f:
content= f.read()
f.close()


f= open('new'+ path2, 'w', encoding=target_coding)
f.write(content)
f.close()

print("done_2")




