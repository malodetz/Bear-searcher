import os

fout = open("names.txt", "w")
names = set()
for name in os.listdir("out"):
    names.add(name[:-7])
for name in names:
    print(name, file=fout)
