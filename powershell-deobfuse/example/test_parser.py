from utils.parser import parser

file_list = ["file1",
             "file2",
             "file3",
             "file4",
             "file5",
            ]

for k in range(len(file_list)):
    data = open(f"./{file_list[k]}", "rb").read()
    print(f'file{k}:')
    for i in range(len(parser(data))):
        print(f'block[{i}]', end=' ')
        print(parser(data)[i][:150])
    print()