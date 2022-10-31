import os

def load_dataset(file_name: str):
    dataset = []
    with open(file_name, "r") as read_handler:
        line = 1 
        while line:
            line = read_handler.readline()
            if len(line) == 0:
                continue
            if line[0] == "#":
                continue
            else:
                line = line.strip(" ")
                line = line.strip("\n")
                try:
                    data = []
                    for x in line.split(" "):
                        if len(x.strip(" ")) > 0:
                            data.append(float(x.strip(" ")))
                    if len(data) > 0:
                        dataset.append(data)
                except:
                    if len(line) == 0:
                        continue
                    else:
                        print(f"convert data:{line} failed in file:{file_name}")
    return dataset
