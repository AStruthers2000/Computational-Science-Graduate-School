last_time = -1
with open("sx-stackoverflow.txt", "r") as file:
    for line in file:
        line = line.strip()
        data = line.split(" ")
        time = int(data[2])
        print(data)
        if time >= last_time:
            last_time = time
        else:
            print(f"not sorted, {time} > {last_time}")

