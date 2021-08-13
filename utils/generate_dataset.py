import glob as gb



window_size = 16
for i in range(36):
    dataset_path = '/home/sensifai/behnam/anomaly/ped1/test/{:03d}*.jpg'.format(i+1)
    train = gb.glob(dataset_path)
    train.sort()
    print(len(train))
    print(train[:16])


    for j in range(len(train)-window_size):
        file_path = 'dataset/16frames/test/{:03d}_{:04d}.txt'.format(i, j)
        with open(file_path, 'w') as txt_file:
            frames = train[i:i+window_size]
            for f in frames:
            #     print(dir(txt_file))
                txt_file.write(f+'\n')
