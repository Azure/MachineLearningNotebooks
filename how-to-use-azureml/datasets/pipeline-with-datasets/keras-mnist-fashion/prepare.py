import os


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    l = open(labelf, "rb")
    o = open(outf, "w")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28 * 28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image) + "\n")
    f.close()
    o.close()
    l.close()


mounted_input_path = os.environ['fashion_ds']
mounted_output_path = os.environ['AZUREML_DATAREFERENCE_prepared_fashion_ds']
os.makedirs(mounted_output_path, exist_ok=True)

convert(os.path.join(mounted_input_path, 'train-images-idx3-ubyte'),
        os.path.join(mounted_input_path, 'train-labels-idx1-ubyte'),
        os.path.join(mounted_output_path, 'mnist_train.csv'), 60000)
convert(os.path.join(mounted_input_path, 't10k-images-idx3-ubyte'),
        os.path.join(mounted_input_path, 't10k-labels-idx1-ubyte'),
        os.path.join(mounted_output_path, 'mnist_test.csv'), 10000)
