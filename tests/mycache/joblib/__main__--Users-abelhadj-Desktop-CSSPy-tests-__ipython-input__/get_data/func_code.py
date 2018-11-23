# first line: 1
@mem.cache
def get_data():
    data = load_svmlight_file("datasets/vectors.dat")
    return data[0].todense(), data[1]
