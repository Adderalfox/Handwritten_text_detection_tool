def load_emnist_mapping(path="./data/EMNIST/mapping/emnist-byclass-mapping.txt"):
    mapping = {}
    with open(path, "r") as f:
        for line in f:
            label_idx, unicode_val = line.strip().split()
            mapping[int(label_idx)] = chr(int(unicode_val))
    return mapping