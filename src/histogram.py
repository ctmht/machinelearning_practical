import matplotlib.pyplot as plt


def load_labels_as_dict(file_name: str):
    labels: dict = {}
    with open(file_name) as file:
        for label in file:
            label = int(label.strip())
            labels[label] = labels.get(label, 0) + 1
    return labels


def print_dict(dictionary):
    for key, value in sorted(dictionary.items()):
        print(key, value)


def plot_histogram(dictionary):
    bins = []
    frequencies = []
    for key, value in sorted(dictionary.items()):
        bins.append(str(key))
        frequencies.append(value)

    plt.bar(bins, frequencies)
    plt.xlabel('Emoji class')
    plt.ylabel('Frequency')
    plt.title('Emoji Label Frequencies from the Train Dataset')
    plt.xticks(rotation=45)
    plt.show()


labels = load_labels_as_dict("train_labels.txt")
print_dict(labels)
plot_histogram(labels)
