import csv
import os
import matplotlib.pyplot as plt


def get_file_data(file_path):
    with open(file_path) as f:
        csvv = csv.reader(f, delimiter="\t")
        value = list(csvv)[0][0]
        if value == "NA":
            return None
        return float(value)


def get_x_y(options_dict, all_options):
    dict2 = {}
    for u in options_dict:
        dict2[u.replace(".tsv", "")] = options_dict[u]
    x = list(range(len(all_options)))
    xx = list(range(len(all_options)))
    for j in xx:
        opt = all_options[j].replace(".tsv", "")
        if opt not in dict2:
            x.remove(j)
    # print(x)
    y = [dict2[all_options[t]] for t in x]
    return x, y


files = os.listdir(".")
names_result = {}

for f in files:
    value = get_file_data(f)
    try:
        option, name, mix, ref = f.split("_")
    except Exception:
        print(f)
    final_option = option + "_" + ref
    if "$" in name:
        splited = name.split("$")
        if "=" not in name:
            name = "$".join([splited[0], splited[1], splited[3]])
        else:
            name = "$".join([splited[0], splited[1], splited[2], splited[4]])
    if name not in names_result:
        names_result[name] = {}
    names_result[name][final_option] = value

all_options = []
refs = ["LM22-Full.tsv", "SkinSignaturesV1.tsv"]
options = ["500", "True", "False"]
for x in options:
    for y in refs:
        all_options.append(x + "_" + y)
all_options = [u.replace(".tsv", "") for u in all_options]

colors = ["b", "g", "r", "c", "m", "y", "k", "silver", "lime", "blueviolet", "magenta", "pink"]
fig, ax = plt.subplots()
ii = 0
for g in names_result:
    g_val = names_result[g]
    x, y = get_x_y(g_val, all_options)
    ax.scatter(x, y, c=colors[ii], label=g, s=100)
    ii += 1

ax.set_title("PBMC2 - rmse error")
ax.set_xticklabels([0] + all_options, rotation="vertical")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

remove = []
for key in names_result:
    if "$" in key:
        if key.split("$")[1] == "Supervised":
            remove.append(key)
for j in remove:
    names_result.pop(j)

if __name__ == "__main__":
    get_x_y(
        {"rmse_LM22-Full.tsv.tsv.tsv": 0.0384962345594855},
        {
            "cibersortGEDIT": {"rmse_LM22-Full.tsv.tsv.tsv": 0.0428992624541942},
            "Dnmf$SupervisedDnmf": {"rmse_LM22-Full.tsv": 0.008183103054761887},
            "nnlsGEDIT": {"rmse_LM22-Full.tsv.tsv.tsv": 0.0384962345594855},
        },
    )
