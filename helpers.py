import numpy as np
import re


def split_comsol_file(filename):
    data = np.genfromtxt(filename, skip_header=9)
    skipped_lines = []
    with open(filename, "r") as fd:
        header = None
        while line := fd.readline():
            if line[0] != "%":
                break
            skipped_lines.append(line)
            header = line
    match = re.findall(r"d(?P<ax>\d)(?P<ay>\d)(?P<az>\d)=1", header)
    multi_indices_raw = [[int(ai) for ai in m] for m in match]
    multi_indices = []
    for m in multi_indices_raw:
        if m not in multi_indices:
            multi_indices.append(m)
    index = 3
    for multi_index in multi_indices:
        out_filename = f"{filename.split('all.txt')[0]}{multi_index[0]}{multi_index[1]}{multi_index[2]}.txt"
        with open(out_filename, "w") as fd:
            for line in skipped_lines:
                fd.write(line)
        d = np.concatenate((data[:, :3], data[:, index:index+6]), axis=-1)
        with open(out_filename, "a") as fd:
            np.savetxt(fd, d)
        index += 6


if __name__ == "__main__":
    # split_comsol_file("../../git_ignore/moments_inverse/data/anisotropic/all.txt")
    split_comsol_file("data/comsol/eps_2/all.txt")
