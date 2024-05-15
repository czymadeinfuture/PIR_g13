i = 0


liste_robot = ["warthog_0", "warthog_1"]


def transform_dict_to_list(d, robot_list):
    result = []
    for key, value in d.items():
        inner_list = []
        for v in value:
            inner_list.append([float(v[0]), float(v[1]), 10.0])
        result.append([robot_list[key - 1], inner_list])
    return result


d = {
    1: [(6, 9), (2, 15), (8, 15), (8, 2)],
    2: [(13, 9), (14, 9), (14, 13), (16, 18), (17, 7)],
}
print(transform_dict_to_list(d, liste_robot))
