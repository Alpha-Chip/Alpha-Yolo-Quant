def max_a(filepath):
    max_a_dict = {}
    with open(filepath, 'r') as f_obj:
        lines = f_obj.readlines()
        for el in lines:
            max_a_dict[el.split(' ')[0][:-1]] = float(el.split(' ')[1].rstrip('\n'))
    return max_a_dict
