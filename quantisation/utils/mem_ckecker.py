import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch


# Память с пропусками
def create_empty_memory(columns):
    memory = torch.full((1, 16, 400, 448), float('nan'))  # 512000 (1, 16, 320, 400)   (1, 16, 336, 400) 537600    (1, 16, 400, 448)
    memory_size = memory.shape[0] * memory.shape[1] * memory.shape[2] * memory.shape[3]
    memory_bit = memory_size * columns / 1024**2
    memory = torch.reshape(memory, (int(memory_size / columns), columns))
    return memory, memory_bit


def write_memory(read_tens, write_tens, r_ind, w_ind, name, two_places=False):
    input_size_r = read_tens.shape[1]
    if not two_places:
        row = f'{name}, r: {r_ind}, w: {w_ind}\n'
    else:
        row = f'{name}, r: {r_ind}, s: {input_size_r}, w: {w_ind}\n'
    with open('./results/memory.txt', 'a') as f_obj:
        f_obj.write(row)


# Создаю пустую память
columns = 8
memory, memory_bit = create_empty_memory(columns)
memory_names = create_empty_memory(columns)[0].tolist()
total_memory_max = []
try:
    os.mkdir('memory')
except FileExistsError:
    pass
# -------------------------------------------------------


def fill_list_with_names(min_row, max_row, mass, name):
    for row in range(min_row, max_row):
        for i in range(columns):
            mass[row][i] = name
    return mass


# Проходим по пустым строкам и смотрим размер
def fit_or_not(unique_ind, input_tensor, place=None):
    # Поиск свободных строк
    empty_memories = {}
    count_empty_memories = 0
    for ind in range(len(unique_ind) - 1):
        if count_empty_memories not in empty_memories.keys():
            empty_memories[count_empty_memories] = []
        if unique_ind[ind+1] - unique_ind[ind] == 1:
            empty_memories[count_empty_memories].append(unique_ind[ind].item())
            if unique_ind[ind+1] == unique_ind[-1]:
                empty_memories[count_empty_memories].append(unique_ind[ind+1].item())
        else:
            empty_memories[count_empty_memories].append(unique_ind[ind].item())
            count_empty_memories += 1

    """
    Проходим по всем пустым блокам памяти, если нашли место, куда влезет тензор - возвращаем индексы, в которые
    можем поместить наш тензор
    """
    fit_empty_memories = {}
    for key, value in empty_memories.items():
        if input_tensor.shape[0] <= len(value):
            fit_empty_memories[key] = value

    for key, value in fit_empty_memories.items():
        if place is None:
            return value[:input_tensor.shape[0]]
        elif key == place:
            return value[:input_tensor.shape[0]]
        elif place == -1:
            value = fit_empty_memories[list(fit_empty_memories.keys())[-1]]
            return value[len(value) - input_tensor.shape[0]:]
        else:
            print(key)
            print(f'НЕТ СВОБОДНОГО МЕСТА: NEED: {input_tensor.shape[0]}, EMPTY: {len(value)}')


def mem_count():
    global memory
    global total_memory_max
    non_nan_count = torch.sum(~torch.isnan(memory)).item() / columns
    total_memory_max.append(non_nan_count)


def append_memory_max():
    global total_memory_max
    with open('./results/final_memory.txt', 'a') as f_obj:
        f_obj.write(f'MAX_MEMORY: {max(total_memory_max)}')


def mem_put(input_tensor, name, place=None):
    global memory
    global memory_names
    input_size = input_tensor.shape
    input_size = input_size[0] * input_size[1] * input_size[2] * input_size[3]
    input_memory = torch.reshape(input_tensor, (int(input_size / columns), columns))


    # Создаем маску для NaN
    mask = torch.isnan(memory)

    # Находим индексы NaN
    indices = torch.nonzero(mask)
    rows = indices[:, 0]

    empty_memory_place = fit_or_not(torch.unique(rows), input_memory, place)
    min_row = min(empty_memory_place)
    max_row = max(empty_memory_place) + 1

    memory[min_row:max_row, :] = input_memory
    memory_names = fill_list_with_names(min_row, max_row, memory_names, name)
    mem_count()


def mem_clean(name, new_name=None, replace=False):
    global memory
    global memory_names

    rows = []
    if replace:
        for index, row in enumerate(memory_names):
            if name in row:
                rows.append(index)
                memory_names[index] = [new_name for i in range(columns)]
        # min_row = min(rows)
        # max_row = max(rows) + 1
        # memory[min_row:max_row, :] = float('nan')
    else:
        for index, row in enumerate(memory_names):
            if name in row:
                rows.append(index)
                memory_names[index] = [float('nan') for i in range(columns)]
        min_row = min(rows)
        max_row = max(rows) + 1
        memory[min_row:max_row, :] = float('nan')


def bottle_clean(bottle_name):
    mem_clean('x1')
    mem_clean('x2')
    mem_clean(bottle_name)


def x1x2_transform(name):
    global memory
    global memory_names

    rows = []
    for index, row in enumerate(memory_names):
        if name in row:
            rows.append(index)
            memory_names[index] = [float('nan') for i in range(columns)]

    # print(len(rows), len(rows[:len(rows) // 2]), len(rows[len(rows) // 2:]))
    x_devide = len(rows[:len(rows) // 2])

    memory_names = fill_list_with_names(rows[0], rows[0] + x_devide+1, memory_names, 'x1')
    memory_names = fill_list_with_names(rows[0] + x_devide, rows[-1]+1, memory_names, 'x2')


def plot_memory(read_name, write_name):
    global memory
    global memory_bit
    memory_copy = (~torch.isnan(memory)).int()
    sns.heatmap(memory_copy)
    plt.title(f'MEM: {memory_bit} | READ: {read_name} | WRITE: {write_name}')
    plt.savefig(f'./memory/{write_name}')
    plt.clf()


def read_write(read_tens, write_tens, read_name, write_name, conv_type=None, two_places=True, place=None):
    global memory
    global memory_names

    # plot_memory(read_name, write_name)
    if conv_type == '3x3':
        for read_ind, name in enumerate(memory_names):
            if read_name in name:
                break
        mem_put(write_tens, write_name, place)
        mem_clean(read_name)
        for write_ind, name in enumerate(memory_names):
            if write_name in name:
                break
    elif conv_type == '1x1':
        mem_put(write_tens, write_name, place)
        for write_ind, name in enumerate(memory_names):
            if write_name in name:
                break
        for read_ind, name in enumerate(memory_names):
            if read_name in name:
                break
        # mem_clean(read_t)
    elif conv_type == 'split_bottle':
        for read_ind, name in enumerate(memory_names):
            if read_name in name:
                break
        mem_put(write_tens, write_name, place)
        for write_ind, name in enumerate(memory_names):
            if write_name in name:
                break
    plot_memory(read_name, write_name)
    write_memory(read_tens, write_tens, read_ind, write_ind, write_name, two_places)
    print(f'READ: {read_ind}, WRITE: {write_ind}, NAMES_0: {memory_names[0]}, NAMES_-1: {memory_names[-1]}')


def read_write_mass(read_tenses: list, write_tens: str, read_names: list, write_name: str, mem_type=None, place=None):
    if mem_type == 'bottle_sum':
        # mem_put(write_tens, write_name, place)
        # plot_memory(read_names[0], write_name)
        for ind, read_tens in enumerate(read_tenses):
            for read_ind, name in enumerate(memory_names):
                if read_names[ind] in name:
                    break
            for write_ind, name in enumerate(memory_names):
                if read_names[-1] in name:
                    break
            write_memory(read_tens, write_tens, read_ind, write_ind, write_name, two_places=True)
        mem_clean(read_names[-1], new_name=write_name, replace=True)
        # mem_put(write_tens, write_name, place)
    else:
        mem_put(write_tens, write_name, place)
        plot_memory(read_names[0], write_name)
        for ind, read_tens in enumerate(read_tenses):
            for read_ind, name in enumerate(memory_names):
                if read_names[ind] in name:
                    break
            for write_ind, name in enumerate(memory_names):
                if write_name in name:
                    break
            write_memory(read_tens, write_tens, read_ind, write_ind, write_name, two_places=True)
            mem_clean(read_names[ind])
    #     read_write(read_tens, write_tens, read_names[ind], write_name, two_places=True, place=place)


def memory_labels():
    global memory_names
    global memory
    res = {}
    for ind, el in enumerate(memory_names):
        if el[0] not in res and type(el[0]) is str:
            res[el[0]] = ind

    # mem = []
    # for row in range(memory.shape[0]):
    #     for column in range(memory.shape[1]):
    #         mem.append(f'[{row}, {column}] = {memory[row, column]}')
    # with open('mem.txt', 'w') as f_obj:
    #     for el in mem:
    #         f_obj.write(f'{el}\n')
    print(f'ОБЪЕКТЫ В ПАМЯТИ: {res}')


def sort_and_remove_w_duplicates(values):
    r_vals, s_vals, w_vals = [], [], []

    for item in values:
        prefix, num_str = item.split(':')
        prefix = prefix.strip()          # 'r', 's' или 'w'
        num = int(num_str.strip())       # переводим число в int

        if prefix == 'r':
            r_vals.append(num)
        elif prefix == 's':
            s_vals.append(num)
        elif prefix == 'w':
            w_vals.append(num)
        else:
            # Если у вас бывают другие префиксы, добавьте их обработку
            pass

    # Удаляем дубликаты только для w
    w_vals = list(set(w_vals))

    # # Сортируем каждый список по возрастанию
    # r_vals.sort()
    # s_vals.sort()
    # w_vals.sort()

    # Склеиваем обратно в порядок r -> s -> w
    result = (
        [f"r: {num}" for num in r_vals] +
        [f"s: {num}" for num in s_vals] +
        [f"w: {num}" for num in w_vals]
    )
    return result


def final_memory_rewrite():
    all_layers = {}
    with open('./results/memory.txt', 'r') as f_obj:
        lines = [tuple(line.strip().split(', ')) for line in f_obj.readlines()]
        for line in lines:
            name = line[0]
            read = line[1]
            size = line[2]
            write = line[3]

            if name not in all_layers:
                all_layers[name] = [read, size, write]
            else:
                all_layers[name].append(read)
                all_layers[name].append(size)
                all_layers[name].append(write)

    with open('./results/final_memory.txt', 'w') as f_obj:
        for key, value in all_layers.items():
            new_value = sort_and_remove_w_duplicates(value)
            new_line = ' | '.join(new_value)
            f_obj.write(f'{key} | {new_line}\n')
    append_memory_max()


# if __name__ == '__main__':
    # columns = 8
    # memory = create_empty_memory(columns)
    # memory_names = create_empty_memory(columns).tolist()


# mem_put(np.random.rand(1, 1, 4, 4), '1')
# mem_put(torch.rand((1, 1, 4, 4)), '2')
# mem_put(torch.rand((1, 1, 8, 4)), '3')
# mem_clean('2')
# mem_put(torch.rand((1, 1, 4, 4)), '4')
# read_write('4', '3', '3x3')
# print(memory)
