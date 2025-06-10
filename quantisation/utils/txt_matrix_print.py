import torch


def clear_txt(filename):
    with open(filename, 'w') as f_obj:
        pass


def matrix_txt(tnsr, layer):
    with open('matrix.txt', 'a') as f_obj:
        for el in tnsr:
            for j in el:
                f_obj.write(f'\n-------------------------{layer}-------------------------\n')
                for i in j:
                    stroka = ''
                    for number in i:
                        stroka += f'  {number.item()  }'
                    # print(f'{stroka}')
                    f_obj.write(f'{stroka}\n')


def matrix_txt_name(tnsr, layer, filename):
    if type(tnsr) != torch.Tensor:
        tnsr = torch.from_numpy(tnsr)
    # with open(filename, 'w') as f_obj:
    #     pass
    with open(filename, 'a') as f_obj:
        for el in tnsr:
            for ind, j in enumerate(el):
                f_obj.write(f'\n-------------------------{layer}_{ind}-------------------------\n')
                for i in j:
                    stroka = ''
                    for number in i:
                        stroka += f'  {number.item()  }'
                    # print(f'{stroka}')
                    f_obj.write(f'{stroka}\n')


def matrix_txt_flatten(img, layer):
    with open('matrix.txt', 'a') as f_obj:
        a = ''
        for ind, el in enumerate(img):
            a += f'{el} '
        f_obj.write(f'\n-------------------------{layer}-------------------------\n')
        f_obj.write(f'{a}\n')


def matrix_txt_flatten_name(img, layer, filename):
    with open(filename, 'w') as f_obj:
        pass
    with open(filename, 'a') as f_obj:
        a = ''
        for ind, el in enumerate(img):
            a += f'{el} '
        f_obj.write(f'\n-------------------------{layer}-------------------------\n')
        f_obj.write(f'{a}\n')
