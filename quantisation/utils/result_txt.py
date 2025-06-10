def result_txt(matrix, fc=0):
    with open('result_quant.txt', 'w') as f_obj:
        pass
    if fc == 0:
        with open('result_quant.txt', 'a') as f_obj:
            for i in range(matrix.shape[1]):
                for stroka in matrix[0, i]:
                    r = ''
                    for width in stroka:
                        r += f'{width}  '
                    f_obj.write(f'{r}\n')
                f_obj.write(f'\n')
                # f_obj.write(f'\n\n')
    else:
        with open('result_quant.txt', 'a') as f_obj:
            r = ''
            for el in matrix:
                r += f'{el}  '
            f_obj.write(f'{r}\n')

