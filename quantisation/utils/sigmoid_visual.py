import seaborn as sns
import matplotlib.pyplot as plt


x = []
y = []
with open('sigmoid_table_8_bit.txt', 'r') as f_obj:
    lines = f_obj.readlines()
    for line in lines:
        if '//' not in line and len(line) != 1:
            line = line.strip()
            split_line = line.split(' = ')
            x.append(float(split_line[0]))
            y.append(float(split_line[1]))

sns.lineplot(
    x=x,
    y=y
)
plt.title('Квантизация сигмоиды')
plt.xlabel('Исходное квантизованное значение')
plt.ylabel('Квантизованное значение сигмоиды')

plt.savefig('C:/Users/Артем Васильев/Documents/YOLOV8/визуализация_сигмоиды')
plt.show()
