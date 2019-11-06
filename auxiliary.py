#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Набор вспомогательных функций.

Version: 0.3
Created: 18/09/2019
Last modified: 23/09/2019
"""

import cv2 as cv
import os
import random
import time
from operator import add
from typing import Union

DEFAULT_DIRS = {
    "images": "images",
    "points": "points",
    'processed': 'processed',
    "marked": "marked",
    "sets": "sets"
}


def is_file_ok(file: Union[os.DirEntry, str], *extensions: str) -> bool:
    """
    Проверяет существование файла и его расширение.

    :param file: файл
    :param extensions: список допустимых расширений
    :return: False, если файл не существует, является каталогом или имеет неправильное расширение. Иначе True
    """
    if isinstance(file, os.DirEntry):
        if not file.is_file():
            return False
        for extension in extensions:
            if not file.name.lower().endswith('.' + extension):
                return False
    else:
        if not os.path.isfile(file):
            return False
        for extension in extensions:
            if not file.lower().endswith('.' + extension):
                return False
    return True


def get_points(file_name: str, path_points=DEFAULT_DIRS["points"]) -> list:
    """
    Возвращает список особых точек, если удалось его получить.
    Иначе возвращает пустой список.

    :param file_name: имя файла данных (расширение не txt) или файла с особыми точками (с расширением txt)
    :param path_points: путь к каталогу файлов с особыми точками
    :return: список особых точек в формате (x, y)
    """
    # Путь к файлу с особыми точками
    if not file_name.endswith('.txt'):
        file_name = '.'.join([file_name.rsplit('.', 1)[0]] + ['txt'])
    file_path = os.path.join(path_points, file_name)

    # Пробуем открыть файл
    try:
        file = open(file_path, 'r')
    except OSError as e:
        print("[ERROR] Ошибка при открытии файла особых точек:", e)
        return []

    # Список особых точек
    points = []

    # Получаем список особых точек
    try:
        for line in file:
            point = tuple(map(int, line.split('\t')))
            if len(point) != 2:
                raise ValueError
            points.append(point)
    except ValueError as e:
        print(f"[ERROR] Неверный формат файла особых точек {file_path}:", e)
        return []

    file.close()

    return points


def mark_points(path_images=DEFAULT_DIRS["images"],
                path_marked=DEFAULT_DIRS["marked"],
                path_points=DEFAULT_DIRS["points"]):
    """
    Помечает особые точки на наборе данных.
    Используется для визуального понимания задачи.

    :param path_images: папка с исходными изображениями
    :param path_points: папка с набором точек
    :param path_marked: папка для изображений с помеченными точками
    :return: None
    """
    # Создаем каталог для маркированных изображений (если его еще нет)
    os.makedirs(path_marked, exist_ok=True)

    # Для вывода данных о скорости работы
    perf_time_total = 0
    perf_files_total = 0

    # Маркируем все исходные изображения
    with os.scandir(path_images) as files:
        for file in files:
            # Проверяем, является ли файл изображением 'jpg'
            if not is_file_ok(file, 'jpg'):
                continue

            perf_file_time = time.perf_counter()

            # Загружаем особые точки для данного файла в список
            points = get_points(file.name, path_points)
            if not points:
                continue

            # Открываем и маркируем изображение. Затем записываем его в файл
            img = cv.imread(file.path, cv.IMREAD_COLOR)
            if img is None:
                continue

            for point in points:
                cv.circle(img, point, 1, (0, 255, 255), -1)
                cv.rectangle(img, tuple(map(add, point, (-5, -5))),
                             tuple(map(add, point, (5, 5))), (0, 0, 255), 1)

            cv.imwrite(os.path.join(path_marked, file.name), img)

            perf_time_total += time.perf_counter() - perf_file_time
            perf_files_total += 1
            break

    if perf_files_total:
        print("[INFO] Промаркировано {0} файл(-ов) за {1:.3} с ({2:.3} с на файл)".format(
                perf_files_total, perf_time_total, perf_time_total / perf_files_total))
    else:
        print("[INFO] Ни один файл не маркирован")


def make_set(test_size=0.2,
             path_images=DEFAULT_DIRS["images"],
             path_sets=DEFAULT_DIRS["sets"],
             seed: int = None) -> dict:
    """
    Создает обучающий и проверочный наборы из общего набора данных.

    :param test_size: относительный размер тестового набора (от 0 до 1)
    :param path_images: папка с исходными изображениями
    :param path_sets: папка с набрами данных
    :param seed: зерно генератора случайных чисел
    :return: словарь путей к наборам
    """
    # Создаем каталог для хранения файлов наборов (если его еще нет)
    os.makedirs(path_sets, exist_ok=True)

    # Список файлов
    file_list = []

    # Делим файлы из общего каталога данных на обучающий и тренировочный наборы
    with os.scandir(path_images) as files:
        for file in files:
            # Проверяем, является ли файл изображением 'jpg'
            if not is_file_ok(file, 'jpg'):
                continue

            file_list.append(file.name)

    # Чтобы из одного и того же общего набора при одинаковом соотношении и зерне
    # формировались одинаковые наборы данных, нужно отсортировать общий список файлов
    file_list.sort()

    if seed is None:
        seed = int(time.time_ns() % 1e+10)  # Последние 10 цифр
    random.seed(seed)

    # Формируем тестовый и тренировочный наборы
    test_size = max(0., test_size)
    test_size = min(1., test_size)

    test_set = random.sample(file_list, int(len(file_list) * test_size))
    test_set.sort()

    training_set = [file for file in file_list if file not in test_set]

    # Пути к файлам набора
    sets = {'seed': seed,
            'training': os.path.join(path_sets, f"{seed}_training.txt"),
            'test': os.path.join(path_sets, f"{seed}_test.txt")}

    # Записываем наборы в соответствующие файлы
    with open(sets['training'], 'w') as file:
        for element in training_set:
            file.write(element + "\n")

    with open(sets['test'], 'w') as file:
        for element in test_set:
            file.write(element + "\n")

    return sets


# TODO Переделать, чтобы mark_points вызывала process_images
def process_images(path_images=DEFAULT_DIRS["images"],
                   path_processed=DEFAULT_DIRS["processed"]):
    # Создаем каталог для обработанных изображений (если его еще нет)
    os.makedirs(path_processed, exist_ok=True)

    # Для вывода данных о скорости работы
    perf_time_total = 0
    perf_files_total = 0

    with os.scandir(path_images) as files:
        for file in files:
            # Проверяем, является ли файл изображением 'jpg'
            if not is_file_ok(file, 'jpg'):
                continue

            perf_file_time = time.perf_counter()

            # Открываем и обрабатываем изображение. Затем записываем его в файл
            img = cv.imread(file.path, cv.IMREAD_GRAYSCALE)
            if img is None:
                continue

            _, img = cv.threshold(img, 210, 255, cv.THRESH_TOZERO_INV)
            img[img == 0] = 255

            cv.imwrite(os.path.join(path_processed, file.name), img)

            perf_time_total += time.perf_counter() - perf_file_time
            perf_files_total += 1
            break

    if perf_files_total:
        print("[INFO] Обработано {0} файл(-ов) за {1:.3} с ({2:.3} с на файл)".format(
            perf_files_total, perf_time_total, perf_time_total / perf_files_total))
    else:
        print("[INFO] Ни один файл не обработан")


def main():
    # mark_points()
    process_images()
    mark_points(DEFAULT_DIRS['processed'], "processed and marked")
    # make_set(seed=100)


if __name__ == '__main__':
    __time_start = time.perf_counter()
    main()
    __time_delta = time.perf_counter() - __time_start
    __TIMES = (('d', 24 * 60 * 60), ('h', 60 * 60), ('m', 60), ('s', 1))
    __times = ''
    for __i in range(len(__TIMES) - 1):
        __t, __time_delta = divmod(__time_delta, __TIMES[__i][1])
        if __t > 0:
            __times += "{} {} ".format(int(__t), __TIMES[__i][0])
    __times += "{:.3} {}".format(__time_delta, __TIMES[~0][0])
    print("\n[Finished in {}]".format(__times))
