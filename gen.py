from check import check_table, find_neibours
import random


def add_one_product(
    st: tuple[int, int],
    door: tuple[int, int],
    mat: list[list[int]],
    pr_type: int,
    pr_cnt: int = 1,
    all_reached: bool = False,
) -> bool:
    n: int = len(mat)
    m: int = len(mat[0])
    placed: list[int] = []
    for j in range(pr_cnt):
        my_nei: list[int] = find_neibours(st, n, m)
        if j == 0:
            my_nei = [st]
        random.shuffle(my_nei)
        good_try: bool = False
        for el in my_nei:
            if mat[el[0]][el[1]] == 0:
                mat[el[0]][el[1]] = pr_type
                if check_table(mat, door, all_reached=all_reached):
                    st = el
                    placed.append(el)
                    good_try = True
                    break
                mat[el[0]][el[1]] = 0
        if not (good_try):
            for el in placed:
                mat[el[0]][el[1]] = 0
            return False
    return True


ms = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]]
print(add_one_product((2, 1), (0, 3), ms, 2))
print(ms)
