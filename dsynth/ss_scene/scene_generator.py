from scene_check import check_table, find_neibours
from collections import deque
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
# print(add_one_product((2, 1), (0, 3), ms, 2, 2))
# print(ms)


def get_rd_point(n: int, m: int):
    return (random.randint(0, n - 1), random.randint(0, m - 1))


def add_many_products(
    door: tuple[int, int],
    mat: list[list[int]],
    name_to_cnt: dict,
    all_reached: bool = False,
) -> tuple[bool, list[list]]:
    n: int = len(mat)
    m: int = len(mat[0])
    id_to_name: list[str] = []
    id: int = 0

    for el in name_to_cnt:
        id_to_name.append(el)
        goodpr = False
        id += 1
        for j in range(2 * n * m):
            s = get_rd_point(n, m)
            if add_one_product(s, door, mat, id, name_to_cnt[el], all_reached):
                goodpr = True
                break
        if not (goodpr):
            return (False, mat)
    for i in range(n):
        for j in range(m):
            if mat[i][j] > 0:
                mat[i][j] = id_to_name[mat[i][j] - 1]

    return (True, mat)


def get_orientation(door: tuple[int, int], mat: list[list[int]]) -> list[list[bool]]:
    n: int = len(mat)
    m: int = len(mat[0])
    had: deque = deque()
    dst: list[list[int]] = [[1e10] * len(mat[0]) for c in range(n)]
    if mat[door[0]][door[1]] != 0:
        return False
    dst[door[0]][door[1]] = 0
    had.append(door)
    while len(had):
        v: int = had.popleft()
        for el in find_neibours(v, n, m):
            if mat[el[0]][el[1]] == 0 and dst[el[0]][el[1]] != 0:
                had.append(el)
                dst[el[0]][el[1]] = 0
    mat_ops: list[list[bool]] = [[0 for j in range(m)] for i in range(n)]
    for i in range(n):
        for j in range(m):
            for el in find_neibours((i, j), n, m):
                if dst[el[0]][el[1]] == 0:
                    if abs(el[0] - i) == 1:
                        mat_ops[i][j] = 0
                    else:
                        mat_ops[i][j] = 1
    return mat_ops


ms = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
tr = [[0 for j in range(7)] for i in range(7)]

my_food = {1: 2, 2: 2, 3: 2, 4: 3, 5: 2, 6: 1, 7: 2, 8: 3, 9: 3}
(a, b) = add_many_products((0, 0), tr, my_food)
