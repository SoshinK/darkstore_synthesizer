from collections import deque


def check_coords(pos: tuple[int, int], n: int, m: int) -> bool:
    if pos[0] >= n or pos[0] < 0:
        return False
    if pos[1] >= m or pos[1] < 0:
        return False
    return True


def find_neibours(pos: tuple[int, int], n: int, m: int) -> list[tuple[int, int]]:
    nei = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if check_coords((pos[0] + i, pos[1] + j), n, m) and (abs(i) + abs(j)) == 1:
                nei.append((pos[0] + i, pos[1] + j))
    return nei


def check_table(
    mat: list[list[int]], pos: tuple[int, int], emp: int = 0, all_reached: bool = False
) -> bool:
    n: int = len(mat)
    m: int = len(mat[0])
    had: deque = deque()
    dst: list[list[int]] = [[1e10] * len(mat[0]) for c in range(n)]
    if mat[pos[0]][pos[1]] != 0:
        return False
    dst[pos[0]][pos[1]] = 0
    had.append(pos)
    while len(had):
        v: int = had.popleft()
        for el in find_neibours(v, n, m):
            if mat[el[0]][el[1]] == emp and dst[el[0]][el[1]] != 0:
                had.append(el)
                dst[el[0]][el[1]] = 0
    for i in range(n):
        for j in range(m):

            if mat[i][j] != emp:
                good: bool = False
                for el in find_neibours((i, j), n, m):
                    good = good or (dst[el[0]][el[1]] == 0)
                if not (good):
                    return False
            if all_reached:
                good: bool = False
                for el in find_neibours((i, j), n, m):
                    good = good or (dst[el[0]][el[1]] == 0)
                if not (good):
                    return False
    return True


ms = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]]
assert check_table(ms, (0, 0), all_reached=False) == True
assert check_table(ms, (0, 0), all_reached=True) == False
