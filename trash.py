NEIG = [[1, 4, 5], [0, 2, 4, 5, 6], [1, 3, 5, 6, 7], [2, 6, 7], [0, 1, 5, 8, 9], [0, 1, 2, 4, 6, 8, 9, 10], [1, 2, 3, 5, 7, 9, 10, 11], [2, 3, 6, 10, 11], [4, 5, 9, 12, 13], [4, 5, 6, 8, 10, 12, 13, 14], [5, 6, 7, 9, 11, 13, 14, 15], [6, 7, 10, 14, 15], [8, 9, 13], [8, 9, 10, 12, 14], [9, 10, 11, 13, 15], [10, 11, 14]]

def enlarge(path):
    if isinstance(path, list):
        for i in NEIG[path[-1]]:
            if i not in path:
                yield path[:] + [i]
    else:
        for i in path:
            yield from enlarge(i)

def paths(length):
    steps = ([i] for i in range(16))  # first steps on every point on board
    for _ in range(length-1):
        nsteps = enlarge(steps)
        steps = nsteps
    yield from steps

#[print(i) for i in paths(3)]
[print(i) for i in enlarge([0])]