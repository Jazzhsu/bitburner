import copy
import random
from typing import Set
from enum import Enum

import torch

ADJ = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class StoneType(Enum):
    EMPTY = 0
    BLOCK = 1
    WHITE = 2
    BLACK = 3

def opposite(stone: StoneType):
    if stone == StoneType.WHITE:
        return StoneType.BLACK

    return StoneType.WHITE


class UnionFind:

    def __init__(self, size: int):
        self._r = [0] * size
        self._p = list(range(size))

    def reset(self, nodes: Set[int]):
        for i in nodes:
            self._p[i] = i
            self._r[i] = 0
    
    def parent(self, i: int):
        # Bounds checking
        if i < 0 or i >= len(self._p):
            raise IndexError(f"Index {i} out of bounds for UnionFind of size {len(self._p)}")
        
        # Find root iteratively (no recursion)
        root = i
        path = []
        
        # Traverse to root, recording path
        while self._p[root] != root:
            path.append(root)
            root = self._p[root]
            
            # Cycle detection safety check
            if len(path) > len(self._p):
                print(self._p)
                raise RuntimeError(f"Cycle detected in UnionFind starting from {i}")
        
        # Path compression - point all nodes directly to root
        for node in path:
            self._p[node] = root
        
        return root

    def union(self, i: int, j: int):
        i = self.parent(i)
        j = self.parent(j)

        if i == j:
            return

        if self._r[i] > self._r[j]:
            self._p[j] = i
        elif self._r[i] < self._r[j]:
            self._p[i] = j
        else:
            self._p[j] = i
            self._r[i] += 1

    def dump(self):
        print(self._p)


class GO:
    def __init__(self, size: int):
        self._size = size
        self._board = [ [ StoneType.EMPTY for _ in range(size) ] for _ in range(size) ]
        self._n_blocks = 0
        self._protected = (-1, -1)
        self._uf = UnionFind(size * size)
        self._lib_map: dict[int, Set[int]] = {}
        self._group_map: dict[int, Set[int]] = {}

        for i in range(size):
            for j in range(size):
                if (i == 0 or j == 0 or i == size - 1 or j == size - 1) and random.random() < 0.2:
                    self._board[i][j] = StoneType.BLOCK
                    self._n_blocks += 1

    def _idx2ij(self, idx: int) -> tuple[int, int]:
        return (idx // self._size, idx % self._size)

    def _ij2idx(self, i: int, j: int) -> int:
        if not self._in_bound(i, j):
            raise ValueError(f"Coordinates ({i}, {j}) out of bounds")
        return i * self._size + j

    def size(self) -> int:
        return self._size

    def print_board(self):
        print_map = {
            StoneType.EMPTY: "·",
            StoneType.BLOCK: "#",
            StoneType.WHITE: "O",
            StoneType.BLACK: "X",
        }

        print("-" * (self._size * 2 - 1));
        for i in range(self._size):
            print(" ".join([ print_map[stone] for stone in self._board[i]]))
        print("-" * (self._size * 2 - 1));

    # return a list of stone of this group and the total liberty the group has
    def group_info(self, i: int, j: int) -> tuple[Set, int]:
        assert self._board[i][j] in (StoneType.WHITE, StoneType.BLACK)
        group = set()
        lib = self._dfs_lib(i, j, self._board[i][j], group)
        return (group, lib)

    def is_valid_move(self, i: int, j: int, stone_type: StoneType) -> bool:
        if not self._in_bound(i, j):
            return False

        if self._board[i][j] != StoneType.EMPTY:
            return False

        groups = set()
        for (di, dj) in ADJ:
            ii, jj = i + di, j + dj
            if not self._in_bound(ii, jj):
                continue

            if self._board[ii][jj] == StoneType.EMPTY:
                return True

            if self._board[ii][jj] == opposite(stone_type) and (ii, jj) != self._protected:
                root = self._uf.parent(self._ij2idx(ii, jj))
                enemy_lib = self._lib_map[root]
                if len(enemy_lib) == 1:
                    return True

            if self._board[ii][jj] == stone_type:
                groups.add(self._uf.parent(self._ij2idx(ii, jj)))

        if len(groups) == 0:
            return False

        my_lib = sum([ len(self._lib_map[idx]) for idx in groups ])
        return my_lib > len(groups)


    def move(self, i: int, j: int, stone_type: StoneType, check_valid: bool = True) -> bool:
        if check_valid and not self.is_valid_move(i, j, stone_type):
            return False

        idx = self._ij2idx(i, j)

        # self.print_board()
        # print(i, j, stone_type)
        # print(self._lib_map)
        # print(self._group_map)
        # self._uf.dump()

        # 1. kill and clean opponent stones
        killed = set()
        for (di, dj) in ADJ:
            ii, jj = i + di, j + dj
            if not self._in_bound(ii, jj):
                continue

            if self._board[ii][jj] == opposite(stone_type):
                root = self._uf.parent(self._ij2idx(ii, jj))
                enemy_lib = self._lib_map[root]
                if idx in enemy_lib:
                    enemy_lib.remove(idx)

                if len(enemy_lib) == 0:
                    killed.add(root)

        self._cleanup(killed, stone_type)
            
        # 2. update my stones
        new_idx = { idx }
        new_libs = set()
        to_cleanup = set()
        for (di, dj) in ADJ:
            ii, jj = i + di, j + dj
            if not self._in_bound(ii, jj):
                continue

            adj_idx = self._ij2idx(ii, jj)
            if self._board[ii][jj] == StoneType.EMPTY:
                new_libs.add(adj_idx)

            if self._board[ii][jj] == stone_type:
                root = self._uf.parent(adj_idx)
                new_idx.update(self._group_map[root])
                new_libs.update(self._lib_map[root])
                to_cleanup.add(root)

        for root in to_cleanup:
            self._uf.union(idx, root)
            self._group_map.pop(root)
            self._lib_map.pop(root)

        root = self._uf.parent(idx)
        if idx in new_libs:
            new_libs.remove(idx)
        self._lib_map[root] = new_libs
        self._group_map[root] = new_idx
        self._board[i][j] = stone_type

        if len(killed) > 0:
            self._protected = (i, j)
        else:
            self._protected = (-1, -1)
        return True

    def _cleanup(self, killed_root: Set, my_type: StoneType):
        for root in killed_root:
            group = self._group_map[root]
            self._uf.reset(group)

            # For each stone in group
            for idx in group:
                i, j = self._idx2ij(idx)

                # 1. Clear the board
                self._board[i][j] = StoneType.EMPTY

                # 2. Update my liberty
                for (di, dj) in ADJ:
                    ii, jj = i + di, j + dj
                    if not self._in_bound(ii, jj):
                        continue

                    if self._board[ii][jj] != my_type:
                        continue

                    self._lib_map[self._uf.parent(self._ij2idx(ii, jj))].add(idx)

            self._group_map.pop(root)
            self._lib_map.pop(root)


    def eval(self) -> dict:
        bl_score = 0
        wh_score = 0

        vis = set()
        for i in range(self._size):
            for j in range(self._size):
                if self._board[i][j] == StoneType.WHITE:
                    wh_score += 1
                if self._board[i][j] == StoneType.BLACK:
                    bl_score += 1

                if self._board[i][j] == StoneType.EMPTY and (i, j) not in vis:
                    cnt, t = self._dfs_eval(i, j, vis)

                    if t == StoneType.WHITE and cnt < self._size ** 2 // 2:
                        wh_score += cnt
                    if t == StoneType.BLACK and cnt < self._size ** 2 // 2:
                        bl_score += cnt

        total_score = self.size() ** 2 - self._n_blocks
        return { StoneType.WHITE: wh_score, StoneType.BLACK: bl_score, StoneType.EMPTY: total_score }

    def serialize(self, player: StoneType) -> str:
        return self._serialize(player, self._board) + f"({self._protected[0]},{self._protected[1]})"

    def _serialize(self, player: StoneType, board: list[list]) -> str:
        res = ""
        for i in range(self._size):
            for j in range(self._size):
                if board[i][j] == StoneType.EMPTY:
                    res += "0"
                elif board[i][j] == StoneType.BLOCK:
                    res += "1"
                else:
                    res += "2" if player == board[i][j] else "3"
        return res


    def get_nn_state(self, player: StoneType) -> torch.Tensor:
        return torch.tensor(self.get_canonical(player, self._board)).permute(2, 0, 1)

    def get_canonical(self, player: StoneType, board: list[list]) -> list:
        base_planes =  [
            [ self._one_hot(player, board[i][j]) for j in range(self._size) ]
            for i in range(self._size)
        ]
        
        # Adding liberty and group size planes
        liberty_plane = self._get_liberty_plane()
        group_size_plane = self._get_group_size_plane()
        
        # Combine all planes
        combined_planes = []
        for i in range(self._size):
            row = []
            for j in range(self._size):
                row.append(base_planes[i][j] + [liberty_plane[i][j], group_size_plane[i][j]])
            combined_planes.append(row)
            
        return combined_planes


    def _get_liberty_plane(self) -> list[list[float]]:
        """Generates a plane representing the number of liberties for each stone's group."""
        plane = [[0.0] * self._size for _ in range(self._size)]
        for i in range(self._size):
            for j in range(self._size):
                stone = self._board[i][j]
                if stone in (StoneType.WHITE, StoneType.BLACK):
                    root = self._uf.parent(self._ij2idx(i, j))
                    if root in self._lib_map:
                        plane[i][j] = len(self._lib_map[root]) / (self._size * self._size)  # Normalize
        return plane

    def _get_group_size_plane(self) -> list[list[float]]:
        """Generates a plane representing the size of each stone's group."""
        plane = [[0.0] * self._size for _ in range(self._size)]
        for i in range(self._size):
            for j in range(self._size):
                stone = self._board[i][j]
                if stone in (StoneType.WHITE, StoneType.BLACK):
                    root = self._uf.parent(self._ij2idx(i, j))
                    if root in self._group_map:
                        plane[i][j] = len(self._group_map[root]) / (self._size * self._size)  # Normalize
        return plane

    def _one_hot(self, player: StoneType, stone_type: StoneType) -> list:
        ret = [0, 0, 0, 0]
        if stone_type == StoneType.EMPTY:
            ret[0] = 1
        elif stone_type == StoneType.BLOCK:
            ret[1] = 1
        else:
            ret[2 if player == stone_type else 3] = 1

        return ret

        
    def _board_eq(self, board1: list[list], board2: list[list]) -> bool:
        for i in range(self._size):
            for j in range(self._size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def _in_bound(self, i: int, j: int) -> bool:
        return i >= 0 and i < self._size and j >= 0 and j < self._size

    def _dfs_lib(self, i: int, j: int, stone_type: StoneType, vis: Set) -> int:
        if i < 0 or j < 0 or i >= self._size or j >= self._size:
            return 0

        if self._board[i][j] == StoneType.EMPTY:
            return 1

        if self._board[i][j] != stone_type:
            return 0

        if (i, j) in vis:
            return 0

        vis.add((i, j))

        lib = 0
        for (di, dj) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            lib += self._dfs_lib(i + di, j + dj, stone_type, vis)
        return lib

    def _dfs_eval(self, i: int, j: int, vis: Set) -> tuple[int, StoneType | None]:
        if self._board[i][j] != StoneType.EMPTY:
            return (0, self._board[i][j])

        vis.add((i, j))

        stone_type = None 
        total = 1
        for (di, dj) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            ii, jj = i + di, j + dj
            if not self._in_bound(ii, jj) or self._board[ii][jj] == StoneType.BLOCK:
                continue

            if (ii, jj) in vis:
                continue

            cnt, t = self._dfs_eval(ii, jj, vis)
            total += cnt

            if stone_type == None:
                stone_type = t
            elif t != None and stone_type != t:
                stone_type = StoneType.EMPTY


        return (total, stone_type)
    
