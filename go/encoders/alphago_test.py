import unittest

from agent.helpers import is_point_an_eye
from goboard_fast import Board, GameState, Move
from gotypes import Player, Point
from encoders.alphago import AlphaGoEncoder


class AlphaGoEncoderTest(unittest.TestCase):
    def test_encoder(self):
        alphago = AlphaGoEncoder()

        start = GameState.new_game(19)
        next_state = start.apply_move(Move.play(Point(16, 16)))
        alphago.encode(next_state)

        self.assertEqual(alphago.name(), 'alphago')
        self.assertEqual(alphago.board_height, 19)
        self.assertEqual(alphago.board_width, 19)
        self.assertEqual(alphago.num_planes, 49)
        self.assertEqual(alphago.shape(), (49, 19, 19))



if __name__ == '__main__':
    unittest.main()
