import unittest
from src.loops.loop1 import Loop1
from src.loops.loop2 import Loop2
from src.loops.loop3 import Loop3
from src.loops.loop4 import Loop4

class TestLoops(unittest.TestCase):

    def setUp(self):
        self.loop1 = Loop1()
        self.loop2 = Loop2()
        self.loop3 = Loop3()
        self.loop4 = Loop4()

    def test_startLoop(self):
        self.assertEqual(self.loop1.startLoop(), "Loop1 started")
        self.assertEqual(self.loop2.startLoop(), "Loop2 started")
        self.assertEqual(self.loop3.startLoop(), "Loop3 started")
        self.assertEqual(self.loop4.startLoop(), "Loop4 started")

    def test_endLoop(self):
        self.assertEqual(self.loop1.endLoop(), "Loop1 ended")
        self.assertEqual(self.loop2.endLoop(), "Loop2 ended")
        self.assertEqual(self.loop3.endLoop(), "Loop3 ended")
        self.assertEqual(self.loop4.endLoop(), "Loop4 ended")

    def test_brainstorming(self):
        self.assertTrue(isinstance(self.loop1.brainstorming(), list))
        self.assertTrue(isinstance(self.loop2.brainstorming(), list))
        self.assertTrue(isinstance(self.loop3.brainstorming(), list))
        self.assertTrue(isinstance(self.loop4.brainstorming(), list))

if __name__ == '__main__':
    unittest.main()