import unittest

%run "source_notebook.ipynb"

class Calculator:
    def add(self, a, b):
        return a + b


class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()

    def test_addition(self):
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)


if __name__ == "__main__":
    unittest.main()
