import unittest

from preprocessing import *


class test_graph_functions(unittest.TestCase):

    def setUp(self):
        # Define test data and relationships for your tests
        self.tables = [pd.DataFrame({'A': ['a1', 'a2', 'a3'], 'B': ['b4', 'b5', 'b6'], 'T': ['t1', 't2', 't3']}), pd.DataFrame({
            'C': ['c7', 'c8', 'c9'], 'D': ['d10', 'd11', 'd12'], 'T': ['t1', 't2', 't3']})]
        self.relationships = [['A', 'B'], ['C', 'D']]
        self.time_col = 'T'
        self.join_token = '::'

        if not isinstance(self.tables, list):
            self.tables = [self.tables]
        if isinstance(self.relationships[0], str):
            self.relationships = [self.relationships]
        if not isinstance(self.relationships[0][0], list):
            self.relationships = [self.relationships] * len(self.tables)
        # Handle the case when time_col is None
        if self.time_col is None:
            self.time_col = [None] * len(self.tables)
        elif isinstance(self.time_col, str):
            self.time_col = [self.time_col] * len(self.tables)
        if len(self.time_col) != len(self.tables):
            self.time_col = self.time_col * len(self.tables)

    def test_matrix_from_tables(self):
        A, attributes = matrix_from_tables(
            self.tables, self.relationships, self.time_col, self.join_token)
        # Add assertions to check if the output is as expected
        # You can use numpy.testing.assert_array_equal for comparing arrays
        self.assertTrue(isinstance(A, sparse.csr_matrix))
        self.assertTrue(isinstance(attributes, list))

    def test_create_edge_list(self):
        edge_list = create_edge_list(
            self.tables, self.relationships, self.time_col, self.join_token)
        # Add assertions to check if the output is as expected
        self.assertTrue(isinstance(edge_list, pd.DataFrame))

    def test_extract_node_time_info(self):
        edge_list = create_edge_list(
            self.tables, self.relationships, self.time_col, self.join_token)
        nodes, partitions, times, node_ids, time_ids = extract_node_time_info(
            edge_list, self.join_token)
        # Add assertions to check if the output is as expected
        self.assertTrue(isinstance(nodes, list))
        self.assertTrue(isinstance(partitions, list))
        self.assertTrue(isinstance(times, list))
        self.assertTrue(isinstance(node_ids, dict))
        self.assertTrue(isinstance(time_ids, dict))

    def test_create_edge_list_correct_columns(self):
        # Define test data and relationships
        df1 = pd.DataFrame(
            {'A': [1, 2, 3], 'B': [4, 5, 6], 'ID': [10, 11, 12]})
        df2 = pd.DataFrame(
            {'B': [7, 8, 9], 'C': [10, 11, 12], 'ID': [13, 14, 15]})
        tables = [df1, df2]
        relationships = [['A', 'B'], ['B', 'C']]
        time_col = 'ID'
        if not isinstance(tables, list):
            tables = [tables]
        if isinstance(relationships[0], str):
            relationships = [relationships]
        if not isinstance(relationships[0][0], list):
            relationships = [relationships] * len(tables)
        # Handle the case when time_col is None
        if time_col is None:
            time_col = [None] * len(tables)
        elif isinstance(time_col, str):
            time_col = [time_col] * len(tables)
        if len(time_col) != len(tables):
            time_col = time_col * len(tables)

        edge_list = create_edge_list(
            tables, relationships, time_col, join_token='::')

        # Check if the columns from dataframes are correctly combined in the edge_list
        expected_columns = ['V1', 'V2', 'T', 'P1', 'P2']
        for col in expected_columns:
            self.assertTrue(col in edge_list.columns)

        # Check if the correct number of rows are generated based on relationships
        # There should be 6 rows in this case
        self.assertEqual(len(edge_list), 6)

    def test_matrix_from_tables_different_inputs(self):
        df1 = pd.DataFrame(
            {'A': [1, 2, 3], 'B': [7, 9, 6], 'C': [2, 3, 4], 'ID': [10, 11, 12]})
        df2 = pd.DataFrame(
            {'B': [7, 8, 9], 'C': [10, 11, 12], 'ID': [13, 14, 15]})
        tables = [df1, df2]

        relationships1 = [['A', 'B'], ['B', 'C']]
        time_col1 = 'ID'
        A1, attributes1 = matrix_from_tables(
            tables, relationships1, time_col1, join_token='::')

        relationships2 = [[['A', 'B'], ['B', 'C']], [['B', 'C']]]
        time_col2 = ['ID', 'ID']
        A2, attributes2 = matrix_from_tables(
            tables, relationships2, time_col2, join_token='::')

        self.assertTrue(np.allclose(A1.toarray(), A2.toarray()))
        self.assertTrue(attributes1 == attributes2)
        self.assertEqual(A1.shape, (13, 78))
        self.assertEqual(A1.shape, (len(attributes1[0]), len(attributes1[1])))

    def test_find_cc_containing_most(self):
        df = pd.DataFrame(
            {'A': ['a1', 'a1', 'a2', 'a2', 'a1', 'a1', 'a2', 'a2'],
             'B': ['b1', 'b2', 'b1', 'b2', 'b1', 'b2', 'b1', 'b2'],
             'ID': [1, 1, 1, 1, 2, 2, 2, 2]})
        relationships = ['A', 'B']
        time_col = 'ID'
        A1, attributes1 = matrix_from_tables(
            df, relationships, dynamic_col=time_col, join_token='::')

        c0, att0 = find_cc_containing_most(A1, attributes1, 'B', dynamic=False)
        c1, att1 = find_cc_containing_most(A1, attributes1, 'A', dynamic=True)

        self.assertEqual(np.sum(c0.todense() == c1.todense()), 8)
        self.assertTrue(att0 == att1)


class test_text_functions(unittest.TestCase):
    self.text = pd.DataFrame(
        ['This is a test sentence', 'This is another test sentence', 'This contains an email address: email_address@email.com'], columns=['data'])
    self.column = 'data'

    Y, attributes = matrix_from_text(
        self.text, self.column, remove_email_addresses=True)
    self.assertTrue(isinstance(Y, sparse.csr_matrix))
    self.assertTrue(isinstance(attributes, list))

    self.assertEqual(Y.shape, (3, 6))
    self.assertEqual(len(attributes[0]), 3)
    self.assertEqual(len(attributes[1]), 6)


if __name__ == "__main__":
    unittest.main()
