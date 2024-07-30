# test_dataset_update.py

import unittest

def update_unseen_dataset(unseen_data, sorted_selected_ids):
    return [item for item in unseen_data if int(item[0]) not in sorted_selected_ids]


def get_related_ids(sample_id, num_hypotheses):
    # 計算每個樣本的第一個ID
    base_sample_id = ((sample_id - 1) // num_hypotheses) * num_hypotheses + 1
    # 返回這個樣本對應的所有ID
    return set(range(base_sample_id, base_sample_id + num_hypotheses))

class TestDatasetUpdate(unittest.TestCase):

    def setUp(self):
        self.selected_ids = set()
        self.unseen_data = [
            ('1', "label_1", "text1"),
            (2, "label_2", "text1"),
            (3, "label_3", "text1"),
            (4, "label_4", "text1"),
            (5, "label_5", "text1"),
            (6, "label_6", "text1"),
            (7, "label_7", "text1"),
            (8, "label_8", "text1"),
            (9, "label_9", "text1"),
            (10, "label_10", "text1"),
            (11, "label_1", "text2"),
            (12, "label_2", "text2"),
            (13, "label_3", "text2"),
            (14, "label_4", "text2"),
            (15, "label_5", "text2"),
            (16, "label_6", "text2"),
            (17, "label_7", "text2"),
            (18, "label_8", "text2"),
            (19, "label_9", "text2"),
            (20, "label_10", "text2"),
            (21, "label_1", "text3"),
            (22, "label_2", "text3"),
            (23, "label_3", "text3"),
            (24, "label_4", "text3"),
            (25, "label_5", "text3"),
            (26, "label_6", "text3"),
            (27, "label_7", "text3"),
            (28, "label_8", "text3"),
            (29, "label_9", "text3"),
            (30, "label_10", "text3"),
        ]
    

    def test_get_related_ids(self):
        print(sorted(get_related_ids(1, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.selected_ids.update(get_related_ids(1, 10))
        print(f'self.selected_ids: {self.selected_ids}')
        print(sorted(get_related_ids(2, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(sorted(get_related_ids(3, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(sorted(get_related_ids(4, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(sorted(get_related_ids(5, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(sorted(get_related_ids(6, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(sorted(get_related_ids(7, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(sorted(get_related_ids(8, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(sorted(get_related_ids(9, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(sorted(get_related_ids(10, 10)))  # 應該輸出 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(sorted(get_related_ids(15, 10))) # 應該輸出 [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        print(sorted(get_related_ids(21, 10))) # 應該輸出 [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        


    def test_update_unseen_dataset(self):
        # 假設我們要移除ID2和相關的ID
        selected_id = 30
        num_hypotheses = 10  # 假設每個樣本有10個相關ID
        # sorted_selected_ids = sorted(get_related_ids(selected_id, num_hypotheses))
        # print(f'sorted_selected_ids: {sorted_selected_ids}')
        self.selected_ids.update(get_related_ids(1, 10))
        # self.selected_ids = list(self.selected_ids)
        print(f'self.selected_ids: {self.selected_ids}')
        # 執行更新
        updated_unseen_data = update_unseen_dataset(self.unseen_data, self.selected_ids)
        print(f'updated_unseen_data: {updated_unseen_data}')
        
        # 检查结果
        self.assertEqual(len(updated_unseen_data), 10)
        self.assertNotIn((2, "label_2", "text1"), updated_unseen_data)
        self.assertIn((1, "label1", "text1"), updated_unseen_data)
        self.assertIn((4, "label4", "text4"), updated_unseen_data)

    def test_empty_dataset(self):
        unseen_data = []
        sorted_selected_ids = [1, 2, 3]
        updated_unseen_data = update_unseen_dataset(unseen_data, sorted_selected_ids)
        self.assertEqual(len(updated_unseen_data), 0)

    def test_no_removal(self):
        unseen_data = [(1, "label1", "text1"), (2, "label2", "text2")]
        sorted_selected_ids = [3, 4, 5]
        updated_unseen_data = update_unseen_dataset(unseen_data, sorted_selected_ids)
        self.assertEqual(len(updated_unseen_data), 2)
        self.assertEqual(unseen_data, updated_unseen_data)

if __name__ == '__main__':
    unittest.main()