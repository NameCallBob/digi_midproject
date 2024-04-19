import pandas as pd
import glob

# 取得資料
def combined_data():

    # 利用pandas concat方法能夠合併目前的資料
    # 將資料進行合併
    # 技術文獻  https://www.learncodewithmike.com/2021/05/pandas-merge-multiple-csv-files.html
    file_list = [pd.read_excel(i) for i in glob.glob("./*.xlsx")]
    file_list[1] = file_list[1]['']
    print(pd.merge(file_list[0],file_list[1],on="日期"))


# 測試資料
class test_data:
    def get_tamp_day():
        return (pd.read_excel('./台北市每日氣溫.xlsx'))
    def get_tamp():
        return (pd.read_excel('./台北市氣溫(2018-2023).xlsx'))
    def get_used():
        return (pd.read_excel('./台北市電力用量(2018-2023).xlsx'))


# 回歸
# 1.Scikit-Learn
# 2.Regression