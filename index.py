from main import *



class data:
  def data_csv_all(self,area):
    import pandas as pd
    if area == 'tp': data = pd.read_csv('./data/tempelecttaipei.csv')
    elif area == 'tn': data = pd.read_csv('./data/tempelecttainan.csv')
    else:
      raise ValueError ("地區輸入錯誤，目前只有台南 = 'tn'、台北 = 'tp'可輸入")
    return data

  def data_tp_all(self,path = ''):
    """
    輸出台北市電力、天氣資訊 (json)
    """
    import pandas as pd
    # 使用檔案
    filelist_taipei = ['台北市氣溫(2018-2023).xlsx','台北市電力用量(2018-2023).xlsx']
    # pandas讀取檔案
    filelist_taipei = [pd.read_excel(i) for i in filelist_taipei]
    # 合併pandas Dataframe
    data = pd.merge(filelist_taipei[0],filelist_taipei[1],on="日期")
    # drop 不必要的欄位
    data = data.drop(['縣市'],axis=1)
    data = data.drop(['住宅部門用電佔比(%)'],axis=1)
    return data
  def data_tp_reshape(self):
    """
    利用目標向量(Target Vector)的方式進行資料處理，讓scikit learn 可讀取
    """
    import numpy as np
    # 將自定義方法導入
    data1 = self.data_tp_all()
    x = np.array(data1['平均氣溫'])
    y = np.array(data1['住宅部門售電量(度)'])

    x = np.reshape(x,(len(x),1))
    y = np.reshape(y,(len(y),1))
    return x , y
  def data_tn_all(self,path= ''):
    """
    @path -> 使用不同路徑請用list填入檔案路徑
    """
    import pandas as pd
    filelist_tainan = ['台南市氣溫.xlsx','台南市電力用量.xlsx']
    if (path != '' ):filelist_tainan = path
    try:
      data = [pd.read_excel(i) for i in filelist_tainan]
    except:
      print('file not found')
    data = pd.merge(data[0],data[1],on="日期")
    data = data.drop(['Unnamed: 2'],axis=1)
    data = data.drop(['縣市'],axis=1)
    data = data.drop(['住宅部門用電佔比(%)'],axis=1)
    data.rename(columns={'氣溫':'平均氣溫'},inplace=True)
    return data
  def data_tn_reshape(self):
    """
    利用目標向量(Target Vector)的方式進行資料處理，讓scikit learn 可讀取
    """
    import numpy as np
    # 將自定義方法導入
    data1 = self.data_tn_all()
    x = np.array(data1['平均氣溫'])
    y = np.array(data1['住宅部門售電量(度)'])

    x = np.reshape(x,(len(x),1))
    y = np.reshape(y,(len(y),1))
    return x , y
  def pre(self,area:str):
    """
    填入class中的方法進行運算
    area -> 地區(tn,tp)
    """
    if area == 'tn' : data_method = self.data_tn_reshape();
    elif area == 'tp' : data_method = self.data_tp_reshape();

    from sklearn.linear_model import LinearRegression
    x_train,y_train = data_method
    # 分割資料 x 溫度 y 使用電
    x_test = x_train[:40]
    x_train = x_train[-20:]
    y_test = y_train[:40]
    y_train = y_train[-20:]
    #
    model = LinearRegression()
    model.fit(x_train,y_train)

    res = model.predict(x_test)
    return res

class draw:


  def predict(self,area):
    """
    用於簡單線性回歸 測試資料
    area -> 'tp','tn'
    (若輸入錯誤會)
    """
    from sklearn.linear_model import LinearRegression

    import matplotlib.pyplot as plt

    d = data().data_csv_all(area=area)


    y = d['electricity'].values.reshape(-1,1)
    X = d['temperature'].values.reshape(-1,1)

    # 資料共64筆，拿前面40筆進行訓練
    # 拿後面20筆進行預測結果
    X_train = X[:40]
    X_test = X[-20:]
    y_train =y[:40]
    y_test =y[-20:]
    model = LinearRegression()
    model.fit(X_train,y_train)
    res = model.predict(X_test)
    # 圖表呈現
    plt.scatter(X_test,y_test)
    plt.plot(X_test,res,color='r')
    plt.title('簡單線性回歸_{}電力、溫度預測圖'.format('台北' if area =='tp' else '台南'))
    plt.ylabel('電力使用',size=14)
    plt.xlabel('溫度',size=14)
    plt.legend(['資料','預測線'])
    """
    結果
    若不考慮外在因素（經濟、政治方面），在溫度提高的情況下，經過模型預測得知電量使用量也會越來越高。
    """
    return d

  def predict_sq(self,area):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    # read
    d = data().data_csv_all('tp' if area=='tp' else 'tn')
    # list轉換
    y = d['electricity'].values.reshape(-1,1)
    X = d['temperature'].values.reshape(-1,1)



    # 資料共64筆，拿前面40筆進行訓練
    # 拿後面20筆進行預測結果
    X_train = X[:40]
    X_test = X[-20:]
    y_train =y[:40]
    y_test =y[-20:]

    for i in range(len(X_train)):
      if X_train[i] <= 20:
        X_train[i] = X[i] ** 2

    model = LinearRegression()
    model.fit(X_train,y_train)
    res = model.predict(X_test)

    # 圖表呈現
    plt.scatter(X_test,y_test)
    plt.plot(X_test,res,color='r')
    plt.title('簡單線性回歸_{}電力、溫度預測圖'.format('台北' if area =='tp' else '台南'))
    plt.ylabel('電力使用',size=14)
    plt.xlabel('溫度',size=14)
    # plt.xticks([10,20,30,40,250,350,400,500])
    plt.legend(['資料','預測線'])

    # 準確度 使用均方誤差

    # def MSE(l,rightlist):
    #   N = len(l)
    #   s = 0
    #   a = sum(rightlist)/len(rightlist)
    #   for i in l:
    #     s += ((i-a)**2)

    #   return (s/N)

    # print('MSE為',MSE(l=res,rightlist=y_test))





# draw().predict_pic('tn',filetype='csv')
# print(draw().predict('tn'))
print(draw().predict_sq('tp'))





# 未使用到的程式碼

 # def predict_pic(self,area,filetype='json'):
  #   """
  #   area -> tn , tp
  #   filetype -> json,csv
  #   """
  #   import pandas as pd
  #   import matplotlib.pyplot as plt

  #   if filetype == "json":
  #     if area == 'tn' :x = data().data_tn_all();pre =data().pre('tn');n='台南';
  #     elif area == 'tp':x = data().data_tp_all();pre=data().pre('tp');n='台北';
  #     else: return 'area not found'
  #   elif filetype == 'csv':
  #     x = data().data_csv_all(area)
  #     n = '台北' if area == 'tp' else '台南'
  #     plt.rcParams['font.family'] = 'Microsoft JhengHei'
  #   try:
  #     y1 = x['住宅部門售電量(度)'] ; y2 = x['平均氣溫']
  #   except:
  #     print('using csv data')
  #     y1 = x['electricity'] ; y2 = x['temperature']
  #   try:
  #     x = x['日期']
  #   except:
  #     x = x['date']

  #   pre = data().pre(area)


  #   fig , ax1 = plt.subplots()
  #   plt.title('{}電力與溫度趨勢圖含預測'.format(n))
  #   plt.xlabel('時間')
  #   ax2 = ax1.twinx()
  #   ax3 = ax1.twinx()

  #   ax1.set_ylabel('電量使用')
  #   ax1.plot(x,y2)
  #   ax1.tick_params(axis='y')

  #   ax2.set_ylabel('氣溫')
  #   ax2.plot(x,y1,color='r')
  #   ax2.tick_params(axis='y')

  #   ax3.plot(x,pre,color='y')
  #   plt.legend([ax1,ax2],['電量使用','氣溫'])
  #   fig.tight_layout() ; plt.show()