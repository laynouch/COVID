
#Concatenating DataFrames
Datasub=pd.read_csv("submission.csv")

frames= [DataTs,Datasub]
result =  pd.concat(frames)
#print(result)
#print(result.head())
#print(result.tail())
result.drop('Province_State', inplace=True, axis=1)
result.set_index('ForecastId', inplace=True)

print(result)
print(result.head())
print(result.tail())

'''DataTr=np.random.random(100)
print(sum(DataTr))'''
print("Statisques")
'''DataTr=np.amin(DataTr)
DataTr=np.amax(DataTr)
DataTr=np.median(DataTr)'''

'''DataTr=np.random.random(100)
print(sum(DataTr))'''
print("Statisques")