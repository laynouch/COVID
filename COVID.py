import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
import seaborn as sns
'''import the data'''
DataTr=pd.read_csv("train.csv")
DataTs=pd.read_csv("test.csv")
Dataset=pd.concat([DataTr, DataTs], axis=0)
Dataset.reset_index(drop=True)
print(Dataset)

DataTr.head()
DataTr.info()
DataTr.tail()
print(DataTr.corr())
print(np.std(DataTr)) 
print("----------------------------------------")
print(DataTr.isnull().sum()) #20700 cellules de la column province state est vide
b=DataTr['Country_Region'].nunique()
print(b)
#184 than 
#Cleaning the data 'train','Test'

to_drop=['province_state']
DataTr.drop('Province_State', inplace=True, axis=1)
DataTs.drop('Province_State', inplace=True, axis=1)

#Convert the date coolumn into datetime

DataTr['Date']=pd.to_datetime(DataTr['Date'])

#use the date as an index in our dataframe

DataTr.set_index('Date', inplace=True)


dfC=pd.pivot_table(DataTr,columns='Country_Region',
                     index='Date',
                     values='ConfirmedCases',
                     aggfunc=np.sum)
print(dfC)

dfF=pd.pivot_table(DataTr,
                              columns='Country_Region',
                              index='Date',
                              values='Fatalities',
                              aggfunc=np.sum)
print(dfF)                              
country_list=[]
value_list=[]
fatality_list=[]

for country in list(dfC.columns):
    country_list.append(country)
    value_list.append(dfC[country].max())
    fatality_list.append(dfF[country].max())
    new={'Country':country_list,'Confirmed':value_list,'Fatality':fatality_list}

df=pd.DataFrame.from_dict(new)
df.set_index('Country',inplace=True)
print("00000000000000000000000000000000000000000000000")
print(df) #data fiha country + confirmedcases +fatalities



#Visualisations 

#descending order of coutries ' confirmedcases'
or_conf = df.sort_values(by=['Confirmed'],ascending=False)
or_fat = df.sort_values(by=['Fatality'],ascending=False)
topconf = or_conf.head(10)
topFat=or_fat.head(10)
print("________________________________________________________")
print(topconf)
print(topFat)
#descending order of coutries ' Fatilities'


#ConfirmedCases by Country
fig, ax = plt.subplots()

plt.bar(topconf.index,topconf['Confirmed'],color=['black','brown','blue', 'blueviolet', 'aqua','orange', 'cornflowerblue', 'yellow','green','red'] ) 
             
plt.title('ConfirmedCases by Country')
plt.xlabel('Country')
plt.ylabel('ConfirmedC')
plt.show()

#Fatilities by Country
plt.bar(topFat.index,topFat['Fatality'], color=['black','brown','blue', 'blueviolet', 'aqua','orange', 'cornflowerblue', 'yellow','green','red']) 

plt.title('Fatality by Country')
plt.xlabel('Country')
plt.ylabel('Fatality')
plt.show()


#we want to add india and pakistane 'countries which has intereset' 'very high nbr of peoples'
listC=list(topconf.index)
listC.append('India')
listC.append('Pakistan')
print("--------------------------------------------------")
print(listC)

#confirmedCases by dates
#Seaborn line chart visualisation





list_countries=list(topconf.index)
list_countries.append('India')
list_countries.append('Pakistan')

Confirm_pivot=pd.pivot_table(DataTr,index='Date',columns='Country_Region',
                             values='ConfirmedCases',aggfunc=np.sum)
Confirm_pivot


plt.figure(figsize=(10,6))
colors=['r','b','g','y','orange','purple','m','hotpink','violet','darkgreen','navy','brown']
for i,country in enumerate(list_countries):
    Confirm=Confirm_pivot[Confirm_pivot[country]>0][country].diff().fillna(0)
    Confirm=Confirm[Confirm>0]
    Confirm.plot(color=colors[i],label=country,markersize=8,lw=3)   
    plt.title('Number of Daily Cases',fontsize=15)
    plt.legend(title='country')
plt.tight_layout()
plt.show()