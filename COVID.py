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
print("________________________________________________________________")
print(df) #data fiha country + confirmedcases +fatalities



#Visualisations 

#descending order of coutries ' confirmedcases' 'Fatality'
or_conf = df.sort_values(by=['Confirmed'],ascending=False)
or_fat = df.sort_values(by=['Fatality'],ascending=False)
topconf = or_conf.head(10)
topFat=or_fat.head(10)
print("________________________________________________________")
print(topconf)
print(topFat)





#ConfirmedCases by Country
fig, ax = plt.subplots()

plt.bar(topconf.index,topconf['Confirmed'],color=['black','brown','blue', 'blueviolet', 'aqua','orange', 'cornflowerblue', 'yellow','green','red'] ) 
             
plt.title('ConfirmedCases by Country', size='16')

plt.xlabel('Country', size='16')
plt.ylabel('ConfirmedC', size='16')
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
listC.append('China') #the beginnig of the covid19
print("--------------------------------------------------")
print(listC)

#Cumulative trend plot for Confirmed Cases
print("___________________________________________________________")
timesercntr = DataTr.groupby(['Date','Country_Region'])['ConfirmedCases'].sum()\
                    .reset_index().set_index('Date')
dfcountries= timesercntr[timesercntr['Country_Region'].isin(listC)]


plt.figure(figsize=(16,12))
ax = sns.lineplot(x=dfcountries.index, 
                  y="ConfirmedCases", 
                  hue="Country_Region", 
                  data=dfcountries,palette='muted').set_title('Cumulative line')

plt.legend(loc=2, prop={'size': 16})
plt.title('Cumulative trend plot for Confirmed Cases')
plt.xticks(rotation=90)
plt.show()


#Daily cases
plt.figure(figsize=(10,6))
colors=['black','brown','blue', 'blueviolet', 'aqua','orange', 'cornflowerblue', 'yellow','green','red','navy','purple','hotpink']
for i,country in enumerate(listC):
    Dcases=dfC[dfC[country]>0][country].diff().fillna(0)
    Dcases=Dcases[Dcases>0]
    Dcases.plot(color=colors[i],label=country,markersize=8,lw=3)   
    plt.title('Daily Cases',fontsize=20)
    plt.legend(title='country')
plt.tight_layout()
plt.show()

#daily cases of each country (top 10)+india pakistane and chaina

plt.figure(figsize=(20,16))
colors=['black','brown','blue', 'blueviolet', 'aqua','orange', 'cornflowerblue', 'yellow','green','red','navy','purple','hotpink']
for i,country in enumerate(listC):
    Dcases=dfC[dfC[country]>0][country].diff()
    Dcases=Dcases[Dcases>0]
    plt.subplot(5,4,i+1)
    Dcases.plot(color=colors[i],label=country,markersize=20,lw=4)    
    plt.xticks()
    plt.legend(title='Country')
    print(end='')
    plt.title('Number of Daily Cases in {}'.format(country.upper()))
plt.tight_layout()
plt.show()


#correlation between confirmed cases and fatalities

ConfC=DataTr.groupby('Date')['ConfirmedCases'].sum()
Fatali=DataTr.groupby('Date')['Fatalities'].sum()
fig, ax = plt.subplots()
ax.scatter(x=ConfC , y= Fatali ,color='r')
ax.set_title('Correlation between confirmedcases and fatalities')
ax.set_xlabel('ConfirmedCases')
ax.set_ylabel('Fatalities')
plt.show()


