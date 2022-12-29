import numpy as np
import pandas as pd 
import matplotlib.pylab as plt
import seaborn as sns
'''import the data'''
print("TRAIN")
DataTr=pd.read_csv("train.csv")
#Before Cleaning the Data
print(DataTr.shape)
print(DataTr.columns)
print(DataTr.head())
print(DataTr.tail())
print(DataTr.info())
print(DataTr.describe())
print(DataTr.corr())
print(DataTr['Country_Region'].describe())
#change Id 
DataTr.set_index('Id', inplace=True)
#
to_drop=['province_state']
DataTr.drop('Province_State', inplace=True, axis=1)
#After Cleaning the Data
print(DataTr.shape)
print(DataTr.columns)
print(DataTr.head())
print(DataTr.info())
print(DataTr.describe())
print(DataTr.tail())
print(DataTr.corr())
print(DataTr['Country_Region'].describe())
#TEST

print("TEST")
DataTs=pd.read_csv("test.csv")

print(DataTs.shape)
DataTs.set_index('ForecastId', inplace=True)

DataTs.drop('Province_State', inplace=True, axis=1)
print(DataTs.head())
print(DataTs.shape)
print(DataTs.columns)
print(DataTs.tail())
print(DataTs.corr())

#Visualisation
DataTr.plot(kind="scatter",x='ConfirmedCases',y='Fatalities')


# create a figure and axis
fig, ax = plt.subplots()

# scatter the confirmedCases against the Fatalities
ax.scatter(DataTr['ConfirmedCases'], DataTr['Fatalities'])
ax.set_title('Covid19')
ax.set_xlabel('confirmedCases')
ax.set_ylabel('Fatalities')
plt.show()

#Country_Region / ConfirmedCases  tji b l2a3mida


#Pairplot
sns.set_style("whitegrid")
sns.pairplot(DataTr,hue="Country_Region",height=5)
plt.show()
#plt.savefig ('D:\GL M1\python 2', dpi=500)

plt.bar(df['variety'],df['sepal.length'],color =
['cornflowerblue'
,
'lightseagreen'
,
'steelblue'])


# get columns to plot
columns = df.columns.drop(['variety'])
# create x data
x_data = range(0, df.shape[0])
# create figure and axis
fig, ax = plt.fig subplots()
# plot each column
for column in columns:
ax.plot(x_data, df[column])
# set title and legend
ax.set_title('Iris Dataset')
ax.legend()
plt.show()


ime= DataTr.groupby(['Date','Country_Region'])['ConfirmedCases'].sum()\
                    .reset_index().set_index('Date')
dftime = time[time['Country_Region'].isin(listC)]

plt.figure(figsize=(16,12))
ax = sns.lineplot(x=dftime.index, 
                  y="ConfirmedCases", 
                  hue="Country_Region", 
                  data=dftime,palette='muted').set_title('Cumulative line')

plt.legend(loc=2, prop={'size': 16})
plt.title('Cumulative trend plot for Confirmed Cases')
plt.xticks(rotation=90)
plt.show()
#pairplot
sns.set_style("whitegrid")
sns.pairplot(dfC,hue="Country_Region",height=3)

plt.show()


df=df.head(10)

df.plot(kind="scatter",x='ConfirmedC',y='Fatality')
plt.show()

# create a figure and axis
fig, ax = plt.subplots()

# scatter the sepal_length against the sepal_width
ax.scatter(df['ConfirmedC'], df['Fatality'])
# set a title and labels
ax.set_title('Iris Dataset')
ax.set_xlabel('sepal.length')
ax.set_ylabel('sepal.width')
plt.show()





pivot=pd.pivot_table(DataTr,columns='Country_Region',
                     index='Date',
                     values='ConfirmedCases',
                     aggfunc=np.sum)

pivot_fatality=pd.pivot_table(DataTr,
                              columns='Country_Region',
                              index='Date',
                              values='Fatalities',
                              aggfunc=np.sum)
country_list=[]
value_list=[]
fatality_list=[]

for country in list(pivot.columns):
    country_list.append(country)
    value_list.append(pivot[country].max())
    fatality_list.append(pivot_fatality[country].max())
    new_dict={'Country':country_list,'ConfirmedC':value_list,'Fatality':fatality_list}

df=pd.DataFrame.from_dict(new_dict)
df.set_index('Country',inplace=True)
import plotly.express as px
fig = px.bar(topconf, x=topconf.index, y='ConfirmedC', 
             labels={'x':'Country'}, color="ConfirmedC", 
             color_continuous_scale=px.colors.sequential.Rainbow_r)
fig.update_layout(title_text='Top 10 Confirmed COVID-19 cases by country')
fig.show()

fig = px.bar(topFat, x=topFat.index, y='Fatality', 
             labels={'x':'Country'}, color="Fatality", 
             color_continuous_scale=px.colors.sequential.Rainbow_r)
fig.update_layout(title_text='Top 10 Fatality COVID-19 cases by country')
fig.show()
