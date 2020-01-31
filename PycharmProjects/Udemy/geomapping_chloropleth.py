import pandas as pd
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, plot, iplot
'''
df = pd.read_csv('https://raw.githubusercontent.com/srijan-das/mlaicrc/master/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2014_World_Power_Consumption')

data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        reversescale = True,
        locations = df['Country'],
        locationmode = "country names",
        z = df['Power Consumption KWH'],
        text = df['Country'],
        colorbar = {'title' : 'Power Consumption KWH'},
      ) 

layout = dict(title = '2014 Power Consumption KWH',
                geo = dict(showframe = False,projection = {'type':'mercator'})
             )

choromap = go.Figure(data = [data],layout = layout)
plot(choromap,validate=False)
'''
'''
df = pd.read_csv('https://raw.githubusercontent.com/srijan-das/mlaicrc/master/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2012_Election_Data')

data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        reversescale = True,
        locations = df['State Abv'],
        locationmode = 'USA-states',
        z = df['Voting-Eligible Population (VEP)'],
        text = df['State'],
        colorbar = {'title' : 'VEP'},
      ) 

layout = dict(title = '2012 Voting-Eligible Population',
                geo = dict(showframe = False,projection = {'type':'mercator'})
             )

choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
'''
'''
usdf = pd.read_csv('https://raw.githubusercontent.com/srijan-das/mlaicrc/master/PycharmProjects/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/09-Geographical-Plotting/2012_Election_Data')
data = dict(type='choropleth',
            colorscale = 'Viridis',
            reversescale = True,
            locations = usdf['State Abv'],
            z = usdf['Voting-Age Population (VAP)'],
            locationmode = 'USA-states',
            text = usdf['State'],
            marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),
            colorbar = {'title':"Voting-Age Population (VAP)"}
            )

layout = dict(title = '2012 General Election Voting Data',
              geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )

choromap = go.Figure(data = [data],layout = layout)
plot(choromap,validate=False)
'''