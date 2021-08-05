# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 22:00:00 2021

@author: Angelo
"""

"""
Clase 2 - Pandas

En la clase inicial realizamos una introducción general a python, donde aprendimos distintos 
tipos de objetos, creación de variables, funciones, operaciones, etc. En esta clase aprenderemos 
a utilizar el modulo Pandas, el cual consiste en una librería open-source que permite realizar manipulación 
y análisis de datos estructurados (tablas, dataframe).

Dentro de los conceptos que revisaremos durante este modulo serán:

    Creación e importación de dataframes
    Índices
    Selección de variables y/o observaciones
    Vista y operación de dataframes
    Principales métodos de un dataframe
    Métodos de agregación y merge
"""

import numpy as np
import pandas as pd

#Carga de datos
first_import = pd.read_csv('C:/Users/Angelo/Dropbox/Mi PC (vngel0-PC)/Desktop/Diplomado DS PUC/2_ H. computacionales y machine learning/Computacional/R/Esteban/anime.csv',delimiter=';')

#Visualización de base cargada
first_import.head(2)

#Resumen del índice
first_import.index

#Asignación de índice a partir de una columna de la base
first_import.set_index('anime_id', inplace = True)

#Carga de datos con índice asignado por una columna de la base
first_import = pd.read_csv('C:/Users/Angelo/Dropbox/Mi PC (vngel0-PC)/Desktop/Diplomado DS PUC/2_ H. computacionales y machine learning/Computacional/R/Esteban/anime.csv', delimiter = ';', index_col=['anime_id', 'type'])

#Carga de datos con índice fecha
fechas_index = pd.read_csv('C:/Users/Angelo/Dropbox/Mi PC (vngel0-PC)/Desktop/Diplomado DS PUC/2_ H. computacionales y machine learning/Computacional/R/Esteban/SPfuture.csv', index_col = 'Date', parse_dates = True)

#Selección de columna con loc
first_import.loc[:, 'name']

#Selección de columna
first_import['name']

#Selección de columna
first_import.name

#Selección de columnas con loc
first_import.loc[:,['name','episodes']]

#Selección de columnas menos una
first_import.loc[:, first_import.columns != 'name']

#Selección de columnas, es mejor con drop al eliminar más de una
first_import.drop(['name', 'genre', 'members'], axis='columns')

#Seleccionar las 3 primeras filas con loc
first_import.loc[[32281, 5114, 28977], :]

#Seleccionar las 3 primeras filas con iloc
first_import.iloc[0:3, :]

#Seleccionar 2 columnas con iloc
first_import.iloc[:,[0,2]]

#Seleccionar las primeras 10 filas de 2 columnas
first_import.iloc[0:10,[0,2]]

#Selección columnas a partir de sus nombres
first_import[['name','episodes']]

#Seleccionar las 1 primeras filas de 2 columnas
first_import[['name','episodes']][0:10]

#Seleccionar las 9 primeras filas de 3 columnas
first_import.loc[0:9,['name','episodes', 'anime_id']]

#Selección por criterio, genera lista true or false
criterio = first_import['episodes'] <= 15

#Selección por criterio, genera lista true or false
criterio2 = first_import['type'] == 'TV'

#Selección por 2 criterios definidos anteriormente
first_import[criterio & criterio2]['rating'].mean()

#Selección por criterio a partir de query
first_import.query("episodes <= 15 & type == 'TV'")['rating'].mean()

#Selección de valores únicos
first_import['type'].unique()

#Recuento de valores únicosd
first_import['type'].nunique()

#Obtención de valores según categoría
first_import['type'].value_counts()

#Ordenar según algún criterio
first_import.sort_values('episodes',ascending = True)

#Selección por criterio
first_import[first_import['type'] == 'Movie']

#Creación de columna, transformando una
first_import['rating10'] = first_import['rating']*10

first_import['rating_str'] = first_import['rating10'].astype(str)

#Creación de columna, creando variable
first_import['ratio'] = first_import['episodes'] / first_import['episodes'].max()

#Str, resumen de dataframe
first_import.info()

# Estadísticas descriptivas por culumnas
first_import.describe(include = np.number).round(1)

#Promedio de una columna
first_import['rating'].mean()

#Desviación estándar de una columna
first_import['rating'].std()

#Mínimo de una columna
first_import['rating'].min()

#Cuantiles de una columna
first_import['rating'].quantile(q = 0.25)

#Método apply
def multiplica_10(value):
    res = value * 10
    return res

def porcentaje(value):
    res = str(value) + ' %'
    return res

def multiplica_N(value, N):
    return value * N

def multiplica_rating_episodes(row):
    return row['rating'] * row['episodes']

#Aplica la función a lo largo de todas las filas
first_import['rating'].apply(multiplica_10)

#Se puede aplicar multiples apply
first_import['rating'].apply(multiplica_10).apply(porcentaje)

#Aplica la función en las filas seleccionadas
list(map(lambda x: x*10,first_import['rating']))[0:4]

#Operación vectorizada
(first_import['rating'] * 10).astype(str) + ' %'

#Duda
first_import.apply(multiplica_rating_episodes, axis='columns')

#Creación de df
df = pd.DataFrame({
        'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
        'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
        'rating': [4, 4, 3.5, 15, 5]
                    })

#Visualización del df
display(df)
df.head()

#Eliminación de duplicados, valores en todas las columnas
df[~df.duplicated()]
df.drop_duplicates()

#Eliminación de duplicados omitiendo una columna
df.drop_duplicates(subset='style')

#Carga de base con valores NA
anime_na = pd.read_csv("C:/Users/Angelo/Dropbox/Mi PC (vngel0-PC)/Desktop/Diplomado DS PUC/2_ H. computacionales y machine learning/Computacional/R/Esteban/animeNA.csv", delimiter = ';')

#Observar datos faltantes
anime_na.info()
anime_na.tail(5)

#Identificar cuantos datos NA por columna hay
anime_na.isna().sum()

#Eliminar todas las filas con almenos un NA
anime_na = anime_na.dropna()
anime_na.info()

#Duda
anime_na['episodes'][anime_na['episodes'] == 'Unknown'] = None # np.nan
anime_na.info()

#Agrupación de categorías de una columna
data_agrupada = anime_na.groupby('type')

#Es necesario usar una función agg para aprovechar groupby
data_agrupada.agg('count')

anime_na.groupby('type').agg(['mean','std']).round(1)

def extension(value):
    if value <= 15:
        res = 'No extenso'
    else:
        res = 'extenso'
    return res

#Duda
anime_na['duracion'] = anime_na['episodes'].apply(extension)
anime_na.head()

#Creación de df
df1 = pd.DataFrame({'Nombre': ['Juan', 'Jorge', 'Lisa', 'Susana'],
                    'Profesion': ['Contador', 'Ingeniero', 'Ingeniero', 'Psicologo']})

df2 = pd.DataFrame({'Nombre': ['Lisa', 'Juan', 'Jorge', 'Susana'],
                    'Contratacion': [2004, 2008, 2012, 2014]})

display(df1)
display(df2)

#Se usa merge para unir 2 bases de datos a partir de un parámetro común
pd.merge(df1,df2)

#En caso de que la llave no tenga el mismo nombre, se usa el parámetro 'on'
df2 = pd.DataFrame({'Empleado': ['Lisa', 'Juan', 'Jorge', 'Susana'],
                    'Contratacion': [2004, 2008, 2012, 2014]})

pd.merge(df1,df2, left_on='Nombre',right_on='Empleado')

df1 = pd.DataFrame({'Nombre': ['Federico', 'Jorge', 'Lisa', 'Susana'],
                    'Profesion': ['Contador', 'Ingeniero', 'Ingeniero', 'Psicologo']})

df2 = pd.DataFrame({'Nombre': ['Lisa', 'Pedro', 'Jorge', 'Susana'],
                    'Contratacion': [2004, 2008, 2012, 2014]})

display(df1)
display(df2)

pd.merge(df1,df2)

#Agregando al df1 (izq de la fórmula) lo que tiene en común el 2
pd.merge(df1,df2,how = 'left')

#Agregando al df2 (der de la fórmula) lo que tiene en común el 1
pd.merge(df1,df2,how = 'right')

#Mezcla ambos df
pd.merge(df1,df2,how = 'outer')

#Creación de df
temp_max = pd.DataFrame({
                    'Ciudad': ['Santiago', 'Talca', 'Concepcion', 'Arica'],
                    'Temperatura': [20,16,15,26],
                    'x': [1, 2, 3, 4]})

temp_min = pd.DataFrame({
                    'Ciudad': ['Santiago', 'Talca', 'Concepcion', 'Arica'],
                    'Temperatura': [12, 8, 2, 20],
                    'y': [2, 3, 4, 5]})

display(temp_max)
display(temp_min)

#Uniendo por la llave
pd.merge(temp_max,temp_min, on = 'Ciudad')

#Renombrando variables con sufijos
pd.merge(temp_max,temp_min,on = 'Ciudad', suffixes=['_maxima','_minima'])

#Uniendo con comando join
display(temp_max.join(temp_min, lsuffix='Ciudad', rsuffix='Ciudad'))

#Uniendo con comando concat
display(pd.concat([temp_max,temp_min], axis=1))

"""
Ejercicios

- state-population.csv
- state-area.csv
- state-abbrevs.csv

Con estos archivos realice los siguientes ejercicios:
- 1) Cargue los tres datasets utilizando la función `read_csv` de pandas
- 2) Revise los datos cargados y realice las modificaciones necesarias para empezar 
    a trabajar
- 3) Elimine las columnas que no aportan información
- 4) Agrupe por state/region y ages, para luego agregar usando promedio, minimo, máximo 
    y contar
- 5) Forme un nuevo dataset llamado df que tenga la información de los 3 sets importados. 
    Use las llaves necesarias.
- 6) Calcule la población por area para cada estado. Para esto agrupe por state/region y 
    calcule el promedio de la población. Guarde dicha información en un data frame con 
    las columnas 'state' y 'pop/area' (hint: puede extraer los valores de un objeto groupby, 
                                       finalizando su sentencia con .values, o extraer sus 
                                       etiquetas con .index)
- 7) Finalmente, cree un dataframe que posea: El estado, el área, la población, la población 
    por área y como índice  la abreviación del nombre del estado.

"""
import numpy as np
import pandas as pd

#Carga de df1
population = pd.read_csv("C:/Users/Angelo/Dropbox/Mi PC (vngel0-PC)/Desktop/Diplomado DS PUC/2_ H. computacionales y machine learning/Computacional/R/Esteban/state-population.csv", delimiter=(';'))
population.columns
#Selección de columnas con información para pregunta 3
population = population.loc[:,['state/region', 'ages', 'year', 'population']]

#Carga de df2
area = pd.read_csv("C:/Users/Angelo/Dropbox/Mi PC (vngel0-PC)/Desktop/Diplomado DS PUC/2_ H. computacionales y machine learning/Computacional/R/Esteban/state-area.csv", delimiter=(';'))
area.columns
#Selección de columnas con información para pregunta 3
area = area.loc[:,['state','area (sq. mi)']]

#Carga de df3
abbrevs = pd.read_csv("C:/Users/Angelo/Dropbox/Mi PC (vngel0-PC)/Desktop/Diplomado DS PUC/2_ H. computacionales y machine learning/Computacional/R/Esteban/state-abbrevs.csv", delimiter=(';'))
abbrevs.columns
#Selección de columnas con información para pregunta 3
abbrevs = abbrevs.loc[:,['state', 'abbreviation']]

#Agrupación por columnas y extracción de indicadores para pregunta 4
p4 = population.groupby(['state/region','ages']).agg(['max','min','sum'])

#iteración 1 para pregunta 5
i1 = pd.merge(abbrevs,area, on = 'state')
i1.columns

#Iteración 2 y solución de pregunta 5
p5 = pd.merge(i1,population, left_on = 'abbreviation', right_on = 'state/region').drop('abbreviation', axis = 1)

#Agrupación para pregunta 6 
p6 = p5.groupby(['state/region']).agg('mean')
p6 = p6.iloc[:,2]

#Pregunta 7
p7 = pd.merge(p5,p6, on = 'state/region', suffixes=['_prop',''], how = 'right')
p7.columns
p7 = p7.loc[:, ['state','area (sq. mi)','population','population_prop']]
