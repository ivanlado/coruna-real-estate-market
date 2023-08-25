import pandas as pd
import numpy as np
from bd import *

def function_replace(x, data_municipios):
    mun = x['municipio']
    dir_x = x['direccion_x']
    dir_y = x['direccion_y']
    if pd.isna(dir_x):
        coor_x = data_municipios.loc[data_municipios['Municipio']==mun, 'direccion_x'].values[0]
        x['direccion_x'] = coor_x
    if pd.isna(dir_y):
        coor_y = data_municipios.loc[data_municipios['Municipio']==mun, 'direccion_y'].values[0]
        x['direccion_y'] = coor_y
    return x

def get_distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def normalize_str(x):
    if x is not None:
        x = x.lower()
        x = x.replace('á', 'a')
        x = x.replace('é', 'e')
        x = x.replace('í', 'i')
        x = x.replace('ó', 'o')
        x = x.replace('ú', 'u')
    return x

def get_zona(x, zonas, municipios):
    titulo = x['titulo'].lower()
    municipio = x['municipio'].lower()
    for z in zonas:
        if z in titulo:
            return z
    for z in zonas:
        if '-' in z:
            zn = z.split('-')
            for zi in zn:
                zi = zi.strip()
                if zi in titulo:
                    return z
    return municipio

def get_data_viviendas():
    db = SQLDatabase(user="postgres", password="1111", host="localhost", port="5432", db_name="corunaRealEstateMarket")
    db_con = db.get_connection()
    db_cursor = db.get_cursor()
    data_viviendas = db.get_data_v1(cols = cols_1)
    db.disconnect()
    return data_viviendas

def get_data_municipios():
    data_municipios = pd.read_csv('./data/ayuntamientos/municipios.csv')
    data_municipios = data_municipios.rename(columns={'Superficie(km²)': 'superficie', 'Densidad(h/km²)(2018)': 'densidad', 'PIB 2016':'pib'})
    data_municipios['Municipio'] = data_municipios['Municipio'].apply(lambda x: normalize_str(x))
    data_municipios['pib_capita'] = data_municipios['pib'] / data_municipios['Población(2015)']
    # list municipios names
    municipios = data_municipios['Municipio'].values.tolist()
    new_municipios = ['a coruña', 'a coruna', 'la coruña', 'la coruna', 'arteixo', 'arteijo']
    municipios.extend(mun for mun in new_municipios if mun not in municipios)
    return data_municipios, municipios

def get_data_zonas():
    data_zonas = pd.read_csv('./data/zonas/zonas.csv')
    data_zonas['zona'] = data_zonas['zona'].apply(lambda x: normalize_str(x))
    data_zonas['municipio'] = data_zonas['municipio'].apply(lambda x: normalize_str(x))
    zonas = data_zonas['zona'].values.tolist()
    zonas = [x.lower() for x in zonas]
    return data_zonas, zonas

def get_data_model_v1():

    data_viviendas = get_data_viviendas()
    data_viviendas['titulo'] = data_viviendas['titulo'].apply(lambda x: normalize_str(x))
    data_viviendas['descripcion'] = data_viviendas['descripcion'].apply(lambda x: normalize_str(x))
    data_viviendas['extra_info'] = data_viviendas['extra_info'].apply(lambda x: normalize_str(x))
    data_viviendas['direccion'] = data_viviendas['direccion'].apply(lambda x: normalize_str(x))
    data_viviendas['landmarks_cercanos'] = data_viviendas['landmarks_cercanos'].apply(lambda x: normalize_str(x))
    data_viviendas['municipio'] = data_viviendas['municipio'].apply(lambda x: normalize_str(x))
    data_municipios, municipios_list = get_data_municipios()
    data_zonas, zonas_list = get_data_zonas()

    data_viviendas['zona'] = data_viviendas.apply(lambda x: get_zona(x, zonas_list, municipios_list), axis=1)

    # replace nan coordinates by the coordinates of the municipio
    data_viviendas = data_viviendas.apply(lambda x: function_replace(x, data_municipios), axis=1)
    data = pd.merge(data_viviendas, data_municipios[['Municipio', 'densidad', 'pib_capita']], how='left', left_on='municipio', right_on='Municipio')
    data = data.drop(columns=['Municipio'], axis=1)
    data['piscina'] = data['descripcion'].str.contains('piscina').astype(bool)
    data['parking'] = data['extra_info'].str.contains('piscina') | data['descripcion'].str.contains('plaza de garaje') | data['descripcion'].str.contains('plaza garaje').astype(bool)
    data['estudiantes'] = data['descripcion'].str.contains('estudiante') | data['descripcion'].str.contains('escolar').astype(bool)
    data['playa'] = data['descripcion'].str.contains('playa').astype(bool)
    data['balcon'] =  data['descripcion'].str.contains('balcon') | data['descripcion'].str.contains('terraza').astype(bool)
    data['trastero'] = data['descripcion'].str.contains('trastero').astype(bool)
    data['vacacional'] = data['descripcion'].str.contains('vacacion').astype(bool)
    data['vistas'] = data['descripcion'].str.contains('vistas').astype(bool)
    data['sin_ascensor'] = data['descripcion'].str.contains('sin ascensor') | data['extra_info'].str.contains('sin ascensor').astype(bool)
    data['profesores'] = data['descripcion'].str.contains('profesor').astype(bool)
    data['amueblado'] = data['descripcion'].str.contains('amueblad').astype(bool)
    data['es_casa'] = data['titulo'].apply(lambda x : 1.0 if(x.lower().startswith('casa') or x.lower().startswith('chalet')) else 0.0)
    data['lujo'] = data['descripcion'].str.contains('lujo') | data['titulo'].str.contains('lujo')
    # data['urbanizacion'] = data['descripcion'].str.contains('urbanizacion') | data['titulo'].str.contains('urbanizacion')
    # data['reformado'] = data['descripcion'].str.contains('reformado') | data['titulo'].str.contains('reformado')
    # data['ubicacion_excepcional'] = data['descripcion'].str.contains('ubicacion excepcional') | data['descripcion'].str.contains('buena situacion')
    # data['duplex'] = data['descripcion'].str.contains('duplex') | data['titulo'].str.contains('duplex')
    # data['atico'] = data['descripcion'].str.contains('atico') | data['titulo'].str.contains('atico') | data['descripcion'].str.contains('ático') | data['titulo'].str.contains('ático')
    # data['villa'] = data['descripcion'].str.contains('villa') | data['titulo'].str.contains('villa')
    # data['climalit'] = data['descripcion'].str.contains('climalit')
    # data['aire_acondicionado'] = data['descripcion'].str.contains('aire acondicionado')
    # data['sin muebles'] = data['descripcion'].str.contains('sin muebles') | data['descripcion'].str.contains('sin amueblar')
    # data['electrico'] = data['descripcion'].str.contains('electrico')
    # data['transporte'] = data['descripcion'].str.contains('estacion') | data['descripcion'].str.contains('parada')
    # data['gastos_incluidos'] = data['descripcion'].str.contains('gastos') & data['descripcion'].str.contains('incluidos')
    data['estrenar'] = data['descripcion'].str.contains('estrenar') | data['titulo'].str.contains('estrenar')
    coruna_x = data_municipios.loc[data_municipios['Municipio']=='a coruña', 'direccion_x'].values[0]
    coruna_y = data_municipios.loc[data_municipios['Municipio']=='a coruña', 'direccion_y'].values[0]
    data['distancia_centro_coruna'] = data.apply(lambda x: get_distance(x['direccion_x'], x['direccion_y'], coruna_x, coruna_y), axis=1)
    oleiros_x = data_municipios.loc[data_municipios['Municipio']=='oleiros', 'direccion_x'].values[0]
    oleiros_y = data_municipios.loc[data_municipios['Municipio']=='oleiros', 'direccion_y'].values[0]
    data['distancia_centro_oleiros'] = data.apply(lambda x: get_distance(x['direccion_x'], x['direccion_y'], oleiros_x, oleiros_y), axis=1)

    return data

def get_data_model_v2():
    ds = get_data_model_v1()
    cols = ds.columns
    cols_exclude = ['id', 'titulo', 'descripcion', 'extra_info', 'direccion', 'landmarks_cercanos', 'municipio']
    cols_check_nas = ['n_banos', 'n_plazas_garaje', 'valoracion']
    for col in cols_check_nas:
        if sum(ds[col]==-1)/len(ds) > 0.25:
            print(f'Column {col} has more than 20% of missing values.')
            cols_exclude.append(col)
    cols_model = [col for col in cols if col not in cols_exclude]
    #convert int and bool columns to float
    for col in cols:
        if ds[col].dtype == 'int64':
            ds[col] = ds[col].astype('float64')
        elif ds[col].dtype == 'object':
            cols_exclude.append(col)
        elif ds[col].dtype == 'bool':
            ds[col] = ds[col].astype('float64')
    for col in cols_check_nas:
        ds[col] = ds[col].fillna(np.mean(ds[col]))
    ds['coruna'] = ds["municipio"].apply(lambda x: 1.0 if 'coru' in x else 0.0)
    ds['oleiros'] = ds["municipio"].apply(lambda x: 1.0 if 'oleiros' in x else 0.0)
    ds['art_berg_camb'] = ds["municipio"].apply(lambda x: 1.0 if ('artei' in x or 'berg' in x or 'cambr' in x) else 0.0)
    cols_model.append('coruna')
    cols_model.append('oleiros')
    return ds, cols_model