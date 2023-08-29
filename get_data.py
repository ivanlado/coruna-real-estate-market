import pandas as pd
import numpy as np
from bd import *
from sklearn.cluster import KMeans


def normalize_str(x):
    if x is not None:
        x = x.lower()
        x = x.replace('á', 'a')
        x = x.replace('é', 'e')
        x = x.replace('í', 'i')
        x = x.replace('ó', 'o')
        x = x.replace('ú', 'u')
    return x

co_1 = ['Monte Alto - Zalaeta - Atocha']
co_2 = ['Ciudad Vieja - Centro']
co_3 = ['Ensanche - Juan Florez']
co_4 = ['Riazor - Los Rosales']
co_5 = ['Agra del Orzán - Ventorrillo', 'Vioño']
co_6 = ['Os Mallos', 'Sagrada Familia']
co_7 = ['Cuatro Caminos - Plaza de la Cubela']
co_8 = ['Falperra-Santa Lucía']
co_9 = ['Paseo de los Puentes-Santa Margarita']
co_10 = ['Someso - Matogrande', 'Eirís']
co_11 = ['Mesoiro']
co_12 = ['Los Castros - Castrillón']
co_13 = ['elviña - a zapateira']
co = [co_1, co_2, co_3, co_4, co_5, co_6, co_7, co_8, co_9, co_10, co_11, co_12, co_13]

ol_1 = ['Nós', 'Dexo-Lorbé', 'Dorneda', 'Mera-Serantes', 'Liáns', 'Iñás']
ol_2 = ['Maianca']
ol_3 = ['Perillo']
ol_4 = ['Oleiros']
ol = [ol_1, ol_2, ol_3, ol_4]

zonas_seleccionadas = {'a coruña': co, 'oleiros': ol}
normalized_zonas = {}
for key, sublist in zonas_seleccionadas.items():
    normalized_sublist = []
    for subsublist in sublist:
        normalized_items = [normalize_str(item) for item in subsublist]
        normalized_sublist.append(normalized_items)
    normalized_zonas[key] = normalized_sublist
zonas_seleccionadas = normalized_zonas

zonas_seleccionadas_2 = {'a coruña': co, 'oleiros': ol}


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


def normalize_zonas_seleccionadas(x):
    for zonas in x.values():
        for zona in zonas:
            zona = normalize_str(zona)

def closest_zona(precio, df_sorted):
    precios = df_sorted['precio_m2'].values
    for i in range(len(precios)):
        if precios[i]  < precio:
            precio_ant = precios[max(i-1, 0)]
            precio_act = precios[i]
            if abs(precio-precio_ant) < abs(precio-precio_act):
                return df_sorted.loc[max(i-1, 0)]['zona_v2']
    return df_sorted.loc[len(precios)-1]['zona_v2']
    

def weighted_mean(group):
    n = group['n_viviendas']
    precio_m2 = group['precio_m2']
    aux = (n*precio_m2).sum()/n.sum()
    return(aux)
    

def cluster_zonas(df, k_clusters=None):
    viv_zonas = df.groupby('zona_v2').agg({'titulo':'count', 'precio_m2':'mean'}).reset_index()
    viv_zonas = viv_zonas.rename(columns={'titulo':'n_viviendas'})
    viv_zonas = viv_zonas.sort_values(['precio_m2'], ascending=False).reset_index(drop=True)
    viv_zonas_original = viv_zonas.copy()
    viv_zonas_excluidas = viv_zonas[viv_zonas['n_viviendas'] <5]
    viv_zonas = viv_zonas[viv_zonas['n_viviendas'] >= 5]
    viv_zonas = viv_zonas.reset_index(drop=True)
    innertia_vals = []
    
    # k-means clustering
    X = viv_zonas[['precio_m2']]
    num_clusters_list = [3, 4, 5, 6]

    for i, num_clusters in enumerate(num_clusters_list):
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        innertia_vals.append(kmeans.inertia_)
    min_index = np.argmin(np.diff(innertia_vals))
    best_k = num_clusters_list[0] + min_index + 1
    k = best_k if k_clusters is None else k_clusters
    kmeans = KMeans(n_clusters=k, n_init=10)
    viv_zonas['cluster'] = kmeans.fit_predict(X)

    viv_zonas_excluidas['cluster'] = viv_zonas_excluidas.apply(lambda x: viv_zonas[viv_zonas['zona_v2']==closest_zona(x['precio_m2'], viv_zonas)]['cluster'].values[0], axis=1)
    viv_zonas = pd.concat([viv_zonas, viv_zonas_excluidas]).sort_values(['precio_m2'], ascending=False).reset_index(drop=True)
    df['zona_cluster'] = df.apply(lambda x: viv_zonas[viv_zonas['zona_v2']==x['zona_v2']]['cluster'].values[0] if x['zona_v2'] is not None else None, axis=1)
    precio_m2_medio_cluster = viv_zonas.groupby('cluster').apply(lambda x: weighted_mean(x))
    df['precio_m2_medio_cluster'] = df.apply(lambda x: precio_m2_medio_cluster[x['zona_cluster']] if x['zona_cluster'] is not None else None, axis=1)
    
    return df, best_k


def get_zona(x, data_zonas, municipios):
    titulo = x['titulo'].lower()
    municipio = x['municipio'].lower()
    zonas = data_zonas[data_zonas['municipio'] == municipio]['zona'].values.tolist() 
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


def get_zona_v2(x):
    zona = x['zona']
    municipio = x['municipio']
    if municipio in zonas_seleccionadas.keys():
        for zonas in zonas_seleccionadas[municipio]:
            if zona in zonas:
                return zonas[0]
    else:
        return municipio

def is_centro_coruna(zona):
    centro_coruna = ['Monte Alto - Zalaeta - Atocha', 'Ensanche - Juan Florez', 'Ciudad Vieja - Centro']
    centro_coruna = [normalize_str(x) for x in centro_coruna]
    return True if zona in centro_coruna else False

def contar_palabras_en_texto(texto, palabras):
    count = 0
    if texto is not None:
        for p in palabras:
            if p in texto:
                count += 1
    return count
    

def precio_medio_zona(x, ds):
    '''
    Función que devuelve el precio por m2 medio de una zona de viviendas de tamaño y número de habitaciones similares
    '''
    min_items_grupo = 3
    id = x['id']
    hab = x['n_habitaciones']
    tam = x['tamano']
    zona_v2 = x['zona_v2']
    mun = x['municipio']
    r = 0.08*tam
    coor_x = x['direccion_x']
    coor_y = x['direccion_y']
    df = ds.copy()
    if hab <= 3:
        df = ds[(ds['n_habitaciones'] == hab)]
    elif hab == 4:
        df = ds[(ds['n_habitaciones'] ==3) | (ds['n_habitaciones'] ==4)]
    df = df[abs(df['tamano']-tam)<r]
    df = df[df['zona_v2'] == zona_v2]
    df = df[df['id'] != id]
    if len(df)>=min_items_grupo:
        df['precio_m2'] = df['precio']/df['tamano']
        return np.mean(df['precio_m2'])
    else:
        df = ds.copy()
        df = df[df['municipio'] == mun]
        if hab <= 3:
            df = df[(df['n_habitaciones'] == hab)]
        elif hab == 4:
            df = df[(df['n_habitaciones'] ==3) | (df['n_habitaciones'] ==4)]
        df = df[df['id'] != id]
        if len(df) >= min_items_grupo:
            df['precio_m2'] = df['precio']/df['tamano']
            return np.mean(df['precio_m2'])
        else: 
            df = ds.copy()
            df = df[df['municipio'] == mun]
            df = df[df['id'] != id]
            if(len(df)>=min_items_grupo):
                df['precio_m2'] = df['precio']/df['tamano']
                return np.mean(df['precio_m2'])
            else:
                df = ds[['precio', 'tamano', 'direccion_x', 'direccion_y', 'id']].copy()
                df = df[df['id']!=id]
                df['dist'] = df.apply(lambda x : get_distance(x['direccion_x'], x['direccion_y'], coor_x, coor_y), axis=1)
                df = df.sort_values('dist', ascending=True)
                df = df.head(7)
                df['precio_m2'] = df['precio']/df['tamano']
                return np.mean(df['precio_m2'])
    

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

def get_data_model_v1(k_clusters=None):

    data_viviendas = get_data_viviendas()
    data_viviendas['titulo'] = data_viviendas['titulo'].apply(lambda x: normalize_str(x))
    data_viviendas['descripcion'] = data_viviendas['descripcion'].apply(lambda x: normalize_str(x))
    data_viviendas['extra_info'] = data_viviendas['extra_info'].apply(lambda x: normalize_str(x))
    data_viviendas['direccion'] = data_viviendas['direccion'].apply(lambda x: normalize_str(x))
    data_viviendas['landmarks_cercanos'] = data_viviendas['landmarks_cercanos'].apply(lambda x: normalize_str(x))
    data_viviendas['municipio'] = data_viviendas['municipio'].apply(lambda x: normalize_str(x))
    data_municipios, municipios_list = get_data_municipios()
    data_zonas, zonas_list = get_data_zonas()

    data_viviendas['zona'] = data_viviendas.apply(lambda x: get_zona(x, data_zonas, municipios_list), axis=1)
    data_viviendas['zona_v2'] = data_viviendas.apply(lambda x: get_zona_v2(x), axis=1)
    data_viviendas['precio_m2'] = data_viviendas['precio']/data_viviendas['tamano']
    data_viviendas, best_k = cluster_zonas(data_viviendas, k_clusters)
    data_viviendas = data_viviendas.drop(columns=['precio_m2'], axis=1)
    print(f'Best k for clustering zones: {best_k}')
    ds_dummies = pd.get_dummies(data_viviendas, columns=['zona_cluster'], prefix='zona_cluster')
    data_viviendas = pd.concat([data_viviendas[['zona_cluster']], ds_dummies], axis=1)
    k = best_k if k_clusters is None else k_clusters
    for i in range(k):
        data_viviendas[f'zona_cluster_{i}'] = data_viviendas[f'zona_cluster_{i}'] * data_viviendas['tamano']

    # data_viviendas['precio_m2_medio_zona_v2'] = data_viviendas.apply(lambda x: precio_medio_zona(x, data_viviendas), axis=1)
    # data_viviendas['precio_medio_zona_v2'] = data_viviendas['precio_m2_medio_zona_v2'] * data_viviendas['tamano']
    # replace nan coordinates by the coordinates of the municipio
    data_viviendas = data_viviendas.apply(lambda x: function_replace(x, data_municipios), axis=1)
    data = pd.merge(data_viviendas, data_municipios[['Municipio', 'densidad', 'pib_capita']], how='left', left_on='municipio', right_on='Municipio')
    data = data.drop(columns=['Municipio'], axis=1)

    coruna_x = data_municipios.loc[data_municipios['Municipio']=='a coruña', 'direccion_x'].values[0]
    coruna_y = data_municipios.loc[data_municipios['Municipio']=='a coruña', 'direccion_y'].values[0]
    data['distancia_centro_coruna'] = data.apply(lambda x: get_distance(x['direccion_x'], x['direccion_y'], coruna_x, coruna_y), axis=1)
    data['distancia_centro_coruna*tam'] = (1-data['distancia_centro_coruna']) * data['tamano']
    oleiros_x = data_municipios.loc[data_municipios['Municipio']=='oleiros', 'direccion_x'].values[0]
    oleiros_y = data_municipios.loc[data_municipios['Municipio']=='oleiros', 'direccion_y'].values[0]
    data['distancia_centro_oleiros'] = data.apply(lambda x: get_distance(x['direccion_x'], x['direccion_y'], oleiros_x, oleiros_y), axis=1)
    data['distancia_centro_oleiros*tam'] = (1-data['distancia_centro_oleiros']) * data['tamano']

    so_far_cols = set(data.columns)
    valoracion_cols_1 = ['parking','balcon', 'trastero', 'vistas', 'amueblado', 'reformado', 'duplex', 'atico', 'climalit', 'aire_acondicionado', 'electrico', 'transporte', 'soleado', 'paseo_maritimo', 'equipado']
    valoracion_cols_2 = ['piscina', 'playa', 'ubicacion_excepcional', 'gastos_incluidos', 'estrenar', 'padel', 'gym']
    valoracion_cols_3 = ['lujo', 'villa', 'jardin', 'exclusivo', 'spa', 'es_chalet', 'urbanizacion', 'de_diseño']
    valoracion_cols_3_neg = ['sin_ascensor', 'sin_muebles']
    caros_cols_1 = ['ensueño', 'privileg', 'unic', 'privad', 'exclusiv', 'singular', 'lujo', 'jardin', 'urbanizacion', 'vist', 'piscin', 'centr', 'mejor zona', 'corazon']
    caros_cols_2 = ['primera linea',  'gimnasio', 'spa', 'solarium', 'padel', 'paddel', 'parcela', 'finca']
    caros_cols_3 = ['natural', 'verd', 'enclave', 'entorno', 'villa', 'cesped_natural', 'vestidor', 'duplex', 'conexion perfecta', 'comod', 'porteria', 'de diseño', 'domoti', 'calidad', 'alto estan', 'gran altura']
    caros_adjetivos = ['clasico', 'idilico', 'gran', 'ampli', 'extraordinari', 'magnifico', 'inmejorable', 'optim', 'emblematic']
    adjetivos = ['enorme', 'espectacular', 'excelente', 'empotrado', 'luminos', 'orientacion sur', 'relajante', 'fantastico', 'bonit', 'excelente', 'condiciones optimas', 'esplendid','fabuloso', 'tranquil', 'impresionante', 'fantastic', 'increible', 'hermos', 'espacio', 'precios']
    # begin boolean columns
    data['vacacional'] = data['descripcion'].str.contains('vacacion') | data['descripcion'].str.contains('verano') | data['descripcion'].str.contains('julio')  | data['descripcion'].str.contains('agosto') | data['descripcion'].str.contains('de temporada') | data['descripcion'].str.contains('quincena')
    data['piscina'] = data['descripcion'].str.contains('piscina').astype(bool)
    data['parking'] = data['extra_info'].str.contains('piscina') | data['descripcion'].str.contains('plaza de garaje') | data['descripcion'].str.contains('plaza garaje').astype(bool)
    data['estudiantes'] = data['descripcion'].str.contains('estudiante') | data['descripcion'].str.contains('escolar').astype(bool)
    data['playa'] = data['descripcion'].str.contains('playa').astype(bool)
    data['balcon'] =  data['descripcion'].str.contains('balcon') | data['descripcion'].str.contains('terraza').astype(bool)
    data['trastero'] = data['descripcion'].str.contains('trastero').astype(bool)
    data['vistas'] = data['descripcion'].str.contains('vistas').astype(bool)
    data['sin_ascensor'] = data['descripcion'].str.contains('sin ascensor') | data['extra_info'].str.contains('sin ascensor').astype(bool)
    data['profesores'] = data['descripcion'].str.contains('profesor').astype(bool)
    data['amueblado'] = data['descripcion'].str.contains('amueblad').astype(bool)
    data['es_casa'] = data['titulo'].apply(lambda x : 1.0 if(x.lower().startswith('casa') or x.lower().startswith('chalet')) else 0.0)
    data['lujo'] = data['descripcion'].str.contains('lujo') | data['titulo'].str.contains('lujo')
    data['urbanizacion'] = data['descripcion'].str.contains('urbanizacion')
    data['reformado'] = data['descripcion'].str.contains('reformado') | data['titulo'].str.contains('reformado')
    data['seminuevo'] = data['descripcion'].str.contains('seminuevo') | data['titulo'].str.contains('seminuevo')
    data['estrenar'] = data['descripcion'].str.contains('estrenar') | data['titulo'].str.contains('estrenar') | data['descripcion'].str.contains('totalmente nuevo')
    data['reformado_estrenar'] = data['reformado'] + data['estrenar']
    data['reformado_estrenar_seminuevo_amueblado'] = data['reformado'] + data['estrenar'] + data['amueblado'] + data['seminuevo']
    data['ubicacion_excepcional'] = data['descripcion'].str.contains('ubicacion excepcional') | data['descripcion'].str.contains('buena situacion') | data['descripcion'].str.contains('buena zona') | data['descripcion'].str.contains('zona privilegiada') | data['descripcion'].str.contains('entorno unico') | data['descripcion'].str.contains('conexion perfecta') | data['descripcion'].str.contains('bien comunicado')
    data['duplex'] = data['descripcion'].str.contains('duplex') | data['titulo'].str.contains('duplex')
    data['atico'] = data['descripcion'].str.contains('atico') | data['titulo'].str.contains('atico') | data['descripcion'].str.contains('ático') | data['titulo'].str.contains('ático')
    data['villa'] = data['descripcion'].str.contains('villa') | data['titulo'].str.contains('villa')
    data['climalit'] = data['descripcion'].str.contains('climalit')
    data['aire_acondicionado'] = data['descripcion'].str.contains('aire acondicionado')
    data['sin_muebles'] = data['descripcion'].str.contains('sin muebles') | data['descripcion'].str.contains('sin amueblar')
    data['electrico'] = data['descripcion'].str.contains('electrico')
    data['transporte'] = data['descripcion'].str.contains('estacion') | data['descripcion'].str.contains('parada')
    data['gastos_incluidos'] = data['descripcion'].str.contains('gastos') & data['descripcion'].str.contains('incluidos')
    data['soleado'] = data['descripcion'].str.contains('soleado') | data['descripcion'].str.contains('luminoso') | data['descripcion'].str.contains('calido') | data['descripcion'].str.contains('orientacion sur') | data['descripcion'].str.contains('ilumin')
    data['paseo_maritimo'] = data['descripcion'].str.contains('paseo maritimo')
    data['equipado'] = data['descripcion'].str.contains('equipad')
    data['jardin'] = data['descripcion'].str.contains('jardin')
    data['exclusivo'] = data['descripcion'].str.contains('exclusiv')
    data['spa'] = data['descripcion'].str.contains('spa') | data['descripcion'].str.contains('jacuzzi')
    data['es_chalet'] = data['titulo'].str.contains('chalet') | data['descripcion'].str.contains('chalet')
    data['padel'] = data['descripcion'].str.contains('padel')
    data['gym'] = data['descripcion'].str.contains('gym')

    data['de_diseño'] = data['descripcion'].str.contains('de diseño')
    # end boolean columns
    add_cols = set(data.columns) - so_far_cols
    for col in add_cols:
        data[col] = data[col].astype('bool')

    data['puntuacion'] = np.sum(data[valoracion_cols_1], axis=1)
    data['puntuacion'] += np.sum(2*data[valoracion_cols_2], axis=1)
    data['puntuacion'] += np.sum(3*data[valoracion_cols_3], axis=1)
    data['puntuacion'] += np.sum(-3*data[valoracion_cols_3_neg], axis=1)
    data['puntuacion_lujo'] = data['descripcion'].apply(lambda x: contar_palabras_en_texto(x, caros_cols_1))
    data['puntuacion_lujo_2'] = data['descripcion'].apply(lambda x: contar_palabras_en_texto(x, caros_cols_1))
    data['puntuacion_lujo_2'] += data['descripcion'].apply(lambda x: contar_palabras_en_texto(x, caros_cols_2))
    data['puntuacion_lujo_2'] += data['descripcion'].apply(lambda x: contar_palabras_en_texto(x, caros_cols_3))
    data['puntuacion_lujo_2'] += data['descripcion'].apply(lambda x: contar_palabras_en_texto(x, caros_adjetivos))
    data['palabras_bonitas'] = data['descripcion'].apply(lambda x: contar_palabras_en_texto(x, adjetivos))

    data['coruna'] = data["municipio"].apply(lambda x: 1.0 if 'coru' in x else 0.0)
    data['centro_coruna'] = data.apply(lambda x: is_centro_coruna(x['zona_v2']), axis=1)
    data['oleiros'] = data["municipio"].apply(lambda x: 1.0 if 'oleiros' in x else 0.0)
    data['maianca'] = data["zona"].apply(lambda x: 1.0 if 'maianca' in x else 0.0)
    data['art_berg_camb'] = data["municipio"].apply(lambda x: 1.0 if ('artei' in x or 'berg' in x or 'cambr' in x) else 0.0)

    # cols * tam
    data['puntuacion*tam'] = data['puntuacion'] * data['tamano']
    data['puntuacion_lujo*tam'] = data['puntuacion_lujo'] * data['tamano']
    data['puntuacion_lujo_2*tam'] = data['puntuacion_lujo_2'] * data['tamano']
    data['palabras_bonitas*tam'] = data['palabras_bonitas'] * data['tamano']
    data['es_chalet*tam'] = data['es_chalet'] * data['tamano']
    data['estrenar*tam'] = data['estrenar'] * data['tamano']
    data['reformado_estrenar_seminuevo_amueblado*tam'] = data['reformado_estrenar_seminuevo_amueblado'] * data['tamano']
    data['amueblado*tam'] = data['amueblado'] * data['tamano']
    data['estudiantes*tam'] = data['estudiantes'] * data['tamano']
    data['vacacional*tam'] = data['vacacional'] * data['tamano']
    data['exclusivo*tam'] = data['exclusivo'] * data['tamano']
    data['piscina*tam'] = data['piscina'] * data['tamano']
    data['sin_ascensor*tam'] = data['sin_ascensor'] * data['tamano']
    data['playa'] = data['playa'] * data['tamano']
    data['vistas'] = data['vistas'] * data['tamano']

    data['oleiros*tam'] = data['oleiros'] * data['tamano']
    data['maianca*tam'] = data['maianca'] * data['tamano']
    data['art_berg_camb*tam'] = data['art_berg_camb'] * data['tamano']
    data['coruna*tam'] = data['coruna'] * data['tamano']
    data['centro_coruna*tam'] = data['centro_coruna'] * data['tamano']
    data['precio_m2_medio_cluster*tam'] = data['precio_m2_medio_cluster'] * data['tamano']

    # polynomial features
    data['tamano^2'] = data['tamano']**2

    return data

def get_data_model_v2(k_clusters=None):
    ds = get_data_model_v1(k_clusters)
    cols = ds.columns
    cols_exclude = ['id', 'titulo', 'descripcion', 'extra_info', 'direccion', 'landmarks_cercanos', 'municipio', 'zona', 'zona_v2', 'direccion_x', 'direccion_y']
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
    x = ds[cols_model].drop('precio', axis=1)
    y = ds['precio']
    corr = abs(x.corrwith(y)).sort_values(ascending=False)
    # get cols whose corr is nan
    cols_corr_nan = corr[corr.isna()].index.tolist()
    for col in cols_corr_nan:
        cols_exclude.append(col)
        if col in cols_model:
            cols_model.remove(col)
    
    for col in cols_check_nas:
        ds[col] = ds[col].fillna(np.mean(ds[col]))

    return ds, cols_model
