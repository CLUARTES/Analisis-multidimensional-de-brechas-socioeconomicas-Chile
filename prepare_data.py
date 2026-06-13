import pandas as pd
import os

data_dir = '.'
out_dir = './data'

print("Cargando CASEN 2024...")
d24 = pd.read_parquet(os.path.join(data_dir, 'casen_2024.parquet'))

print("Generando datos Mapa y KPIs...")
df_mapa = d24[['region', 'area', 'ypchtotcor', 'pobreza', 'expr', 'edad', 'activ']]
df_mapa.to_parquet(os.path.join(out_dir, 'casen_mapa.parquet'))

print("Generando datos Ingreso por Educación...")
df_ing = d24[(d24['edad'] >= 15) & (d24['educc'].between(0, 6))][['region', 'area', 'edad', 'educc', 'sexo', 'ypchtotcor']]
df_ing.to_parquet(os.path.join(out_dir, 'casen_ingreso_edu.parquet'))

print("Generando datos Empleo por Educación...")
df_emp = d24[d24['edad'] >= 15][['region', 'area', 'edad', 'educc', 'sexo', 'activ']]
df_emp.to_parquet(os.path.join(out_dir, 'casen_empleo_edu.parquet'))

print("Generando datos Evolución 2024...")
df_evo = d24[['region', 'area', 'ypchtotcor', 'expr']]
df_evo.to_parquet(os.path.join(out_dir, 'casen_evolucion_24.parquet'))

print("Generando datos Brecha Sector...")
df_sec = d24[d24['activ'] == 1][['region', 'area', 'sexo', 'ypchtotcor', 'rama1']]
df_sec.to_parquet(os.path.join(out_dir, 'casen_brecha_sector.parquet'))

print("Generando datos Brecha Oficio...")
df_ofi = d24[d24['activ'] == 1][['region', 'area', 'sexo', 'ypchtotcor', 'oficio1_08']]
df_ofi.to_parquet(os.path.join(out_dir, 'casen_brecha_oficio.parquet'))

print("Cargando CASEN 2020...")
d20 = pd.read_parquet(os.path.join(data_dir, 'casen_2020.parquet'), columns=['region', 'ypchtotcor', 'expr'])
d20.dropna(subset=['ypchtotcor', 'expr']).to_parquet(os.path.join(out_dir, 'casen_evolucion_20.parquet'))

print("Cargando CASEN 2022...")
try:
    d22 = pd.read_parquet(os.path.join(data_dir, 'casen_2022.parquet'), columns=['region', 'ypchtotcor', 'expr'])
except ValueError:
    d22 = pd.read_parquet(os.path.join(data_dir, 'casen_2022.parquet'), columns=['ypchtotcor', 'expr'])
d22.dropna(subset=['ypchtotcor', 'expr']).to_parquet(os.path.join(out_dir, 'casen_evolucion_22.parquet'))

print("¡Listo!")
